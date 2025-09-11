import os
import sys
import time
import traceback
import faulthandler
from typing import Iterator
from multiprocessing import Process, Event, JoinableQueue
from queue import Empty
import pandas as pd
from ...utils import LogRedirector
from ...general_params import watcher_config as config
from .model_loaders import build_watcher, build_interpreter
from .generation import queued_monte_carlo, make_sim_dataframe


LOGDIR = "watcher_orchestration_logs"
TENSOR_PROD_LOGDIR = "tensor_producer_logs"
POSTPROC_LOGDIR = "postprocess_logs"
TASK_GEN_LOGDIR = "task_generator_logs"
SENTINEL = "sentinel"
TIMEOUT = 600


class WatcherOrchestrator:
    """Start processes that orchestrate autoregressive inference."""

    def __init__(
        self,
        task_generator: Iterator,
        log_dir: str,
        blueprint: str,
        gpu_ids: list[str],
        max_batch_size: int = 256,
        stride: int = 64,
        stop_vocab: list[int] = None,
        max_length: int = 5000,
        return_generated_parts_only: bool = True,
        return_unfinished: bool = False,
        compile_model: bool = False,
        temperature: float = 1.0,
    ) -> None:
        """
        Initializes the orchestrator.
        Args:
            task_generator (Iterator): Iterator object (generator or list) that yields:
                timeline, catalog_ids, n_iter, eval_time, horizon_td, product_id.
            log_dir (str): Directory for saving log files.
            blueprint (str): Path to a model's blueprint.
            gpu_ids (list[str]): List of GPU UUIDs to be used.
            max_batch_size (int): Maximum batch size.
            stride (int): Stride of sliding window during autoregressive inference beyond the max sequence length.
            stop_vocab (list[int], optional): Vocabularies given in catalog indexes to stop generation.
            max_length (int, optional): Maximum length of generated timelines.
            return_generated_parts_only (bool, optional): If true, only the generated part of timelines are returned with demographic heads.
            return_unfinished (bool, optional): If true, unfinished timelines are returned when they reach the max length (max_length).
            compile_model (bool, optional): If true, torch.compile() is applied.
            temperature (float, optional): Model's temperature parameter.
        """
        # Set attributes
        self.task_generator = task_generator
        self.log_dir = log_dir
        self.blueprint = blueprint
        if not os.path.exists(blueprint):
            raise ValueError(f"{blueprint} ('blueprint') does not exist.")
        self.gpu_ids = gpu_ids
        self.max_batch_size = max_batch_size
        self.mc_kwargs = {
            "stop_vocab": stop_vocab,
            "max_length": max_length,
            "stride": stride,
            "return_generated_parts_only": return_generated_parts_only,
            "return_unfinished": return_unfinished,
            "temperature": temperature,
            "compile_model": compile_model,
            "logits_filter": "default",
        }
        self.started_proc = []

        # Initialize other items
        self.task_q = None
        self.intermediate_product_q = None
        self.final_product_q = None
        self.end_signal = None

    def start(self):
        """Enters the context manager, redirecting stdout and stderr to the log file."""
        # multiprocessing.set_start_method("spawn")
        # multiprocessing.set_start_method("spawn")
        # Initialize queues
        self.task_q = JoinableQueue()
        self.intermediate_product_q = JoinableQueue()
        self.final_product_q = JoinableQueue()
        # Initialize signals
        self.end_signal = Event()

        # Create producer processes
        # NOTE: Each producer puts a sentinel (SENTINE) to the intermediate product queue.
        gpu_users = []
        for gpu_id in self.gpu_ids:
            p = Process(
                target=_generate_timelines,
                kwargs={
                    "log_dir": self.log_dir,
                    "blueprint": self.blueprint,
                    "gpu_id": gpu_id,
                    "task_q": self.task_q,
                    "intermediate_product_q": self.intermediate_product_q,
                    "mc_kwargs": self.mc_kwargs,
                    "end_signal": self.end_signal,
                },
                # NOTE: If you compile the model, set daemon=False here, because torch.complie() creates child processes.
                #       Daemonic processes are not allowed to have children.
                daemon=False,
                name=f"watcher_orch_tensor_producer_{gpu_id}",
            )
            gpu_users.append(p)
        # Create task generator process
        task_gen_p = Process(
            target=_generate_tasks,
            kwargs={
                "log_dir": self.log_dir,
                "task_generator": self.task_generator,
                "task_q": self.task_q,
                "max_batch_size": self.max_batch_size,
                "n_consumers": len(self.gpu_ids),
                "end_signal": self.end_signal,
            },
            daemon=True,
            name="watcher_orch_task_generator_worker",
        )

        # Create post-processing processe
        postproc_p = Process(
            target=_postprocess_timelines,
            kwargs={
                "log_dir": self.log_dir,
                "blueprint": self.blueprint,
                "intermediate_product_q": self.intermediate_product_q,
                "final_product_q": self.final_product_q,
                "n_producers": len(gpu_users),
                "end_signal": self.end_signal,
            },
            daemon=True,
            name="watcher_orch_postprocessing_worker",
        )

        # Start processes
        task_gen_p.start()
        self.started_proc.append(task_gen_p)
        for p in gpu_users:
            p.start()
            self.started_proc.append(p)
        postproc_p.start()
        self.started_proc.append(postproc_p)

    def terminate(self):
        """Terminate all child processes."""
        # Kill started child processes
        try:
            while self.started_proc:
                proc = self.started_proc.pop(0)
                proc.kill()
                del proc
        finally:
            self.started_proc = []

    def _ensure_closing(self):
        """Ensures that all the child processes are stopped."""
        print("Ensuring all other processes finishes ")
        if not self.end_signal.is_set():
            self.end_signal.set()
        time.sleep(3)
        for _ in range(100):
            n_active = 0
            for p in self.started_proc:
                if p.exitcode is None:
                    print(f"{p.name} is still active.")
                    n_active += 1
            time.sleep(1)
            if n_active == 0:
                print("Child processes stopped.")
                break
        else:
            print("Child processes hang too long. They are forcefully terminated.")
            print("Process details:")
            for p in self.started_proc:
                print(p)
                print(p.exitcode)
            self.terminate()

    def _watch_processes(self) -> bool:
        """Check status of the child processes."""
        for p in self.started_proc:
            if p.exitcode is not None:
                print("A child process stopped.")
                break
        # All processes working
        else:
            return False

        # Abnormal stop
        for p in self.started_proc:
            print(p)
            print("Exitcode:", p.exitcode)
        self._ensure_closing()
        return True

    def __iter__(self):
        """Iterates products.
        The order of the returned products is not guranteed to be the same as that of the tasks.
        Therefore, the product (dataframe) is returned together with a product id.
        If the product queue is empty, this method always checks is the child processes are ok to prevent hanging.
        """
        processes_closed = False
        while not processes_closed:
            try:
                prod = self.final_product_q.get(timeout=60)
                self.final_product_q.task_done()

            # Check child process status if the queue is empty
            except Empty:
                processes_closed = self._watch_processes()
                if not processes_closed:
                    continue
                else:
                    raise RuntimeError(
                        "Child processes finished abnornally during generation."
                    )

            # Yield
            if prod != SENTINEL:
                df, product_id = prod
                yield df, product_id

            # End of iteration
            else:
                print("Orchestrator reached to the end.")
                self.end_signal.set()
                print("Wait for child processes to stop.")
                # Check for other processes to finish
                self._ensure_closing()
                processes_closed = True
                # Exit the main loop
                break


def _generate_tasks(
    log_dir: str | None,
    task_generator: Iterator,
    task_q: JoinableQueue,
    max_batch_size: int,
    n_consumers: int,
    end_signal: Event,
):
    """Core function for the task generator process."""
    # Log config
    if log_dir is not None:
        log_dir = os.path.join(log_dir, LOGDIR, TASK_GEN_LOGDIR)
        pid = os.getpid()
        file_name = f"log_task_generator_{pid}.txt"
    else:
        file_name = None

    with LogRedirector(log_dir=log_dir, file_name=file_name):
        try:
            # Task loading loop
            for t in task_generator:
                (
                    timeline,
                    catalog_ids,
                    n_repeat,
                    horizon_start,
                    time_horizon,
                    product_id,
                ) = t
                # Check batch size
                if n_repeat > max_batch_size:
                    print(
                        """Warning! `n_repeat` is larger than `max_batch_size`. 
                        `n_repeat` is decreased to `max_batch_size`."""
                    )
                    n_repeat = max_batch_size
                task_q.put(
                    (
                        timeline,
                        catalog_ids,
                        n_repeat,
                        horizon_start,
                        time_horizon,
                        product_id,
                    ),
                    # Set timeout in case the queue crashes
                    timeout=TIMEOUT,
                )
                # Wait if the task queue is full
                while task_q.qsize() >= n_consumers * 10:
                    if end_signal.is_set():
                        break
                    else:
                        time.sleep(1)

                # Check for end signal
                if end_signal.is_set():
                    print("An error event detected. Finishing...")
                    break

            if not end_signal.is_set():
                # At the end, put sentinel values as many as consumers so that all consumers can get one.
                for _ in range(n_consumers):
                    task_q.put(SENTINEL, timeout=TIMEOUT)
                # Join
                # NOTE: Another option to join the queue (task_q.join())
                print("aiting for other processes to stop...")
                end_signal.wait()
                print("Task generation finished.")

        except Exception as e:
            end_signal.set()
            print("**********************")
            print("Error duting the task generation:")
            print(e)
            traceback.print_exc()
            print("**********************")
            sys.exit(1)


def _postprocess_timelines(
    log_dir: str | None,
    blueprint: str,
    intermediate_product_q: JoinableQueue,
    final_product_q: JoinableQueue,
    n_producers: int,
    end_signal: Event,
):
    """Collects finished products from the queue, and postprocess them."""
    # TODO (Yu Akagi): Make this function parallel if this can be a bottleneck.

    # Log config
    if log_dir is not None:
        log_dir = os.path.join(log_dir, LOGDIR, POSTPROC_LOGDIR)
        pid = os.getpid()
        file_name = f"log_postprocessing_{pid}.txt"
    else:
        file_name = None
    with LogRedirector(log_dir=log_dir, file_name=file_name):
        print("postprocess started")
        try:
            # Create an interpreter
            interpreter = build_interpreter(blueprint)
            # Get the products and postprocess them
            n_sentinel = 0
            while True:
                if not end_signal.is_set():
                    prod = intermediate_product_q.get()
                    intermediate_product_q.task_done()
                    # Checking sentinel
                    if prod == SENTINEL:
                        n_sentinel += 1
                        if n_sentinel == n_producers:
                            break
                    # Collect products
                    else:
                        prod_timelines, prod_catalog_ids, product_id = prod
                        df = make_sim_dataframe(
                            interpreter=interpreter,
                            prod_timelines=prod_timelines,
                            prod_catalog_ids=prod_catalog_ids,
                        )
                        # Put the final product in the queue
                        final_product_q.put((df, product_id), timeout=TIMEOUT)

                # Finish once the end singnal is detected.
                else:
                    break

            if not end_signal.is_set():
                # At the end, put sentinel values as many as consumers so that all consumers can get one.
                # NOTE: Currently, the number of consumer is just one.
                final_product_q.put(SENTINEL, timeout=TIMEOUT)
                # Join
                print("waiting for other processes to stop...")
                end_signal.wait()
                print("Postporcessing process finished.")

        except Exception as e:
            end_signal.set()
            print("**********************")
            print("Error duting postprocessing:")
            print(e)
            traceback.print_exc()
            print("**********************")
            sys.exit(1)


def _generate_timelines(
    log_dir: str | None,
    blueprint: str,
    gpu_id: str,
    task_q: JoinableQueue,
    intermediate_product_q: JoinableQueue,
    mc_kwargs: dict,
    end_signal: Event,
):
    """
    Run a process that performs inference.
    This process tries to use KV-cache when possible.
    """
    # NOTE: Do not exclude rows outside the test-val period, especially the ones before the period,
    #       to simulate the actual patient care.

    # Log config
    if log_dir is not None:
        log_dir = os.path.join(log_dir, LOGDIR, TENSOR_PROD_LOGDIR)
        pid = os.getpid()
        file_name = f"log_tensor_producer_{pid}.txt"
    else:
        file_name = None
    with LogRedirector(log_dir=log_dir, file_name=file_name):
        faulthandler.enable()
        try:
            # GPU and model settings
            # NOTE: device_id here is device's UUID
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            model = build_watcher(blueprint, train=False).cuda()

            # ********************
            # * Monte Carlo loop *
            # ********************
            queued_monte_carlo(
                task_queue=task_q,
                product_queue=intermediate_product_q,
                sentinel=SENTINEL,
                model=model,
                end_signal=end_signal,
                **mc_kwargs,
            )

            if not end_signal.is_set():
                # At the end, put sentinel values as many as consumers so that all consumers can get one.
                intermediate_product_q.put(SENTINEL)
                # join
                print("waiting for other processes to finish...")
                end_signal.wait()
                print("all done")

        except Exception as e:
            end_signal.set()
            print("**********************")
            print("Error during tensor generation:")
            print(e)
            traceback.print_exc()
            print("**********************")
            sys.exit(1)
