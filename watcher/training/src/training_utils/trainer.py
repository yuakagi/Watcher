"""The main trainer class"""

import os
import socket
import time
import json
import shutil
import traceback
import math
from typing import Optional
from datetime import datetime
from tqdm import tqdm
import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from .batch_scheduler import BatchScheduler
from .criteria import WatcherLoss
from ..datasets import TimelineDataset
from ....models import Watcher
from ....general_params import watcher_config as config
from ....general_params import get_settings

TIME_FORMAT_READABLE: str = "%Y/%m/%d %H:%M:%S"
TIME_FORMAT_STR: str = "%Y%m%d%H%M%S"


# TODO: Deprecate
def custom_trace_handler(
    dir_name: str,
    rank: int,
    record_memory_timeline: bool,
    worker_name: Optional[str] = None,
    use_gzip: bool = False,
):
    """
    ** Modified from the pytorch source code. **
    This custom trace handler exports memory timeline.
    """
    main_folder = os.path.join(dir_name, "watcher")
    sub_folder = os.path.join(dir_name, "etc")
    for d in [main_folder, sub_folder]:
        if not os.path.isdir(d):
            try:
                os.makedirs(d, exist_ok=True)
            except Exception as e:
                raise RuntimeError("Can't create directory: " + d) from e

    def handler_fn(prof) -> None:
        nonlocal worker_name
        if not worker_name:
            worker_name = f"{socket.gethostname()}_{os.getpid()}"
        # Use nanosecond here to avoid naming clash when exporting the trace
        file_name = f"main_profiles_{worker_name}.{time.time_ns()}.pt.trace.json"
        if use_gzip:
            file_name = file_name + ".gz"
        prof.export_chrome_trace(os.path.join(main_folder, file_name))
        # Construct the memory timeline file.
        # Export the memory logs only from rank0
        if rank == 0:
            if record_memory_timeline:
                file_name = f"memory_timeline_{worker_name}.{time.time_ns()}.html"
                prof.export_memory_timeline(
                    os.path.join(sub_folder, file_name), device="cuda:0"
                )

    return handler_fn


class LogitsTrainer:
    """The trainer class to train a Watcher model"""

    def __init__(
        self,
        rank: int,
        world_size: int,
        training_report: dict,
        update: bool = False,
        snapshot_path: str = None,
        initial_weight_path: str = None,
        criteria: torch.nn.Module | None = None,
        dataset_class: torch.utils.data.Dataset = None,
        record_memory_timeline: bool = True,
    ) -> None:
        # Define training settings
        training_config = training_report["training_config"]
        self.gpu_id = rank
        self.device = f"cuda:{rank}"
        self.world_size = world_size
        self.training_report = training_report
        self.save_every = training_config["checkpoint_interval"]
        self.validate_every = training_config["validation_interval"]
        self.early_stopping_epochs = training_config["early_stopping_epochs"]
        self.initiated_time = training_report["initiated_time"]
        self.ignore_index = training_report["etc"]["ignored_index"]
        if criteria is None:
            self.criteria = WatcherLoss(
                ignore_index=self.ignore_index,
                # Focal loss gamma (fall back to CE if gamma == 0.0)
                gamma=0.0,
                # Scaling
                scaled=False,
            )
        else:
            self.criteria = criteria
        # Define paths
        self.active_tb_log_dir = get_settings("ACTIVE_TENSORBOARD_DIR")
        self.training_dir = get_settings("TRAINING_DIR")
        self.snapshots_dir = get_settings("SNAPSHOTS_DIR")
        self.profiling_dir = get_settings("PROFILING_DIR")
        self.main_training_report_path = get_settings("MAIN_TRAINING_REPORT_PTH")
        # Initialize variables
        self.n_step = 0
        self.start_epoch = 0
        self.last_epoch_steps = 0
        self.cumulative_loss = torch.tensor(0.0).to(self.device)
        self.last_avg_loss = None
        self.prof = None
        # Instantiate objects
        self.model = Watcher(
            **training_report["hyperparameters"],
            catalogs_dir=get_settings("CATALOGS_DIR"),
            lab_stats_dir=get_settings("LAB_STATS_DIR"),
        )
        if initial_weight_path is not None:
            print("Loading initial weights...")
            # Initialize model with pretrained weights
            pretrained_checkpoint = torch.load(initial_weight_path)
            self.model.load_state_dict(pretrained_checkpoint)
        self.model = self.model.to(rank)
        self.optimizer = self._load_optimizer(training_report, self.model)
        self.scaler = GradScaler(enabled=True)
        self.writer = SummaryWriter(
            log_dir=self.active_tb_log_dir, filename_suffix=f"rank{rank}"
        )

        # Prepare data loaders
        train_loader = self._prepare_dataloader(
            training_report,
            self.model,
            train=True,
            update=update,
            dataset_class=dataset_class,
        )
        self.val_loader = self._prepare_dataloader(
            training_report,
            self.model,
            train=False,
            update=update,
            dataset_class=dataset_class,
        )
        # Wrap the training dataloader with batchsize scheduler
        self.batch_scheduler = BatchScheduler(
            dataloader=train_loader,
            world_size=world_size,
            total_epochs=self.training_report["training_config"]["total_epochs"],
            step_scale=self.training_report["training_config"]["batch_size_step_scale"],
            active_phase=self.training_report["training_config"][
                "batch_size_active_phase"
            ],
            min_batch_size=self.training_report["training_config"]["min_batch_size"],
            enabled=self.training_report["training_config"]["batch_schedule_enabled"],
        )
        # Scheduler
        # NOTE: Estimated total steps is influenced by the batch size schedule. Therefore, it is calculated by BatchScheduler.
        total_steps = self.batch_scheduler.estimated_total_steps
        self.training_report["estimated_total_steps"] = total_steps
        self.scheduler = self._load_scheduler(
            training_report,
            self.optimizer,
            self.batch_scheduler,
            enabled=self.training_report["training_config"]["lr_scheduler_enabled"],
        )

        # Clear tensorboard log
        if rank == 0:
            for file in os.listdir(self.active_tb_log_dir):
                file_path = os.path.join(self.active_tb_log_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

        # Restore the last training state
        if snapshot_path is not None:
            self._restore_from_snapshot(snapshot_path)

        # Initialize steps for a new training
        else:
            prof_log_dir = self.profiling_dir
            prof_wait = 3
            prof_warmup = 3
            prof_active = 10
            prof_repeat = 1
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    skip_first=10,
                    wait=prof_wait,
                    warmup=prof_warmup,
                    active=prof_active,
                    repeat=prof_repeat,
                ),
                on_trace_ready=custom_trace_handler(
                    dir_name=prof_log_dir,
                    rank=rank,
                    record_memory_timeline=record_memory_timeline,
                ),
                # NOTE: record_shapes increases reference to tensor elements, which may result in undesired profiling results.
                record_shapes=record_memory_timeline,
                with_stack=record_memory_timeline,
                profile_memory=True,
            )

        # DDP wrapping
        # NOTE: This part ensures that DDP instantiates on rank 0 first.
        self.model = DDP(
            self.model,
            device_ids=[rank],
            gradient_as_bucket_view=True,
        )
        # Save initial training state
        if (rank == 0) and (self.start_epoch == 0):
            self._take_snapshot(epoch=-1, name="initial")
            print("The initial model state was saved by rank 0.")

    def _prepare_dataloader(
        self,
        training_report: dict,
        model: torch.nn.Module,
        train: bool,
        update: bool,
        dataset_class: torch.utils.data.Dataset = None,
    ) -> DataLoader:
        """Instantiates a dataloader class.

        Args:
            training_report (dict): Dictionary for training configurations.
                See 'initialize_training_report' for details.
            model (torch.nn.Module): Watcher instance
            train (bool): If true, the returned dataloader loads train dataset.
                Otherwise, validation dataset is loaded.
            update (bool): If true, the dataset for update finetuning is loaded instead of training dataset.
            dataset_class (Dataset): Custom dataset class.
                The default is none, and TimelineDataset is used.
                The custom dataset class must take the same arguments for instantiation as those
                with TimelineDataset.
        Returns:
            dataloader (DataLoader): Pytorch DataLoader instance.
        """
        # Load params
        train_config = training_report["training_config"]
        max_batch_size = train_config["max_batch_size"]
        num_workers = train_config["dataloader_workers"]
        # Determine paths
        if train:
            # Train dataset
            if not update:
                data_dir = get_settings("TENSORS_TRAIN_DIR")
            # Update dataset
            else:
                data_dir = get_settings("TENSORS_UPDATE_TRAIN_DIR")

        # Validation dataset
        else:
            if update:
                data_dir = get_settings("TENSORS_UPDATE_VAL_DIR")
            else:
                data_dir = get_settings("TENSORS_VAL_DIR")

        # Instantiate
        if dataset_class is None:
            dataset_class = TimelineDataset
        dataset = dataset_class(
            data_dir=data_dir,
            model=model,
            train=train,
        )
        sampler = DistributedSampler(
            dataset=dataset, shuffle=True, drop_last=True, rank=self.gpu_id
        )
        dataloader = DataLoader(
            dataset,
            batch_size=max_batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            sampler=sampler,
        )

        return dataloader

    def _load_optimizer(
        self, training_report: dict, model: torch.nn.Module
    ) -> Optimizer:
        """Instantiate an optimizer"""
        # Load params
        train_config = training_report["training_config"]
        weight_decay = train_config["weight_decay"]
        max_lr = train_config["max_lr"]
        # Weight decay is not applied to biases and layer norm weights
        no_decay = ["bias", "LayerNorm.weight", "norm"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        # Instantiate
        optimizer = AdamW(optimizer_grouped_parameters, lr=max_lr)

        return optimizer

    def _load_scheduler(
        self,
        training_report: dict,
        optimizer: Optimizer,
        batch_scheduler: BatchScheduler,
        enabled: bool = True,
    ) -> LRScheduler:
        """Instantiate a learning-rate scheduler"""
        # Load params
        total_steps = training_report["estimated_total_steps"]
        train_config = training_report["training_config"]
        lr_warmup = train_config["lr_warmup"]
        anealing_ratio = train_config["lr_decay"]
        max_lr = float(train_config["max_lr"])
        min_lr = float(train_config["min_lr"])
        min_lr_ratio = min_lr / max_lr
        warmup_steps = batch_scheduler.estimate_n_steps(
            lr_warmup
        )  # int(total_steps * lr_warmup)
        anealing_steps = batch_scheduler.estimate_n_steps(
            anealing_ratio
        )  # int(total_steps * anealing_ratio)
        coeff = (1 - min_lr_ratio) / 2
        const = (1 + min_lr_ratio) / 2

        # Disable scheduler to always get the minimum learning rate
        if not enabled:
            warmup_steps = 0
            anealing_steps = 0

        def lr_lambda(current_step: int):
            # Linear warmup
            if current_step < warmup_steps:
                ratio = float(current_step) / float(max(1, warmup_steps))

            # Cosine anealing
            elif current_step < anealing_steps:
                progress = float(current_step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                ratio = coeff * math.cos(math.pi * progress) + const
                ratio = max(min_lr_ratio, ratio)

            # Learning rate plateaued
            else:
                ratio = min_lr_ratio

            return ratio

        # Instantiate
        scheduler = LambdaLR(optimizer, lr_lambda)

        return scheduler

    def _run_batch(self, batch: torch.Tensor, scaling_factors: torch.Tensor):

        # Step
        self.optimizer.zero_grad(set_to_none=True)
        timeline_tensor = batch[:, :, :-1]
        labels = batch[:, :, -1].long()
        scaling_factors = scaling_factors.view(-1, 1).float()
        # Compute logits (autocast is inside .forward())
        logits = self.model(timeline_tensor)
        # Scaled cross-entropy loss
        loss = self.criteria(logits, labels, scaling_factors)

        self.scaler.scale(loss).backward()

        # *** Gradient clipping ***
        # TODO: Consider adding control over gradient clipping params via args
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0, norm_type=2
        )
        # *************************
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.cumulative_loss += loss
        self.scheduler.step()

        # Record loss every 50 steps
        if self.n_step % 50 == 0:
            # Log other training variables
            if self.gpu_id == 0:
                self.writer.add_scalar(
                    f"Loss on cuda:{self.gpu_id}", loss.item(), self.n_step
                )
                last_lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar("Learning rate", last_lr, self.n_step)
                self.writer.add_scalar(
                    "Batch size", self.batch_scheduler.batch_size, self.n_step
                )
            # Detect abnormal surge in loss
            if self.last_avg_loss is not None:
                if loss.item() >= (self.last_avg_loss + config.LOSS_SURGE_THRESHOLD):
                    raise RuntimeError(
                        f"Abrupt loss surge detected on rank {self.gpu_id}. Restarting training from the last check point."
                    )

    def _run_epoch(self, epoch: int):
        # Shuffle and resampling
        self.batch_scheduler.dataloader.sampler.set_epoch(epoch)
        self.batch_scheduler.dataloader.dataset.resample_data(epoch)
        self.val_loader.dataset.resample_data(epoch)

        # The main loop
        for batch, scaling_factors in self.batch_scheduler:
            if self.prof is not None:
                self.prof.step()
            batch = batch.to(self.device)
            scaling_factors = scaling_factors.to(self.device)
            self._run_batch(batch, scaling_factors)
            self.n_step += 1

        # Calculate the average loss
        steps_in_epoch = self.n_step - self.last_epoch_steps
        average_loss = self.cumulative_loss / steps_in_epoch
        dist.barrier()
        dist.all_reduce(average_loss, dist.ReduceOp.SUM, async_op=False)
        average_loss /= self.world_size
        average_loss = average_loss.item()

        # Save the avarage loss
        if self.gpu_id == 0:
            train_avg_loss = {"Training": average_loss}
            self.writer.add_scalars("Average loss", train_avg_loss, global_step=epoch)

        # Set variables for the next epoch
        self.cumulative_loss = torch.tensor(0.0, device=self.device)
        self.last_epoch_steps = self.n_step
        self.last_avg_loss = average_loss

    def _run_val(self, epoch: int) -> float:
        # Set the model to evaluation mode
        self.model.eval()

        # Validation loop
        n_val_iter = 0
        total_val_loss = torch.tensor(0.0).to(self.device)
        with torch.no_grad():
            # Run validation
            for batch, scaling_factors in self.val_loader:
                batch = batch.to(self.device)
                scaling_factors = scaling_factors.to(self.device)
                timeline_tensor = batch[:, :, :-1]
                labels = batch[:, :, -1].long()
                scaling_factors = scaling_factors.view(-1, 1).float()
                logits = self.model(timeline_tensor)
                loss = self.criteria(logits, labels, scaling_factors)
                total_val_loss += loss
                n_val_iter += 1

            # Calculate the mean validation loss
            if n_val_iter > 0:
                val_loss_mean = total_val_loss / n_val_iter
            else:
                val_loss_mean = total_val_loss
            # Collect validation losses
            dist.barrier()
            dist.all_reduce(val_loss_mean, dist.ReduceOp.SUM, async_op=False)
            val_loss_mean /= self.world_size
            val_loss_mean = val_loss_mean.item()
            if self.gpu_id == 0:
                # Record the validation loss
                val_avg_loss = {"Validation": val_loss_mean}
                self.writer.add_scalars("Average loss", val_avg_loss, global_step=epoch)

        # Set the model back to train mode at the end of each validation
        self.model.train()

        return val_loss_mean

    def _take_snapshot(self, epoch: int, name: str | None = None):
        # Save checkpoint
        try:
            snapshot_time = datetime.now().strftime(TIME_FORMAT_STR)
            model_state = self.model.module.state_dict()
            training_state = {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "batch_scheduler": self.batch_scheduler.state_dict(),
                "epoch": epoch,
                "steps": self.n_step,
                "snapshot_time": snapshot_time,
                "average_loss": self.last_avg_loss,
            }

            # Determine paths to save results
            if name is not None:
                snp_tag = name
            else:
                snp_tag = f"epoch_{epoch}"
            snapshot_dir = os.path.join(self.snapshots_dir, snp_tag)
            blueprint_dir = os.path.join(snapshot_dir, config.DIR_BLUEPRINT)
            tb_log_dir = os.path.join(snapshot_dir, config.DIR_TENSORBOARD_LOGS)
            model_state_path = os.path.join(blueprint_dir, config.MODEL_STATE)
            training_report_path = os.path.join(blueprint_dir, config.TRAINING_REPORT)
            train_state_path = os.path.join(snapshot_dir, config.TRAINING_STATE)
            # Clear old files
            if os.path.exists(snapshot_dir):
                shutil.rmtree(snapshot_dir)
            # Create directories
            os.mkdir(snapshot_dir)
            os.mkdir(blueprint_dir)
            # Save the current state of training
            torch.save(model_state, model_state_path)
            torch.save(training_state, train_state_path)
            with open(training_report_path, "w", encoding="utf-8") as f:
                json.dump(self.training_report, f, indent=2)
            shutil.copytree(
                self.active_tb_log_dir,
                tb_log_dir,
            )
            # Copy catalogs and related files from the dateset into the blueprint
            # NOTE: Catalogs are copied to ensure that each blueprint has everything needed to instantiate a model.
            src_catalog_dir = get_settings("CATALOGS_DIR")
            src_lab_stats_dir = get_settings("LAB_STATS_DIR")
            dst_catalog_dir = os.path.join(blueprint_dir, "catalogs")
            dst_lab_stats_dir = os.path.join(blueprint_dir, config.DIR_LAB_STATS)
            shutil.copytree(src_catalog_dir, dst_catalog_dir)
            shutil.copytree(src_lab_stats_dir, dst_lab_stats_dir)

            # Wirte the last checkpoint path to the environment, once everything is successfully saved.
            self.training_report["previous_snapshot"] = os.path.abspath(snapshot_dir)

        except Exception as e:
            print(f"Failed to take the snapshot on epoch {epoch}: {e}")
            traceback.print_exc()
            raise

    def _restore_from_snapshot(self, snapshot_path: str):
        """Restores the previous training state using a snapshot"""
        # Load checkpoints
        model_state_path = os.path.join(
            snapshot_path, config.DIR_BLUEPRINT, config.MODEL_STATE
        )
        train_state_path = os.path.join(snapshot_path, config.TRAINING_STATE)
        # NOTE: Function 'build_wathcer()' is not called here to instantiate the model for better performance.
        model_state = torch.load(model_state_path, map_location={"cuda:0": self.device})
        training_state = torch.load(
            train_state_path, map_location={"cuda:0": self.device}
        )
        # Restore training progress
        self.last_avg_loss = training_state["average_loss"]
        last_epoch = training_state["epoch"]
        self.n_step = training_state["steps"]
        self.last_epoch_steps = training_state["steps"]
        self.start_epoch = last_epoch + 1
        self.scheduler.load_state_dict(training_state["scheduler"])
        self.optimizer.load_state_dict(training_state["optimizer"])
        self.batch_scheduler.load_state_dict(training_state["batch_scheduler"])
        self.scaler.load_state_dict(training_state["scaler"])
        # Restore model weights
        self.model.load_state_dict(model_state)
        # Restore tensorboard log
        if self.gpu_id == 0:
            last_tensorboard_log = os.path.join(
                snapshot_path,
                config.DIR_TENSORBOARD_LOGS,
            )
            shutil.rmtree(self.active_tb_log_dir)
            shutil.copytree(last_tensorboard_log, self.active_tb_log_dir)
        # Write the time of restarting to the report
        resumed_time = datetime.strftime(datetime.now(), format=TIME_FORMAT_STR)
        self.training_report["resumed_at"].append(resumed_time)
        # Skip profiling
        self.prof = None

    def train(self) -> str:
        """Main method for training
        Returns:
            best_model_state_path (str): Path to the best model weight.
        """
        # *****************
        # * Training loop *
        # *****************
        # Configs for the debug mode
        if self.gpu_id == 0:
            debug = bool(get_settings("DEBUG_MODE"))
            never_disruped = not bool(self.training_report["disrupted_at"])
            raise_error_for_debug = debug and never_disruped
        else:
            raise_error_for_debug = False

        # Ensure that the model is set to the train mode.
        self.model.train()
        # Set up variables
        total_epochs = self.training_report["training_config"]["total_epochs"]
        epochs = list(range(self.start_epoch, total_epochs))
        disable_pbar = not self.gpu_id == 0
        completed = False
        early_stop_flag = torch.tensor([0], device=self.device).long()
        # Start pytorch profiler
        if self.prof is not None:
            self.prof.start()
        try:
            with tqdm(epochs, disable=disable_pbar) as pbar:
                print(f"Rank {self.gpu_id} entered the training loop.")
                for epoch in pbar:
                    # Check for early stopping
                    dist.barrier()
                    dist.all_reduce(early_stop_flag, dist.ReduceOp.SUM, async_op=False)
                    if early_stop_flag.item() != 0:
                        print(f"Early stopping detected on rank {self.gpu_id}.")
                        break

                    # Update pbar
                    if self.gpu_id == 0:
                        epoch_desc = f"[Epoch {epoch}/{total_epochs-1}]"
                        pbar.set_description(epoch_desc)

                    # Run epoch
                    self._run_epoch(epoch)

                    # Run val
                    best_updated = False
                    if ((epoch + 1) % self.validate_every == 0) or (
                        epoch == total_epochs - 1
                    ):
                        mean_val_loss = self._run_val(epoch)
                        best_val_loss = self.training_report.get("best_val_loss")
                        # Check for early stopping
                        if best_val_loss is not None:
                            if mean_val_loss <= best_val_loss:
                                self.training_report["best_val_loss"] = mean_val_loss
                                self.training_report["best_epoch"] = epoch
                                best_updated = True
                            else:
                                if self.gpu_id == 0:
                                    best_epoch = self.training_report["best_epoch"]
                                    epochs_since_best = epoch - best_epoch
                                    if epochs_since_best >= self.early_stopping_epochs:
                                        # Early stopping
                                        print("Early stopping criteria met.")
                                        print(f"Stopping training at epoch {epoch}.")
                                        print(
                                            f"Lowest validation loss at epoch {best_epoch}."
                                        )
                                        # NOTE: This value is shared among other ranks by dist.all_reduce at the start of next epoch
                                        early_stop_flag[0] = 1

                        else:
                            best_updated = True
                            self.training_report["best_val_loss"] = mean_val_loss
                            self.training_report["best_epoch"] = epoch

                    # Save checkpoint
                    if (
                        ((epoch + 1) % self.save_every == 0)
                        or best_updated  # Every best val-loss
                        or (epoch == total_epochs - 1)  # Final epoch
                    ):
                        if self.gpu_id == 0:
                            self._take_snapshot(epoch)
                            if best_updated:
                                self.training_report["best_snapshot"] = (
                                    self.training_report["previous_snapshot"]
                                )

                    # Artificially raise an error for debug at the end of epoch 2 to test the training restoration
                    if raise_error_for_debug:
                        if epoch >= 2:
                            raise RuntimeError(
                                "*****Artificial error raised for debugging*****"
                            )

            # Write training completion status to the training report
            if self.gpu_id == 0:
                completed_time = datetime.strftime(
                    datetime.now(), format=TIME_FORMAT_READABLE
                )
                self.training_report["completed_at"] = completed_time
                self.training_report["final_state"] = "completed"
                print(f"Training completed at rank {self.gpu_id}.")

            completed = True

            # **********************************************************
            # * NOTE: Codes in the 'finally' clause are executed here. *
            # **********************************************************

        except Exception as e:
            print("Error during the training loop:")
            print(e)
            traceback.print_exc()
            # Write training disruption status to the training report
            if self.gpu_id == 0:
                disrupted_time = datetime.strftime(
                    datetime.now(), format=TIME_FORMAT_READABLE
                )
                self.training_report["disrupted_at"].append(disrupted_time)
                self.training_report["final_state"] = f"disrupted (details:{e})"

        finally:
            # Save the main training report
            if self.gpu_id == 0:
                with open(self.main_training_report_path, "w", encoding="utf-8") as f:
                    json.dump(self.training_report, f, indent=2)
            # Stop profiler
            if self.prof is not None:
                self.prof.stop()
            # Raise an error if training is not completed
            if not completed:
                raise RuntimeError("Training was disrupted.")
