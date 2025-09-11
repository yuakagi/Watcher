import json
import uuid
import time
import signal
import atexit
import traceback
from multiprocessing import Process, JoinableQueue, Manager
from multiprocessing.managers import DictProxy
from datetime import datetime, timedelta
import pandas as pd
import torch
from flask import Flask, make_response, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_limiter.errors import RateLimitExceeded
from ...models import (
    build_watcher,
    Watcher,
    WatcherOrchestrator,
)
from ...preprocess import preprocess_for_inference
from ...general_params import watcher_config as config

TIMEOUT = 60
EXPIRES_AFTER = 600
STATE_204 = {
    "status": 204,
    "progress": "Empty",
    "errors": ["Request not found"],
    "updated": datetime.now().isoformat(),
}

import atexit


@atexit.register
def cleanup_all():
    print("Shutting down on exit.")
    _graceful_shutdown("atexit", None)


# ==================
# Flask app setup
# ==================
# region: Flask apps
# Initialize Flask app
app = Flask(__name__)
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="redis://redis:6379",  # container name used as hostname
    app=app,
)
limiter.init_app(app)


# Rate-limit handler for Flask-limiter
@app.errorhandler(RateLimitExceeded)
def ratelimit_handler(e):
    """Handles rate limits."""
    return make_response(
        json.dumps(
            {
                "status": 429,
                "errors": ["Rate limit exceeded."],
                "retry_after_seconds": e.retry_after,
            }
        ),
        429,
    )


# API to add a request in the queue
@app.route("/watcher_api/monte_carlo", methods=["POST"])
@limiter.limit("1 per 5 seconds")
def submit_monte_carlo_request():
    """API to perform Monte Carlo simulation."""
    try:
        # Variables
        request_queue = app.config["REQUEST_QUEUE"]
        status_board = app.config["STATUS_BOARD"]

        # Generate simulation id (UUID)
        simulation_id = uuid.uuid4().hex

        # Get data
        data = request.json.get("data", {})
        patient_id = data.get("patient_id", None)
        n_iter_str = data.get("n_iter", None)
        time_horizon_str = data.get("time_horizon", None)
        sim_start_str = data.get("sim_start", None)

        # Put the request in the queue
        # NOTE: Any unexpected data is put as 'None'
        request_queue.put(
            (patient_id, sim_start_str, time_horizon_str, n_iter_str, simulation_id),
            timeout=TIMEOUT,
        )

        # Write the request status
        status_board[simulation_id] = {
            "status": 202,
            "progress": "Waiting for data preprocessing",
            "errors": [],
            "updated": datetime.now().isoformat(),
        }

        # Return the simulation id
        # NOTE: The task status is '202', but the json response is returned with 200
        # because submission was successful
        return make_response(json.dumps({"simulation_id": simulation_id}), 200)

    except Exception as e:
        # Error handling
        error_response = make_response(json.dumps({"errors": str(e)}), 500)
        error_response.headers["Content-Type"] = "application/json"
        traceback.print_exc()
        return error_response


# API to recieve the product
@app.route("/watcher_api/result/<simulation_id>", methods=["GET"])
@limiter.limit("1 per 1 seconds")
def get_result(simulation_id):
    """Fetches products if ready."""
    # Check request status
    status_board = app.config["STATUS_BOARD"]
    state = status_board.get(simulation_id)
    if state is not None:
        status = state["status"]
        if status in [200, 400]:
            # Pop out the final status
            final_state = status_board.pop(simulation_id, STATE_204)
            # Successful response
            if status == 200:
                # Get products
                product_store = app.config["PRODUCT_STORE"]
                result = product_store.pop(simulation_id, None)
                if result is not None:
                    return make_response(result, 200)
                else:
                    # Products not found (e.g., expired)
                    return make_response(STATE_204, 204)
            # Invalid request (400)
            else:
                return make_response(json.dumps(final_state), 400)

        # Pending response (202)
        else:
            return make_response(json.dumps(status_board[simulation_id]), 202)

    # simulation id not found (e.g., expired)
    else:
        return make_response(json.dumps(STATE_204), 204)


# endregion


# ==========================
# Simulator background utils
# ==========================
# region: Simulator utils
def _graceful_shutdown(signum, frame):
    print(f"Received signal {signum}. Terminating simulator processes...")

    # 1. Terminate child processes
    started_processes = app.config.get("STARTED_PROC")
    if started_processes:
        for proc in app.config["STARTED_PROC"]:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)  # force shutdown if not clean

    print("All simulator processes terminated.")

    # 2. Terminate orchestrator
    active_orch = app.config.get("ORCH")
    if active_orch is not None:
        try:
            print("Terminating WatcherOrchestrator...")
            active_orch.terminate()
        except Exception as e:
            print(f"Failed to terminate orchestrator cleanly: {e}")


def _setup_signal_handlers():
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)
    atexit.register(lambda: _graceful_shutdown(signum="atexit", frame=None))


def _preprocess_requests(
    request_queue: JoinableQueue,
    temp_queue: JoinableQueue,
    status_board: DictProxy,
    max_n_iter: int,
    model: Watcher,
    db_schema: str,
):
    """Monitors completed requests and stores simulation output in a shared product store."""
    torch.set_num_threads(1)
    while True:
        try:
            errors = []
            # Get one request
            req = request_queue.get()
            patient_id, sim_start_str, time_horizon_str, n_iter_str, simulation_id = req
            # ==========================
            # Input validation
            # ==========================
            # Validate patient ID
            # NOTE: If the patient ID is not in the database, currently, ValueError is raised by `preprocess_for_inference()`
            if patient_id is None:
                errors.append("Patient ID not in data.")
            # Validate sim_start
            if sim_start_str is None:
                errors.append("Time of simulation start not in data.")
                sim_start = None
            else:
                try:
                    sim_start = datetime.strptime(sim_start_str, config.DATETIME_FORMAT)
                except ValueError:
                    errors.append("Time of simulation has an irregular format")
                    sim_start = None
            # Validate time horizon
            if time_horizon_str is None:
                errors.append("Time horizon not in data.")
                time_horizon = None
            else:
                try:
                    time_horizon = int(time_horizon_str)
                    if time_horizon <= 0:
                        errors.append("Time horizon must be a positive integer.")
                except (ValueError, TypeError):
                    errors.append("Time horizon must be a positive integer.")
                    time_horizon = None
            # Validate n_iter
            if n_iter_str is None:
                errors.append("Number of iterations not in data.")
                n_iter = None
            else:
                try:
                    n_iter = int(n_iter_str)
                    if (n_iter <= 0) or (n_iter > max_n_iter):
                        errors.append(f"Number of iterations must be 1 ~ {max_n_iter}.")
                except (ValueError, TypeError):
                    errors.append("Time horizon must be a positive integer.")
                    n_iter = None

            # Do not proceed if obvious request errors are detected
            if errors:
                status = 400
                progress = "Aborted."

            # ===================================
            # Try to download and preprocess data
            # ===================================
            else:
                try:
                    timeline, catalog_ids, dob = preprocess_for_inference(
                        patient_id=patient_id,
                        model=model,
                        start=None,
                        end=sim_start_str,  # This arg expects strings,
                        db_schema=db_schema,
                    )
                    preprocess_success = True

                # NOTE: If patient_id is not found, `preprocess_for_inference()` raises ValueError
                except ValueError as e:
                    errors.append(str(e))
                    status = 400
                    progress = "Aborted."
                    preprocess_success = False

                # Put in the queue
                if preprocess_success:
                    horizon_start = sim_start - dob
                    temp_queue.put(
                        (
                            timeline,
                            catalog_ids,
                            n_iter,
                            horizon_start,
                            timedelta(days=time_horizon),
                            simulation_id,
                        ),
                        timeout=TIMEOUT,
                    )
                    status = 202
                    progress = "Data preprocessing completed."

            # ===================================
            # Update the request status
            # ===================================
            status_board[simulation_id] = {
                "status": status,
                "progress": progress,
                "errors": errors,
                "updated": datetime.now().isoformat(),
            }
            request_queue.task_done()

        # Catch all other errors
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError("Error during request process") from e


def _watch_products(
    orch: WatcherOrchestrator,
    status_board: DictProxy,
    product_store: DictProxy,
):
    """Gets products and put them in a dict."""
    for prod in orch:
        df, simulation_id = prod
        # NOTE:
        #   product_store must be updated first, otherwise, the status_board may be read first while the
        #   product has not yet been saved.
        # Make everyhting to strings for consistency
        df = df.astype(str)
        # Add the product to the store
        product_store[simulation_id] = df.to_json(orient="records")

        # Update status
        # NOTE: As mentioned above, this must follow the product_store update
        status_board[simulation_id] = {
            "status": 200,
            "progress": "Simulation completed",
            "errors": [],
            "updated": datetime.now().isoformat(),
        }


def _watch_expired(
    status_board: DictProxy,
    product_store: DictProxy,
):
    """Removes expired requests from memory store."""
    while True:
        # DictProxy -> Dict (make a copy)
        status_dict = dict(status_board)
        # Check expired
        df = pd.DataFrame.from_dict(status_dict, orient="index")
        df["simulation_id"] = df.index
        if not df.empty:
            df["updated"] = pd.to_datetime(df["updated"])
            expired = df["status"].isin([200, 400]) & (
                (datetime.now() - df["updated"]).dt.total_seconds() > EXPIRES_AFTER
            )
            expired_ids = df.loc[expired, "simulation_id"].tolist()
            # Remove expired requests
            for req in expired_ids:
                _ = status_board.pop(req, None)
                _ = product_store.pop(req, None)
        else:
            json_data = "[]"
        # Wait for the next round
        time.sleep(EXPIRES_AFTER)


# endregion


# ===========================
# Main functions to be called
# ===========================
# region: Main functions
def start_simulators(
    blueprint: str,
    log_dir: str,
    gpu_ids: list[str],
    n_preprocess_workers: int,
    db_schema: str = "public",
    max_batch_size: int = 256,
    max_length: int = 10000,
    return_generated_parts_only: bool = True,
    return_unfinished: bool = False,
) -> Flask:
    """
    Starts all background simulator processes including:
      - Preprocessing workers
      - WatcherOrchestrator
      - Expiration manager

    This function should be called separately from the Flask app,
    typically before launching Gunicorn or from a separate process.


    **Example Usage**

        .. code-block:: python

            from watcher.watcher_api import start_simulators

            start_simulators(...)

    Args:
        blueprint (str): Model blueprint name.
        log_dir (str): Path to directory for saving logs.
        gpu_ids (list[str]): List of GPU device IDs to assign for simulation.
        n_preprocess_workers (int): Number of multiprocessing workers for data preprocessing.
        db_schema (str, optional): PostgreSQL schema name. Defaults to "public".
        max_batch_size (int, optional): Max batch size for simulation. Defaults to 256.
        max_length (int, optional): Max sequence length for simulation input. Defaults to 10000.
        return_generated_parts_only (bool, optional): If True, returns only generated output. Defaults to True.
        return_unfinished (bool, optional): If True, includes unfinished trajectories. Defaults to False.
    """
    if not hasattr(app, "_simulators_started"):
        print("Please wait for API initialization...")
        # Setup objects
        manager = Manager()
        status_board = manager.dict()
        product_store = manager.dict()
        temp_queue = JoinableQueue()
        request_queue = JoinableQueue()

        # Generator -> orchestrator
        def _gen_task(temp_queue):
            while True:
                yield temp_queue.get()

        task_gen = _gen_task(temp_queue)

        print("Building watcher orchestrator...")
        orch = WatcherOrchestrator(
            task_generator=task_gen,
            log_dir=log_dir,
            blueprint=blueprint,
            gpu_ids=gpu_ids,
            max_batch_size=max_batch_size,
            max_length=max_length,
            return_generated_parts_only=return_generated_parts_only,
            return_unfinished=return_unfinished,
            compile_model=False,
            temperature=1,
        )
        orch.start()
        print("Orchestrator started")

        # Store in app config (Flask will pick these up when app starts)
        app.config["MAX_N_ITER"] = max_batch_size
        app.config["DB_SCHEMA"] = db_schema
        app.config["REQUEST_QUEUE"] = request_queue
        app.config["PRODUCT_STORE"] = product_store
        app.config["STATUS_BOARD"] = status_board
        app.config["STARTED_PROC"] = []

        # Start preprocessors
        dummy_model = build_watcher(blueprint=blueprint, train=False).cpu()
        for _ in range(n_preprocess_workers):
            p = Process(
                target=_preprocess_requests,
                kwargs={
                    "request_queue": request_queue,
                    "temp_queue": temp_queue,
                    "status_board": status_board,
                    "max_n_iter": max_batch_size,
                    "model": dummy_model,
                    "db_schema": db_schema,
                },
                daemon=True,
            )
            p.start()
            # Add to the list of started processes
            app.config["STARTED_PROC"].append(p)

        # Setup orchestrator

        app.config["ORCH"] = orch
        print("Orch done")

        p_prod = Process(
            target=_watch_products,
            kwargs={
                "orch": orch,
                "status_board": status_board,
                "product_store": product_store,
            },
            daemon=False,
        )
        p_prod.start()
        print("Prodoct watcher started")
        app.config["STARTED_PROC"].append(p_prod)

        p_exp = Process(
            target=_watch_expired,
            kwargs={
                "status_board": status_board,
                "product_store": product_store,
            },
            daemon=True,
        )
        p_exp.start()
        print("Expiration watcher started")
        app.config["STARTED_PROC"].append(p_exp)

        # Falg for simulators started
        app._simulators_started = True

        # Final message
        print("API is ready.")

    # Register handlers
    # _setup_signal_handlers()

    return app


# endregion
