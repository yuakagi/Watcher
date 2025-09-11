"""The main module for the model training"""

import os
import sys
from typing import Literal
import json
import traceback
from datetime import datetime
from multiprocessing import active_children
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group
import torch.multiprocessing as mp
from .training_utils import (
    ddp_setup,
    initialize_training_report,
    LogitsTrainer,
)
from ...general_params import watcher_config as config
from ...general_params import TrainSettingsManager, get_settings


def _training_branch(
    rank: int,
    world_size: int,
    snapshot_path: str,
    initial_weight_path: str,
    update: bool,
    training_report: dict,
    log_dir: str,
    trainer_class: object,
):
    """Helper function for training."""
    # Set up DDP
    ddp_setup(rank, world_size)
    # Set up logs
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    logfile = os.path.join(log_dir, f"training_log_rank{rank}.txt")
    f = open(logfile, "a", encoding="utf-8")
    sys.stdout = f
    sys.stderr = f

    completed = False
    try:
        # Instantiate trainer
        if trainer_class is None:
            trainer_class = LogitsTrainer
        trainer = trainer_class(
            rank=rank,
            world_size=world_size,
            training_report=training_report,
            update=update,
            snapshot_path=snapshot_path,
            initial_weight_path=initial_weight_path,
        )
        # Wait for all processes to finish initialization
        dist.barrier()
        # Start training
        trainer.train()
        # Set flag of training completion
        completed = True

    except Exception as e:
        print(f"Error at rank {rank}:")
        print(e)
        traceback.print_exc()

    finally:
        destroy_process_group()
        # Close the log file
        if f is not None:
            f.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        # Raise error if training was disrupted
        if not completed:
            raise RuntimeError("Training was disrupted.")


def train_watcher(
    dataset_dir: str,
    output_dir: str,
    max_gpus: int = 4,
    # These arguments are used by calling locals(), pylint: disable=unused-argument
    embedding_dim: int = 1024,
    num_layers: int = 16,
    num_heads: int = 16,
    ff_hidden_dim: int = 3072,
    dropout_rate: float = 0.1,
    total_epochs: int = 100,
    weight_decay: float = 0.01,
    max_lr: float = 1.0e-4,
    min_lr: float = 1.0e-5,
    lr_scheduler_enabled: bool = True,
    lr_warmup: float = 0.01,
    lr_decay: float = 0.9,
    batch_size_step_scale: int = 8,
    batch_size_active_phase: float = 0.03,
    max_batch_size: int = 64,
    min_batch_size: int = 16,
    batch_schedule_enabled: bool = True,
    dataloader_workers: int = 2,
    checkpoint_interval: int = 5,
    validation_interval: int = 5,
    early_stopping_epochs: int = 5,
    snapshot_path: str | None = None,
    initial_weight_path: str | None = None,
    precision: Literal["float32", "float16", "bfloat16"] = "bfloat16",
    update: bool = False,
    restart_limit: int = 10,
    debug: bool = False,
    debug_chunks: int = 10,
    trainer_class: object | None = None,
) -> str:
    """
    Train a Watcher model.

    This function initiates training and stops either when the maximum number of epochs is reached,
    or when early stopping is triggered due to no improvement in validation loss.

    Warnings:

        - Training requires GPU devices with large memory capacity. We recommend using NVIDIA A100 or newer GPUs.
        - If training fails, it may be due to an OutOfMemoryError. In that case, consider reducing the batch size or using a smaller model configuration.

    Note:

        - Training may take from several hours to several days, depending on dataset size and hardware specifications.
        - For reference, in an experiment with approximately 370,000 patients using four NVIDIA A100 80GB GPUs, training completed in about 48 hours.

    Example:

        **Pretraining**

        .. code-block:: python

            from watcher.training import train_watcher

            best_weights_path = train_watcher(
                dataset_dir="/path/to/prepared_dataset",
                output_dir="/path/to/save_training_outputs",
                max_gpus=4,
                embedding_dim=1024,
                num_layers=16,
                num_heads=16,
                ff_hidden_dim=3072,
                dropout_rate=0.1,
                total_epochs=100,
                weight_decay=0.01,
                max_lr=1e-4,
                min_lr=1e-5,
                lr_scheduler_enabled=True,
                lr_warmup=0.01,
                lr_decay=0.9,
                batch_size_step_scale=8,
                batch_size_active_phase=0.03,
                max_batch_size=64,
                min_batch_size=16,
                batch_schedule_enabled=True,
                dataloader_workers=2,
                checkpoint_interval=5,
                validation_interval=5,
                early_stopping_epochs=10,
                snapshot_path=None,
                initial_weight_path=None,
                precision="bfloat16",
                restart_limit=10,
            )

            print(f"Best model weights saved at: {best_weights_path}")


        **Fine tuning**

        .. code-block:: python

            from watcher.training import train_watcher

            best_weights_path = train_watcher(
                dataset_dir="/path/to/prepared_dataset",  # Use the same dataset as pretraining
                output_dir="/path/to/save_finetuning_outputs",
                max_gpus=4,
                embedding_dim=1024,  # Use the same model hyperparameters as pretraining
                num_layers=16,       # Must match pretraining
                num_heads=16,        # Must match pretraining
                ff_hidden_dim=3072,  # Must match pretraining
                dropout_rate=0.1,
                total_epochs=20,
                weight_decay=0.01,
                max_lr=1e-5,
                min_lr=1e-5,
                lr_scheduler_enabled=False,  # Constant LR = min_lr
                batch_schedule_enabled=False,  # Constant batch size
                max_batch_size=64,
                min_batch_size=64,
                dataloader_workers=2,
                checkpoint_interval=5,
                validation_interval=5,
                early_stopping_epochs=5,
                initial_weight_path="/path/to/pretrained_model.pt",  # Set pretrained weight path
                precision="bfloat16",
                update=True,  # Flag to use fine-tuning dataset
                restart_limit=10,
            )

            print(f"Best model weights saved at: {best_weights_path}")

    The following directory structure is created during training:

        .. code-block:: text

            output_dir
            ├── main_training_report.json     # Training summary
            ├── profiling/
            │   └── ...
            ├── snapshots/
            │   ├── epoch_0/
            │   │   ├── training_state.pt
            │   │   ├── tensorboard_logs/
            │   │   └── watcher_blueprint/
            │   │       ├── catalogs/         # CSV files containing model vocabulary
            │   │       ├── laboratory_stats/ # CSV files of lab test stats
            │   │       ├── model_state.pt    # Model weights
            │   │       └── training_report.json
            │   └── ...
            └── tensorboard_active/
                └── ... (TensorBoard logs)

    The `watcher_blueprint` directory is the main product of training. Each blueprint contains
    everything needed to re-instantiate the Watcher model.
    To monitor training progress with TensorBoard, set the `tensorboard_active` directory as the logdir.

    Args:
        dataset_dir (str): Path to the dataset created by :meth:`watcher.preprocess.create_dataset()`.
        output_dir (str): Directory where training results are saved.
        max_gpus (int, optional): Maximum number of GPUs to use for training.
        embedding_dim (int, optional): Dimensionality of the model embeddings (`d_model`).
        num_layers (int, optional): Number of transformer blocks in the model.
        num_heads (int, optional): Number of attention heads per transformer layer.
        ff_hidden_dim (int, optional): Hidden layer size of the feedforward network (`d_ff`).
        dropout_rate (float, optional): Dropout rate applied during training.
        total_epochs (int, optional): Maximum number of training epochs.
        weight_decay (float, optional): Weight decay for regularization.
        max_lr (float, optional): Peak learning rate.
        min_lr (float, optional): Minimum learning rate.
        lr_scheduler_enabled (bool, optional): Whether to use a learning rate scheduler. If False, `min_lr` is used throughout training.
        lr_warmup (float, optional): Warm-up duration as a fraction of total training data.
        lr_decay (float, optional): Learning rate decay duration after warm-up.
        batch_size_step_scale (int, optional): Scaling factor for batch size scheduling.
        batch_size_active_phase (float, optional): Fraction of data (0 to 1.0) during which the batch size increases.
        max_batch_size (int, optional): Maximum batch size.
        min_batch_size (int, optional): Initial batch size.
        batch_schedule_enabled (bool, optional): If False, batch size is fixed to `max_batch_size` throughout training.
        dataloader_workers (int, optional): Number of worker processes for data loading.
        checkpoint_interval (int, optional): Epoch interval at which training snapshots are saved.
        validation_interval (int, optional): Epoch interval at which validation is performed.
        early_stopping_epochs (int, optional): Number of consecutive epochs without validation loss improvement required to stop training early.
        snapshot_path (str, optional): Path to a training snapshot to resume from. Ensure that all training parameters match the previous run.
        initial_weight_path (str, optional): Path to pretrained weights for model initialization. Required if `update=True`.
        precision (Literal["float32", "float16", "bfloat16"], optional): Floating point precision for training.
        update (bool, optional): If True, fine-tunes the model using the current dataset.
        restart_limit (int, optional): Maximum number of automatic training restarts after runtime errors.
        debug (bool, optional): If True, enables debug mode. Dataloaders will yield only `debug_chunks` number of samples.
        debug_chunks (int, optional): Number of samples yielded per loader in debug mode.
        trainer_class (object, optional): [Deprecated] Custom trainer class.

    Returns:
        best_weights (str): Path to the best-performing model weights.
    """

    # Record the time of training initiation
    initiated_time = datetime.strftime(datetime.now(), format="%Y%m%d%H%M%S")

    # Validate arguments
    if (snapshot_path is not None) and (initial_weight_path is not None):
        message = """snapshot_path and initial_weight_path are mutual exclusive.
            Do not path them both at a time."""
        raise ValueError(message)

    # Ensure that paths are absolute
    dataset_dir = os.path.abspath(dataset_dir)
    output_dir = os.path.abspath(output_dir)
    if initial_weight_path is not None:
        initial_weight_path = os.path.abspath(initial_weight_path)
    if snapshot_path is not None:
        snapshot_path = os.path.abspath(snapshot_path)

    # Validate precision
    if precision.lower() not in ["float32", "float16", "bfloat16"]:
        raise KeyError("precision must be either float32, float16, or bfloat16")

    # Determine world size
    if max_gpus > 0:
        world_size = min(torch.cuda.device_count(), max_gpus)
    else:
        world_size = 1

    # Determine training report and previous checkpoint
    n_restart = 0
    if snapshot_path is not None:
        # Load an old trianing report and restore the initiated time
        training_report_path = os.path.join(
            snapshot_path, config.DIR_BLUEPRINT, config.TRAINING_REPORT
        )
        with open(training_report_path, "r", encoding="utf-8") as f:
            training_report = json.load(f)
        initiated_time = training_report["initiated_time"]
    else:
        # Initialize everything
        training_report = None
    # Write environment variables
    settings_manager = TrainSettingsManager(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        initiated_time=initiated_time,
        debug=debug,
        debug_chunks=debug_chunks,
    )
    settings_manager.create_dirs()
    settings_manager.write()

    # Redirect logs
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_dir = get_settings("TRAINING_LOG_DIR")
    logfile = os.path.join(log_dir, "main_training_log.txt")
    f = open(logfile, "a", encoding="utf-8")
    print(f"Standard outputs and errors are saved in {log_dir}")
    print("See the file for details.")
    sys.stdout = f
    sys.stderr = f

    try:
        if training_report is None:
            training_report = initialize_training_report(**locals())
        # Start training
        main_tr_path = get_settings("MAIN_TRAINING_REPORT_PTH")
        while True:
            try:
                # Run distributed training
                arguments = (
                    world_size,
                    snapshot_path,
                    initial_weight_path,
                    update,
                    training_report,
                    log_dir,
                    trainer_class,
                )
                mp.spawn(
                    _training_branch,
                    args=arguments,
                    nprocs=world_size,
                )
                break

            # Handle disrupted training
            except Exception as e:
                print("Training disruped")
                # Count up
                n_restart += 1
                # Terminate all child processes
                active = active_children()
                if active:
                    print(
                        f"Killing child processes... (active child processes: {len(active)})"
                    )
                    for child in active:
                        child.terminate()
                    for child in active:
                        child.join()
                    print("Successfully terminated child processes.")
                # Try to get the latest checkpoint path
                try:
                    print("Trying to get the latest checkpoint path...")
                    with open(main_tr_path, "r", encoding="utf-8") as f:
                        training_report = json.load(f)
                        # Update checkpoint path
                        snapshot_path = training_report["previous_snapshot"]
                        print("Latest snapshot:", os.path.basename(snapshot_path))

                    # Once a checkpoint is saved, disable the initial weight path
                    if snapshot_path is not None:
                        initial_weight_path = None

                except FileNotFoundError:
                    print("Failed to load the latest checkpoint path.")
                    print("Start the training all over")
                    snapshot_path = None

                # Forcefully terminate training if restart limit is reached
                if n_restart > restart_limit:
                    print("Max number restarting attempts reached.")
                    print("Giving up training")
                    raise RuntimeError("Training failed too many times.") from e

                # Restart training
                print("Restarting training...")

        # Close the log file
        if f is not None:
            f.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Get the best model weights and return it
        print("Training done")
        with open(main_tr_path, "r", encoding="utf-8") as f:
            training_report = json.load(f)
            best_snapshot = training_report.get("best_snapshot")
            best_weights = os.path.join(
                best_snapshot, config.DIR_BLUEPRINT, config.MODEL_STATE
            )
        return best_weights

    finally:
        settings_manager.delete()
