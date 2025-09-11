"""Training utilities"""

import os
import json
import torch
from torch.distributed import init_process_group
from ....general_params import watcher_config as config
from ....general_params import get_settings
from ....utils import load_model_params_from_dataset


def ddp_setup(rank: int, world_size: int) -> None:
    """Initializes distributed parallelism.

    Configure 'MASTER_ADD' and 'MASTER_PORT'.
    If you are using a Docker container, refer to your Dockerfile or docker-compose.yml to find the port number.
    Args:
        rank (int): Unique identifier given to each process.
            This is implicitly handled by 'torch.multiprocessing.spawn'.
        world_size (int): Number of processes involved, technically equivalent to the number of GPUs used.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # The training script works without the line 'torch.cuda.set_device(rank)'.
    # However, RAM usage on rank 0 may increase significantly


def initialize_training_report(
    initiated_time: str,
    world_size: int,
    embedding_dim: int,
    num_layers: int,
    num_heads: int,
    ff_hidden_dim: int,
    dropout_rate: float,
    total_epochs: int,
    max_batch_size: int,
    weight_decay: float,
    max_lr: float,
    min_lr: float,
    lr_scheduler_enabled: bool,
    lr_warmup: float,
    lr_decay: float,
    batch_size_step_scale: int,
    batch_size_active_phase: float,
    min_batch_size: int,
    batch_schedule_enabled: bool,
    dataloader_workers: int,
    checkpoint_interval: int,
    validation_interval: int,
    early_stopping_epochs: int,
    precision: str,
    debug: bool,
    debug_chunks: int,
    **kwargs,
) -> dict:
    """Initializes the dictionary that stores all the settings necessary for the training."""
    # Load params
    train_tensors_dir = get_settings("TENSORS_TRAIN_DIR")
    stats_path = os.path.join(
        train_tensors_dir, config.TRAJECTORY_STATS_PATTERN
    ).replace("*", "train")
    with open(stats_path, "r", encoding="utf-8") as f:
        timeline_stats = json.load(f)
    n_included_patients = timeline_stats["included_patients"]
    n_excluded_patients = timeline_stats["excluded_patients"]

    # Create a new dict
    training_report = {}
    # Record the time of training initiation
    training_report["initiated_time"] = initiated_time
    # Record model hyperparameters
    manual_hyperparams = {
        "precision": precision.lower(),
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "ff_hidden_dim": ff_hidden_dim,
        "dropout_rate": dropout_rate,
    }
    dataset_defined_hyperparams = load_model_params_from_dataset()
    training_report["hyperparameters"] = {
        **manual_hyperparams,
        **dataset_defined_hyperparams,
    }
    # Record training configs
    training_report["training_config"] = {
        "total_epochs": total_epochs,
        "weight_decay": weight_decay,
        "max_lr": max_lr,
        "min_lr": min_lr,
        "lr_warmup": lr_warmup,
        "lr_decay": lr_decay,
        "lr_scheduler_enabled": lr_scheduler_enabled,
        "batch_size_step_scale": batch_size_step_scale,
        "batch_size_active_phase": batch_size_active_phase,
        "min_batch_size": min_batch_size,
        "max_batch_size": max_batch_size,
        "batch_schedule_enabled": batch_schedule_enabled,
        "n_gpus": world_size,
        "dataloader_workers": dataloader_workers,
        "checkpoint_interval": checkpoint_interval,
        "validation_interval": validation_interval,
        "early_stopping_epochs": early_stopping_epochs,
        "debug": debug,
        "debug_chunks": debug_chunks,
    }
    # Record patient info
    training_report["patients"] = {
        "included": n_included_patients,
        "excluded": n_excluded_patients,
    }
    # Other params
    training_report["etc"] = {
        "timedelta_dim": len(config.TIMEDELTA_COMPONENT_COLS),
        "numeric_dim": config.NUMERIC_DIM,
        "n_demographic_rows": config.DEMOGRAPHIC_ROWS,
        "ignored_index": config.LOGITS_IGNORE_INDEX,
    }
    training_report["previous_snapshot"] = None
    # Training resumption and disruption histories
    training_report["final_state"] = None
    training_report["disrupted_at"] = []
    training_report["resumed_at"] = []

    return training_report
