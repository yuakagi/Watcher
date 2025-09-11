"""Functions to load models"""

import os
import json
import torch
from .watcher import Watcher, WatcherInterpreter
from ...general_params import watcher_config as config


def build_watcher(blueprint: str, train: bool = False) -> Watcher:
    """Instantiate a pretrained model using a blueprint

    Args:
        blueprint (str): Path to a blueprint.
        train (bool, optional): If false, the model is initialized for inference.
    Returns:
        model (Watcher): Pretrained Watcher model.
    """
    # Load the training report
    tr_path = os.path.join(blueprint, config.TRAINING_REPORT)
    with open(tr_path, "r", encoding="utf-8") as f:
        training_report = json.load(f)

    # Hyperparameters
    hyperparams = training_report["hyperparameters"]
    other_params = {
        "catalogs_dir": os.path.join(blueprint, config.DIR_CATALOGS),
        "lab_stats_dir": os.path.join(blueprint, config.DIR_LAB_STATS),
    }
    params = {**hyperparams, **other_params}

    # Model weights
    mw_path = os.path.join(blueprint, config.MODEL_STATE)
    model_weights = torch.load(mw_path, map_location=torch.device("cpu"))

    # Instantiate a model
    model = Watcher(**params)
    model.load_state_dict(model_weights)
    if not train:
        model.eval(for_inference=True)
    else:
        model.train()

    return model


def build_interpreter(blueprint: str) -> WatcherInterpreter:
    """
    Args:
        blueprint (str): Path to a blueprint.
    Returns:
        interpreter (WatcherInterpreter): WatcherInterpreter object.
    """
    # Load the training report
    tr_path = os.path.join(blueprint, config.TRAINING_REPORT)
    with open(tr_path, "r", encoding="utf-8") as f:
        training_report = json.load(f)

    # Hyperparameters
    hyperparams = training_report["hyperparameters"]
    n_numeric_bins = hyperparams["n_numeric_bins"]

    # Instantiate
    interpreter = WatcherInterpreter(
        n_numeric_bins=n_numeric_bins,
        catalogs_dir=os.path.join(blueprint, config.DIR_CATALOGS),
        lab_stats_dir=os.path.join(blueprint, config.DIR_LAB_STATS),
    )
    return interpreter
