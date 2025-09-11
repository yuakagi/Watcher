"""Main module for data generation."""

import math
from datetime import timedelta, datetime
from multiprocessing import Queue, Event
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm
from .watcher import Watcher
from .quantize import quantize_linear_weights
from ...general_params import watcher_config as config
from ...utils import shuffle_timeline_matrix_indexes

TIMEOUT = 600


# ******************
# * Model forwards *
# ******************
@torch.no_grad()
def _forward_prefill(model: Watcher, input_tensor: Tensor) -> Tensor:
    """Model forward for the first inference of autoregressive decoding.

    Use with torch.compile(dynamic=True, fullgraph=True).

    Input Shape:
        - (N, L, E):
            - N: Batch size
            - L: Target sequence length
            - E: Embedding dim
    """
    logits = model(input_tensor, True, False)
    logits = logits.view(logits.size(0), -1)
    return logits


@torch.no_grad()
def _forward_one_step(model: Watcher, input_tensor: Tensor) -> Tensor:
    """Model forward for autoregressive steps.

    Use with torch.compile(mode="reduce-overhead" fullgraph=False).

            Input Shape:
                - (N, 1, E):
                    - N: Batch size
                    - E: Embedding dim
    """
    logits = model(input_tensor, True, False)
    logits = logits.view(logits.size(0), -1)
    return logits


@torch.no_grad()
def _forward_sliding(model: Watcher, input_tensor: Tensor) -> Tensor:
    """Model forward for the sliding window.

    Use with torch.compile(mode="reduce-overhead" fullgraph=False).

    Input Shape:
        - (N, Lw, E):
            - N: Batch size
            - Lw: Max sequence length - stride
            - E: Embedding dim
    """
    logits = model(input_tensor, True, False)
    logits = logits.view(logits.size(0), -1)
    return logits


# **************************
# * General decoding utils *
# **************************
def generate_from_batch(
    model: Watcher,
    timeline_batch: torch.Tensor,
    catalog_ids_batch: list[list[int]],
    time_horizon: timedelta | None = None,
    horizon_start: timedelta | None = None,
    stride: int = 64,
    stop_vocab: list[int] | None = None,
    max_length: int = 10000,
    return_generated_parts_only: bool = True,
    return_unfinished: bool = False,
    compile_model: bool = False,
    quantize_weights: bool = False,
    logits_filter: str = "default",
    temperature: float = 1.0,
    show_pbar: bool = False,
    compiled_fn: dict | None = None,
) -> pd.DataFrame:
    """Generates timelines from a given batch of timelines."""
    # Input validation
    if timeline_batch.ndim != 3:
        raise ValueError(
            f"Input timeline must be a 3D tensor, but got {timeline_batch.ndim}D."
        )
    if timeline_batch.size(1) > model.max_sequence_length:
        raise ValueError(
            f"Input timeline length ({timeline_batch.size(1)}) exceeds the model's max sequence length ({model.max_sequence_length})."
        )
    # NOTE: Use numpy's timedelta consistently. Currently, timedelta data types are mixed.
    if isinstance(horizon_start, timedelta):
        horizon_start = pd.to_timedelta(horizon_start)
    time_anchor = horizon_start

    # Ensure the toensor device mathces the model's device
    with torch.device(model.device):
        # Prepare static objects
        if stop_vocab is None:
            stop_vocab = [model.eot_index]
        else:
            if isinstance(stop_vocab, int):
                stop_vocab = [stop_vocab]
            stop_vocab += [model.eot_index]
        stop_vocab = torch.tensor(stop_vocab).long()
        # Clean and truncate the input timeline
        prompt_ts, prompt_cs, latest_times = [], [], []
        ts, cs = [], []
        prompt_t_length, prompt_c_length = None, None
        init_td_cat_idxs, initial_td_rows = [], []
        for timeline, catalog_ids in zip(timeline_batch, catalog_ids_batch):
            timeline = timeline.unsqueeze(0)  # Add batch dimension
            prompt_t, prompt_c, latest_time = model.preprocess_prompt(
                timeline, catalog_ids, time_anchor
            )
            # Shuffle
            if config.SHUFFLE_INPUT and prompt_t.size(1) > config.DEMOGRAPHIC_ROWS:
                shuffled_indexes = shuffle_timeline_matrix_indexes(
                    timeline_matrix=prompt_t.squeeze(0),
                    pad_start=None,  # model.preprocess_prompt should have removed paddings
                    dsc_idx=model.dsc_index,
                    eot_idx=model.eot_index,
                    lab_code_token_idx=model.lab_token_index,
                    k=1,
                )[0]
                prompt_t = prompt_t[:, shuffled_indexes]
                prompt_c = (np.array(prompt_c)[shuffled_indexes]).tolist()

            # Truncate
            t, c = model.truncate_timeline(
                x=prompt_t,
                catalog_ids=prompt_c,
                length=model.max_sequence_length,
            )
            # Initialize objects for truncated generation
            if return_generated_parts_only:
                init_td_id_candid = torch.tensor(c)
                candid_mask = init_td_id_candid >= model.small_step_start_idx
                if candid_mask.any():
                    init_td_cat_idx = init_td_id_candid[candid_mask][-1].item()
                else:
                    init_td_cat_idx = 0  # <= Defaults to initial age
                initial_td_row = model.create_timedelta_rows(
                    np.array([latest_time], dtype=object).astype("timedelta64[m]")
                )
            else:
                init_td_cat_idx, initial_td_row = None, None

            # Validate prompt lengths
            if prompt_t_length is None:
                prompt_t_length = prompt_t.size(1)
            elif prompt_t_length != prompt_t.size(1):
                raise ValueError("All input timelines must have the same length.")
            if prompt_c_length is None:
                prompt_c_length = len(prompt_c)
            elif prompt_c_length != len(prompt_c):
                raise ValueError("All catalog IDs must have the same length.")
            # Append to the lists
            prompt_ts.append(prompt_t)
            prompt_cs.append(prompt_c)
            latest_times.append(latest_time)
            ts.append(t)
            cs.append(c)
            init_td_cat_idxs.append(init_td_cat_idx)
            initial_td_rows.append(initial_td_row)

        # Concatenate
        prompt_ts = torch.cat(prompt_ts, dim=0)
        ts = torch.cat(ts, dim=0)
        prompt_cs = torch.tensor(prompt_cs, dtype=torch.long)
        cs = torch.tensor(cs, dtype=torch.long)
        current_time = np.array(latest_times, dtype="timedelta64[m]")
        if return_generated_parts_only:
            init_td_cat_idxs = torch.tensor(init_td_cat_idxs)
            initial_td_rows = torch.stack(initial_td_rows, dim=0)
        else:
            init_td_cat_idxs, initial_td_rows = None, None
        # Vars
        bs = ts.size(0)

        # Initialize main dynamic objects
        pos = torch.tensor(ts.size(1) - 1)
        input_tensor = torch.zeros(bs, model.max_sequence_length, ts.size(2)).float()
        input_tensor[:, : ts.size(1), :] = ts  # padded to max sequence length
        last_ids = cs[:, -1].clone().long()
        # Prepare other objects
        actives = torch.arange(0, bs)
        gen_timelines = torch.empty((bs, 0, ts.size(2))).float()
        gen_catalog_ids = torch.empty((bs, 0)).long()
        if time_horizon is not None:
            # NOTE: if current_time is used, period_end is numpy array. Otherwise, it is timedelta.
            period_end = time_horizon + (horizon_start or current_time)
        else:
            period_end = None
        # Compile
        # NOTE: This is not used currently, as this did not show significant performance improvement.
        static_input = compiled_fn or compile_model
        if compiled_fn:
            prefill_fn = compiled_fn["_forward_prefill"]
            sliding_fn = compiled_fn["_forward_sliding"]
            one_step_fn = compiled_fn["_forward_one_step"]
        elif compile_model:
            # NOTE: _forward_prefill is not compiled here, because it is only called once
            prefill_fn = _forward_prefill
            one_step_fn = torch.compile(
                _forward_one_step, mode="reduce-overhead", fullgraph=True
            )
            sliding_fn = _forward_sliding
        else:
            prefill_fn = _forward_prefill
            one_step_fn = _forward_one_step
            sliding_fn = _forward_sliding

        # Quantize
        if quantize_weights:
            model = quantize_linear_weights(model)

        # Initialize empty products
        prod_timelines = []
        prod_catalog_ids = []

        # Inference loop
        with tqdm(total=bs, disable=not show_pbar) as pbar:
            while actives.size(0):
                # *****************
                # * Forward calls *
                # *****************
                # Case 1. First forward call
                if model.cache_length == 0:
                    model.setup_cache(batch_size=bs)
                    logits = prefill_fn(
                        model=model, input_tensor=input_tensor[:, : pos + 1, :]
                    )
                # Case 2. First forward call after sliding window
                elif model.cache_length == model.max_sequence_length:
                    model.empty_cache()
                    logits = sliding_fn(
                        model=model, input_tensor=input_tensor[:, : pos + 1, :]
                    )
                # Case 3. All other steps
                else:
                    logits = one_step_fn(
                        model=model, input_tensor=input_tensor[:, pos : pos + 1, :]
                    )

                # ***************
                # * Id sampling *
                # ***************
                probs = model.compute_probs(
                    logits=logits[actives],
                    input_tensor=input_tensor[actives],
                    last_ids=last_ids,
                    pos=pos,
                    current_time=current_time,
                    time_anchor=time_anchor,
                    logits_filter=logits_filter,
                    temperature=temperature,
                )
                sampled_ids = model.sample_from_probs(probs=probs)

                # ******************
                # * Update objects *
                # ******************
                # Check timedelta anchoring status
                if time_anchor is not None:
                    if (current_time >= time_anchor).all():
                        time_anchor = None
                # Update next inputs, current time, and admission status
                next_inputs, current_time = model.generate_next_input(
                    sampled_ids=sampled_ids,
                    input_tensor=input_tensor[actives],
                    pos=pos,
                    current_time=current_time,
                    time_anchor=time_anchor,
                )
                # Update timelines and catalog indexes
                gen_timelines = torch.cat(
                    [gen_timelines, next_inputs.unsqueeze(1)], dim=1
                )
                gen_catalog_ids = torch.cat(
                    [gen_catalog_ids, sampled_ids.unsqueeze(1)], dim=1
                )
                last_ids = gen_catalog_ids[:, -1]
                # Update input tensor
                pos += 1
                if pos == model.max_sequence_length:
                    # Sliding window
                    shifts = -(stride + 1)
                    input_tensor[:, model.n_demographic_rows :, :] = input_tensor[
                        :, model.n_demographic_rows :, :
                    ].roll(shifts=shifts, dims=1)
                    # Shift back the position
                    pos -= stride + 1
                input_tensor[actives, pos] = next_inputs

                # **********************
                # * Check terminations *
                # **********************
                finished_mask = torch.isin(sampled_ids, stop_vocab)
                if period_end is not None:
                    stopped_by_td = (
                        current_time >= period_end
                    )  # NOTE: Here, period_end can by either a numpy array or a timedelta. But in any case, stop_by_td is a numpy array.
                    finished_mask |= torch.tensor(stopped_by_td)
                limit_reached = gen_timelines.size(1) > max_length
                if limit_reached:
                    finished_mask.fill_(True)
                finished_indexes = finished_mask.nonzero().squeeze(-1).tolist()
                finished_indexes = sorted(finished_indexes)

                # ****************************
                # * Remove finished products *
                # ****************************
                for i in finished_indexes:
                    if (not limit_reached) or return_unfinished:
                        # Pick up item
                        fin_tl = gen_timelines[i].unsqueeze(0)
                        fin_id = gen_catalog_ids[i].tolist()
                        prompt_t = prompt_ts[i].unsqueeze(0)  # 3D Tensor
                        prompt_c = prompt_cs[i].tolist()  # 1D List
                        # Case 1. Returning the generated products only (without prompts)
                        # NOTE: Even in this case, the demographic rows from the prompts are prepended.
                        if return_generated_parts_only:
                            init_td_cat_idx = init_td_cat_idxs[i].item()
                            initial_td_row = initial_td_rows[i]
                            # Append demographic rows to the generated products
                            first_id = fin_id[0]
                            # If the truncated part does not begin with a timedelta row, insert an timedelta row.

                            if (prompt_t.size(1) > model.n_demographic_rows) and (
                                first_id < model.small_step_start_idx
                            ):
                                fin_id = (
                                    prompt_c[: model.n_demographic_rows]
                                    + [init_td_cat_idx]
                                    + fin_id
                                )
                                fin_tl = torch.cat(
                                    [
                                        prompt_t[:, : model.n_demographic_rows],
                                        initial_td_row.unsqueeze(0),
                                        fin_tl,
                                    ],
                                    dim=1,
                                )
                                assert len(fin_id) == fin_tl.size(
                                    1
                                ), f"Product timeline ({fin_tl.size(1)}) and catalog ids ({len(fin_id)}) have different lengths. (Error within first timedelta row handlings.)"
                            else:
                                fin_id = prompt_c[: model.n_demographic_rows] + fin_id
                                fin_tl = torch.cat(
                                    [prompt_t[:, : model.n_demographic_rows], fin_tl],
                                    dim=1,
                                )
                        # Case 2. Return the generated products with the prompts
                        else:
                            fin_id = prompt_c + fin_id
                            fin_tl = torch.cat([prompt_t, fin_tl], dim=1)

                        assert len(fin_id) == fin_tl.size(
                            1
                        ), f"Product timeline ({fin_tl.size(1)}) and catalog ids ({len(fin_id)}) have different lengths."
                        # Add the product to the product list
                        prod_timelines.append(fin_tl.clone())
                        prod_catalog_ids.append(fin_id)

                # ****************
                # * Trim objects *
                # ****************
                if finished_indexes:
                    # Create mask to select active ones
                    unfinished = torch.ones_like(actives, dtype=torch.bool)
                    unfinished[finished_indexes] = False
                    unfinished = unfinished.tolist()
                    # Select active ones only
                    gen_timelines = gen_timelines[unfinished]
                    gen_catalog_ids = gen_catalog_ids[unfinished]
                    last_ids = last_ids[unfinished]
                    current_time = current_time[unfinished]
                    actives = actives[unfinished]
                    # Change the shape of `input_tensor` if torch.compile() is not used.
                    if (not static_input) and actives.size(0):
                        next_bs = math.ceil(actives.size(0) / 64) * 64
                        if next_bs < input_tensor.size(0):
                            # Sort and trim the input tensor
                            input_tensor[: actives.size(0)] = input_tensor[actives]
                            input_tensor = input_tensor[:next_bs]
                            # Sort and trim caceh
                            model.trim_cache(actives=actives, new_size=next_bs)
                            # Refresh actives
                            actives = torch.arange(0, actives.size(0))
                    # period_end
                    if isinstance(period_end, np.ndarray):
                        # NOTE: This only happens if horizon_start is None but time_horizon is used.
                        period_end = period_end[unfinished]

                    # Prompts
                    prompt_ts = prompt_ts[unfinished]
                    prompt_cs = prompt_cs[unfinished]

                # *****************
                # * Progress bar *
                # ****************
                if show_pbar:
                    n_finished = len(finished_indexes)
                    pbar.update(n_finished)
                    pbar.set_description(
                        f"[Generated lengths: {str(gen_timelines.size(1))} (input pos = {pos.item()})]"
                    )

        # ************
        # * Clean up *
        # ************
        model.delete_cache()
        # Convert to DataFrame
        df = make_sim_dataframe(
            interpreter=model.interpreter,
            prod_timelines=prod_timelines,
            prod_catalog_ids=prod_catalog_ids,
        )

        return df


def monte_carlo(
    model: Watcher,
    timeline: Tensor,
    catalog_ids: list[int],
    n_iter: int = 256,
    time_horizon: timedelta | None = None,
    horizon_start: timedelta | None = None,
    stride: int = 64,
    stop_vocab: list[int] | None = None,
    max_length: int = 10000,
    return_generated_parts_only: bool = True,
    return_unfinished: bool = False,
    compile_model: bool = False,
    quantize_weights: bool = False,
    logits_filter: str = "default",
    temperature: float = 1.0,
    show_pbar: bool = False,
    compiled_fn: dict | None = None,
):
    """Performs Monte Carlo simulations using a single timeline as input.

    This function is designed for a single patient.

    Args:
        model (Watcher): Instantiated model.
        timeline (Tensor): Input timeline. This must be a 3D tensor with the first dimension is one.
        catalog_ids (list[int]): Sequence of vocabulary catalog indexes.
        n_iter (int): Number of simulations to be performed.
        time_horizon (timedelta): Time horizon of the simulation.
        horizon_start (timedelta): Start of the time horizon.
        stride (int): Stride of sliding window during autoregressive inference beyond the max sequence length.
        stop_vocab (list[int]): Simulation completes once one of the vocabularies listed in this argument is reached.
            Vocabularies must be given in ids (integers).
        max_length (int): Once inference reaches this length, the simulation result is discarded.
        return_generated_parts_only (bool): If true, only the generated parts with demographic rows are returned.
        return_unfinished (bool): If true, timelines that reached the max length (max_length) are returned instead of
            being discarded.
        compile_model (bool): If true, torch.compile() is activated.
        quantize_weights (bool): If true, inear layers are quantized using Int8.
        logits_filter (str): Filter to limit the logits.
        temperature (float): Temperature.
        show_pbar (bool): If true, a progress bar appears.

    """
    # NOTE: Use numpy's timedelta consistently. Currently, timedelta data types are mixed.
    if isinstance(horizon_start, timedelta):
        horizon_start = pd.to_timedelta(horizon_start)
    time_anchor = horizon_start
    with torch.device(model.device):
        # Prepare static objects
        if stop_vocab is None:
            stop_vocab = [model.eot_index]
        else:
            if isinstance(stop_vocab, int):
                stop_vocab = [stop_vocab]
            stop_vocab += [model.eot_index]
        stop_vocab = torch.tensor(stop_vocab).long()
        # Clean and truncate the input timeline
        prompt_t, prompt_c, latest_time = model.preprocess_prompt(
            timeline, catalog_ids, time_anchor
        )
        # Shuffle
        if config.SHUFFLE_INPUT and prompt_t.size(1) > config.DEMOGRAPHIC_ROWS:
            shuffled_indexes = shuffle_timeline_matrix_indexes(
                timeline_matrix=prompt_t.squeeze(0),
                pad_start=None,  # model.preprocess_prompt should have removed paddings
                dsc_idx=model.dsc_index,
                eot_idx=model.eot_index,
                lab_code_token_idx=model.lab_token_index,
                k=1,
            )[0]
            prompt_t = prompt_t[:, shuffled_indexes]
            prompt_c = (np.array(prompt_c)[shuffled_indexes]).tolist()

        # Truncate
        t, c = model.truncate_timeline(
            x=prompt_t,
            catalog_ids=prompt_c,
            length=model.max_sequence_length,
        )
        # Initialize objects for truncated generation
        if return_generated_parts_only:
            init_td_id_candid = torch.tensor(c)
            candid_mask = init_td_id_candid >= model.small_step_start_idx
            if candid_mask.any():
                init_td_cat_idx = init_td_id_candid[candid_mask][-1].item()
            else:
                init_td_cat_idx = 0  # <= Defaults to initial age
            initial_td_row = model.create_timedelta_rows(
                np.array([latest_time], dtype=object).astype("timedelta64[m]")
            )
        else:
            init_td_cat_idx, initial_td_row = None, None

        # Initialize main dynamic objects
        pos = torch.tensor(t.size(1) - 1)
        input_tensor = torch.zeros(n_iter, model.max_sequence_length, t.size(2)).float()
        input_tensor[:, : t.size(1), :] = t
        current_time = np.full(n_iter, latest_time, dtype=object).astype(
            "timedelta64[m]"
        )
        last_ids = torch.full((n_iter,), c[-1]).long()
        # Prepare other objects
        actives = torch.arange(0, n_iter)
        gen_timelines = torch.empty((n_iter, 0, t.size(2))).float()
        gen_catalog_ids = torch.empty((n_iter, 0)).long()
        if time_horizon is not None:
            period_end = time_horizon + (horizon_start or latest_time)
        else:
            period_end = None
        # Compile
        static_input = compiled_fn or compile_model
        if compiled_fn:
            prefill_fn = compiled_fn["_forward_prefill"]
            sliding_fn = compiled_fn["_forward_sliding"]
            one_step_fn = compiled_fn["_forward_one_step"]
        elif compile_model:
            # NOTE: _forward_prefill is not compiled here, because it is only called once
            prefill_fn = _forward_prefill
            one_step_fn = torch.compile(
                _forward_one_step, mode="reduce-overhead", fullgraph=True
            )
            sliding_fn = _forward_sliding
        else:
            prefill_fn = _forward_prefill
            one_step_fn = _forward_one_step
            sliding_fn = _forward_sliding

        # Quantize
        if quantize_weights:
            model = quantize_linear_weights(model)

        # Initialize empty products
        prod_timelines = []
        prod_catalog_ids = []

        # Inference loop
        with tqdm(total=n_iter, disable=not show_pbar) as pbar:
            while actives.size(0):
                # *****************
                # * Forward calls *
                # *****************
                # Case 1. First forward call
                if model.cache_length == 0:
                    model.setup_cache(batch_size=n_iter)
                    logits = prefill_fn(
                        model=model, input_tensor=input_tensor[:, : pos + 1, :]
                    )
                # Case 2. First forward call after sliding window
                elif model.cache_length == model.max_sequence_length:
                    model.empty_cache()
                    logits = sliding_fn(
                        model=model, input_tensor=input_tensor[:, : pos + 1, :]
                    )
                # Case 3. All other steps
                else:
                    logits = one_step_fn(
                        model=model, input_tensor=input_tensor[:, pos : pos + 1, :]
                    )

                # ***************
                # * Id sampling *
                # ***************
                probs = model.compute_probs(
                    logits=logits[actives],
                    input_tensor=input_tensor[actives],
                    last_ids=last_ids,
                    pos=pos,
                    current_time=current_time,
                    time_anchor=time_anchor,
                    logits_filter=logits_filter,
                    temperature=temperature,
                )
                sampled_ids = model.sample_from_probs(probs=probs)

                # ******************
                # * Update objects *
                # ******************
                # Check timedelta anchoring status
                if time_anchor is not None:
                    if (current_time >= time_anchor).all():
                        time_anchor = None
                # Update next inputs, current time, and admission status
                next_inputs, current_time = model.generate_next_input(
                    sampled_ids=sampled_ids,
                    input_tensor=input_tensor[actives],
                    pos=pos,
                    current_time=current_time,
                    time_anchor=time_anchor,
                )
                # Update timelines and catalog indexes
                gen_timelines = torch.cat(
                    [gen_timelines, next_inputs.unsqueeze(1)], dim=1
                )
                gen_catalog_ids = torch.cat(
                    [gen_catalog_ids, sampled_ids.unsqueeze(1)], dim=1
                )
                last_ids = gen_catalog_ids[:, -1]
                # Update input tensor
                pos += 1
                if pos == model.max_sequence_length:
                    # Sliding window
                    shifts = -(stride + 1)
                    input_tensor[:, model.n_demographic_rows :, :] = input_tensor[
                        :, model.n_demographic_rows :, :
                    ].roll(shifts=shifts, dims=1)
                    # Shift back the position
                    pos -= stride + 1
                input_tensor[actives, pos] = next_inputs

                # **********************
                # * Check terminations *
                # **********************
                finished_mask = torch.isin(sampled_ids, stop_vocab)
                if period_end is not None:
                    stopped_by_td = current_time >= period_end
                    finished_mask |= torch.tensor(stopped_by_td)
                limit_reached = gen_timelines.size(1) > max_length
                if limit_reached:
                    finished_mask.fill_(True)
                finished_indexes = finished_mask.nonzero().squeeze(-1).tolist()
                finished_indexes = sorted(finished_indexes)

                # ****************************
                # * Remove finished products *
                # ****************************
                for i in finished_indexes:
                    if (not limit_reached) or return_unfinished:
                        # Pick up item
                        fin_tl = gen_timelines[i].unsqueeze(0)
                        fin_id = gen_catalog_ids[i].tolist()
                        # Case 1. Returning the generated products only (without prompts)
                        # NOTE: Even in this case, the demographic rows from the prompts are prepended.
                        if return_generated_parts_only:
                            # Append demographic rows to the generated products
                            first_id = fin_id[0]
                            # If the truncated part does not begin with a timedelta row, insert an timedelta row.
                            if (prompt_t.size(1) > model.n_demographic_rows) and (
                                first_id < model.small_step_start_idx
                            ):
                                fin_id = (
                                    prompt_c[: model.n_demographic_rows]
                                    + [init_td_cat_idx]
                                    + fin_id
                                )
                                fin_tl = torch.cat(
                                    [
                                        prompt_t[:, : model.n_demographic_rows],
                                        initial_td_row.unsqueeze(0),
                                        fin_tl,
                                    ],
                                    dim=1,
                                )
                                assert len(fin_id) == fin_tl.size(
                                    1
                                ), f"Product timeline ({fin_tl.size(1)}) and catalog ids ({len(fin_id)}) have different lengths. (Error within first timedelta row handlings.)"
                            else:
                                fin_id = prompt_c[: model.n_demographic_rows] + fin_id
                                fin_tl = torch.cat(
                                    [prompt_t[:, : model.n_demographic_rows], fin_tl],
                                    dim=1,
                                )
                        # Case 2. Return the generated products with the prompts
                        else:
                            fin_id = prompt_c + fin_id
                            fin_tl = torch.cat([prompt_t, fin_tl], dim=1)

                        assert len(fin_id) == fin_tl.size(
                            1
                        ), f"Product timeline ({fin_tl.size(1)}) and catalog ids ({len(fin_id)}) have different lengths."
                        # Add the product to the product list
                        prod_timelines.append(fin_tl.clone())
                        prod_catalog_ids.append(fin_id)

                # ****************
                # * Trim objects *
                # ****************
                if finished_indexes:
                    # Create mask to select active ones
                    unfinished = torch.ones_like(actives, dtype=torch.bool)
                    unfinished[finished_indexes] = False
                    unfinished = unfinished.tolist()
                    # Select active ones only
                    gen_timelines = gen_timelines[unfinished]
                    gen_catalog_ids = gen_catalog_ids[unfinished]
                    last_ids = last_ids[unfinished]
                    current_time = current_time[unfinished]
                    actives = actives[unfinished]
                    # Change the shape of `input_tensor` if torch.compile() is not used.
                    if (not static_input) and actives.size(0):
                        next_bs = math.ceil(actives.size(0) / 64) * 64
                        if next_bs < input_tensor.size(0):
                            # Sort and trim the input tensor
                            input_tensor[: actives.size(0)] = input_tensor[actives]
                            input_tensor = input_tensor[:next_bs]
                            # Sort and trim caceh
                            model.trim_cache(actives=actives, new_size=next_bs)
                            # Refresh actives
                            actives = torch.arange(0, actives.size(0))

                # *****************
                # * Progress bar *
                # ****************
                if show_pbar:
                    n_finished = len(finished_indexes)
                    pbar.update(n_finished)
                    pbar.set_description(
                        f"[Generated lengths: {str(gen_timelines.size(1))} (input pos = {pos.item()})]"
                    )

        # ************
        # * Clean up *
        # ************
        model.delete_cache()

        return prod_timelines, prod_catalog_ids


def queued_monte_carlo(
    task_queue: Queue,
    product_queue: Queue,
    sentinel: str,
    model: Watcher,
    stride: int = 64,
    stop_vocab: list[int] | None = None,
    max_length: int = 10000,
    return_generated_parts_only: bool = True,
    return_unfinished: bool = False,
    compile_model: bool = False,
    quantize_weights: bool = False,
    logits_filter: str = "default",
    temperature: float = 1.0,
    end_signal: Event = None,
) -> None:
    """Repeats Monte Carlo simulations using a task queue.

    .. warning::
        This function does not join the product queue (`product_queue`) by itself.

    Args:
        task_queue (Queue|JoinableQueue): Task queue.
            Each item in the queue is expected to have the following items:

                - timeline (Tensor): Prompt timeline.
                - catalog_ids (Tensor): Catalog ids paired with the timeline.
                - time_horizon (timedelta): Time horizon.
                - horizon_start (timedelta): Starting time of the simulation.
                - product_id (str|Any): Object that identifies the products.
                    Product IDs of strings are expected, but they can be anything as long as they can be put in the queue.

        product_queue (Queue|JoinableQueue): Generated products are placed in this queue.
            Each item in the queue contains the following items:

                - gen_timelines (list[Tensor]): List of generated timelines.
                - gen_catalog_ids (list[list[int]]): List of catalog id sequences.

        sentinel (str): Sentinel values to make the function to exit.
            Ensure to put this sentinel value in the product queue at the end of all tasks so that the process can safely exit.
        model (Watcher): Instantiated model.
        stride (int): Stride of sliding window during autoregressive inference beyond the max sequence length.
        stop_vocab (list[int]): Simulation is completed once one of the vocabularies listed in this argument.
            Vocabularies must be given in ids (integers).
        max_length (int): Once inference reaches this length, the simulation result is discarded.
        return_generated_parts_only (bool): If true, only the generated parts with demographic rows are returned.
        return_unfinished (bool): If true, timelines that reached the max length (max_length) are returned instead of
            being discarded.
        compile_model (bool): If true, torch.compile() is activated.
        quantize_weights (bool): If true, weights of the linear layers are quantized using Int8.
        logits_filter (str): Filter to limit the logits.
        temperature (float): Temperature.
        end_signal (Event): Event signal that terminates the process.
    Returns:
        None
    """

    def _compile_forward_calls():
        compiled_fn = {
            "_forward_prefill": _forward_prefill,
            "_forward_sliding": _forward_sliding,
            "_forward_one_step": torch.compile(
                _forward_one_step, mode="reduce-overhead", fullgraph=True
            ),
        }
        return compiled_fn

    # Compile forward methods
    if compile_model:
        compiled_fn = _compile_forward_calls()
    else:
        compiled_fn = None

    # Quantize (weight-only)
    if quantize_weights:
        model = quantize_linear_weights(model)

    # Generation loop
    last_n_iter = 0
    while True:
        # Check event
        if end_signal.is_set():
            print("An error event detected. Aborting...")
            break
        # Get a new task
        task = task_queue.get()
        # Checking for the sentinel value
        if task == sentinel:
            break
        # Unpack the task
        if len(task) == 6:
            timeline = task[0]
            catalog_ids = task[1]
            n_iter = task[2]
            horizon_start = task[3]
            time_horizon = task[4]
            product_id = task[5]
        else:
            n_items = len(task)
            raise ValueError(f"Number of items in a task must be 5, but got {n_items}")
        # Reset compiler if necessary
        if compile_model:
            if last_n_iter != n_iter:
                # Reset compiler if the batch sizes changes.
                compiled_fn = _compile_forward_calls()
        last_n_iter = n_iter
        # Generate
        prod_timelines, prod_catalog_ids = monte_carlo(
            model=model,
            timeline=timeline,
            catalog_ids=catalog_ids,
            time_horizon=time_horizon,
            horizon_start=horizon_start,
            n_iter=n_iter,
            stride=stride,
            stop_vocab=stop_vocab,
            max_length=max_length,
            logits_filter=logits_filter,
            temperature=temperature,
            return_generated_parts_only=return_generated_parts_only,
            return_unfinished=return_unfinished,
            show_pbar=False,
            compile_model=False,
            quantize_weights=False,
            compiled_fn=compiled_fn,
        )
        # Ensure tensors are on cpu
        prod_timelines = [t.cpu() for t in prod_timelines]
        # Put prodcuts to the queue
        product_queue.put(
            (prod_timelines, prod_catalog_ids, product_id), timeout=TIMEOUT
        )


def make_sim_dataframe(interpreter, prod_timelines, prod_catalog_ids) -> pd.DataFrame:
    # Create a table and save it
    df = interpreter.create_table(
        prod_timelines,
        prod_catalog_ids,
        readable_timedelta=False,
        patient_id_start=0,
    )

    df[config.COL_AGE] = pd.to_timedelta(df[config.COL_AGE], errors="coerce")
    # Sort the dataframe (Undo shuffling)

    if config.SHUFFLE_INPUT:
        df[config.COL_PID] = df[config.COL_PID].astype(int)
        df = df.sort_values(
            by=[config.COL_PID, config.COL_AGE, config.COL_TYPE, config.COL_CODE],
            ascending=[True, True, True, True],
            ignore_index=True,
        )
    # Modify patient ID
    df[config.COL_PID] = "simulation" + df[config.COL_PID].astype(str)

    return df
