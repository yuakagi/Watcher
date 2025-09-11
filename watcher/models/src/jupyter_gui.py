"""Module to explore Watcher's functionality on Jupyter notebook."""

import os
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import seaborn as sns
from IPython.display import display
from ipywidgets import (
    Button,
    IntSlider,
    FloatText,
    Dropdown,
    HBox,
    VBox,
    interactive_output,
)
from .model_loaders import build_watcher
from ...utils import (
    extract_timedelta_sequence,
    set_figure_params,
)
from ...general_params import watcher_config as config

FONT = "Arial"
FONT_S = 5
FONT_L = 7
LINE_W = 1
LINE_N = 0.5

DEFAULT_ARGS = {
    "start": 0,
    "n_shown": 20,
    "inference_start": 0,
    "n_candidates": 10,
    "logits_filter": "default",
    "temperature": 1.0,
}


class WatcherGui:
    """A graphical user interface designed for interacting with Watcher within Jupyter notebooks."""

    def interpret_predicted_indexes(self, indexes: list, lab_code: str = None) -> list:
        """Interprets catalog indexes into human-readable text.

        Args:
            indexes (list): List of catalog indexes to interpret.
            lab_code (str, optional): medication code used for interpreting lab results.
                Defaults to None.
        Returns:
            list: List of interpreted texts corresponding to the given indexes.
        """

        # Ensure that 'indexes' is a list
        if not isinstance(indexes, list):
            indexes = indexes.tolist()

        # Create a series of plain texts for the given indexes
        texts = self.model.interpreter.catalogs["full"].loc[indexes, config.COL_TEXT]

        # Interpret laboratory test result values if a lab code is given
        if lab_code is not None:
            original_vals = self.model.interpreter.catalogs["full"].loc[
                indexes, config.COL_ORIGINAL_VALUE
            ]

            # Non-numerics
            df = pd.DataFrame(
                {config.COL_TEXT: texts, config.COL_ORIGINAL_VALUE: original_vals}
            )
            df[config.COL_ITEM_CODE] = lab_code
            # Map the nonnumeric laboratory value tokens back into texts
            df = pd.merge(
                df,
                self.model.interpreter.nonnumeric_result_table,
                left_on=[config.COL_ITEM_CODE, config.COL_ORIGINAL_VALUE],
                right_on=[config.COL_ITEM_CODE, config.COL_TOKEN],
                how="left",
            )
            nonnum_mask = ~df[config.COL_NONNUMERIC].isna()
            # Overwrite texts
            df[config.COL_TEXT] = df[config.COL_TEXT].mask(
                nonnum_mask, df[config.COL_NONNUMERIC]
            )

            # Numerics
            # <-- Z-score --> (depricated)
            # <-- Percentile -->
            num_series = pd.to_numeric(df[config.COL_ORIGINAL_VALUE], errors="coerce")
            num_mask = num_series.notna()
            nums = num_series.loc[num_mask].values
            unscaled = self.model.interpreter.translate_numerics(
                lab_codes=np.full(nums.shape, lab_code), numerics=nums
            )
            df.loc[num_mask, config.COL_TEXT] = unscaled

            texts = df[config.COL_TEXT]

        # Finalize
        texts = texts.to_list()

        return texts

    def _refresh_attributes(self):
        self.latest_args = DEFAULT_ARGS
        self.top_indexes = None
        self.probs = None
        self.inference_browsing_df = None
        self.display_numeric = False

    def __init__(
        self,
        blueprint: str,
        temperature=1.0,
        device=None,
    ):
        """Initialization"""
        # Attributes
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.temperature = temperature
        colors = ["white", "#9b2726"]
        custom_cmap = LinearSegmentedColormap.from_list("CustomReds", colors)
        self.cmap = custom_cmap
        self.blueprint = blueprint
        self.model = build_watcher(blueprint=blueprint, train=False)
        self.model = self.model.to(self.device)
        self.latest_args = DEFAULT_ARGS
        self.top_indexes = None
        self.probs = None
        self.inference_browsing_df = None
        self.display_numeric = False
        self.catalog_indexes = None
        self.input_sequence = None
        self.texts = None
        self.codes = None
        self.numeric_sequence = None
        self.timedelta_indexes = None
        self.timedelta_sequence = None
        self.sequence_length = None
        self.max_sequence_start = None
        self.timeline_browsing_df = None
        self.next_label = None
        self.next_label_index = None
        self.numeric_scores = None
        self.numeric_label = None
        self.fig = None
        self.last_x = None

    def set_timeline(self, timeline: torch.Tensor, catalog_indexes: list):
        """
        Loads a patient timeline and prepares it for further analysis.

        Args:
            timeline (torch.Tensor): Tensor representing the patient timeline.
            catalog_indexes (list): List of catalog indexes corresponding to the timeline.
        Returns:
            None
        """

        # Assign data to attributes
        self.input_sequence, self.catalog_indexes = self.model.remove_padding_rows(
            timeline, catalog_indexes
        )
        self.input_sequence = self.input_sequence.to(self.model.device)
        (
            self.texts,
            self.codes,
            self.numeric_sequence,
            self.timedelta_indexes,
            _,
            _,
        ) = self.model.interpreter.interpret_timeline(
            self.input_sequence, self.catalog_indexes, readable_timedelta=True
        )
        self.timedelta_sequence = extract_timedelta_sequence(self.input_sequence)[0]
        self.numeric_sequence = np.array(self.numeric_sequence)
        self.sequence_length = self.input_sequence.size(1)
        self.max_sequence_start = max(
            0, self.sequence_length - self.latest_args["n_shown"]
        )
        # Create a template dataframe for timeline browser
        self.timeline_browsing_df = pd.DataFrame({config.COL_TEXT: self.texts})
        self.timeline_browsing_df["masked"] = False
        self.timeline_browsing_df["attention"] = np.nan
        # Add double spaces before texts other than timedeltas for better visualization
        self.timeline_browsing_df[config.COL_TEXT] = (
            "  " + self.timeline_browsing_df[config.COL_TEXT]
        )
        self.timeline_browsing_df.loc[self.timedelta_indexes, config.COL_TEXT] = (
            self.timeline_browsing_df.loc[
                self.timedelta_indexes, config.COL_TEXT
            ].str.strip()
        )
        # Refresh other common attributes
        self._refresh_attributes()

    def create_table(self) -> pd.DataFrame:
        """
        Creates a tabular representation of a patient timeline.
        Returns:
            pd.DataFrame: DataFrame representing the timeline in tabular format.
        """
        df = self.model.interpreter.create_table(
            timeline_list=[self.input_sequence],
            catalog_idx_list=[self.catalog_indexes],
            readable_timedelta=True,
        )
        return df

    @torch.inference_mode()
    def _demo_retrospective(
        self,
        sequence_start=0,
        n_shown=20,
        inference_start=0,
        n_candidates=10,
        logits_filter="default",
        temperature=1.0,
        debug_mode="default",
    ):
        """
        Retrospectively analyzes a patient timeline using Watcher in a Jupyter notebook environment.

        Args:
            sequence_start (int): Starting point of the timeline to display.
            n_shown (int): Number of sequence elements to show.
            inference_start (int): Point in the sequence to begin inference.
            n_candidates (int): Number of next-step candidates to show in the inference output.
            logits_filter (str): Filter applied to logits (e.g., 'default', 'diagnoses').
            temperature (float): Temperature used for sampling logits.
            debug_mode (str): Specifies the debug mode for inference (e.g., 'default', 'use_cache').
        Returns:
            None
        """
        # Get figsize from the environment
        width, heigt = os.environ["DEMO_FIGSIZE"].split("_")
        dpi = int(os.environ["DEMO_DPI"])
        figsize = (float(width), float(heigt))
        # Set general props
        _ = set_figure_params(dpi=dpi)
        # Initialize fig and ax
        fig = plt.figure(figsize=figsize, facecolor="#f7f5ee")
        plt.suptitle("Simulation Visualizer", fontsize=FONT_L)
        ax0 = plt.subplot(121)
        renderer = fig.canvas.get_renderer()
        self.max_sequence_start = self.sequence_length - n_shown
        self.temperature = temperature

        # *************
        # * Inference *
        # *************
        # Detect changes
        inference_needed = False
        refresh_needed = self.latest_args["n_candidates"] != n_candidates
        inspected_keys = [
            "inference_start",
            "logits_filter",
            "temperature",
            "debug_mode",
        ]
        inspected_vals = [inference_start, logits_filter, temperature]
        for key, val in zip(inspected_keys, inspected_vals):
            if self.latest_args[key] != val:
                inference_needed = True
                break

        if inference_needed:
            # Debug params
            use_cache = debug_mode == "use_cache"

            if inference_start > config.DEMOGRAPHIC_ROWS:
                # Separate key rows
                demographic_rows = self.input_sequence[
                    :, 0 : config.DEMOGRAPHIC_ROWS, :
                ]
                last_row = self.input_sequence[
                    :, inference_start : inference_start + 1, :
                ]
                slice_end = inference_start

                # Slice out intermediate rows so that the total input length is max_sequence_length - 1 or shorter
                min_slice_start = (
                    slice_end
                    - self.model.max_sequence_length
                    + 1
                    + config.DEMOGRAPHIC_ROWS
                )
                slice_start = max(config.DEMOGRAPHIC_ROWS, min_slice_start)
                intermediate_rows = self.input_sequence[:, slice_start:slice_end, :]
                # Concatenate all rows to create an inference input
                truncated_input = torch.cat(
                    [demographic_rows, intermediate_rows, last_row], dim=1
                )
                # Record sliced indexes
                input_indexes = (
                    list(range(config.DEMOGRAPHIC_ROWS))
                    + list(range(slice_start, slice_end))
                    + [inference_start]
                )
            else:
                truncated_input = self.input_sequence[:, 0 : inference_start + 1, :]
                input_indexes = list(range(inference_start + 1))

            # Prepare for inference
            with torch.device(self.model.device):
                last_ids = torch.tensor(
                    self.catalog_indexes[inference_start]
                ).unsqueeze(0)
                x = truncated_input

            # Main inference
            self.last_x = truncated_input
            if use_cache and (x.size(1) > 1):
                # Simulate caching
                self.model.setup_cache(x.size(0))
                for i in range(0, x.size(1) - 1):
                    _ = self.model(
                        x[:, i : i + 1, :],
                        use_kv_cache=True,
                        record_attention_weights=False,
                    )
                logits = logits = self.model(
                    x[:, -1:, :],
                    use_kv_cache=True,
                    record_attention_weights=True,
                )
                self.model.delete_cache()
            else:
                logits = self.model(
                    x,
                    use_kv_cache=False,
                    record_attention_weights=True,
                )
            logits = logits[:, -1:, :]
            self.probs = self.model.compute_probs(
                input_tensor=x,
                last_ids=last_ids,
                pos=-1,
                logits=logits,
                current_time=None,
                time_anchor=None,
                logits_filter=logits_filter,
                temperature=self.temperature,
            ).view(-1)

            # Get attention map
            attention_map = self.model.get_attention_map()
            attention_map = attention_map[: len(input_indexes)]
            max_attention = attention_map.max()
            min_attention = attention_map.min()
            diff = max_attention - min_attention
            attention_map = (attention_map - min_attention) / diff
            attention_map = attention_map.tolist()
            attention_map = pd.Series(data=attention_map, index=input_indexes)

            # Update the dataframe for timeline browsing
            self.timeline_browsing_df["masked"] = True
            self.timeline_browsing_df["attention"] = np.nan
            self.timeline_browsing_df.loc[input_indexes, "masked"] = False
            self.timeline_browsing_df.loc[input_indexes, "attention"] = attention_map

            # Get the label to be predicted
            if inference_start != self.sequence_length - 1:
                self.next_label = self.texts[inference_start + 1]
                self.next_label_index = self.catalog_indexes[inference_start + 1]
            else:
                self.next_label = None
                self.next_label_index = None

        # **********************
        # * Timeline browser *
        # **********************
        # Set axes
        ax0.set_title("Timeline browser", fontsize=FONT_L)
        ax0.set_facecolor(mcolors.to_rgba("#cce7ee", alpha=0.5))
        ax0.set_ylim(0, 100)
        ax0.set_xlim(0, 10)
        ax0.xaxis.set_tick_params(bottom=False, labelbottom=False)
        ax0.yaxis.set_tick_params(bottom=False, labelleft=False)
        ax0.vlines(1, 0, 100, color="#ca3f3f")  # The vertical stem
        ys = np.linspace(10, 90, num=n_shown)
        ys = np.flip(ys)

        # Slice out dataframe for plotting
        rows_shown = self.timeline_browsing_df.iloc[
            sequence_start : sequence_start + n_shown, :
        ]

        # Plot
        for i, row in rows_shown.iterrows():
            # Initialize variables
            text, masked, attention = row
            # Determine y axis position
            y = ys[i - sequence_start]
            # Set annotation parameters
            text_alpha = 0.5 if masked else 1.0
            if np.isnan(attention):
                box_color = "none"
            else:
                box_color = self.cmap(attention)
            # Plot the row number and a marker for it
            ax0.text(
                0.9,
                y,
                i,
                horizontalalignment="right",
                verticalalignment="center",
                fontsize=FONT_S,
            )
            ax0.plot(
                1,
                y,
                "-o",
                markersize=1.5,
                markeredgewidth=0.5,
                color="k",
                markerfacecolor="w",
            )
            # Plot text
            plotted_annot = ax0.annotate(
                text,
                (1.2, y),
                alpha=text_alpha,
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=FONT_S,
                bbox={
                    "facecolor": box_color,
                    "edgecolor": "none",
                    "alpha": 0.5,
                    "pad": 1,
                },
            )
            # Annotate the end of input sequence
            if (self.probs is not None) and (
                i in [inference_start, inference_start + 1]
            ):
                if i == inference_start:
                    anot_face_color = "#c8cdda"
                    anot_text = "Prompt end"
                else:
                    anot_face_color = "#a0ca79"
                    anot_text = "Target"
                text_box = plotted_annot.get_window_extent(renderer)
                arrow_pos = ax0.transData.inverted().transform(
                    (
                        text_box.x1,
                        text_box.y0 + text_box.height / 2,
                    )
                )
                arrow_end_x, arrow_end_y = arrow_pos
                text_position_x = arrow_end_x + 0.5
                text_position_y = arrow_end_y
                ax0.annotate(
                    anot_text,
                    xy=arrow_pos,
                    xytext=(text_position_x, text_position_y),
                    horizontalalignment="left",
                    verticalalignment="center",
                    fontsize=FONT_S,
                    bbox={
                        "facecolor": anot_face_color,
                        "edgecolor": "black",
                        "linewidth": LINE_N,
                        "alpha": 1,
                        "pad": 1,
                    },
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "black",
                        "lw": LINE_N,
                    },
                )

        # ******************
        # * Inference view *
        # ******************
        # Show the inference window only if any inference has been already performed
        if (self.probs is not None) and (inference_needed or refresh_needed):
            # Pick up next prediction candidates
            self.top_indexes = torch.topk(self.probs, n_candidates).indices.tolist()
            numeric_indexes = self.model.interpreter.catalog_index_lists[
                "numeric_lab_values"
            ]
            # See if the top candidate is numeric
            if self.top_indexes[0] in numeric_indexes:
                # Set the flag for numeric prediction display to True
                self.display_numeric = True
                # Get numeric value distribution
                self.numeric_scores = self.probs[numeric_indexes].float().cpu().numpy()
                # Get the next numeric label
                if self.next_label is not None:
                    self.numeric_label = self.numeric_sequence[inference_start + 1]
                else:
                    self.numeric_label = None
            else:
                self.display_numeric = False

            last_category_index = self.catalog_indexes[inference_start]
            top_scores = self.probs[self.top_indexes].float().cpu().numpy()
            if (
                last_category_index
                in self.model.interpreter.catalog_index_lists[config.LAB_CODE]
            ):
                lab_code = self.codes[inference_start]
            else:
                lab_code = None
            # Candidates
            candidates = self.interpret_predicted_indexes(self.top_indexes, lab_code)
            self.inference_browsing_df = pd.DataFrame(
                {
                    "candidate": candidates,
                    "indexes": self.top_indexes,
                    "score": top_scores,
                }
            )
            self.inference_browsing_df = self.inference_browsing_df.astype(
                dtype={"candidate": str, "indexes": int, "score": float}
            )
            self.inference_browsing_df = self.inference_browsing_df.sort_values(
                "score", ascending=False
            )

        # Create axes
        if self.display_numeric:
            # With numeric distribution
            # ax1 = plt.subplot(222)
            gs = gridspec.GridSpec(
                2, 2, height_ratios=[1, 2]
            )  # Set height_ratios for rows
            ax1 = fig.add_subplot(gs[0, 1])  # Smaller subplot in the top-right
            ax2 = fig.add_subplot(gs[1, 1])  # Larger subplot in the bottom-right

            # ax1.get_xaxis().set_visible(False)
            ax1.xaxis.set_tick_params(labelsize=FONT_S)
            ax2 = plt.subplot(224)
            ax2.xaxis.set_tick_params(labelsize=FONT_S)
            ax2.yaxis.set_tick_params(labelsize=FONT_S)
            # <-- Percentile -->
            ax2.set_xlabel("Percentile", fontsize=FONT_S)
            ax2.set_ylabel("Probability", fontsize=FONT_S)
            plt.subplots_adjust(hspace=0.5)

        else:
            # Without numeric distribution
            ax1 = plt.subplot(122)
            ax1.xaxis.set_tick_params(labelsize=FONT_S)

        ax1.get_yaxis().set_visible(False)
        ax1.set_xlabel("Probability", fontsize=FONT_S)
        ax1.set_ylabel("Candidates", fontsize=FONT_S)
        ax1.set_title("Candidates with probability", fontsize=FONT_L)

        # Plot next token candidates and confidence scores
        if isinstance(self.inference_browsing_df, pd.DataFrame):
            sns.set_color_codes("pastel")
            sns.barplot(
                x="score",
                y="candidate",
                data=self.inference_browsing_df,
                label="Probability",
                color="#9ccbec",
                ax=ax1,
                errorbar=None,
                # Hide legend
                legend=False,
            )
            max_score = self.inference_browsing_df["score"].max()
            # Plot each candidate
            for i in range(n_candidates):
                text = self.inference_browsing_df["candidate"].iloc[i]
                index = self.inference_browsing_df["indexes"].iloc[i]
                plotted_annot = ax1.annotate(
                    " " + text,
                    (
                        0,
                        i,
                    ),
                    horizontalalignment="left",
                    verticalalignment="center",
                    fontsize=FONT_S,
                )
                # Annotate if the next prediction label appears in the candidates
                if index == self.next_label_index:
                    text_box = plotted_annot.get_window_extent(renderer)
                    bbox_position = ax1.transData.inverted().transform(
                        (text_box.x1, text_box.y0)
                    )
                    bbox_position[0] += max_score * 0.05
                    ax1.annotate(
                        "Target",
                        (bbox_position),
                        horizontalalignment="left",
                        verticalalignment="bottom",
                        fontsize=FONT_S,
                        bbox={
                            "facecolor": "#a0ca79",
                            "edgecolor": "black",
                            "linewidth": 0.5,
                            "alpha": 1,
                            "pad": 1,
                        },
                    )

        # Disable the inference window if no inference has been performed
        else:
            ax1.set_facecolor("grey")
            ax1.text(
                0.5,
                0.5,
                "Disabled",
                horizontalalignment="center",
                verticalalignment="center",
                color="white",
                fontsize=FONT_L,
            )

        # Plot numeric distribution
        if self.display_numeric:
            # Plot hist
            ax2.set_title("Numeric value prediction", fontsize=FONT_L)

            # <-- Percentile -->
            stem_xs = self.model.interpreter.percentile_steps * 100
            ax2.set_xlim(0, 100)
            stem_container = ax2.stem(
                stem_xs,
                self.numeric_scores,
                label="Probability",
                linefmt="#0172b3",
                markerfmt=" ",
                basefmt="#625f4a",
            )

            # Set the line width
            plt.setp(stem_container.stemlines, linewidth=LINE_W)
            plt.setp(stem_container.baseline, linewidth=LINE_W)

        # Update values
        self.latest_args = {
            "sequence_start": sequence_start,
            "inference_start": inference_start,
            "n_shown": n_shown,
            "n_candidates": n_candidates,
            "logits_filter": logits_filter,
            "temperature": temperature,
            "debug_mode": debug_mode,
        }

        # Save the current fig as an attribute
        self.fig = fig

        plt.show()

    def demo_retrospective(self, figsize=(7.08661, 3.14961), dpi=1000, debug=False):
        """Launch the retrospective demo on Jupyter."""
        # Write the figsize to the environment
        # NOTE: You can not pass figsize (a tuple) through 'interactive_output()', therefore it is written here.
        os.environ["DEMO_FIGSIZE"] = f"{figsize[0]}_{figsize[1]}"
        os.environ["DEMO_DPI"] = str(dpi)

        # Define the interface modules
        filter_choices = [
            "default",
            "diagnoses",
            "drugs",
            "labs_and_results",
            "numeric",
        ]
        debug_choices = ["default", "use_cache"]
        start_bar = IntSlider(
            min=0,
            max=self.max_sequence_start,
            description="Shown from",
        )
        n_shown_bar = IntSlider(min=20, max=50, step=1, description="N shown")
        inf_point_bar = IntSlider(
            min=0,
            max=self.sequence_length - 1,
            description="Inference->",
        )
        candid_bar = IntSlider(min=10, max=50, step=1, description="N candidates")
        filter_drop = Dropdown(options=filter_choices, description="Filter")
        save_fig_button = Button(description="Save Fig")
        temp_txt = FloatText(value=1.0, description="Temperature")
        debug_drop = Dropdown(options=debug_choices, description="Debug mode")
        # Button to save figure
        save_fig_button = Button(description="Save Fig")

        def save_figure_callback(button):
            """Function to handle button click and save figure."""
            print("Saving figure...")  # Debugging output
            fig_dir = os.path.join("gui_figures/")
            os.makedirs(fig_dir, exist_ok=True)
            time_str = datetime.now().strftime("%Y%m%d%H%M")
            self.fig.savefig(
                os.path.join(fig_dir, f"gui_{time_str}.svg"),
                format="svg",
                transparent=False,
                bbox_inches="tight",
                dpi=dpi,
            )
            print("Figure saved.")

        save_fig_button.on_click(save_figure_callback)  # Bind function to button click

        if debug:
            # Replacing the masking choices for debug
            h1 = HBox([start_bar, n_shown_bar, debug_drop, save_fig_button])
        else:
            h1 = HBox([start_bar, n_shown_bar, save_fig_button])
        h2 = HBox([inf_point_bar, candid_bar, filter_drop, temp_txt])
        interface = VBox([h1, h2])

        # Start demo
        out = interactive_output(
            self._demo_retrospective,
            {
                "sequence_start": start_bar,
                "n_shown": n_shown_bar,
                "inference_start": inf_point_bar,
                "n_candidates": candid_bar,
                "logits_filter": filter_drop,
                "temperature": temp_txt,
                "debug_mode": debug_drop,
            },
        )

        display(out, interface)
