#!/usr/bin/env python3
"""Generate notebook-sourced composite figures for analysis v5.

The notebook remains the source of truth for analysis calculations and the
individual plots. This script executes only the required notebook cells, sends
their normal PNG/CSV outputs to the supplement, and builds the requested main
composites under ``analysis/outputs/paper_plot_v5``.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from PIL import Image, ImageChops, ImageDraw, ImageFont
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "analysis" / "analysis_v5.ipynb"
OUTPUT_ROOT = REPO_ROOT / "analysis" / "outputs" / "paper_plot_v5"
SUPPLEMENT_ROOT = OUTPUT_ROOT / "supplement"
DPI = 600

CONTROLLED_DIR = OUTPUT_ROOT / "controlled_temporal_belief_construction"
REVISION_DIR = OUTPUT_ROOT / "conversational_revision_and_stability"
LONG_HORIZON_DIR = OUTPUT_ROOT / "long_horizon_task_completion"
CROSS_DOMAIN_DIR = OUTPUT_ROOT / "cross_domain_diagnostics"

NOTEBOOK_HEADINGS = [
    "## Setup and shared data assets",
    "### Shared same-session trajectory asset",
    "### Panel 2. Track 1 Active Performance Relative to Full Context",
    "### Panel 3. Temporal Map Component Profile",
    "### Panel 6. Semantic Temporal Belief Profile and Component-Level Active Versus Full-Context Gap",
    "### Panel 7. Semantic Revision and Belief Inertia",
    "### Panel 8. Same-Session Semantic Belief Trajectory and Stability",
    "### Panel 10. Long-Horizon Active Task and Variant Profile",
    "### Panel 11. Navigation and Marginal Completion-Stage Profile",
    "### Panel 13. Evidence Acquisition, Efficiency, and Belief Dissociation",
    "### Panel 14. Uncertainty Awareness and Supported Commitment",
]

REQUIRED_NAMESPACE = {
    "active_1c",
    "gap_1c_significance",
    "gap_1c_support",
    "gap_1c_table",
    "profile_1c_support",
    "profile_1c_table",
    "profile_table",
    "lines",
    "revision",
    "summary_1c",
    "t2_profile",
    "track2_cell_order",
    "general_summary",
    "combined",
    "combined_scatter",
    "MODEL_COLORS",
    "short_model_label",
    "t2_efficiency_long",
    "track1_uncertainty",
    "track1_uncertainty_significance",
    "track1_uncertainty_support",
    "track1_uncertainty_table",
    "t2_uncertainty",
    "t2_uncertainty_support",
    "t2_uncertainty_table",
}


def _cell_source_after_heading(notebook: dict[str, Any], heading: str) -> tuple[str, str]:
    """Return the first code cell following one exact markdown heading."""
    cells = notebook.get("cells")
    assert isinstance(cells, list), "Notebook has no cell list"
    matches = []
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        lines = {line.strip() for line in "".join(cell.get("source", [])).splitlines()}
        if heading in lines:
            matches.append(index)
    assert len(matches) == 1, f"Expected one notebook heading {heading!r}; found {len(matches)}"
    for cell in cells[matches[0] + 1 :]:
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            assert source.strip(), f"Code cell following {heading!r} is empty"
            return str(cell.get("id", "unknown")), source
    raise AssertionError(f"No code cell follows notebook heading {heading!r}")


def _execute_notebook_cells() -> dict[str, Any]:
    """Execute the selected current notebook cells in a shared namespace."""
    assert NOTEBOOK_PATH.is_file(), f"Missing source notebook: {NOTEBOOK_PATH}"
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    selected = [
        (heading, *_cell_source_after_heading(notebook, heading))
        for heading in NOTEBOOK_HEADINGS
    ]
    namespace: dict[str, Any] = {
        "__file__": str(NOTEBOOK_PATH),
        "__name__": "__analysis_v5_notebook__",
    }
    previous_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT)
        setup_heading, setup_id, setup_source = selected[0]
        exec(
            compile(setup_source, f"{NOTEBOOK_PATH}:{setup_id}", "exec"),
            namespace,
        )
        global _rename_v5_figure_text
        _rename_v5_figure_text = namespace["rename_v5_figure_text"]
        namespace["ANALYSIS_OUTPUT_DIR"] = SUPPLEMENT_ROOT
        namespace["display"] = lambda *_args, **_kwargs: None
        namespace["ipython_display"] = lambda *_args, **_kwargs: None
        namespace["plt"].show = lambda *_args, **_kwargs: None
        print(f"Executed {setup_heading} ({setup_id})")

        for heading, cell_id, source in selected[1:]:
            try:
                exec(
                    compile(source, f"{NOTEBOOK_PATH}:{cell_id}", "exec"),
                    namespace,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed while executing notebook cell for {heading!r} ({cell_id})"
                ) from exc
            finally:
                namespace["plt"].close("all")
            print(f"Executed {heading} ({cell_id})")
    finally:
        os.chdir(previous_cwd)

    missing = sorted(REQUIRED_NAMESPACE - namespace.keys())
    assert not missing, f"Notebook execution did not define required objects: {missing}"
    return namespace


def _mkdirs() -> None:
    for directory in [
        SUPPLEMENT_ROOT,
        CONTROLLED_DIR,
        REVISION_DIR,
        LONG_HORIZON_DIR,
        CROSS_DOMAIN_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def _save_figure(
    figure: plt.Figure,
    path: Path,
    *,
    tight: bool = True,
) -> None:
    assert path.suffix == ".png", f"Expected PNG output, got {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    _rename_v5_figure_text(figure)
    figure.savefig(
        path,
        format="png",
        dpi=DPI,
        bbox_inches="tight" if tight else None,
        facecolor="white",
    )
    plt.close(figure)
    assert path.is_file(), f"Figure was not written: {path}"


def _copy_asset(source: Path, destination: Path) -> None:
    assert source.is_file(), f"Missing notebook-generated asset: {source}"
    assert source.suffix in {".png", ".csv"}, f"Unsupported source asset: {source}"
    assert destination.suffix == source.suffix
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    assert destination.is_file(), f"Asset was not copied: {destination}"
    if source.suffix == ".csv":
        assert_frame_equal(
            pd.read_csv(source),
            pd.read_csv(destination),
            check_dtype=False,
        )
    else:
        assert source.read_bytes() == destination.read_bytes()


def _selective_annotations(
    namespace: dict[str, Any],
    values: pd.DataFrame,
    counts: pd.DataFrame,
) -> pd.DataFrame:
    return namespace["selective_support_annotations"](values, counts)


def _annotate_missing(
    namespace: dict[str, Any],
    axis: plt.Axes,
    values: pd.DataFrame,
    counts: pd.DataFrame | None = None,
) -> None:
    namespace["annotate_missing_heatmap_cells"](axis, values, counts)


def _render_controlled_composite(namespace: dict[str, Any]) -> Path:
    acoustic = namespace["profile_table"].copy()
    conversational = namespace["profile_1c_table"].copy()
    conversational_support = namespace["profile_1c_support"].copy()
    acoustic_column_parts = [str(column).rsplit("\n", 1) for column in acoustic.columns]
    assert all(len(parts) == 2 for parts in acoustic_column_parts), (
        "Panel 3 display columns must end with a track label"
    )
    acoustic_components = [parts[0] for parts in acoustic_column_parts]
    acoustic_tracks = [parts[1] for parts in acoustic_column_parts]
    component_order = list(dict.fromkeys(acoustic_components))
    assert all(
        acoustic_components.count(component) == 2 for component in component_order
    ), "Panel 3 must expose one Track 1A/1B pair per component"
    assert acoustic_components == [
        component for component in component_order for _ in range(2)
    ], "Panel 3 component columns must remain in adjacent Track 1A/1B pairs"
    model_order = sorted(set(acoustic.index) | set(conversational.index))
    acoustic = acoustic.reindex(model_order)
    conversational = conversational.reindex(model_order)
    conversational_support = conversational_support.reindex(model_order).fillna(0)

    figure = plt.figure(figsize=(16.2, 5.8))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[acoustic.shape[1], conversational.shape[1]],
        left=0.07,
        right=0.93,
        bottom=0.22,
        top=0.88,
        wspace=0.045,
    )
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    colorbar_axis = figure.add_axes([0.945, 0.25, 0.014, 0.56])

    sns.heatmap(
        100 * acoustic,
        mask=acoustic.isna(),
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        cbar=True,
        cbar_ax=colorbar_axis,
        ax=axes[0],
    )
    axes[0].set_facecolor("#e6e6e6")
    _annotate_missing(namespace, axes[0], acoustic)
    for boundary in range(2, acoustic.shape[1], 2):
        axes[0].axvline(boundary, color="white", linewidth=2.5)
    axes[0].set(
        title="(a) Controlled acoustic temporal mapping",
        xlabel="",
        ylabel="",
    )
    axes[0].set_xticklabels(
        acoustic_tracks,
        rotation=0,
    )
    for group_index, component in enumerate(component_order):
        axes[0].text(
            2 * group_index + 1,
            -0.14,
            component,
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )

    conversational_annotations = _selective_annotations(
        namespace,
        conversational,
        conversational_support,
    )
    sns.heatmap(
        100 * conversational,
        mask=conversational.isna(),
        annot=conversational_annotations,
        fmt="",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        cbar=False,
        ax=axes[1],
    )
    axes[1].set_facecolor("#e6e6e6")
    _annotate_missing(
        namespace,
        axes[1],
        conversational,
        conversational_support,
    )
    # Keep the enlarged score/count annotations within their existing rows.
    for annotation in axes[1].texts:
        annotation.set_linespacing(1.0)
    axes[1].set(
        title="(b) Controlled conversational belief profile",
        xlabel="",
        ylabel="",
    )
    axes[1].tick_params(axis="x", rotation=0, labelsize=9)
    axes[1].tick_params(axis="y", labelleft=False)
    colorbar_axis.set_ylabel("Score (%)")

    output = (
        CONTROLLED_DIR / "plot_controlled_temporal_belief_construction.png"
    )
    _save_figure(figure, output)
    return output


def _render_panel_6b(namespace: dict[str, Any]) -> Path:
    values = namespace["gap_1c_table"].copy()
    support = namespace["gap_1c_support"].copy()
    significance = namespace["gap_1c_significance"].copy()
    finite = values.to_numpy(dtype=float)
    gap_limit = max(0.05, np.nanmax(np.abs(finite))) if finite.size else 0.05

    figure, axis = plt.subplots(figsize=(7.4, 5.5))
    annotations = _selective_annotations(namespace, values, support)
    sns.heatmap(
        100 * values,
        mask=values.isna(),
        annot=annotations,
        fmt="",
        cmap="vlag",
        center=0,
        vmin=-100 * gap_limit,
        vmax=100 * gap_limit,
        cbar_kws={"label": "Gap (pp)", "shrink": 0.85},
        ax=axis,
    )
    axis.set_facecolor("#e6e6e6")
    _annotate_missing(namespace, axis, values, support)
    namespace["mark_significant_heatmap_cells"](axis, significance)
    axis.set(
        title="Panel 6B. Matched full context - active",
        xlabel="Component",
        ylabel="Model",
    )
    axis.tick_params(axis="x", rotation=0, labelsize=9)
    namespace["add_significance_caption"](
        figure,
        "all displayed Panel 6B contrast cells",
    )
    figure.tight_layout(rect=[0, 0.05, 1, 1])

    output = (
        SUPPLEMENT_ROOT
        / "panel_06"
        / "plot_component_active_vs_full_context_gap.png"
    )
    _save_figure(figure, output)
    return output


def _font(size: int) -> ImageFont.FreeTypeFont:
    font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
    return ImageFont.truetype(font_path, size=size)


def _stack_images(
    panels: list[tuple[str, Path]],
    output: Path,
) -> list[tuple[Path, tuple[int, int, int, int]]]:
    """Stack PNGs without resampling and verify every embedded pixel region."""
    assert panels, "At least one image is required"
    images: list[tuple[str, Path, Image.Image]] = []
    for label, path in panels:
        assert path.is_file(), f"Missing source plot: {path}"
        with Image.open(path) as source:
            images.append((label, path, source.convert("RGB").copy()))

    width = max(image.width for _, _, image in images)
    label_font = _font(max(44, round(width / 125)))
    header_height = max(100, round(width / 70))
    inter_panel_padding = max(36, round(width / 220))
    height = sum(
        header_height + image.height + inter_panel_padding
        for _, _, image in images
    ) - inter_panel_padding
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    placements: list[tuple[Path, tuple[int, int, int, int]]] = []
    y = 0
    for label, path, image in images:
        draw.text(
            (inter_panel_padding, y + inter_panel_padding // 2),
            label,
            fill="black",
            font=label_font,
        )
        image_y = y + header_height
        image_x = (width - image.width) // 2
        canvas.paste(image, (image_x, image_y))
        placements.append(
            (
                path,
                (
                    image_x,
                    image_y,
                    image_x + image.width,
                    image_y + image.height,
                ),
            )
        )
        y = image_y + image.height + inter_panel_padding

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", dpi=(DPI, DPI), compress_level=6)
    assert output.is_file(), f"Composite was not written: {output}"

    with Image.open(output) as rendered:
        rendered_rgb = rendered.convert("RGB")
        for source_path, bounds in placements:
            with Image.open(source_path) as source:
                source_rgb = source.convert("RGB")
                embedded = rendered_rgb.crop(bounds)
                difference = ImageChops.difference(source_rgb, embedded)
                assert difference.getbbox() is None, (
                    f"Composite resampled or changed source pixels from {source_path}"
                )
    return placements


def _render_revision_composite(namespace: dict[str, Any]) -> Path:
    revision = namespace["revision"].dropna(
        subset=["target_evidence_found", "update_success"]
    ).copy()
    lines = namespace["lines"].copy()
    short_model_label = namespace["short_model_label"]
    model_colors = namespace["MODEL_COLORS"]

    revision["Model"] = revision["model"].map(short_model_label)
    revision["Stale answer rate"] = (
        revision["stale_answer_rate_given_evidence"].fillna(0).clip(0, 1)
    )
    revision["Stale-rate status"] = np.where(
        revision["stale_answer_rate_given_evidence"].notna(),
        "Evaluable",
        "N/A",
    )
    scenario_order = revision["scenario"].drop_duplicates().tolist()
    model_order = revision["Model"].drop_duplicates().tolist()
    assert len(scenario_order) == 6
    assert len(model_order) == 8
    assert revision[["target_evidence_found", "update_success"]].apply(
        lambda column: column.between(0, 1).all()
    ).all()
    missing_model_colors = sorted(set(model_order) - set(model_colors))
    assert not missing_model_colors, (
        f"Missing revision model colors: {missing_model_colors}"
    )

    trajectory_models = sorted(lines["model"].unique(), key=short_model_label)
    assert len(trajectory_models) == 7
    assert lines["normalized_progress"].nunique() == 6
    assert lines[["normalized_progress", "trajectory_score"]].apply(
        lambda column: column.between(0, 1).all()
    ).all()
    trajectory_colors = {
        model: model_colors[short_model_label(model)]
        for model in trajectory_models
    }

    original_rc_params = matplotlib.rcParams.copy()
    matplotlib.rcdefaults()
    matplotlib.rcParams.update(
        {
            "font.size": 5.6,
            "axes.titlesize": 6.6,
            "axes.labelsize": 6.3,
            "xtick.labelsize": 5.3,
            "ytick.labelsize": 5.3,
            "legend.fontsize": 4.8,
            "legend.title_fontsize": 5.2,
            "axes.linewidth": 0.5,
            "axes.grid": False,
        }
    )
    figure = plt.figure(figsize=(7.15, 3.6))
    outer = figure.add_gridspec(
        2,
        2,
        width_ratios=[2.25, 1.0],
        hspace=0.15,
        wspace=0.16,
        left=0.058,
        right=0.985,
        bottom=0.18,
        top=0.83,
    )
    scatter_grid = outer[:, 0].subgridspec(
        2,
        3,
        hspace=0.34,
        wspace=0.12,
    )
    scatter_axes = np.array(
        [
            [
                figure.add_subplot(scatter_grid[row, column])
                for column in range(3)
            ]
            for row in range(2)
        ]
    )
    scenario_titles = {
        "Cancellation and deletion": "Cancellation\nand deletion",
        "Disagreement and final resolution": (
            "Disagreement and\nfinal resolution"
        ),
        "Temporal instruction sequence": (
            "Temporal instruction\nsequence"
        ),
        "Correction and confirmation": "Correction and\nconfirmation",
        "Delayed answer and deferred resolution": (
            "Delayed answer and\ndeferred resolution"
        ),
        "Topic switch with distractors": (
            "Topic switch\nwith distractors"
        ),
    }
    assert set(scenario_order) == set(scenario_titles)

    def stale_rate_size(value: Any) -> Any:
        return 7 + 42 * value / 0.5

    for index, scenario in enumerate(scenario_order):
        row, column = divmod(index, 3)
        axis = scatter_axes[row, column]
        block = revision[revision["scenario"].eq(scenario)]
        for status, marker in [("Evaluable", "o"), ("N/A", "X")]:
            plotted = block[block["Stale-rate status"].eq(status)]
            if plotted.empty:
                continue
            axis.scatter(
                plotted["target_evidence_found"],
                plotted["update_success"],
                s=stale_rate_size(plotted["Stale answer rate"]),
                c=plotted["Model"].map(model_colors),
                marker=marker,
                edgecolor="white",
                linewidth=0.3,
                alpha=0.6,
                clip_on=False,
                zorder=4,
            )
        axis.scatter(
            1,
            1,
            marker="*",
            s=25,
            color="red",
            edgecolor="white",
            linewidth=0.25,
            clip_on=False,
            zorder=7,
        )
        axis.set_title(
            scenario_titles[scenario],
            pad=2,
            fontsize=5.35,
            fontweight="semibold",
        )

    for row in range(2):
        for column in range(3):
            axis = scatter_axes[row, column]
            axis.set(
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel="",
                ylabel="",
            )
            axis.set_xticks([0, 0.5, 1])
            axis.set_yticks([0, 0.5, 1])
            axis.set_xticks([0.2, 0.4, 0.6, 0.8], minor=True)
            axis.set_yticks([0.2, 0.4, 0.6, 0.8], minor=True)
            axis.xaxis.set_major_formatter(
                matplotlib.ticker.FormatStrFormatter("%.1f")
            )
            axis.yaxis.set_major_formatter(
                matplotlib.ticker.FormatStrFormatter("%.1f")
            )
            axis.tick_params(which="minor", length=0)
            axis.grid(
                which="minor",
                color="#d2d2d2",
                linewidth=0.45,
            )
            axis.grid(which="major", visible=False)
            axis.axvline(
                0.5,
                color="#888888",
                linestyle=(0, (3, 2)),
                linewidth=0.65,
                zorder=1,
            )
            axis.axhline(
                0.5,
                color="#888888",
                linestyle=(0, (3, 2)),
                linewidth=0.65,
                zorder=1,
            )
            for spine in axis.spines.values():
                spine.set_color("#c7c7c7")
                spine.set_linewidth(0.55)
            if column:
                axis.tick_params(axis="y", labelleft=False)
            if row == 0:
                axis.tick_params(axis="x", labelbottom=False)
    scatter_axes[0, 0].set_ylabel("Update success")
    scatter_axes[1, 0].set_ylabel("Update success")
    scatter_axes[1, 1].set_xlabel(
        "Target evidence found",
        labelpad=1.2,
    )

    panel_label_y = 0.90
    figure.text(
        0.002,
        panel_label_y,
        "(a)",
        fontsize=7.1,
        fontweight="bold",
        ha="left",
        va="center",
    )
    model_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=3,
            markerfacecolor=model_colors[model],
            markeredgecolor="none",
            label=model,
        )
        for model in model_order
    ]
    model_legend = figure.legend(
        handles=model_handles,
        ncol=8,
        loc="upper left",
        bbox_to_anchor=(0.058, 0.968),
        frameon=False,
        title="Model",
        handletextpad=0.28,
        columnspacing=0.58,
        borderaxespad=0,
        labelspacing=0.1,
    )
    model_legend._legend_box.align = "left"

    size_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=np.sqrt(stale_rate_size(value)),
            markerfacecolor="#555555",
            markeredgecolor="none",
            alpha=0.6,
            label=f"{value:.1f}",
        )
        for value in [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ]
    figure.legend(
        handles=size_handles,
        ncol=6,
        loc="lower left",
        bbox_to_anchor=(0.058, 0.018),
        frameon=False,
        title="Stale answer rate",
        handletextpad=0.18,
        columnspacing=0.42,
        borderaxespad=0,
        labelspacing=0.1,
    )
    status_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=3,
            markerfacecolor="#555555",
            markeredgecolor="none",
            alpha=0.6,
            label="Evaluable",
        ),
        Line2D(
            [],
            [],
            marker="X",
            linestyle="",
            markersize=3,
            markerfacecolor="#555555",
            markeredgecolor="none",
            alpha=0.6,
            label="N/A",
        ),
        Line2D(
            [],
            [],
            marker="*",
            linestyle="",
            markersize=4.2,
            markerfacecolor="red",
            markeredgecolor="none",
            label="Ideal (1, 1)",
        ),
    ]
    figure.legend(
        handles=status_handles,
        ncol=3,
        loc="lower right",
        bbox_to_anchor=(0.66, 0.018),
        frameon=False,
        title="Status / reference",
        handletextpad=0.22,
        columnspacing=0.5,
        borderaxespad=0,
        labelspacing=0.1,
    )

    line_axis = figure.add_subplot(outer[:, 1])
    for model in trajectory_models:
        block = lines[lines["model"].eq(model)].sort_values(
            "normalized_progress"
        )
        line_axis.plot(
            100 * block["normalized_progress"],
            100 * block["trajectory_score"],
            marker="o",
            linewidth=0.9,
            markersize=2.1,
            color=trajectory_colors[model],
            zorder=3,
        )
    endpoints = lines[lines["normalized_progress"].eq(1)].sort_values(
        "trajectory_score"
    )
    label_positions: dict[str, float] = {}
    previous = -np.inf
    for endpoint in endpoints.itertuples(index=False):
        label_positions[endpoint.model] = max(
            100 * float(endpoint.trajectory_score),
            previous + 3.4,
        )
        previous = label_positions[endpoint.model]
    actual_upper = 100 * lines["trajectory_score"].max()
    overflow = max(label_positions.values()) - (actual_upper + 6)
    if overflow > 0:
        label_positions = {
            model: value - overflow
            for model, value in label_positions.items()
        }
    for endpoint in endpoints.itertuples(index=False):
        endpoint_y = 100 * float(endpoint.trajectory_score)
        label_y = label_positions[endpoint.model]
        color = trajectory_colors[endpoint.model]
        line_axis.plot(
            [100, 103],
            [endpoint_y, label_y],
            color=color,
            linewidth=0.55,
        )
        line_axis.text(
            104,
            label_y,
            short_model_label(endpoint.model),
            color=color,
            va="center",
            fontsize=4.3,
        )
    plotted_scores = 100 * lines["trajectory_score"]
    line_axis.set(
        xlim=(-5, 124),
        ylim=(
            max(-2, plotted_scores.min() - 5),
            max(
                plotted_scores.max() + 5,
                max(label_positions.values()) + 3,
            ),
        ),
        xlabel=(
            "Normalized progress through each episode’s\n"
            "observed checkpoint sequence"
        ),
        ylabel="Mean semantic belief score (%)",
        title=(
            "Same-session semantic state"
        ),
    )
    line_axis.xaxis.label.set_fontsize(4.8)
    line_axis.set_xticks(
        [0, 20, 40, 60, 80, 100],
        labels=["0%", "20%", "40%", "60%", "80%", "100%"],
    )
    line_axis.tick_params(axis="x", labelsize=4.5, pad=1)
    line_axis.tick_params(axis="y", labelsize=4.8)
    line_axis.grid(
        which="major",
        color="#d2d2d2",
        linewidth=0.45,
    )
    for spine in line_axis.spines.values():
        spine.set_color("#c7c7c7")
        spine.set_linewidth(0.55)
    line_position = line_axis.get_position()
    figure.text(
        line_position.x0 - 0.03,
        panel_label_y,
        "(b)",
        fontsize=7.1,
        fontweight="bold",
        ha="left",
        va="center",
    )

    output = REVISION_DIR / "plot_conversational_revision_and_stability.png"
    _save_figure(figure, output, tight=False)
    _copy_asset(output, REVISION_DIR / "preview_plot_conversational_revision_and_stability.png")
    matplotlib.rcParams.update(original_rc_params)
    with Image.open(output) as image:
        expected_size = (round(7.15 * DPI), round(3.6 * DPI))
        assert image.size == expected_size, (
            f"Unexpected revision plot dimensions: {image.size}"
        )
    return output


def _write_reduced_panel_10(namespace: dict[str, Any]) -> Path:
    profile = namespace["t2_profile"].copy()
    cell_order = list(namespace["track2_cell_order"])
    short_model_label = namespace["short_model_label"]
    model_order = sorted(profile["model"].unique(), key=short_model_label)
    reduced = (
        profile.pivot(index="model", columns="cell", values="strict_completion")
        .reindex(index=model_order, columns=cell_order)
        .reset_index()
    )
    reduced.columns.name = None
    assert reduced.shape == (len(model_order), len(cell_order) + 1)
    assert list(reduced.columns[1:]) == cell_order
    expected = (
        profile.set_index(["model", "cell"])["strict_completion"]
        .reindex(
            pd.MultiIndex.from_product(
                [model_order, cell_order],
                names=["model", "cell"],
            )
        )
        .unstack("cell")
        .reindex(columns=cell_order)
        .reset_index()
    )
    expected.columns.name = None
    assert_frame_equal(reduced, expected, check_dtype=False)

    output = LONG_HORIZON_DIR / "table_task_variant_strict_completion.csv"
    reduced.to_csv(output, index=False)
    assert_frame_equal(reduced, pd.read_csv(output), check_dtype=False)
    return output


def _render_simplified_panel_14(namespace: dict[str, Any]) -> Path:
    track1_values = namespace["track1_uncertainty_table"].copy()
    track1_support = namespace["track1_uncertainty_support"].copy()
    track1_significance = namespace["track1_uncertainty_significance"].copy()
    track2_values = namespace["t2_uncertainty_table"].copy()
    track2_support = namespace["t2_uncertainty_support"].copy()
    selected_track2 = [
        "Found\nsupport ↑",
        "Absence\nsupported ↑",
        "Calibration ↑",
    ]
    missing_columns = sorted(set(selected_track2) - set(track2_values.columns))
    assert not missing_columns, (
        f"Panel 14 no longer exposes required simplified metrics: {missing_columns}"
    )
    track2_values = track2_values[selected_track2]
    track2_support = track2_support[selected_track2]

    model_order = sorted(set(track1_values.index) | set(track2_values.index))
    track1_values = track1_values.reindex(model_order)
    track1_support = track1_support.reindex(model_order).fillna(0)
    track1_significance = track1_significance.reindex(model_order).fillna(False)
    track2_values = track2_values.reindex(model_order)
    track2_support = track2_support.reindex(model_order).fillna(0)
    finite = track1_values.to_numpy(dtype=float)
    bias_limit = max(0.05, np.nanmax(np.abs(finite))) if finite.size else 0.05

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.3),
        gridspec_kw={"width_ratios": [3, 3]},
    )
    track1_annotations = _selective_annotations(
        namespace,
        track1_values,
        track1_support,
    )
    sns.heatmap(
        100 * track1_values,
        mask=track1_values.isna(),
        annot=track1_annotations,
        fmt="",
        cmap="vlag",
        center=0,
        vmin=-100 * bias_limit,
        vmax=100 * bias_limit,
        cbar_kws={"label": "Known - unknown (pp)", "shrink": 0.85},
        ax=axes[0],
    )
    namespace["mark_significant_heatmap_cells"](axes[0], track1_significance)
    axes[0].set_facecolor("#e6e6e6")
    _annotate_missing(namespace, axes[0], track1_values, track1_support)
    axes[0].set(
        title="Track 1 known-minus-unknown bias",
        xlabel="Track",
        ylabel="Model",
    )
    axes[0].tick_params(axis="x", rotation=0)

    track2_annotations = _selective_annotations(
        namespace,
        track2_values,
        track2_support,
    )
    sns.heatmap(
        100 * track2_values,
        mask=track2_values.isna(),
        annot=track2_annotations,
        fmt="",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Score / rate (%)", "shrink": 0.85},
        ax=axes[1],
    )
    axes[1].set_facecolor("#e6e6e6")
    _annotate_missing(namespace, axes[1], track2_values, track2_support)
    axes[1].set(
        title="Track 2 supported commitment and uncertainty",
        xlabel="Metric",
        ylabel="",
    )
    axes[1].tick_params(axis="x", rotation=0, labelsize=9)
    axes[1].tick_params(axis="y", labelleft=False)
    figure.suptitle("Uncertainty awareness and supported commitment", y=0.99)
    namespace["add_significance_caption"](
        figure,
        "all displayed Track 1 contrast cells",
    )
    figure.tight_layout(rect=[0, 0.05, 1, 0.97])

    output = (
        SUPPLEMENT_ROOT
        / "panel_14"
        / "plot_uncertainty_awareness_and_supported_commitment_simplified.png"
    )
    _save_figure(figure, output)
    return output


def _write_reduced_panel_14(namespace: dict[str, Any]) -> Path:
    metrics = [
        "found_claim_support_score",
        "absence_claim_support_score",
        "calibration_score",
    ]
    source = namespace["t2_uncertainty"].copy()
    missing = sorted(set(["model", *metrics]) - set(source.columns))
    assert not missing, f"Panel 14 table is missing columns: {missing}"
    model_order = sorted(source["model"].unique(), key=namespace["short_model_label"])
    reduced = source.set_index("model")[metrics].reindex(model_order).reset_index()
    output = (
        CROSS_DOMAIN_DIR / "table_track2_supported_commitment_simplified.csv"
    )
    reduced.to_csv(output, index=False)
    assert_frame_equal(reduced, pd.read_csv(output), check_dtype=False)
    return output


def _write_cross_domain_heatmap_support(namespace: dict[str, Any]) -> Path:
    track1_support = namespace["track1_uncertainty_support"].copy()
    track2_support = namespace["t2_uncertainty_support"].copy()
    track1_columns = ["1a", "1b", "1c"]
    track2_columns = [
        "Found\nsupport ↑",
        "Absence\nsupported ↑",
        "Calibration ↑",
    ]
    missing_track1 = sorted(set(track1_columns) - set(track1_support.columns))
    missing_track2 = sorted(set(track2_columns) - set(track2_support.columns))
    assert not missing_track1, (
        f"Track 1 support table is missing columns: {missing_track1}"
    )
    assert not missing_track2, (
        f"Track 2 support table is missing columns: {missing_track2}"
    )

    model_order = sorted(set(track1_support.index) | set(track2_support.index))
    assert len(model_order) == 10, (
        f"Cross-domain heatmap requires ten models; found {len(model_order)}"
    )
    track1_support = track1_support.reindex(
        index=model_order,
        columns=track1_columns,
    )
    track2_support = track2_support.reindex(
        index=model_order,
        columns=track2_columns,
    )
    assert track1_support.notna().all().all()
    assert track2_support.notna().all().all()

    track1_support.index.name = "model"
    track2_support.index.name = "model"
    track1_long = (
        track1_support.rename(
            columns={"1a": "Track 1A", "1b": "Track 1B", "1c": "Track 1C"}
        )
        .reset_index()
        .melt(id_vars="model", var_name="metric", value_name="n_evaluable")
    )
    track1_long["metric_group"] = "Track 1"
    track2_long = (
        track2_support.rename(
            columns={
                "Found\nsupport ↑": "Found support",
                "Absence\nsupported ↑": "Absence supported",
                "Calibration ↑": "Calibration",
            }
        )
        .reset_index()
        .melt(id_vars="model", var_name="metric", value_name="n_evaluable")
    )
    track2_long["metric_group"] = "Track 2"
    support = pd.concat([track1_long, track2_long], ignore_index=True)[
        ["model", "metric_group", "metric", "n_evaluable"]
    ]
    assert support.shape == (60, 4)
    assert not support.duplicated(["model", "metric_group", "metric"]).any()
    numeric_support = pd.to_numeric(support["n_evaluable"], errors="raise")
    assert numeric_support.ge(0).all()
    assert np.equal(numeric_support, np.floor(numeric_support)).all()
    support["n_evaluable"] = numeric_support.astype(int)

    output = CROSS_DOMAIN_DIR / "table_cross_domain_heatmap_support.csv"
    support.to_csv(output, index=False)
    assert_frame_equal(support, pd.read_csv(output), check_dtype=False)
    return output


def _render_cross_domain_composite(namespace: dict[str, Any]) -> Path:
    scatter = namespace["combined_scatter"].copy()
    scatter_columns = {
        "model",
        "track",
        "master_score",
        "evidence_reach_score",
        "temporal_belief_score",
        "budget_used",
        "replay_rate",
        "diagnosis",
    }
    missing_scatter = sorted(scatter_columns - set(scatter.columns))
    assert not missing_scatter, (
        f"Cross-domain scatter data are missing columns: {missing_scatter}"
    )
    track_order = ["1a/1b", "1c", "2"]
    track_titles = {
        "1a/1b": "Tracks 1A/1B mean",
        "1c": "Track 1C",
        "2": "Track 2",
    }
    assert set(scatter["track"]) == set(track_order)
    assert not scatter.duplicated(["model", "track"]).any()
    assert scatter.groupby("model")["track"].nunique().eq(3).all()

    short_model_label = namespace["short_model_label"]
    scatter["Model"] = scatter["model"].map(short_model_label)
    model_mapping = scatter[["model", "Model"]].drop_duplicates()
    assert model_mapping["model"].is_unique
    assert model_mapping["Model"].is_unique, "Short model labels must be unique"
    model_order = sorted(scatter["Model"].unique())
    assert len(model_order) == 10, (
        f"Cross-domain scatter grid requires ten models; found {len(model_order)}"
    )
    model_colors = namespace["MODEL_COLORS"]
    missing_colors = sorted(set(model_order) - set(model_colors))
    assert not missing_colors, f"Missing model colors: {missing_colors}"
    model_palette = {model: model_colors[model] for model in model_order}

    diagnosis_order = [
        "acquisition-limited",
        "belief-limited",
        "inspect priors/metric mismatch",
        "robust",
    ]
    observed_diagnoses = set(scatter["diagnosis"].dropna())
    unknown_diagnoses = sorted(observed_diagnoses - set(diagnosis_order))
    assert not unknown_diagnoses, f"Unknown diagnoses: {unknown_diagnoses}"
    plotted_diagnoses = set(
        scatter.dropna(
            subset=[
                "evidence_reach_score",
                "temporal_belief_score",
                "master_score",
                "diagnosis",
            ]
        )["diagnosis"]
    )
    diagnosis_order = [
        diagnosis for diagnosis in diagnosis_order
        if diagnosis in plotted_diagnoses
    ]
    diagnosis_colors = {
        "acquisition-limited": "#4C72B0",
        "belief-limited": "#DD8452",
        "inspect priors/metric mismatch": "#55A868",
        "robust": "#8172B2",
    }

    bounded_metrics = [
        "master_score",
        "evidence_reach_score",
        "temporal_belief_score",
        "budget_used",
        "replay_rate",
    ]
    for metric in bounded_metrics:
        finite = pd.to_numeric(scatter[metric], errors="coerce").dropna()
        assert finite.between(0, 1).all(), (
            f"Cross-domain scatter metric {metric!r} must be on the 0-1 scale"
        )

    track1_values = namespace["track1_uncertainty_table"].copy()
    track1_significance = namespace[
        "track1_uncertainty_significance"
    ].copy()
    track2_source = namespace["t2_uncertainty_table"].copy()
    track1_columns = ["1a", "1b", "1c"]
    track2_columns = [
        "Found\nsupport ↑",
        "Absence\nsupported ↑",
        "Calibration ↑",
    ]
    missing_track1 = sorted(set(track1_columns) - set(track1_values.columns))
    missing_track2 = sorted(set(track2_columns) - set(track2_source.columns))
    assert not missing_track1, (
        f"Track 1 heatmap is missing columns: {missing_track1}"
    )
    assert not missing_track2, (
        f"Track 2 heatmap is missing columns: {missing_track2}"
    )
    assert track1_values.index.is_unique
    assert track2_source.index.is_unique
    heatmap_model_order = sorted(
        set(track1_values.index) | set(track2_source.index)
    )
    assert heatmap_model_order == model_order, (
        "Scatter and heatmap model sets must match exactly"
    )
    track1_values = track1_values.reindex(
        index=model_order,
        columns=track1_columns,
    )
    track1_significance = track1_significance.reindex(
        index=model_order,
        columns=track1_columns,
    ).fillna(False)
    track2_values = track2_source.reindex(
        index=model_order,
        columns=track2_columns,
    )
    assert track1_values.shape == (10, 3)
    assert track2_values.shape == (10, 3)
    track1_finite = track1_values.to_numpy(dtype=float)
    track1_finite = track1_finite[np.isfinite(track1_finite)]
    track2_finite = track2_values.to_numpy(dtype=float)
    track2_finite = track2_finite[np.isfinite(track2_finite)]
    assert track1_finite.size
    assert track2_finite.size
    assert np.logical_and(track1_finite >= -1, track1_finite <= 1).all()
    assert np.logical_and(track2_finite >= 0, track2_finite <= 1).all()
    bias_limit = max(0.05, float(np.max(np.abs(track1_finite))))

    original_rc_params = matplotlib.rcParams.copy()
    matplotlib.rcParams.update(
        {
            "font.size": 5.6,
            "axes.titlesize": 6.6,
            "axes.labelsize": 6.3,
            "xtick.labelsize": 5.3,
            "ytick.labelsize": 5.3,
            "legend.fontsize": 4.8,
            "legend.title_fontsize": 5.2,
            "axes.linewidth": 0.5,
            "axes.grid": False,
        }
    )
    figure = plt.figure(figsize=(7.15, 3.6))
    outer = figure.add_gridspec(
        2,
        2,
        width_ratios=[2.25, 1.0],
        hspace=0.15,
        wspace=0.16,
        left=0.058,
        right=0.985,
        bottom=0.18,
        top=0.83,
    )
    scatter_grid = outer[:, 0].subgridspec(
        2,
        3,
        hspace=0.34,
        wspace=0.12,
    )
    heatmap_grid = outer[:, 1].subgridspec(1, 2, wspace=0.08)
    scatter_axes = np.array(
        [
            [figure.add_subplot(scatter_grid[row, column]) for column in range(3)]
            for row in range(2)
        ]
    )

    for column, track in enumerate(track_order):
        axis = scatter_axes[0, column]
        plotted = scatter[scatter["track"].eq(track)].dropna(
            subset=[
                "budget_used",
                "replay_rate",
                "evidence_reach_score",
            ]
        )
        axis.scatter(
            plotted["budget_used"],
            plotted["replay_rate"],
            s=7 + 42 * plotted["evidence_reach_score"],
            c=plotted["Model"].map(model_palette),
            edgecolor="white",
            linewidth=0.3,
            alpha=0.6,
            clip_on=False,
            zorder=4,
        )
        axis.scatter(
            0,
            0,
            marker="*",
            s=25,
            color="red",
            edgecolor="white",
            linewidth=0.25,
            clip_on=False,
            zorder=7,
        )
        axis.set_title(track_titles[track], pad=2.5, fontweight="semibold")

    for column, track in enumerate(track_order):
        axis = scatter_axes[1, column]
        plotted = scatter[scatter["track"].eq(track)].dropna(
            subset=[
                "evidence_reach_score",
                "temporal_belief_score",
                "master_score",
                "diagnosis",
            ]
        )
        axis.scatter(
            plotted["evidence_reach_score"],
            plotted["temporal_belief_score"],
            s=7 + 42 * plotted["master_score"],
            c=plotted["diagnosis"].map(diagnosis_colors),
            edgecolor="white",
            linewidth=0.3,
            alpha=0.6,
            clip_on=False,
            zorder=4,
        )
        axis.scatter(
            1,
            1,
            marker="*",
            s=25,
            color="red",
            edgecolor="white",
            linewidth=0.25,
            clip_on=False,
            zorder=7,
        )

    for row in range(2):
        for column in range(3):
            axis = scatter_axes[row, column]
            axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="", ylabel="")
            axis.set_xticks([0, 0.5, 1])
            axis.set_yticks([0, 0.5, 1])
            axis.set_xticks([0.2, 0.4, 0.6, 0.8], minor=True)
            axis.set_yticks([0.2, 0.4, 0.6, 0.8], minor=True)
            axis.xaxis.set_major_formatter(
                matplotlib.ticker.FormatStrFormatter("%.1f")
            )
            axis.yaxis.set_major_formatter(
                matplotlib.ticker.FormatStrFormatter("%.1f")
            )
            axis.tick_params(which="minor", length=0)
            axis.grid(which="minor", color="#d2d2d2", linewidth=0.45)
            axis.grid(which="major", visible=False)
            axis.axvline(
                0.5,
                color="#888888",
                linestyle=(0, (3, 2)),
                linewidth=0.65,
                zorder=1,
            )
            axis.axhline(
                0.5,
                color="#888888",
                linestyle=(0, (3, 2)),
                linewidth=0.65,
                zorder=1,
            )
            for spine in axis.spines.values():
                spine.set_color("#c7c7c7")
                spine.set_linewidth(0.55)
            if column:
                axis.tick_params(axis="y", labelleft=False)
    scatter_axes[0, 0].set_ylabel("Replay rate", labelpad=3)
    scatter_axes[1, 0].set_ylabel("Temporal belief", labelpad=3)
    scatter_axes[0, 1].set_xlabel("Budget used", labelpad=1.2)
    scatter_axes[1, 1].set_xlabel("Evidence reach", labelpad=1.2)
    for label, row in [("(a)", 0), ("(b)", 1)]:
        position = scatter_axes[row, 0].get_position()
        figure.text(
            0.002,
            position.y1 + 0.036,
            label,
            fontsize=7.1,
            fontweight="bold",
            ha="left",
            va="center",
        )

    model_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=3,
            markerfacecolor=model_palette[model],
            markeredgecolor="none",
            label=model,
        )
        for model in model_order
    ]
    model_legend = figure.legend(
        handles=model_handles,
        ncol=5,
        loc="upper left",
        bbox_to_anchor=(0.058, 0.988),
        frameon=False,
        title="Model",
        handletextpad=0.28,
        columnspacing=0.68,
        borderaxespad=0,
        labelspacing=0.18,
    )
    model_legend._legend_box.align = "left"

    def size_handles(values: list[float]) -> list[Line2D]:
        return [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=np.sqrt(7 + 42 * value),
                markerfacecolor="#555555",
                markeredgecolor="none",
                alpha=0.6,
                label=f"{value:.1f}",
            )
            for value in values
        ]

    figure.legend(
        handles=size_handles([0, 0.5, 1]),
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.565, 0.976),
        frameon=False,
        title="Evidence reach",
        handletextpad=0.2,
        columnspacing=0.55,
        borderaxespad=0,
        labelspacing=0.1,
    )
    diagnosis_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=3,
            markerfacecolor=diagnosis_colors[diagnosis],
            markeredgecolor="none",
            label=diagnosis,
        )
        for diagnosis in diagnosis_order
    ]
    diagnosis_handles.append(
        Line2D(
            [],
            [],
            marker="*",
            linestyle="",
            markersize=4.2,
            markerfacecolor="red",
            markeredgecolor="none",
            label="Ideal",
        )
    )
    figure.text(
        0.058,
        0.080,
        "Diagnosis",
        fontsize=5.2,
        ha="left",
        va="bottom",
    )
    figure.legend(
        handles=diagnosis_handles,
        ncol=len(diagnosis_handles),
        loc="lower left",
        bbox_to_anchor=(0.058, 0.022),
        frameon=False,
        handletextpad=0.28,
        columnspacing=0.65,
        borderaxespad=0,
        labelspacing=0.1,
    )
    figure.legend(
        handles=size_handles([0, 0.5, 1]),
        ncol=3,
        loc="lower right",
        bbox_to_anchor=(0.66, 0.019),
        frameon=False,
        title="Master score",
        handletextpad=0.2,
        columnspacing=0.5,
        borderaxespad=0,
        labelspacing=0.1,
    )

    track1_axis = figure.add_subplot(heatmap_grid[0, 0])
    track2_axis = figure.add_subplot(heatmap_grid[0, 1])
    figure.text(
        0.692,
        0.925,
        "(c)",
        fontsize=7.1,
        fontweight="bold",
        ha="left",
        va="center",
    )
    sns.heatmap(
        100 * track1_values,
        mask=track1_values.isna(),
        cmap="vlag",
        center=0,
        vmin=-100 * bias_limit,
        vmax=100 * bias_limit,
        cbar=False,
        annot=False,
        linewidths=0.22,
        linecolor="white",
        ax=track1_axis,
    )
    sns.heatmap(
        100 * track2_values,
        mask=track2_values.isna(),
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        cbar=False,
        annot=False,
        linewidths=0.22,
        linecolor="white",
        ax=track2_axis,
    )
    for axis, values, significance in [
        (track1_axis, track1_values, track1_significance),
        (track2_axis, track2_values, None),
    ]:
        axis.set_facecolor("#e6e6e6")
        axis.set(xlabel="", ylabel="")
        axis.tick_params(length=0, pad=0.8)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values.iloc[row_index, column_index]
                if pd.isna(value):
                    axis.text(
                        column_index + 0.5,
                        row_index + 0.5,
                        "N/A",
                        ha="center",
                        va="center",
                        fontsize=4.15,
                        color="#555555",
                    )
                    continue
                shown = 100 * value
                white_text = (
                    axis is track1_axis
                    and abs(shown) > 0.53 * 100 * bias_limit
                ) or (axis is track2_axis and shown > 62)
                axis.text(
                    column_index + 0.5,
                    row_index + 0.53,
                    f"{shown:.1f}",
                    ha="center",
                    va="center",
                    fontsize=4.2,
                    color="white" if white_text else "#222222",
                )
                if significance is not None and bool(
                    significance.iloc[row_index, column_index]
                ):
                    axis.text(
                        column_index + 0.5,
                        row_index + 0.17,
                        "•",
                        ha="center",
                        va="center",
                        fontsize=4.9,
                        color="black",
                    )
    track1_axis.set_xticklabels(
        ["1A", "1B", "1C"],
        rotation=90,
        fontsize=4.75,
    )
    track2_axis.set_xticklabels(
        track2_columns,
        rotation=90,
        fontsize=4.75,
    )
    track1_axis.set_yticklabels(
        model_order,
        rotation=0,
        fontsize=4.75,
    )
    track2_axis.tick_params(axis="y", labelleft=False)
    track1_axis.set_title(
        "Track 1",
        pad=31,
        fontsize=6.1,
        fontweight="semibold",
    )
    track2_axis.set_title(
        "Track 2",
        pad=31,
        fontsize=6.1,
        fontweight="semibold",
    )

    colorbar_specs = [
        (
            track1_axis,
            "vlag",
            matplotlib.colors.Normalize(
                vmin=-100 * bias_limit,
                vmax=100 * bias_limit,
            ),
            [-round(100 * bias_limit, 1), 0, round(100 * bias_limit, 1)],
            "Bias (pp)",
        ),
        (
            track2_axis,
            "YlGnBu",
            matplotlib.colors.Normalize(vmin=0, vmax=100),
            [0, 50, 100],
            "Score (%)",
        ),
    ]
    for axis, cmap, normalization, ticks, title in colorbar_specs:
        position = axis.get_position()
        colorbar_axis = figure.add_axes(
            [
                position.x0 + 0.08 * position.width,
                position.y1 + 0.035,
                0.84 * position.width,
                0.011,
            ]
        )
        colorbar = figure.colorbar(
            matplotlib.cm.ScalarMappable(norm=normalization, cmap=cmap),
            cax=colorbar_axis,
            orientation="horizontal",
        )
        colorbar.set_ticks(ticks)
        colorbar.ax.tick_params(labelsize=4, length=1, pad=0.6)
        colorbar.outline.set_linewidth(0.3)
        colorbar.ax.set_title(title, fontsize=4.45, pad=1.5)

    output = OUTPUT_ROOT / "cross_track_diagnostics" / "plot_cross_track_diagnostics.png"
    _save_figure(figure, output, tight=False)
    matplotlib.rcParams.update(original_rc_params)
    with Image.open(output) as image:
        expected_size = (round(7.15 * DPI), round(3.6 * DPI))
        assert image.size == expected_size, (
            f"Unexpected cross-domain plot dimensions: {image.size}"
        )
    _copy_asset(output, CROSS_DOMAIN_DIR / "plot_cross_domain_diagnostics.png")
    return output


def _copy_main_tables_and_plot() -> None:
    copies = [
        (SUPPLEMENT_ROOT / "panel_02" / "plot_active_vs_full_context_slopes.png",
         OUTPUT_ROOT / "plots" / "plot_active_vs_full_context_slopes.png"),
        (
            SUPPLEMENT_ROOT
            / "panel_03"
            / "table_temporal_map_component_profile.csv",
            CONTROLLED_DIR / "table_temporal_map_component_profile.csv",
        ),
        (
            SUPPLEMENT_ROOT
            / "panel_06"
            / "table_active_semantic_belief_profile.csv",
            CONTROLLED_DIR / "table_active_semantic_belief_profile.csv",
        ),
        (
            SUPPLEMENT_ROOT
            / "panel_07"
            / "table_semantic_revision_and_belief_inertia.csv",
            REVISION_DIR / "table_semantic_revision_and_belief_inertia.csv",
        ),
        (
            SUPPLEMENT_ROOT
            / "panel_08"
            / "table_same_session_belief_stability.csv",
            REVISION_DIR / "table_same_session_belief_stability.csv",
        ),
        (
            SUPPLEMENT_ROOT
            / "panel_11"
            / "plot_marginal_completion_stage_profile.png",
            LONG_HORIZON_DIR / "plot_marginal_completion_stage_profile.png",
        ),
        (
            SUPPLEMENT_ROOT
            / "panel_11"
            / "table_marginal_completion_stage_profile.csv",
            LONG_HORIZON_DIR / "table_marginal_completion_stage_profile.csv",
        ),
        (
            SUPPLEMENT_ROOT
            / "panel_13"
            / "table_combined_evidence_efficiency_belief_diagnostics.csv",
            CROSS_DOMAIN_DIR
            / "table_combined_evidence_efficiency_belief_diagnostics.csv",
        ),
        (
            SUPPLEMENT_ROOT
            / "panel_13"
            / "table_track2_efficiency_components.csv",
            CROSS_DOMAIN_DIR / "table_track2_efficiency_components.csv",
        ),
        (
            SUPPLEMENT_ROOT
            / "panel_14"
            / "table_track1_known_unknown_scores.csv",
            CROSS_DOMAIN_DIR / "table_track1_known_unknown_scores.csv",
        ),
    ]
    for source, destination in copies:
        _copy_asset(source, destination)


def _validate_outputs(expected: list[Path]) -> None:
    missing = [str(path) for path in expected if not path.is_file()]
    assert not missing, f"Missing expected paper assets: {missing}"
    unsupported = sorted(
        str(path.relative_to(OUTPUT_ROOT))
        for path in OUTPUT_ROOT.rglob("*")
        if path.is_file() and path.suffix not in {".png", ".csv"}
    )
    assert not unsupported, f"Unsupported paper asset formats: {unsupported}"
    for path in OUTPUT_ROOT.rglob("*.png"):
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            dpi = image.info.get("dpi")
            if dpi is not None:
                assert all(abs(value - DPI) < 1 for value in dpi), (
                    f"Unexpected DPI metadata for {path}: {dpi}"
                )


def main() -> None:
    _mkdirs()
    namespace = _execute_notebook_cells()
    controlled_plot = _render_controlled_composite(namespace)
    panel_6b = _render_panel_6b(namespace)
    revision_plot = _render_revision_composite(namespace)
    reduced_panel_10 = _write_reduced_panel_10(namespace)
    simplified_panel_14 = _render_simplified_panel_14(namespace)
    reduced_panel_14 = _write_reduced_panel_14(namespace)
    cross_domain_support = _write_cross_domain_heatmap_support(namespace)
    cross_domain_plot = _render_cross_domain_composite(namespace)
    _copy_main_tables_and_plot()

    expected = [
        controlled_plot,
        panel_6b,
        revision_plot,
        reduced_panel_10,
        LONG_HORIZON_DIR / "plot_marginal_completion_stage_profile.png",
        LONG_HORIZON_DIR / "table_marginal_completion_stage_profile.csv",
        simplified_panel_14,
        reduced_panel_14,
        cross_domain_support,
        cross_domain_plot,
    ]
    _validate_outputs(expected)
    print(f"Generated PNG/CSV paper assets under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
