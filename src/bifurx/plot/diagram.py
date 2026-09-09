from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from bifurx.continuation.arclength import ContinuationResult


def plot_diagram(
    results: ContinuationResult | list[ContinuationResult],
    state_index: int = 0,
    title: str = "Bifurcation Diagram",
    xlabel: str = "Continuation Parameter $p$",
    ylabel: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    ax: Any | None = None,
    show_special_points: bool = True,
    annotate_points: bool = True,
    save_path: str | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Plot bifurcation diagram with stable/unstable branches and marked bifurcation points.

    Stable segments are plotted with solid lines, and unstable segments with dashed lines.
    Bifurcation points are marked with distinctive markers:
      - Fold / Limit Point (LP): purple circle 'o'
      - Branch Point (BP): crimson square 's'
      - Hopf Bifurcation (HB): teal triangle '^'

    Parameters
    ----------
    results : ContinuationResult or list[ContinuationResult]
        Continuation trajectory or list of branches to plot.
    state_index : int, default=0
        Index of the state variable u[i] to plot on the y-axis.
    title : str, default="Bifurcation Diagram"
        Plot title.
    xlabel : str, default="Continuation Parameter $p$"
        X-axis label.
    ylabel : str | None, optional
        Y-axis label (defaults to "$u_{state_index}$").
    figsize : tuple[float, float], default=(10, 6)
        Figure size in inches.
    ax : matplotlib.axes.Axes | None, optional
        Existing axes to draw on.
    show_special_points : bool, default=True
        Whether to highlight detected bifurcation points.
    annotate_points : bool, default=True
        Whether to print text annotations next to bifurcation points.
    save_path : str | None, optional
        File path to save the generated plot.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
    else:
        fig = ax.get_figure()

    if isinstance(results, ContinuationResult):
        branch_list = [results]
    else:
        branch_list = list(results)

    y_label_str = ylabel if ylabel is not None else f"State $u_{{{state_index}}}$"

    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for b_idx, branch in enumerate(branch_list):
        p_arr = np.asarray(branch.p)
        u_arr = np.asarray(branch.u)[:, state_index]
        st_arr = np.asarray(branch.stability)

        base_color = default_colors[b_idx % len(default_colors)]
        branch_name = branch.branch_name

        # Draw contiguous segments of identical stability
        idx = 0
        while idx < len(st_arr):
            is_st = st_arr[idx]
            end_idx = idx + 1
            while end_idx < len(st_arr) and st_arr[end_idx] == is_st:
                end_idx += 1
            slice_end = min(end_idx + 1, len(st_arr))

            label = f"{branch_name} ({'Stable' if is_st else 'Unstable'})" if idx == 0 else ""

            ax.plot(
                p_arr[idx:slice_end],
                u_arr[idx:slice_end],
                linestyle="-" if is_st else "--",
                linewidth=2.2 if is_st else 1.8,
                color=base_color,
                label=label,
            )
            idx = end_idx

        # Mark bifurcation points
        if show_special_points:
            for pt in branch.bifurcation_points:
                p_val = pt.p
                u_val = float(np.asarray(pt.u)[state_index])

                if pt.bif_type == "LP":
                    ax.scatter(
                        p_val,
                        u_val,
                        color="purple",
                        s=120,
                        zorder=5,
                        marker="o",
                        label="Fold / Limit Point (LP)",
                    )
                    if annotate_points:
                        ax.annotate(
                            f"LP (p={p_val:.3f})",
                            (p_val, u_val),
                            textcoords="offset points",
                            xytext=(10, -10),
                            weight="bold",
                            fontsize=9,
                        )
                elif pt.bif_type == "BP":
                    ax.scatter(
                        p_val,
                        u_val,
                        color="crimson",
                        s=120,
                        zorder=5,
                        marker="s",
                        label="Branch Point (BP)",
                    )
                    if annotate_points:
                        ax.annotate(
                            f"BP (p={p_val:.3f})",
                            (p_val, u_val),
                            textcoords="offset points",
                            xytext=(-20, 12),
                            weight="bold",
                            fontsize=9,
                        )
                elif pt.bif_type == "HB":
                    ax.scatter(
                        p_val,
                        u_val,
                        color="teal",
                        s=140,
                        zorder=5,
                        marker="^",
                        label="Hopf Bifurcation (HB)",
                    )
                    if annotate_points:
                        ax.annotate(
                            f"HB (p={p_val:.3f})",
                            (p_val, u_val),
                            textcoords="offset points",
                            xytext=(-25, 12),
                            weight="bold",
                            fontsize=9,
                        )

    # Deduplicate legend items
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="best", framealpha=0.9)

    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(y_label_str, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax
