"""Generate Excalidraw-compatible JSON sources and hand-drawn Excalidraw-style PNG diagrams."""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "docs/theory/figures"
SRC_DIR = "docs/theory/figures/src"


def _sketch_poly(pts: np.ndarray, seed: int, rough: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """Generate two slightly perturbed hand-drawn passes along a polyline."""
    rng = np.random.default_rng(seed)
    passes = []
    for _ in range(2):
        out = []
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            dist = float(np.linalg.norm(p1 - p0))
            n_seg = max(4, int(dist * 6))
            t = np.linspace(0.0, 1.0, n_seg, endpoint=(i == len(pts) - 2))
            seg = p0[None, :] * (1.0 - t[:, None]) + p1[None, :] * t[:, None]
            # Subtle mid-segment wobble, pinned near endpoints
            env = np.sin(np.pi * t)[:, None]
            jitter = rng.normal(0.0, rough, size=seg.shape) * (0.35 + 0.65 * env)
            out.append(seg + jitter)
        passes.append(np.vstack(out))
    return passes[0], passes[1]


def _rounded_rect_pts(x: float, y: float, w: float, h: float, r: float = 0.16) -> np.ndarray:
    r = min(r, w * 0.25, h * 0.25)
    angles = [
        (x + w - r, y + h - r, 0, np.pi / 2),
        (x + r, y + h - r, np.pi / 2, np.pi),
        (x + r, y + r, np.pi, 3 * np.pi / 2),
        (x + w - r, y + r, 3 * np.pi / 2, 2 * np.pi),
    ]
    pts = []
    for cx, cy, a0, a1 in angles:
        th = np.linspace(a0, a1, 7)
        pts.append(np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)]))
    arr = np.vstack(pts)
    return np.vstack([arr, arr[:1]])


def _diamond_pts(cx: float, cy: float, hw: float, hh: float) -> np.ndarray:
    return np.array(
        [
            [cx, cy + hh],
            [cx + hw, cy],
            [cx, cy - hh],
            [cx - hw, cy],
            [cx, cy + hh],
        ]
    )


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    *,
    fill: str = "#dbeafe",
    stroke: str = "#1e3a5f",
    seed: int = 1,
    title_size: float = 11.5,
    sub_size: float = 9.0,
) -> dict[str, Any]:
    pts = _rounded_rect_pts(x, y, w, h)
    # Soft pastel fill
    patch = mpatches.Polygon(pts, closed=True, facecolor=fill, edgecolor="none", zorder=2)
    ax.add_patch(patch)
    # Two rough ink strokes
    p1, p2 = _sketch_poly(pts, seed=seed, rough=0.022)
    ax.plot(p1[:, 0], p1[:, 1], color=stroke, lw=1.75, solid_capstyle="round", zorder=3)
    ax.plot(p2[:, 0], p2[:, 1], color=stroke, lw=1.1, alpha=0.65, solid_capstyle="round", zorder=3)

    cx, cy = x + w / 2.0, y + h / 2.0
    if subtitle:
        ax.text(
            cx,
            cy + h * 0.21,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=stroke,
            family="Chalkboard SE",
            zorder=4,
        )
        ax.text(
            cx,
            cy - h * 0.14,
            subtitle,
            ha="center",
            va="center",
            fontsize=sub_size,
            color="#334155",
            family="Chalkboard SE",
            linespacing=1.25,
            zorder=4,
        )
    else:
        ax.text(
            cx,
            cy,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=stroke,
            family="Chalkboard SE",
            linespacing=1.2,
            zorder=4,
        )
    return {
        "type": "rectangle",
        "x": x * 60,
        "y": -y * 60,
        "width": w * 60,
        "height": h * 60,
        "strokeColor": stroke,
        "backgroundColor": fill,
        "roughness": 1,
        "label": f"{title}\n{subtitle}".strip(),
    }


def _draw_diamond(
    ax: plt.Axes,
    cx: float,
    cy: float,
    hw: float,
    hh: float,
    label: str,
    *,
    fill: str = "#fef3c7",
    stroke: str = "#92400e",
    seed: int = 10,
    size: float = 10.0,
) -> dict[str, Any]:
    pts = _diamond_pts(cx, cy, hw, hh)
    patch = mpatches.Polygon(pts, closed=True, facecolor=fill, edgecolor="none", zorder=2)
    ax.add_patch(patch)
    p1, p2 = _sketch_poly(pts, seed=seed, rough=0.025)
    ax.plot(p1[:, 0], p1[:, 1], color=stroke, lw=1.75, solid_capstyle="round", zorder=3)
    ax.plot(p2[:, 0], p2[:, 1], color=stroke, lw=1.05, alpha=0.65, solid_capstyle="round", zorder=3)
    ax.text(
        cx,
        cy,
        label,
        ha="center",
        va="center",
        fontsize=size,
        fontweight="bold",
        color=stroke,
        family="Chalkboard SE",
        linespacing=1.15,
        zorder=4,
    )
    return {
        "type": "diamond",
        "x": (cx - hw) * 60,
        "y": -(cy + hh) * 60,
        "width": 2 * hw * 60,
        "height": 2 * hh * 60,
        "strokeColor": stroke,
        "backgroundColor": fill,
        "roughness": 1,
        "label": label,
    }


def _draw_arrow(
    ax: plt.Axes,
    pts: list[tuple[float, float]],
    *,
    color: str = "#334155",
    seed: int = 20,
    lw: float = 1.7,
    ls: str = "-",
) -> dict[str, Any]:
    arr = np.array(pts, dtype=float)
    p1, p2 = _sketch_poly(arr, seed=seed, rough=0.018)
    ax.plot(p1[:, 0], p1[:, 1], color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=3)
    ax.plot(
        p2[:, 0], p2[:, 1], color=color, lw=lw * 0.6, ls=ls, alpha=0.55, solid_capstyle="round", zorder=3
    )
    # Hand-drawn arrowhead at final segment
    v = arr[-1] - arr[-2]
    v_norm = float(np.linalg.norm(v))
    if v_norm > 1e-9:
        u = v / v_norm
        perp = np.array([-u[1], u[0]])
        head_len = 0.22
        head_w = 0.11
        tip = arr[-1]
        left = tip - head_len * u + head_w * perp
        right = tip - head_len * u - head_w * perp
        ax.plot([left[0], tip[0], right[0]], [left[1], tip[1], right[1]], color=color, lw=lw, solid_capstyle="round", zorder=4)
    return {"type": "arrow", "points": [[p[0] * 60, -p[1] * 60] for p in pts], "strokeColor": color}


def _save_excalidraw(name: str, elements: list[dict[str, Any]]) -> None:
    doc = {
        "type": "excalidraw",
        "version": 2,
        "source": "bifurx-docs",
        "elements": elements,
        "appState": {"viewBackgroundColor": "#ffffff"},
    }
    with open(os.path.join(SRC_DIR, f"{name}.excalidraw"), "w") as fh:
        json.dump(doc, fh, indent=2)


def make_e1_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.9), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 14.4)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    elems: list[dict[str, Any]] = []

    # Top core box
    elems.append(
        _draw_box(
            ax,
            3.35,
            3.95,
            7.70,
            1.75,
            "bifurx.homotopy  (Standalone Pure-JAX Core)",
            "track_arclength_scan  •  Euler / Padé (2,1) Predictors  •  Weighted Norm\n"
            "Kantorovich Bordered Newton  •  Timme Adaptive Step-Size Control",
            fill="#dbeafe",
            stroke="#1e3a5f",
            seed=101,
            title_size=12.5,
            sub_size=9.2,
        )
    )

    # Hook pill annotation
    ax.text(
        7.20,
        3.28,
        "Pluggable hooks:  chart_fn   •   bordered_solver   •   accept_fn   •   step_observer_fn",
        ha="center",
        va="center",
        fontsize=9.3,
        color="#0f766e",
        family="Chalkboard SE",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f0fdfa", edgecolor="#14b8a6", lw=1.2),
        zorder=5,
    )

    # Bottom 4 module boxes
    modules = [
        (
            0.30,
            "Equilibria",
            "bifurx.continuation\nLP / BP / HB detection\nABE branch switching",
            "#e0f2fe",
            "#0369a1",
            102,
        ),
        (
            3.85,
            "Codim-2 Curves",
            "bifurx.codim2\nLP & HB 2-param curves\nCP / BT / GH / ZH",
            "#ede9fe",
            "#5b21b6",
            103,
        ),
        (
            7.40,
            "Periodic Orbits",
            "bifurx.bvp\nGauss collocation\nO(N n^3) QR & Periodic Schur",
            "#dcfce7",
            "#15803d",
            104,
        ),
        (
            10.95,
            "iPRC / Phase",
            "bifurx.prc\nBlock-condensed linear\nadjoint BVP & continuation",
            "#ffedd5",
            "#c2410c",
            105,
        ),
    ]

    for idx, (x, title, sub, fill, stroke, seed) in enumerate(modules):
        elems.append(_draw_box(ax, x, 0.45, 3.15, 2.05, title, sub, fill=fill, stroke=stroke, seed=seed))
        cx = x + 1.575
        top_x = 4.6 + idx * 1.73
        elems.append(_draw_arrow(ax, [(top_x, 3.95), (cx, 2.55)], color=stroke, seed=200 + idx))

    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT_DIR, "bifurx_architecture.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    _save_excalidraw("bifurx_architecture", elems)


def make_e2_tracker_flow() -> None:
    fig, ax = plt.subplots(figsize=(11.8, 5.1), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    elems: list[dict[str, Any]] = []

    ax.text(
        7.5,
        6.12,
        "Single XLA-Compiled Step inside track_arclength_scan (jax.lax.scan)",
        ha="center",
        va="center",
        fontsize=13.0,
        fontweight="bold",
        color="#1e3a5f",
        family="Chalkboard SE",
    )

    # Top row: Carry -> 1. Predictor -> 2. Weighted Norm -> 3. Bordered Newton
    elems.append(
        _draw_box(
            ax,
            0.25,
            3.85,
            2.55,
            1.45,
            "Carry State",
            "(u_k, p_k, tau_k,\nchart_k, ds_k, w_k)",
            fill="#f1f5f9",
            stroke="#475569",
            seed=301,
            title_size=10.8,
            sub_size=8.6,
        )
    )
    elems.append(
        _draw_box(
            ax,
            3.40,
            3.85,
            2.95,
            1.45,
            "1. Predictor",
            "Euler tangent or\nPadé (2,1) rational",
            fill="#dbeafe",
            stroke="#1d4ed8",
            seed=302,
            title_size=10.8,
            sub_size=8.8,
        )
    )
    elems.append(
        _draw_box(
            ax,
            6.95,
            3.85,
            2.80,
            1.45,
            "2. Norm Update",
            "Scale-invariant\nweights w_k (EMA)",
            fill="#e0f2fe",
            stroke="#0369a1",
            seed=303,
            title_size=10.8,
            sub_size=8.8,
        )
    )
    elems.append(
        _draw_box(
            ax,
            10.35,
            3.85,
            4.35,
            1.45,
            "3. Bordered Newton Corrector",
            "Kantorovich contraction check\n(Dense LU / Block LU / OperatorSolver)",
            fill="#ede9fe",
            stroke="#6d28d9",
            seed=304,
            title_size=10.8,
            sub_size=8.6,
        )
    )

    elems.append(_draw_arrow(ax, [(2.82, 4.58), (3.36, 4.58)], color="#334155", seed=310))
    elems.append(_draw_arrow(ax, [(6.37, 4.58), (6.91, 4.58)], color="#334155", seed=311))
    elems.append(_draw_arrow(ax, [(9.77, 4.58), (10.31, 4.58)], color="#334155", seed=312))

    # Decision diamond below step 3
    elems.append(
        _draw_diamond(
            ax,
            12.52,
            2.20,
            1.65,
            0.85,
            "Converged &\naccept_fn?",
            fill="#fef3c7",
            stroke="#b45309",
            seed=305,
            size=9.5,
        )
    )
    elems.append(_draw_arrow(ax, [(12.52, 3.83), (12.52, 3.08)], color="#334155", seed=313))

    # Reject branch (middle-left): Timme contraction -> loop back to Predictor
    elems.append(
        _draw_box(
            ax,
            5.05,
            1.95,
            3.95,
            1.15,
            "Reject: Timme Contraction",
            "Shrink ds_k analytically -> retry step",
            fill="#fee2e2",
            stroke="#b91c1c",
            seed=306,
            title_size=10.2,
            sub_size=8.5,
        )
    )
    elems.append(_draw_arrow(ax, [(10.84, 2.20), (9.04, 2.52)], color="#b91c1c", seed=314, ls="--"))
    ax.text(10.05, 2.66, "No (abort / fail)", fontsize=8.8, color="#b91c1c", fontweight="bold", family="Chalkboard SE", ha="center")
    elems.append(_draw_arrow(ax, [(5.02, 2.52), (4.85, 2.52), (4.85, 3.81)], color="#b91c1c", seed=315, ls="--"))

    # Accept branch (bottom row): Step 4 -> Step 5 -> Next carry
    elems.append(
        _draw_box(
            ax,
            5.05,
            0.25,
            4.25,
            1.25,
            "4. Update Chart, Tangent & Taylor3",
            "chart_fn(u, p)  •  orient <tau_{k+1}, tau_k> > 0",
            fill="#dcfce7",
            stroke="#15803d",
            seed=307,
            title_size=10.2,
            sub_size=8.4,
        )
    )
    elems.append(
        _draw_box(
            ax,
            0.25,
            0.25,
            4.15,
            1.25,
            "5. Observer & Adaptive ds_{k+1}",
            "step_observer_fn (eigvals / Floquet)\nTimme optimal ds_{k+1}  ->  next step",
            fill="#d1fae5",
            stroke="#047857",
            seed=308,
            title_size=10.2,
            sub_size=8.4,
        )
    )

    elems.append(_draw_arrow(ax, [(12.52, 1.32), (12.52, 0.87), (9.34, 0.87)], color="#15803d", seed=316))
    ax.text(11.10, 1.12, "Yes (accept)", fontsize=8.8, color="#15803d", fontweight="bold", family="Chalkboard SE", ha="center")
    elems.append(_draw_arrow(ax, [(5.02, 0.87), (4.44, 0.87)], color="#15803d", seed=317))
    elems.append(_draw_arrow(ax, [(1.52, 1.53), (1.52, 3.81)], color="#047857", seed=318))

    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT_DIR, "homotopy_tracker_flow.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    _save_excalidraw("homotopy_tracker_flow", elems)


def make_e3_iprc_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.1), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 5.1)
    ax.axis("off")

    elems: list[dict[str, Any]] = []

    ax.text(
        7.25,
        4.75,
        "continuation_iprc: Coupled Periodic-Orbit Continuation & Linear Adjoint Pipeline",
        ha="center",
        va="center",
        fontsize=12.2,
        fontweight="bold",
        color="#1e3a5f",
        family="Chalkboard SE",
    )

    # Top row: Orbit continuation
    elems.append(
        _draw_box(
            ax,
            0.25,
            2.60,
            2.80,
            1.55,
            "Seed Orbit",
            "(u_0, T_0, p_0)\nHopf or user guess",
            fill="#f1f5f9",
            stroke="#475569",
            seed=401,
            title_size=10.8,
            sub_size=8.8,
        )
    )
    elems.append(
        _draw_box(
            ax,
            3.70,
            2.60,
            4.65,
            1.55,
            "Periodic Orbit Continuation",
            "continue_periodic_orbit (Gauss collocation)\nO(N n^3) cyclic block-QR  •  adaptive remesh",
            fill="#dbeafe",
            stroke="#1e3a5f",
            seed=402,
            title_size=10.8,
            sub_size=8.7,
        )
    )
    elems.append(
        _draw_box(
            ax,
            9.05,
            2.60,
            5.15,
            1.55,
            "One-Shot Block-Condensed Adjoint",
            "_solve_adjoint_collocation_jax (same mesh!)\nParallel vmap LU + rank-1 border alpha f0_unit",
            fill="#ffedd5",
            stroke="#c2410c",
            seed=403,
            title_size=10.8,
            sub_size=8.6,
        )
    )

    elems.append(_draw_arrow(ax, [(3.07, 3.38), (3.66, 3.38)], color="#334155", seed=410))
    elems.append(_draw_arrow(ax, [(8.38, 3.38), (9.01, 3.38)], color="#1e3a5f", seed=411))
    ax.text(8.70, 3.85, "(u*_k, T_k)\nat p_k", fontsize=7.6, color="#1e3a5f", family="Chalkboard SE", ha="center", linespacing=1.05)

    # Bottom output box
    elems.append(
        _draw_box(
            ax,
            3.70,
            0.35,
            10.50,
            1.35,
            "IPRCContinuationResult:  Full Manifold Z(theta; p)",
            "Orbits u*(p)   •   Periods T(p)   •   iPRC waveforms Z(theta; p)   •   Peak amplitudes & zero crossings",
            fill="#dcfce7",
            stroke="#15803d",
            seed=404,
            title_size=11.0,
            sub_size=8.9,
        )
    )

    elems.append(_draw_arrow(ax, [(11.62, 2.57), (11.62, 1.74)], color="#15803d", seed=412))
    ax.text(12.60, 2.15, "Z_k(theta) at each p_k", fontsize=8.6, color="#15803d", family="Chalkboard SE", ha="center")

    # Loop-back arrow showing parameter stepping p_0 -> p_1 -> ... -> p_end
    elems.append(_draw_arrow(ax, [(6.02, 2.57), (6.02, 2.10), (4.65, 2.10), (4.65, 2.57)], color="#2563eb", seed=413, ls="--"))
    ax.text(5.35, 1.88, "next step p_k -> p_{k+1}", fontsize=8.3, color="#2563eb", family="Chalkboard SE", ha="center")

    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT_DIR, "iprc_continuation_pipeline.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    _save_excalidraw("iprc_continuation_pipeline", elems)


def make_e4_jax_autodiff_patterns() -> None:
    fig, ax = plt.subplots(figsize=(11.8, 5.2), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    elems: list[dict[str, Any]] = []

    ax.text(
        7.5,
        6.25,
        "Differentiable Programming in bifurx: Tensor-Free Nested JVPs & Static XLA Loops",
        ha="center",
        va="center",
        fontsize=12.6,
        fontweight="bold",
        color="#1e3a5f",
        family="Chalkboard SE",
    )

    # Left source box: pure JAX residual F(u, p)
    elems.append(
        _draw_box(
            ax,
            0.25,
            2.35,
            3.10,
            2.15,
            "Pure-JAX Model",
            "F(u, p) : R^n x R -> R^n\nPyTree Problem &\nOptions (float64)",
            fill="#f1f5f9",
            stroke="#334155",
            seed=501,
            title_size=11.0,
            sub_size=8.9,
        )
    )

    # Top-right: 4 tensor-free JVP consumers
    jvp_boxes = [
        (
            4.35,
            4.55,
            4.85,
            1.20,
            "1st-Order JVP  (O(n) mem)",
            "Matrix-free GMRES matvec & operator\nresidual verification (homotopy.bordered)",
            "#dbeafe",
            "#1d4ed8",
            502,
        ),
        (
            9.70,
            4.55,
            5.00,
            1.20,
            "2nd-Order Nested JVP",
            "ABE coefficients c_11, c_12, c_22 &\nCusp test <w, D^2F[v, v]> (bifurcations / codim2)",
            "#e0f2fe",
            "#0369a1",
            503,
        ),
        (
            4.35,
            2.90,
            4.85,
            1.20,
            "3rd-Order Nested JVP",
            "Padé (2,1) Taylor coeffs (z_2, z_3) &\nKuznetsov l_1 forms B, C (no O(n^4) tensors!)",
            "#ede9fe",
            "#6d28d9",
            504,
        ),
        (
            9.70,
            2.90,
            5.00,
            1.20,
            "vmap + jacfwd Parallelism",
            "Parallel subinterval Jacobians & LU\nin de Boor-Swartz condensation (bvp / prc)",
            "#dcfce7",
            "#15803d",
            505,
        ),
    ]
    for x, y, w, h, title, sub, fill, stroke, seed in jvp_boxes:
        elems.append(
            _draw_box(
                ax, x, y, w, h, title, sub, fill=fill, stroke=stroke, seed=seed, title_size=10.2, sub_size=8.3
            )
        )

    # Arrows from Pure-JAX Model to the 4 autodiff patterns
    elems.append(_draw_arrow(ax, [(3.38, 3.95), (4.30, 5.15)], color="#1d4ed8", seed=510))
    elems.append(_draw_arrow(ax, [(3.38, 3.55), (4.30, 3.50)], color="#6d28d9", seed=511))
    elems.append(_draw_arrow(ax, [(9.23, 5.15), (9.66, 5.15)], color="#0369a1", seed=512, ls="--"))
    elems.append(_draw_arrow(ax, [(9.23, 3.50), (9.66, 3.50)], color="#15803d", seed=513, ls="--"))

    # Bottom banner: Static-shape XLA execution model
    elems.append(
        _draw_box(
            ax,
            0.25,
            0.35,
            14.45,
            1.55,
            "Static-Shape XLA Execution (jax.lax.scan + jax.lax.while_loop)",
            "Fixed-length scan with `is_active` mask & `jnp.where` state freezing  •  Zero Python overhead per step\n"
            "Adaptive step retry inside scan  •  Outer chunking only for topology-changing mesh redistribution",
            fill="#fef3c7",
            stroke="#b45309",
            seed=506,
            title_size=10.8,
            sub_size=8.7,
        )
    )

    elems.append(_draw_arrow(ax, [(6.75, 2.86), (6.75, 1.95)], color="#b45309", seed=514))
    elems.append(_draw_arrow(ax, [(12.20, 2.86), (12.20, 1.95)], color="#b45309", seed=515))

    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT_DIR, "jax_autodiff_patterns.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    _save_excalidraw("jax_autodiff_patterns", elems)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SRC_DIR, exist_ok=True)
    make_e1_architecture()
    make_e2_tracker_flow()
    make_e3_iprc_pipeline()
    make_e4_jax_autodiff_patterns()
    print("Generated E1, E2, E3, E4 PNGs and .excalidraw sources.")
