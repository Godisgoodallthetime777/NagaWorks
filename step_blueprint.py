"""Orthographic blueprint sheets from tessellated STEP models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from step_mesh_viewer import SolidModel

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Blueprint export requires matplotlib. Install dependencies:\n"
        "  pip install -r requirements.txt"
    ) from exc

# (horizontal axis index, vertical axis index) in vertex XYZ
VIEW_PROJECTIONS: dict[str, tuple[int, int]] = {
    "top": (0, 1),
    "front": (0, 2),
    "right": (1, 2),
    "iso": None,  # handled separately
}

VIEW_LABELS: dict[str, str] = {
    "top": "TOP",
    "front": "FRONT",
    "right": "RIGHT",
    "iso": "ISOMETRIC",
}

BLUEPRINT_VIEWS = ("top", "front", "right", "iso")
FIGURE_DPI = 150
SUBPLOT_SIZE = 4.0


@dataclass(frozen=True)
class BlueprintSnapshot:
    """Copied mesh edges for thread-safe / main-thread matplotlib export."""

    source_name: str
    segments_by_view: dict[str, np.ndarray]


def build_blueprint_snapshot(model: SolidModel) -> BlueprintSnapshot:
    """Copy all view segments on the main thread before export."""
    segments: dict[str, np.ndarray] = {}
    for name in BLUEPRINT_VIEWS:
        segs = get_view_segments(model, name)
        if segs.size:
            segs = np.ascontiguousarray(segs, dtype=np.float64)
        segments[name] = segs
    return BlueprintSnapshot(
        source_name=Path(model.source_path).name,
        segments_by_view=segments,
    )


def get_view_segments(model: SolidModel, view: str) -> np.ndarray:
    """Return Nx2x2 edge segments for a named view (top, front, right, iso)."""
    if view == "iso":
        return _isometric_segments(model)
    axes_idx = VIEW_PROJECTIONS.get(view)
    if axes_idx is None:
        raise ValueError(f"Unknown view: {view}")
    return _segments_for_view(model, axes_idx[0], axes_idx[1])


def _segments_for_view(
    model: SolidModel,
    axis_h: int,
    axis_v: int,
) -> np.ndarray:
    all_segments: list[np.ndarray] = []
    for part in model.parts:
        if len(part.faces) == 0:
            continue
        v = part.vertices
        edges: set[tuple[int, int]] = set()
        for tri in part.faces:
            for i in range(3):
                a, b = int(tri[i]), int(tri[(i + 1) % 3])
                edges.add((a, b) if a < b else (b, a))
        if not edges:
            continue
        segs = np.empty((len(edges), 2, 2), dtype=np.float64)
        for idx, (a, b) in enumerate(edges):
            segs[idx, 0, 0] = v[a, axis_h]
            segs[idx, 0, 1] = v[a, axis_v]
            segs[idx, 1, 0] = v[b, axis_h]
            segs[idx, 1, 1] = v[b, axis_v]
        all_segments.append(segs)
    if not all_segments:
        return np.empty((0, 2, 2), dtype=np.float64)
    return np.vstack(all_segments)


def _isometric_segments(model: SolidModel) -> np.ndarray:
    """Cabinet-style isometric: x right, y up, z depth."""
    cos30 = np.cos(np.pi / 6)
    sin30 = np.sin(np.pi / 6)
    all_segments: list[np.ndarray] = []

    for part in model.parts:
        if len(part.faces) == 0:
            continue
        v = part.vertices
        projected = np.column_stack(
            [
                v[:, 0] - v[:, 2] * cos30,
                v[:, 1] + v[:, 2] * sin30,
            ]
        )
        edges: set[tuple[int, int]] = set()
        for tri in part.faces:
            for i in range(3):
                a, b = int(tri[i]), int(tri[(i + 1) % 3])
                edges.add((a, b) if a < b else (b, a))
        if not edges:
            continue
        segs = np.empty((len(edges), 2, 2), dtype=np.float64)
        for idx, (a, b) in enumerate(edges):
            segs[idx, 0] = projected[a]
            segs[idx, 1] = projected[b]
        all_segments.append(segs)

    if not all_segments:
        return np.empty((0, 2, 2), dtype=np.float64)
    return np.vstack(all_segments)


def _draw_view(ax: plt.Axes, segments: np.ndarray, title: str) -> None:
    ax.set_facecolor("white")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10, fontweight="bold", color="#222")
    ax.tick_params(colors="#444", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#888")

    if len(segments) == 0:
        ax.text(0.5, 0.5, "No geometry", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    segments = np.ascontiguousarray(segments, dtype=np.float64)
    lc = LineCollection(
        segments,
        colors="#111111",
        linewidths=0.35,
        capstyle="round",
    )
    ax.add_collection(lc)

    xs = segments[:, :, 0].ravel()
    ys = segments[:, :, 1].ravel()
    margin = max((xs.max() - xs.min()), (ys.max() - ys.min()), 1e-6) * 0.08
    ax.set_xlim(xs.min() - margin, xs.max() + margin)
    ax.set_ylim(ys.min() - margin, ys.max() + margin)

    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    ax.text(
        0.02,
        0.02,
        f"W {w:.2g}  H {h:.2g}",
        transform=ax.transAxes,
        fontsize=7,
        color="#555",
        va="bottom",
    )


def export_blueprint(
    model: SolidModel,
    path: Path,
    *,
    views: tuple[str, ...] = BLUEPRINT_VIEWS,
) -> Path:
    """Save a 2×2 blueprint sheet (PNG or PDF) with orthographic views."""
    return export_blueprint_snapshot(build_blueprint_snapshot(model), path, views=views)


def export_blueprint_snapshot(
    snapshot: BlueprintSnapshot,
    path: Path,
    *,
    views: tuple[str, ...] = BLUEPRINT_VIEWS,
) -> Path:
    """
    Save blueprint from pre-copied segments (call on the Qt main thread).

    Returns the path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(SUBPLOT_SIZE * 2, SUBPLOT_SIZE * 2), dpi=FIGURE_DPI)
    fig.patch.set_facecolor("white")
    source_name = snapshot.source_name

    view_axes = {
        "top": axes[0, 0],
        "front": axes[0, 1],
        "right": axes[1, 0],
        "iso": axes[1, 1],
    }

    for name in views:
        ax = view_axes.get(name)
        if ax is None:
            continue
        segments = snapshot.segments_by_view.get(name, np.empty((0, 2, 2)))
        _draw_view(ax, segments, VIEW_LABELS.get(name, name.upper()))

    for ax in axes.ravel():
        if ax not in view_axes.values():
            ax.set_visible(False)

    fig.suptitle(
        f"Blueprint — {source_name}",
        fontsize=12,
        fontweight="bold",
        color="#222",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        "Orthographic projection from mesh preview • dimensions in model units",
        ha="center",
        fontsize=8,
        color="#666",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        fig.savefig(path, format="pdf", facecolor="white", bbox_inches="tight")
    else:
        if suffix not in {".png", ".jpg", ".jpeg", ".svg"}:
            path = path.with_suffix(".png")
        fig.savefig(path, facecolor="white", bbox_inches="tight")

    plt.close(fig)
    return path
