"""Pencil-sketch style renderings from tessellated STEP models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from step_blueprint import get_view_segments
from step_mesh_viewer import SolidModel

MeshPartArrays = tuple[np.ndarray, np.ndarray]  # vertices, faces

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Pencil sketch export requires matplotlib. Install dependencies:\n"
        "  pip install -r requirements.txt"
    ) from exc

PAPER_COLOR = "#f3efe4"
PENCIL_DARK = "#2a2520"
PENCIL_MID = "#5c534a"
PENCIL_LIGHT = "#8a8076"
FIGURE_DPI = 200
FIGURE_INCHES = 8.0
DEFAULT_VIEW = "iso"
SKETCH_PASSES = 4
LIGHT_DIR = np.array([0.45, 0.65, 0.55], dtype=np.float64)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)


def _project_isometric(points: np.ndarray) -> np.ndarray:
    cos30 = np.cos(np.pi / 6)
    sin30 = np.sin(np.pi / 6)
    return np.column_stack(
        [
            points[:, 0] - points[:, 2] * cos30,
            points[:, 1] + points[:, 2] * sin30,
        ]
    )


def _project_view(points: np.ndarray, view: str) -> np.ndarray:
    if view == "iso":
        return _project_isometric(points)
    from step_blueprint import VIEW_PROJECTIONS

    axes_idx = VIEW_PROJECTIONS.get(view)
    if axes_idx is None:
        raise ValueError(f"Unknown view: {view}")
    h, v = axes_idx
    return np.column_stack([points[:, h], points[:, v]])


def _add_paper_texture(ax: plt.Axes, rng: np.random.Generator) -> None:
    """Subtle grain on the drawing paper."""
    n = 2800
    xs = rng.uniform(0, 1, n)
    ys = rng.uniform(0, 1, n)
    sizes = rng.uniform(0.15, 1.2, n)
    alphas = rng.uniform(0.02, 0.07, n)
    ax.scatter(
        xs,
        ys,
        s=sizes,
        c="#9a9080",
        alpha=alphas,
        transform=ax.transAxes,
        zorder=0,
        linewidths=0,
    )


@dataclass(frozen=True)
class SketchSnapshot:
    """Copied mesh data for main-thread matplotlib export."""

    source_stem: str
    view: str
    segments: np.ndarray
    parts: tuple[MeshPartArrays, ...]


def build_sketch_snapshot(model: SolidModel, view: str = DEFAULT_VIEW) -> SketchSnapshot:
    segs = get_view_segments(model, view)
    if segs.size:
        segs = np.ascontiguousarray(segs, dtype=np.float64)
    parts = tuple(
        (
            np.ascontiguousarray(part.vertices, dtype=np.float64),
            np.ascontiguousarray(part.faces, dtype=np.int64),
        )
        for part in model.parts
        if len(part.faces) > 0
    )
    return SketchSnapshot(
        source_stem=Path(model.source_path).stem,
        view=view,
        segments=segs,
        parts=parts,
    )


def _add_tone_shading(
    ax: plt.Axes,
    parts: tuple[MeshPartArrays, ...],
    view: str,
    rng: np.random.Generator,
) -> None:
    """Soft graphite shading from face normals (drawn under the linework)."""
    polys: list[np.ndarray] = []
    tones: list[float] = []

    for vertices, faces in parts:
        if len(faces) == 0:
            continue
        v = vertices
        for tri in faces:
            idx = tri.astype(int)
            pts = v[idx]
            e1 = pts[1] - pts[0]
            e2 = pts[2] - pts[0]
            normal = np.cross(e1, e2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-12:
                continue
            normal = normal / norm_len
            brightness = float(np.dot(normal, LIGHT_DIR))
            if brightness <= 0.05:
                continue
            proj = _project_view(pts, view)
            polys.append(proj)
            # Slight randomness so shading feels hand-applied.
            tones.append(0.06 + 0.14 * brightness + rng.uniform(-0.015, 0.015))

    if not polys:
        return

    facecolors = [(0.35, 0.32, 0.28, min(0.35, t)) for t in tones]
    pc = PolyCollection(
        polys,
        facecolors=facecolors,
        edgecolors="none",
        zorder=1,
    )
    ax.add_collection(pc)


def _sketchy_line_collections(
    segments: np.ndarray,
    rng: np.random.Generator,
    jitter_scale: float,
) -> list[LineCollection]:
    if len(segments) == 0:
        return []

    collections: list[LineCollection] = []
    charcoals = [PENCIL_LIGHT, PENCIL_MID, PENCIL_MID, PENCIL_DARK]
    widths = [0.55, 0.65, 0.75, 1.05]
    alphas = [0.18, 0.28, 0.38, 0.72]

    for pass_i in range(SKETCH_PASSES):
        jittered = segments.copy()
        noise = rng.normal(0.0, jitter_scale, jittered.shape)
        jittered += noise
        collections.append(
            LineCollection(
                jittered,
                colors=charcoals[pass_i],
                linewidths=widths[pass_i],
                alpha=alphas[pass_i],
                capstyle="round",
                joinstyle="round",
                zorder=3 + pass_i,
            )
        )

    # Crisp definition pass (minimal jitter).
    outline = segments.copy()
    outline += rng.normal(0.0, jitter_scale * 0.15, outline.shape)
    collections.append(
        LineCollection(
            outline,
            colors=PENCIL_DARK,
            linewidths=0.95,
            alpha=0.88,
            capstyle="round",
            zorder=8,
        )
    )
    return collections


def _draw_pencil_sketch(
    ax: plt.Axes,
    parts: tuple[MeshPartArrays, ...],
    segments: np.ndarray,
    *,
    view: str,
    title: str,
) -> None:
    rng = np.random.default_rng()

    ax.set_facecolor(PAPER_COLOR)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    _add_paper_texture(ax, rng)

    if len(segments) == 0:
        ax.text(
            0.5,
            0.5,
            "No geometry",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=PENCIL_MID,
            fontsize=12,
        )
        return

    _add_tone_shading(ax, parts, view, rng)

    xs = segments[:, :, 0].ravel()
    ys = segments[:, :, 1].ravel()
    span = max(xs.max() - xs.min(), ys.max() - ys.min(), 1e-6)
    jitter_scale = span * 0.0028

    for lc in _sketchy_line_collections(segments, rng, jitter_scale):
        ax.add_collection(lc)

    margin = span * 0.12
    ax.set_xlim(xs.min() - margin, xs.max() + margin)
    ax.set_ylim(ys.min() - margin, ys.max() + margin)

    ax.set_title(
        title,
        fontsize=14,
        fontweight="normal",
        fontstyle="italic",
        color=PENCIL_DARK,
        pad=12,
        family="serif",
    )


def export_pencil_sketch(
    model: SolidModel,
    path: Path,
    *,
    view: str = DEFAULT_VIEW,
) -> Path:
    """Save a single pencil-sketch image (PNG, JPEG, or PDF)."""
    return export_pencil_sketch_snapshot(build_sketch_snapshot(model, view), path)


def export_pencil_sketch_snapshot(snapshot: SketchSnapshot, path: Path) -> Path:
    """
    Save sketch from pre-copied mesh data (call on the Qt main thread).

    Returns the path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(FIGURE_INCHES, FIGURE_INCHES),
        dpi=FIGURE_DPI,
    )
    fig.patch.set_facecolor(PAPER_COLOR)

    _draw_pencil_sketch(
        ax,
        snapshot.parts,
        snapshot.segments,
        view=snapshot.view,
        title=f"{snapshot.source_stem} — pencil sketch",
    )

    fig.tight_layout()

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        fig.savefig(path, format="pdf", facecolor=PAPER_COLOR, bbox_inches="tight")
    elif suffix in {".jpg", ".jpeg"}:
        fig.savefig(path, format="jpeg", facecolor=PAPER_COLOR, bbox_inches="tight", quality=92)
    else:
        if suffix not in {".png", ".svg"}:
            path = path.with_suffix(".png")
        fig.savefig(path, facecolor=PAPER_COLOR, bbox_inches="tight")

    plt.close(fig)
    return path
