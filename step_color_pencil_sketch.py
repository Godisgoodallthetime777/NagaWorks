"""Colored-pencil sketch style renderings from tessellated STEP models."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from step_mesh_viewer import SolidModel
from step_pencil_sketch import (
    DEFAULT_VIEW,
    FIGURE_DPI,
    FIGURE_INCHES,
    LIGHT_DIR,
    PAPER_COLOR,
    PENCIL_DARK,
    PENCIL_MID,
    SketchSnapshot,
    _add_paper_texture,
    _project_view,
    _sketchy_line_collections,
    build_sketch_snapshot,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Color pencil sketch export requires matplotlib. Install dependencies:\n"
        "  pip install -r requirements.txt"
    ) from exc

# Muted colored-pencil palette (light blue, pink, orange, cyan, coral).
PENCIL_COLORS: tuple[tuple[float, float, float, float], ...] = (
    (0.53, 0.78, 0.91, 0.52),
    (0.91, 0.47, 0.66, 0.48),
    (0.91, 0.65, 0.35, 0.50),
    (0.55, 0.82, 0.88, 0.46),
    (0.85, 0.55, 0.45, 0.48),
)
ACCENT_RED = (0.84, 0.28, 0.28, 0.62)
STROKE_ANGLES = (np.radians(32), np.radians(58))
MAX_HATCH_FACES = 3500
HATCH_FACE_STRIDE = 3


def _part_triangle_areas(parts: tuple[tuple[np.ndarray, np.ndarray], ...]) -> list[float]:
    areas: list[float] = []
    for vertices, faces in parts:
        total = 0.0
        for tri in faces:
            idx = tri.astype(int)
            pts = vertices[idx]
            e1 = pts[1] - pts[0]
            e2 = pts[2] - pts[0]
            total += float(np.linalg.norm(np.cross(e1, e2))) * 0.5
        areas.append(total)
    return areas


def _point_in_triangle_batch(
    points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """Vectorized point-in-triangle for columns of 2D points."""
    v0 = c - a
    v1 = b - a
    v2 = points - a
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot11 = float(np.dot(v1, v1))
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-18:
        return np.zeros(len(points), dtype=bool)
    inv = 1.0 / denom
    dot02 = v2 @ v0
    dot12 = v2 @ v1
    u = dot02 * dot11 - dot01 * dot12
    v = dot00 * dot12 - dot01 * dot02
    u *= inv
    v *= inv
    return (u >= -0.02) & (v >= -0.02) & ((u + v) <= 1.02)


def _color_pencil_hatch_lines(
    poly: np.ndarray,
    rng: np.random.Generator,
    spacing: float,
    angle: float,
) -> np.ndarray:
    """Diagonal pencil strokes clipped roughly to a triangle."""
    a, b, c = poly
    minx, miny = poly.min(axis=0)
    maxx, maxy = poly.max(axis=0)
    diag = max(float(np.hypot(maxx - minx, maxy - miny)), 1e-9)

    ca, sa = np.cos(angle), np.sin(angle)
    perp = np.array([-sa, ca])
    center = poly.mean(axis=0)
    n_steps = int(diag / max(spacing, 1e-9)) + 3

    segments: list[list[np.ndarray]] = []
    for i in range(-n_steps, n_steps + 1):
        jitter = rng.uniform(-spacing * 0.25, spacing * 0.25)
        offset = center + perp * (i * spacing + jitter)
        p1 = offset - np.array([ca, sa]) * diag * 2.5
        p2 = offset + np.array([ca, sa]) * diag * 2.5

        ts = np.linspace(0.0, 1.0, 24)
        pts = p1[None, :] + ts[:, None] * (p2 - p1)[None, :]
        inside = _point_in_triangle_batch(pts, a, b, c)
        if not inside.any():
            continue
        idx = np.where(inside)[0]
        segments.append([pts[idx[0]], pts[idx[-1]]])

    if not segments:
        return np.empty((0, 2, 2))
    return np.asarray(segments, dtype=np.float64)


def _add_color_pencil_shading(
    ax: plt.Axes,
    parts: tuple[tuple[np.ndarray, np.ndarray], ...],
    view: str,
    rng: np.random.Generator,
    *,
    tick: Callable[[], None] | None = None,
) -> None:
    """Soft colored fills with visible pencil-stroke texture."""
    part_areas = _part_triangle_areas(parts)
    if part_areas:
        area_threshold = min(part_areas) * 1.35 + max(part_areas) * 0.008
    else:
        area_threshold = 0.0

    fill_polys: list[np.ndarray] = []
    fill_colors: list[tuple[float, float, float, float]] = []
    hatch_by_color: dict[str, list[np.ndarray]] = {}
    hatch_style: dict[str, tuple[float, float]] = {}

    visible_faces = 0
    for vertices, faces in parts:
        if len(faces) == 0:
            continue
        for tri in faces:
            idx = tri.astype(int)
            pts3 = vertices[idx]
            e1 = pts3[1] - pts3[0]
            e2 = pts3[2] - pts3[0]
            normal = np.cross(e1, e2)
            if float(np.linalg.norm(normal)) < 1e-12:
                continue
            normal = normal / np.linalg.norm(normal)
            if float(np.dot(normal, LIGHT_DIR)) > 0.04:
                visible_faces += 1

    hatch_stride = 1
    if visible_faces > MAX_HATCH_FACES:
        hatch_stride = max(HATCH_FACE_STRIDE, visible_faces // MAX_HATCH_FACES)

    processed = 0
    for part_idx, (vertices, faces) in enumerate(parts):
        if len(faces) == 0:
            continue

        use_accent = part_areas[part_idx] <= area_threshold and len(parts) > 1
        base_rgba = ACCENT_RED if use_accent else PENCIL_COLORS[part_idx % len(PENCIL_COLORS)]
        stroke_rgb = base_rgba[:3]
        color_key = (
            f"#{int(stroke_rgb[0]*255):02x}"
            f"{int(stroke_rgb[1]*255):02x}"
            f"{int(stroke_rgb[2]*255):02x}"
        )
        angle = STROKE_ANGLES[part_idx % len(STROKE_ANGLES)]

        for tri_idx, tri in enumerate(faces):
            idx = tri.astype(int)
            pts3 = vertices[idx]
            e1 = pts3[1] - pts3[0]
            e2 = pts3[2] - pts3[0]
            normal = np.cross(e1, e2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-12:
                continue
            normal = normal / norm_len
            brightness = float(np.dot(normal, LIGHT_DIR))
            if brightness <= 0.04:
                continue

            proj = _project_view(pts3, view)
            fill_polys.append(proj)

            alpha = base_rgba[3] * (0.55 + 0.45 * brightness)
            alpha += rng.uniform(-0.06, 0.06)
            alpha = float(np.clip(alpha, 0.22, 0.72))
            rgb = [
                float(np.clip(c * (0.82 + 0.18 * brightness) + rng.uniform(-0.03, 0.03), 0.0, 1.0))
                for c in base_rgba[:3]
            ]
            fill_colors.append((rgb[0], rgb[1], rgb[2], alpha))

            if hatch_stride == 1 or tri_idx % hatch_stride == 0:
                tri_area = float(np.linalg.norm(np.cross(e1, e2))) * 0.5
                spacing = max(np.sqrt(tri_area) * 0.09, 0.35)
                segs = _color_pencil_hatch_lines(proj, rng, spacing, angle)
                if len(segs):
                    hatch_by_color.setdefault(color_key, []).append(segs)
                    if color_key not in hatch_style:
                        stroke_strength = 0.35 + 0.25 * brightness
                        hatch_style[color_key] = (
                            rng.uniform(0.35, 0.65),
                            float(np.clip(stroke_strength + rng.uniform(-0.08, 0.08), 0.18, 0.55)),
                        )

            processed += 1
            if tick is not None and processed % 2500 == 0:
                tick()

    if fill_polys:
        ax.add_collection(
            PolyCollection(
                fill_polys,
                facecolors=fill_colors,
                edgecolors="none",
                zorder=1,
            )
        )

    for color_key, segment_groups in hatch_by_color.items():
        width, alpha = hatch_style.get(color_key, (0.5, 0.35))
        merged = np.vstack(segment_groups)
        ax.add_collection(
            LineCollection(
                merged,
                colors=color_key,
                linewidths=width,
                alpha=alpha,
                capstyle="round",
                zorder=2,
            )
        )


def _draw_color_pencil_sketch(
    ax: plt.Axes,
    parts: tuple[tuple[np.ndarray, np.ndarray], ...],
    segments: np.ndarray,
    *,
    view: str,
    title: str,
    tick: Callable[[], None] | None = None,
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

    _add_color_pencil_shading(ax, parts, view, rng, tick=tick)

    xs = segments[:, :, 0].ravel()
    ys = segments[:, :, 1].ravel()
    span = max(xs.max() - xs.min(), ys.max() - ys.min(), 1e-6)
    jitter_scale = span * 0.0032

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


def export_color_pencil_sketch(
    model: SolidModel,
    path: Path,
    *,
    view: str = DEFAULT_VIEW,
    tick: Callable[[], None] | None = None,
) -> Path:
    """Save a single color-pencil sketch image (PNG, JPEG, or PDF)."""
    return export_color_pencil_sketch_snapshot(
        build_sketch_snapshot(model, view),
        path,
        tick=tick,
    )


def export_color_pencil_sketch_snapshot(
    snapshot: SketchSnapshot,
    path: Path,
    *,
    tick: Callable[[], None] | None = None,
) -> Path:
    """
    Save color-pencil sketch from pre-copied mesh data (call on the Qt main thread).

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

    _draw_color_pencil_sketch(
        ax,
        snapshot.parts,
        snapshot.segments,
        view=snapshot.view,
        title=f"{snapshot.source_stem} — color pencil",
        tick=tick,
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
