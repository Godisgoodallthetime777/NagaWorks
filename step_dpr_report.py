"""DPR — Detailed Print Report image for 3D printing reference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from step_blueprint import (
    VIEW_LABELS,
    build_blueprint_snapshot,
)
from step_mesh_viewer import SolidModel
from step_viewport_scale import DISPLAY_UNITS, convert_mm, format_length

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.gridspec import GridSpec
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "DPR export requires matplotlib. Install dependencies:\n"
        "  pip install -r requirements.txt"
    ) from exc

try:
    import trimesh
except ImportError:
    trimesh = None  # type: ignore

FIGURE_DPI = 150
REPORT_UNITS = ("mm", "cm", "m", "in", "nm")


@dataclass(frozen=True)
class DPRSnapshot:
    """All data needed to render a print report (copied on the main thread)."""

    source_name: str
    source_path: str
    generated_at: str
    extent_mm: tuple[float, float, float]
    bmin: tuple[float, float, float]
    bmax: tuple[float, float, float]
    part_count: int
    vertex_count: int
    display_faces: int
    original_faces: int
    volume_mm3: float
    surface_mm2: float
    part_names: tuple[str, ...]
    segments_by_view: dict[str, np.ndarray]


def _mesh_volume_surface(model: SolidModel) -> tuple[float, float]:
    if trimesh is None:
        return 0.0, 0.0
    volume = 0.0
    area = 0.0
    for part in model.parts:
        if len(part.faces) == 0:
            continue
        try:
            mesh = trimesh.Trimesh(
                vertices=np.ascontiguousarray(part.vertices, dtype=np.float64),
                faces=np.ascontiguousarray(part.faces, dtype=np.int64),
                process=False,
            )
            if mesh.is_volume:
                volume += float(abs(mesh.volume))
            area += float(mesh.area)
        except Exception:
            continue
    return volume, area


def build_dpr_snapshot(model: SolidModel) -> DPRSnapshot:
    bmin, bmax = model.bounds
    bmin_a = np.asarray(bmin, dtype=np.float64)
    bmax_a = np.asarray(bmax, dtype=np.float64)
    extent = bmax_a - bmin_a
    volume, surface = _mesh_volume_surface(model)
    bp = build_blueprint_snapshot(model)
    return DPRSnapshot(
        source_name=Path(model.source_path).name,
        source_path=str(model.source_path),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        extent_mm=(float(extent[0]), float(extent[1]), float(extent[2])),
        bmin=(float(bmin_a[0]), float(bmin_a[1]), float(bmin_a[2])),
        bmax=(float(bmax_a[0]), float(bmax_a[1]), float(bmax_a[2])),
        part_count=model.part_count,
        vertex_count=model.vertex_count,
        display_faces=model.display_faces,
        original_faces=model.original_faces,
        volume_mm3=volume,
        surface_mm2=surface,
        part_names=tuple(p.name for p in model.parts),
        segments_by_view=bp.segments_by_view,
    )


def _draw_mini_view(ax: plt.Axes, segments: np.ndarray, title: str) -> None:
    ax.set_facecolor("#fafafa")
    ax.set_title(title, fontsize=8, fontweight="bold", color="#333")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=6, colors="#666")
    for spine in ax.spines.values():
        spine.set_color("#bbb")

    if segments.size == 0:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    segs = np.ascontiguousarray(segments, dtype=np.float64)
    ax.add_collection(
        LineCollection(segs, colors="#224466", linewidths=0.25, capstyle="round")
    )
    xs = segs[:, :, 0].ravel()
    ys = segs[:, :, 1].ravel()
    margin = max(xs.max() - xs.min(), ys.max() - ys.min(), 1e-6) * 0.1
    ax.set_xlim(xs.min() - margin, xs.max() + margin)
    ax.set_ylim(ys.min() - margin, ys.max() + margin)


def _format_volume_mm3(value_mm3: float, unit_key: str) -> str:
    if value_mm3 <= 0:
        return "—"
    # volume scales as length^3
    _label, factor = DISPLAY_UNITS.get(unit_key, ("mm", 1.0))
    converted = value_mm3 * (factor**3)
    if unit_key == "mm":
        if converted >= 1_000_000:
            return f"{converted / 1_000_000:.4g} cm³"
        return f"{converted:.4g} mm³"
    if unit_key == "cm":
        return f"{converted:.4g} cm³"
    if unit_key == "m":
        return f"{converted:.4g} m³"
    if unit_key == "in":
        return f"{converted:.4g} in³"
    return f"{converted:.4g} {unit_key}³"


def _format_area_mm2(value_mm2: float, unit_key: str) -> str:
    if value_mm2 <= 0:
        return "—"
    _label, factor = DISPLAY_UNITS.get(unit_key, ("mm", 1.0))
    converted = value_mm2 * (factor**2)
    if unit_key == "mm":
        if converted >= 100:
            return f"{converted / 100:.4g} cm²"
        return f"{converted:.4g} mm²"
    if unit_key == "cm":
        return f"{converted:.4g} cm²"
    if unit_key == "m":
        return f"{converted:.4g} m²"
    if unit_key == "in":
        return f"{converted:.4g} in²"
    return f"{converted:.4g} {unit_key}²"


def export_dpr_snapshot(snapshot: DPRSnapshot, path: Path) -> Path:
    """Save a Detailed Print Report image (PNG or PDF)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dx, dy, dz = snapshot.extent_mm
    diag_mm = float(np.linalg.norm([dx, dy, dz]))

    fig = plt.figure(figsize=(11.0, 8.5), dpi=FIGURE_DPI)
    fig.patch.set_facecolor("white")
    gs = GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[2.2, 1.5, 1.4],
        hspace=0.42,
        wspace=0.32,
        left=0.07,
        right=0.96,
        top=0.96,
        bottom=0.05,
    )

    # Large isometric preview
    ax_iso = fig.add_subplot(gs[0, 0])
    _draw_mini_view(
        ax_iso,
        snapshot.segments_by_view.get("iso", np.empty((0, 2, 2))),
        VIEW_LABELS.get("iso", "ISOMETRIC"),
    )

    # Dimension table
    ax_dims = fig.add_subplot(gs[0, 1:])
    ax_dims.axis("off")
    dim_lines = [
        f"Model: {snapshot.source_name}   •   Generated: {snapshot.generated_at}",
        "",
        "PRINT DIMENSIONS (bounding box)",
        "",
        f"  Width  (X):  {format_length(convert_mm(dx, 'mm'), 'mm'):>16}  |  "
        f"{format_length(convert_mm(dx, 'cm'), 'cm'):>14}  |  "
        f"{format_length(convert_mm(dx, 'in'), 'in'):>14}",
        f"  Depth  (Y):  {format_length(convert_mm(dy, 'mm'), 'mm'):>16}  |  "
        f"{format_length(convert_mm(dy, 'cm'), 'cm'):>14}  |  "
        f"{format_length(convert_mm(dy, 'in'), 'in'):>14}",
        f"  Height (Z):  {format_length(convert_mm(dz, 'mm'), 'mm'):>16}  |  "
        f"{format_length(convert_mm(dz, 'cm'), 'cm'):>14}  |  "
        f"{format_length(convert_mm(dz, 'in'), 'in'):>14}",
        f"  Diagonal:    {format_length(convert_mm(diag_mm, 'mm'), 'mm'):>16}  |  "
        f"{format_length(convert_mm(diag_mm, 'cm'), 'cm'):>14}  |  "
        f"{format_length(convert_mm(diag_mm, 'in'), 'in'):>14}",
        "",
        "ALL UNITS",
    ]
    for unit in REPORT_UNITS:
        dim_lines.append(
            f"  [{unit.upper():>2}]  X={format_length(convert_mm(dx, unit), unit):>12}  "
            f"Y={format_length(convert_mm(dy, unit), unit):>12}  "
            f"Z={format_length(convert_mm(dz, unit), unit):>12}"
        )
    dim_lines.extend(
        [
            "",
            f"  Min corner (X,Y,Z):  "
            f"({snapshot.bmin[0]:.4g}, {snapshot.bmin[1]:.4g}, {snapshot.bmin[2]:.4g}) mm",
            f"  Max corner (X,Y,Z):  "
            f"({snapshot.bmax[0]:.4g}, {snapshot.bmax[1]:.4g}, {snapshot.bmax[2]:.4g}) mm",
            "",
            f"  Volume (solid):   {_format_volume_mm3(snapshot.volume_mm3, 'mm'):>14}  "
            f"({_format_volume_mm3(snapshot.volume_mm3, 'cm')})",
            f"  Surface area:     {_format_area_mm2(snapshot.surface_mm2, 'mm'):>14}  "
            f"({_format_area_mm2(snapshot.surface_mm2, 'cm')})",
        ]
    )
    ax_dims.text(
        0.0,
        1.0,
        "\n".join(dim_lines),
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        color="#222",
        linespacing=1.35,
        transform=ax_dims.transAxes,
    )

    # Orthographic row
    for col, view in enumerate(("top", "front", "right")):
        ax = fig.add_subplot(gs[1, col])
        _draw_mini_view(
            ax,
            snapshot.segments_by_view.get(view, np.empty((0, 2, 2))),
            VIEW_LABELS.get(view, view.upper()),
        )

    # Stats & print checklist (bottom row)
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis("off")
    simplified = ""
    if snapshot.original_faces > snapshot.display_faces:
        simplified = (
            f" (preview mesh {snapshot.display_faces:,} / "
            f"source {snapshot.original_faces:,} triangles)"
        )
    parts_preview = ", ".join(snapshot.part_names[:8])
    if len(snapshot.part_names) > 8:
        parts_preview += f", … (+{len(snapshot.part_names) - 8} more)"

    stats_text = (
        "MESH & PRINT CHECKLIST\n"
        f"  Parts: {snapshot.part_count}  •  Vertices: {snapshot.vertex_count:,}  •  "
        f"Triangles{simplified}\n"
        f"  Part names: {parts_preview or '—'}\n"
        "\n"
        "  Before slicing for your 3D printer:\n"
        "  □ Confirm units match your slicer (mm recommended)\n"
        "  □ Scale model if needed to fit build volume\n"
        "  □ Export STL/3MF from CAD if mesh preview was simplified\n"
        "  □ Check wall thickness, overhangs, and support requirements\n"
        "  □ Verify volume and footprint against filament / resin budget"
    )
    ax_stats.text(
        0.0,
        1.0,
        stats_text,
        va="top",
        ha="left",
        fontsize=8,
        color="#333",
        linespacing=1.4,
        transform=ax_stats.transAxes,
    )

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        fig.savefig(path, format="pdf", facecolor="white", bbox_inches="tight")
    else:
        if suffix not in {".png", ".jpg", ".jpeg"}:
            path = path.with_suffix(".png")
        fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def export_dpr_report(model: SolidModel, path: Path) -> Path:
    return export_dpr_snapshot(build_dpr_snapshot(model), path)
