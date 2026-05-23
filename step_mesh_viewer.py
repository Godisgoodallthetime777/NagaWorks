"""Load STEP files into simplified meshes for fast GPU preview."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    import trimesh
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "3D preview requires trimesh. Install dependencies:\n"
        "  pip install -r requirements.txt"
    ) from exc

# Total triangle budget for smooth interaction (OpenGL / GPU).
DEFAULT_FACE_BUDGET = 50_000
MIN_FACES_PER_PART = 500


@dataclass(frozen=True)
class MeshPart:
    name: str
    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class SolidModel:
    parts: tuple[MeshPart, ...]
    bounds: tuple[np.ndarray, np.ndarray]
    source_path: str
    display_faces: int
    original_faces: int

    @property
    def part_count(self) -> int:
        return len(self.parts)

    @property
    def vertex_count(self) -> int:
        return sum(len(p.vertices) for p in self.parts)


PART_COLORS = [
    "#8cb3e6",
    "#d98c85",
    "#8cd1a0",
    "#c7a0e0",
    "#ebd072",
    "#80c9d1",
    "#e0aec8",
    "#b3b3b3",
]


def load_solid_model(path: Path, *, face_budget: int = DEFAULT_FACE_BUDGET) -> SolidModel:
    """Parse a STEP file into tessellated, simplified mesh parts."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in {".stp", ".step"}:
        raise ValueError(f"Not a STEP file: {path.name}")

    mtime = path.stat().st_mtime
    return _load_cached(str(path), mtime, face_budget)


@lru_cache(maxsize=12)
def _load_cached(path_str: str, mtime: float, face_budget: int) -> SolidModel:
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    loaded = trimesh.load(path_str, force="mesh")
    raw_parts: list[tuple[str, trimesh.Trimesh]] = []

    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("STEP file contains no displayable geometry.")
        for name, geom in loaded.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                raw_parts.append((str(name), geom))
            else:
                combined = trimesh.util.concatenate(tuple(loaded.geometry.values()))
                raw_parts = [(path.stem, combined)]
                break
    elif isinstance(loaded, trimesh.Trimesh):
        raw_parts.append((path.stem, loaded))
    else:
        raise ValueError(f"Unsupported geometry type: {type(loaded).__name__}")

    if not raw_parts:
        raise ValueError("No mesh data could be extracted from this file.")

    original_faces = sum(len(m.faces) for _, m in raw_parts)
    simplified = _simplify_parts(raw_parts, face_budget)

    parts = tuple(_mesh_to_part(name, mesh) for name, mesh in simplified)
    display_faces = sum(len(p.faces) for p in parts)
    bounds = _combined_bounds(list(parts))
    return SolidModel(
        parts=parts,
        bounds=bounds,
        source_path=path_str,
        display_faces=display_faces,
        original_faces=original_faces,
    )


def _simplify_parts(
    raw_parts: list[tuple[str, trimesh.Trimesh]],
    face_budget: int,
) -> list[tuple[str, trimesh.Trimesh]]:
    total_faces = sum(len(m.faces) for _, m in raw_parts)
    if total_faces <= face_budget:
        return [(name, mesh.copy()) for name, mesh in raw_parts]

    per_part_budget = max(MIN_FACES_PER_PART, face_budget // max(len(raw_parts), 1))
    result: list[tuple[str, trimesh.Trimesh]] = []
    remaining_budget = face_budget

    for index, (name, mesh) in enumerate(raw_parts):
        mesh = mesh.copy()
        mesh.remove_unreferenced_vertices()
        n_faces = len(mesh.faces)
        if index == len(raw_parts) - 1:
            target = max(MIN_FACES_PER_PART, remaining_budget)
        else:
            share = max(1, n_faces) / max(total_faces, 1)
            target = max(MIN_FACES_PER_PART, int(face_budget * share))
            target = min(target, remaining_budget - MIN_FACES_PER_PART * (len(raw_parts) - index - 1))
            target = min(target, n_faces)

        if n_faces > target:
            mesh = _decimate(mesh, target)
        result.append((name, mesh))
        remaining_budget -= len(mesh.faces)

    return result


def _decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    target_faces = max(4, int(target_faces))
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=target_faces)
    except Exception:
        ratio = target_faces / len(mesh.faces)
        return mesh.simplify_quadric_decimation(percent= max(0.01, min(0.99, ratio)))


def _mesh_to_part(name: str, mesh: trimesh.Trimesh) -> MeshPart:
    mesh.remove_unreferenced_vertices()
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Part '{name}' has no faces.")
    return MeshPart(
        name=name,
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
    )


def _combined_bounds(parts: list[MeshPart]) -> tuple[np.ndarray, np.ndarray]:
    mins = []
    maxs = []
    for part in parts:
        if len(part.vertices) == 0:
            continue
        mins.append(part.vertices.min(axis=0))
        maxs.append(part.vertices.max(axis=0))
    if not mins:
        origin = np.zeros(3)
        return origin, origin
    return np.vstack(mins).min(axis=0), np.vstack(maxs).max(axis=0)


def part_color(index: int) -> str:
    return PART_COLORS[index % len(PART_COLORS)]
