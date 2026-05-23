"""GPU-accelerated 3D viewer (VTK / PyVista) embedded in Qt."""

from __future__ import annotations

import os
from typing import Callable

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QResizeEvent, QShortcut
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from pyvistaqt import QtInteractor

from app_logging import get_logger, log_exception
from step_mesh_viewer import SolidModel
from step_viewport_scale import ViewportScaleOverlay

log = get_logger()

STANDARD_VIEWS: dict[str, Callable[[pv.Plotter], None]] = {
    "top": lambda p: p.view_xy(),
    "bottom": lambda p: p.view_xy(negative=True),
    "front": lambda p: p.view_xz(),
    "back": lambda p: p.view_xz(negative=True),
    "right": lambda p: p.view_yz(),
    "left": lambda p: p.view_yz(negative=True),
    "iso": lambda p: p.view_isometric(),
}

DEFAULT_MODEL_COLOR = "#8cb3e6"
DEFAULT_OPACITY = 1.0
VIEWER_BACKGROUND = "#ffffff"
MESH_ACTOR_PREFIX = "part::"


class Vtk3DWidget(QWidget):
    """OpenGL 3D viewport — color, transparency, standard views."""

    ZOOM_FACTOR = 1.2

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        pv.set_plot_theme("document")
        self._model: SolidModel | None = None
        self._wireframe = False
        self._initialized = False
        self._display_color = DEFAULT_MODEL_COLOR
        self._opacity = DEFAULT_OPACITY
        self._cached_meshes: list[pv.PolyData] = []
        self._fullscreen_toggle: Callable[[], None] | None = None
        self._render_paused = False
        self._pending_fit: tuple[int, int] | None = None
        self._ruler_enabled = False
        self._ruler_unit = "mm"
        self._ruler_update_pending = False
        self._ruler_timer = QTimer(self)
        self._ruler_timer.setSingleShot(True)
        self._ruler_timer.setInterval(80)
        self._ruler_timer.timeout.connect(self._apply_ruler_update)

        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background-color: {VIEWER_BACKGROUND};")

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        self.plotter.background_color = VIEWER_BACKGROUND
        self.plotter.show_axes()

        interactor = self.plotter.interactor
        interactor.setMinimumSize(200, 200)
        interactor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        interactor.setStyleSheet(f"background-color: {VIEWER_BACKGROUND};")
        self._layout.addWidget(interactor, stretch=1)
        self._layout.setStretch(0, 1)

        self.setFocusPolicy(Qt.StrongFocus)
        interactor.setFocusPolicy(Qt.StrongFocus)
        self._bind_shortcuts()

        self._scale_overlay = ViewportScaleOverlay(self)
        self._scale_overlay.set_providers(
            plotter_provider=lambda: self.plotter,
            bounds_provider=self._model_bounds,
        )
        self._scale_overlay.hide()
        self._hook_interaction_updates()

    def _hook_interaction_updates(self) -> None:
        """VTK fires observers off the GUI thread — always defer ruler work to Qt."""
        try:
            iren = self.plotter.interactor

            def _on_camera_changed(*_args) -> None:
                self._schedule_ruler_update()

            iren.AddObserver("EndInteractionEvent", _on_camera_changed)
            iren.AddObserver("MouseWheelForwardEvent", _on_camera_changed)
            iren.AddObserver("MouseWheelBackwardEvent", _on_camera_changed)
        except Exception:
            log_exception("_hook_interaction_updates")

    def _model_bounds(self):
        if self._model is None:
            return None
        return self._model.bounds

    def set_ruler_visible(self, visible: bool) -> None:
        self._ruler_enabled = visible
        if visible:
            self._scale_overlay.set_unit(self._ruler_unit)
            self._scale_overlay.set_active(True)
            self._position_scale_overlay()
        else:
            self._scale_overlay.set_active(False)

    def set_ruler_unit(self, unit_key: str) -> None:
        self._ruler_unit = unit_key
        self._scale_overlay.set_unit(unit_key)
        if self._ruler_enabled:
            self._scale_overlay.refresh()

    def _schedule_ruler_update(self) -> None:
        if not self._ruler_enabled:
            return
        if not self._ruler_update_pending:
            self._ruler_update_pending = True
            QTimer.singleShot(0, self._ruler_timer.start)

    def _apply_ruler_update(self) -> None:
        self._ruler_update_pending = False
        if not self._ruler_enabled or self._render_paused:
            return
        try:
            self._position_scale_overlay()
            self._scale_overlay.refresh()
        except Exception:
            log_exception("_apply_ruler_update")

    def _position_scale_overlay(self) -> None:
        if hasattr(self, "_scale_overlay"):
            self._scale_overlay.reposition()

    def set_fullscreen_toggle(self, callback: Callable[[], None]) -> None:
        self._fullscreen_toggle = callback

    def set_render_paused(self, paused: bool) -> None:
        """Pause VTK resize/render during layout changes (main window schedules reflow)."""
        self._render_paused = paused
        if not paused and self._initialized and self.isVisible():
            w, h = self.width(), self.height()
            if w >= 16 and h >= 16:
                QTimer.singleShot(0, lambda: self.fit_viewport(w, h))
            elif self._pending_fit is not None:
                pw, ph = self._pending_fit
                QTimer.singleShot(0, lambda: self.fit_viewport(pw, ph))

    def discard_pending_fit(self) -> None:
        self._pending_fit = None

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        size = event.size()
        if size.width() < 16 or size.height() < 16:
            return
        if self._render_paused:
            self._pending_fit = (size.width(), size.height())
            return
        if self._initialized:
            QTimer.singleShot(0, lambda: self.fit_viewport(size.width(), size.height()))
        self._schedule_ruler_update()

    def get_display_color(self) -> str:
        return self._display_color

    def get_opacity(self) -> float:
        return self._opacity

    def set_display_color(self, color: str) -> None:
        self._display_color = color
        self._update_actor_properties()

    def set_opacity(self, opacity: float) -> None:
        self._opacity = max(0.05, min(1.0, float(opacity)))
        self._update_actor_properties()

    def _bind_shortcuts(self) -> None:
        self._shortcuts: list[QShortcut] = []
        bindings: list[tuple[QKeySequence | str, Callable[[], None]]] = [
            (QKeySequence.ZoomIn, self.zoom_in),
            (QKeySequence(Qt.CTRL | Qt.Key_Equal), self.zoom_in),
            (QKeySequence.ZoomOut, self.zoom_out),
            (QKeySequence(Qt.CTRL | Qt.Key_Minus), self.zoom_out),
            (QKeySequence(Qt.CTRL | Qt.Key_Underscore), self.zoom_out),
            (QKeySequence(Qt.CTRL | Qt.Key_0), self.reset_camera),
            (QKeySequence(Qt.Key_F12), self._on_f12),
        ]
        for key in (Qt.Key_Plus, getattr(Qt, "Key_Pad_Plus", Qt.Key_Plus)):
            bindings.append((QKeySequence(Qt.CTRL | key), self.zoom_in))

        for seq, slot in bindings:
            shortcut = QShortcut(seq, self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
            self._shortcuts.append(shortcut)

    def _on_f12(self) -> None:
        if self._fullscreen_toggle is not None:
            self._fullscreen_toggle()

    def _ensure_gl_ready(self) -> None:
        if self._initialized:
            return
        self.plotter.render()
        self.plotter.interactor.Initialize()
        self._initialized = True

    def clear(self) -> None:
        self._model = None
        self._cached_meshes.clear()
        self.plotter.clear()
        self.plotter.background_color = VIEWER_BACKGROUND
        self.plotter.render()
        if hasattr(self, "_scale_overlay"):
            self._scale_overlay.set_active(False)

    def set_wireframe(self, enabled: bool) -> None:
        self._wireframe = enabled
        self._rebuild_actors(reset_camera=False)

    def show_model(self, model: SolidModel) -> None:
        self._ensure_gl_ready()
        self._model = model
        self._build_mesh_cache()
        self._rebuild_actors(reset_camera=True)

    def _build_mesh_cache(self) -> None:
        self._cached_meshes.clear()
        if self._model is None:
            return
        for part in self._model.parts:
            if len(part.faces) == 0:
                continue
            faces = np.hstack(
                [np.full((len(part.faces), 1), 3, dtype=np.int64), part.faces]
            )
            mesh = pv.PolyData(part.vertices, faces)
            if mesh.n_points > 0:
                self._cached_meshes.append(mesh)

    def _rebuild_actors(self, *, reset_camera: bool) -> None:
        if not self._cached_meshes:
            return

        self.plotter.clear()
        self.plotter.background_color = VIEWER_BACKGROUND
        self.plotter.show_axes()

        style = "wireframe" if self._wireframe else "surface"
        for index, mesh in enumerate(self._cached_meshes):
            self.plotter.add_mesh(
                mesh,
                color=self._display_color,
                opacity=self._opacity,
                style=style,
                show_edges=self._wireframe,
                smooth_shading=False,
                lighting=True,
                name=f"{MESH_ACTOR_PREFIX}{index}",
            )

        if reset_camera:
            self.set_standard_view("iso")
        else:
            self._force_render()
        self._schedule_ruler_update()

    def _update_actor_properties(self) -> None:
        if not self._cached_meshes:
            return
        rgb = list(pv.Color(self._display_color).float_rgb)
        for key, actor in list(self.plotter.renderer.actors.items()):
            if not str(key).startswith(MESH_ACTOR_PREFIX):
                continue
            prop = actor.GetProperty()
            prop.SetColor(rgb)
            prop.SetOpacity(self._opacity)
        self._force_render()

    def set_standard_view(self, name: str) -> None:
        if self._model is None:
            return
        view_fn = STANDARD_VIEWS.get(name.lower())
        if view_fn is None:
            return
        view_fn(self.plotter)
        self.plotter.reset_camera()
        self._force_render()
        self._schedule_ruler_update()

    def fit_viewport(self, width: int, height: int) -> None:
        """Resize VTK render area to fill the available widget space."""
        if not self._initialized or width < 16 or height < 16:
            return
        if self._render_paused or not self.isVisible():
            self._pending_fit = (int(width), int(height))
            return
        try:
            w, h = int(width), int(height)
            interactor = self.plotter.interactor
            interactor.resize(w, h)
            rw = self.plotter.render_window
            if rw is not None:
                rw.SetSize(w, h)
            try:
                interactor.GetRenderWindow().SetSize(w, h)
            except Exception:
                pass
            self.plotter.render()
            self._schedule_ruler_update()
        except Exception:
            log_exception("fit_viewport")

    def _force_render(self) -> None:
        if not self._initialized or self._render_paused:
            return
        try:
            self.plotter.render()
            rw = self.plotter.render_window
            if rw is not None:
                rw.Render()
            self.plotter.interactor.Render()
            self._schedule_ruler_update()
        except Exception:
            log_exception("_force_render")

    def zoom_in(self) -> None:
        self.plotter.camera.zoom(self.ZOOM_FACTOR)
        self._force_render()

    def zoom_out(self) -> None:
        self.plotter.camera.zoom(1.0 / self.ZOOM_FACTOR)
        self._force_render()

    def reset_camera(self) -> None:
        self.set_standard_view("iso")

    def closeEvent(self, event) -> None:
        self._ruler_enabled = False
        if hasattr(self, "_scale_overlay"):
            self._scale_overlay.set_active(False)
        self._ruler_timer.stop()
        try:
            self.plotter.close()
        except Exception:
            pass
        super().closeEvent(event)
