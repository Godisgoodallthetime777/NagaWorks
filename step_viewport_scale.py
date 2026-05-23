"""Futuristic scale / ruler overlay for the 3D viewport."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from app_logging import log_exception

# STEP mesh coordinates are treated as millimeters (common CAD export default).
SOURCE_UNIT_MM = 1.0

DISPLAY_UNITS: dict[str, tuple[str, float]] = {
    "nm": ("nm", 1_000_000.0),
    "mm": ("mm", 1.0),
    "cm": ("cm", 0.1),
    "m": ("m", 0.001),
    "in": ("in", 1.0 / 25.4),
}

ACCENT = QColor("#00e5ff")
ACCENT_DIM = QColor("#0099bb")
PANEL_BG = QColor(12, 20, 32, 215)
PANEL_BORDER = QColor("#00e5ff")
TEXT_COLOR = QColor("#e8f4ff")
TICK_COLOR = QColor("#66ccee")


def convert_mm(value_mm: float, unit_key: str) -> float:
    _label, factor = DISPLAY_UNITS.get(unit_key, ("mm", 1.0))
    return float(value_mm) * factor


def format_length(value: float, unit_key: str) -> str:
    label, _ = DISPLAY_UNITS.get(unit_key, ("mm", 1.0))
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f} {label}"
    if magnitude >= 100:
        return f"{value:.0f} {label}"
    if magnitude >= 10:
        return f"{value:.1f} {label}"
    if magnitude >= 1:
        return f"{value:.2f} {label}"
    if magnitude >= 0.01:
        return f"{value:.3f} {label}"
    return f"{value:.3e} {label}"


def pick_nice_bar_length_mm(raw_mm: float) -> float:
    if raw_mm <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw_mm))
    base = 10.0**exponent
    for mult in (1.0, 2.0, 5.0, 10.0):
        candidate = mult * base
        if candidate >= raw_mm * 0.82:
            return candidate
    return base * 10.0


def world_mm_per_pixel(plotter) -> float:
    """Approximate model mm represented by one screen pixel (vertical axis)."""
    renderer = plotter.renderer
    _w, height = renderer.GetSize()
    if height <= 0:
        return 1.0

    camera = plotter.camera
    if getattr(camera, "parallel_projection", False):
        half_height = float(getattr(camera, "parallel_scale", 1.0))
        if half_height > 0:
            return (2.0 * half_height) / float(height)

    cam_pos = np.asarray(camera.position, dtype=np.float64)
    focal = np.asarray(camera.focal_point, dtype=np.float64)
    dist = float(np.linalg.norm(cam_pos - focal))
    if dist < 1e-9:
        dist = 1.0
    view_angle = float(getattr(camera, "view_angle", 30.0))
    visible_height = 2.0 * dist * math.tan(math.radians(view_angle * 0.5))
    if visible_height <= 1e-9:
        visible_height = 1.0
    return visible_height / float(height)


class ViewportScaleOverlay(QWidget):
    """Corner HUD: dynamic scale bar + model size in selected units."""

    BAR_TARGET_PX = 130
    FIXED_WIDTH = 248
    FIXED_HEIGHT = 96

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(self.FIXED_WIDTH, self.FIXED_HEIGHT)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._unit_key = "mm"
        self._bar_mm = 10.0
        self._bar_px = self.BAR_TARGET_PX
        self._model_size_text = ""
        self._subtitle = "CAD units → mm basis"
        self._plotter_provider: Callable | None = None
        self._bounds_provider: Callable | None = None
        self._paint_frozen = False

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(500)
        self._refresh_timer.timeout.connect(self.refresh)

    def set_providers(
        self,
        plotter_provider: Callable,
        bounds_provider: Callable,
    ) -> None:
        self._plotter_provider = plotter_provider
        self._bounds_provider = bounds_provider

    def set_unit(self, unit_key: str) -> None:
        if unit_key in DISPLAY_UNITS:
            self._unit_key = unit_key
            self.refresh()

    def set_active(self, active: bool) -> None:
        if active:
            self.show()
            self.raise_()
            if not self._paint_frozen:
                self._refresh_timer.start()
                self.refresh()
        else:
            self._refresh_timer.stop()
            self.hide()

    def set_paint_frozen(self, frozen: bool) -> None:
        """Skip paint/refresh while VTK or matplotlib export is in progress."""
        self._paint_frozen = frozen
        if frozen:
            self._refresh_timer.stop()
        elif self.isVisible():
            self._refresh_timer.start()

    def reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        margin = 12
        x = max(margin, parent.width() - self.width() - margin)
        y = max(margin, parent.height() - self.height() - margin)
        self.move(x, y)
        self.raise_()

    def refresh(self) -> None:
        if self._paint_frozen or not self.isVisible() or self._plotter_provider is None:
            return
        try:
            plotter = self._plotter_provider()
            if plotter is None:
                return

            try:
                mm_per_px = world_mm_per_pixel(plotter)
                raw_bar_mm = mm_per_px * self.BAR_TARGET_PX
                self._bar_mm = pick_nice_bar_length_mm(raw_bar_mm)
                self._bar_px = max(
                    40,
                    min(self.BAR_TARGET_PX + 20, int(self._bar_mm / max(mm_per_px, 1e-12))),
                )
            except Exception:
                self._bar_mm = 10.0
                self._bar_px = self.BAR_TARGET_PX

            self._model_size_text = ""
            if self._bounds_provider is not None:
                bounds = self._bounds_provider()
                if bounds is not None:
                    bmin, bmax = bounds
                    extent_mm = np.asarray(bmax, dtype=np.float64) - np.asarray(bmin, dtype=np.float64)
                    ex = convert_mm(extent_mm[0], self._unit_key)
                    ey = convert_mm(extent_mm[1], self._unit_key)
                    ez = convert_mm(extent_mm[2], self._unit_key)
                    self._model_size_text = (
                        f"Δ {format_length(ex, self._unit_key)} × "
                        f"{format_length(ey, self._unit_key)} × "
                        f"{format_length(ez, self._unit_key)}"
                    )

            self.update()
        except Exception:
            log_exception("scale_overlay refresh")

    def paintEvent(self, _event) -> None:
        if self._paint_frozen:
            return
        try:
            self._paint_scale_overlay()
        except Exception:
            log_exception("scale_overlay paintEvent")

    def _paint_scale_overlay(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(1, 1, -2, -2)
        painter.setBrush(PANEL_BG)
        painter.setPen(QPen(PANEL_BORDER, 1.5))
        painter.drawRoundedRect(rect, 8, 8)

        # Corner accents
        accent_len = 14
        painter.setPen(QPen(ACCENT, 2))
        painter.drawLine(rect.topLeft() + QPoint(6, 6), rect.topLeft() + QPoint(6 + accent_len, 6))
        painter.drawLine(rect.topLeft() + QPoint(6, 6), rect.topLeft() + QPoint(6, 6 + accent_len))
        painter.drawLine(
            rect.bottomRight() + QPoint(-6, -6),
            rect.bottomRight() + QPoint(-6 - accent_len, -6),
        )
        painter.drawLine(
            rect.bottomRight() + QPoint(-6, -6),
            rect.bottomRight() + QPoint(-6, -6 - accent_len),
        )

        painter.setPen(TEXT_COLOR)
        title_font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.2)
        painter.setFont(title_font)
        unit_label, _ = DISPLAY_UNITS.get(self._unit_key, ("mm", 1.0))
        painter.drawText(12, 18, f"SCALE // {unit_label.upper()}")

        painter.setFont(QFont("Consolas", 7))
        painter.setPen(ACCENT_DIM)
        painter.drawText(12, 30, self._subtitle)

        # Scale bar
        bar_y = 52
        bar_x0 = 14
        bar_x1 = bar_x0 + self._bar_px
        painter.setPen(QPen(ACCENT, 2))
        painter.drawLine(bar_x0, bar_y, bar_x1, bar_y)
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            tx = int(bar_x0 + self._bar_px * t)
            tick_h = 8 if t in (0.0, 0.5, 1.0) else 5
            painter.setPen(QPen(TICK_COLOR, 1.5))
            painter.drawLine(tx, bar_y - tick_h, tx, bar_y + tick_h)

        display_len = convert_mm(self._bar_mm, self._unit_key)
        painter.setPen(TEXT_COLOR)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(bar_x0, bar_y + 22, format_length(display_len, self._unit_key))

        if self._model_size_text:
            painter.setFont(QFont("Consolas", 7))
            painter.setPen(QColor("#9ab8cc"))
            painter.drawText(12, self.height() - 10, self._model_size_text)

        painter.end()
