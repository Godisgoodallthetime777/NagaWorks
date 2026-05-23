"""Detect system printers (including 3D brands) and print DPR sheets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPageLayout, QPageSize, QPainter, QPixmap
from PySide6.QtPrintSupport import QPrintDialog, QPrinter, QPrinterInfo
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

# Name/description keywords → display brand (order: more specific first).
THREE_D_PRINTER_BRANDS: tuple[tuple[str, str], ...] = (
    ("bambu lab", "Bambu Lab"),
    ("bambulab", "Bambu Lab"),
    ("bambu", "Bambu Lab"),
    ("x1 carbon", "Bambu Lab"),
    ("x1c", "Bambu Lab"),
    ("p1s", "Bambu Lab"),
    ("p1p", "Bambu Lab"),
    ("a1 mini", "Bambu Lab"),
    ("prusa", "Prusa"),
    ("mk4", "Prusa"),
    ("mk3", "Prusa"),
    ("creality", "Creality"),
    ("ender", "Creality"),
    ("k1 max", "Creality"),
    ("k1", "Creality"),
    ("ultimaker", "Ultimaker"),
    ("anycubic", "Anycubic"),
    ("kobra", "Anycubic"),
    ("flashforge", "FlashForge"),
    ("raise3d", "Raise3D"),
    ("snapmaker", "Snapmaker"),
    ("octoprint", "OctoPrint"),
    ("klipper", "Klipper"),
    ("moonraker", "Klipper"),
    ("voron", "Voron"),
    ("lulzbot", "LulzBot"),
    ("makerbot", "MakerBot"),
    ("formlabs", "Formlabs"),
    ("markforged", "Markforged"),
    ("3d printer", "3D Printer"),
    ("3dprint", "3D Printer"),
    ("fdm", "FDM"),
    ("resin", "Resin"),
)


@dataclass(frozen=True)
class DetectedPrinter:
    name: str
    description: str
    location: str
    is_3d_printer: bool
    brand: str | None
    is_default: bool

    @property
    def display_label(self) -> str:
        if self.is_3d_printer and self.brand:
            return f"[3D • {self.brand}]  {self.name}"
        if self.is_3d_printer:
            return f"[3D]  {self.name}"
        return self.name

    @property
    def detail_text(self) -> str:
        parts = []
        if self.description:
            parts.append(self.description)
        if self.location:
            parts.append(self.location)
        if self.is_default:
            parts.append("Windows default")
        return "  •  ".join(parts) if parts else "System printer"


def _classify_printer(name: str, description: str, location: str) -> tuple[bool, str | None]:
    combined = f"{name} {description} {location}".lower()
    combined = re.sub(r"\s+", " ", combined)
    for keyword, brand in THREE_D_PRINTER_BRANDS:
        if keyword in combined:
            return True, brand
    return False, None


def detect_printers() -> list[DetectedPrinter]:
    """List printers installed in Windows, with 3D-brand detection."""
    default_name = QPrinterInfo.defaultPrinter().printerName()
    found: list[DetectedPrinter] = []

    for info in QPrinterInfo.availablePrinters():
        name = info.printerName()
        desc = info.description() or ""
        loc = info.location() or ""
        is_3d, brand = _classify_printer(name, desc, loc)
        found.append(
            DetectedPrinter(
                name=name,
                description=desc,
                location=loc,
                is_3d_printer=is_3d,
                brand=brand,
                is_default=(name == default_name),
            )
        )

    found.sort(key=lambda p: (not p.is_3d_printer, not p.is_default, p.name.lower()))
    return found


def print_image(image_path: Path, printer_name: str, *, parent=None) -> bool:
    """Send an image file to the chosen printer (shows page setup dialog)."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Print image not found: {path}")

    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        raise ValueError(f"Could not load image: {path}")

    printer = QPrinter(QPrinter.PrinterMode.HighResolution)
    printer.setPrinterName(printer_name)
    printer.setPageOrientation(QPageLayout.Orientation.Portrait)
    printer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))

    dialog = QPrintDialog(printer, parent)
    dialog.setWindowTitle("Print — Detailed Print Report")
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return False

    painter = QPainter(printer)
    try:
        page_rect = painter.viewport()
        scaled = pixmap.scaled(
            page_rect.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = page_rect.x() + (page_rect.width() - scaled.width()) // 2
        y = page_rect.y() + (page_rect.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
    finally:
        painter.end()

    return True


class PrinterSelectDialog(QDialog):
    """Pick a printer — 3D printers (Bambu Lab, etc.) listed first when detected."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Print to 3D / System Printer")
        self.resize(520, 420)
        self._printers: list[DetectedPrinter] = []
        self.selected_printer: DetectedPrinter | None = None

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Printers installed on this PC are listed below. "
            "3D printers (Bambu Lab, Prusa, Creality, …) are detected by name and shown at the top."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._summary = QLabel()
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        row = QHBoxLayout()
        row.addWidget(QLabel("Select printer:"))
        row.addStretch()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_printers)
        row.addWidget(refresh_btn)
        layout.addLayout(row)

        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(self._accept_selection)
        layout.addWidget(self._list, stretch=1)

        self._detail = QLabel("")
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet("color: #555;")
        layout.addWidget(self._detail)

        hint = QLabel(
            "Tip: Add your Bambu Lab or other 3D printer in Windows Settings → "
            "Bluetooth & devices → Printers & scanners, or via the manufacturer app "
            "(e.g. Bambu Studio)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(hint)

        buttons = QDialogButtonBox()
        self._print_btn = buttons.addButton("Print DPR", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._list.currentItemChanged.connect(self._on_selection_changed)
        self._refresh_printers()

    def _refresh_printers(self) -> None:
        self._printers = detect_printers()
        self._list.clear()

        if not self._printers:
            self._summary.setText("No printers found on this computer.")
            self._print_btn.setEnabled(False)
            self._detail.setText("")
            return

        count_3d = sum(1 for p in self._printers if p.is_3d_printer)
        if count_3d:
            self._summary.setText(
                f"Found {len(self._printers)} printer(s) — "
                f"{count_3d} likely 3D printer(s) detected."
            )
        else:
            self._summary.setText(
                f"Found {len(self._printers)} printer(s). "
                "No 3D printer name matched; all system printers are still available."
            )

        for printer in self._printers:
            item = QListWidgetItem(printer.display_label)
            item.setData(Qt.ItemDataRole.UserRole, printer.name)
            item.setToolTip(printer.detail_text)
            if printer.is_3d_printer:
                item.setForeground(Qt.GlobalColor.darkCyan)
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            if printer.is_default:
                item.setText(f"{item.text()}  ★ default")
            self._list.addItem(item)

        self._list.setCurrentRow(0)
        self._print_btn.setEnabled(True)

    def _on_selection_changed(self) -> None:
        printer = self._current_printer()
        if printer is None:
            self._detail.setText("")
            return
        kind = f"3D printer ({printer.brand})" if printer.is_3d_printer else "System printer"
        self._detail.setText(f"{kind}  —  {printer.detail_text}")

    def _current_printer(self) -> DetectedPrinter | None:
        item = self._list.currentItem()
        if item is None:
            return None
        name = item.data(Qt.ItemDataRole.UserRole)
        for printer in self._printers:
            if printer.name == name:
                return printer
        return None

    def _accept_selection(self) -> None:
        self.selected_printer = self._current_printer()
        if self.selected_printer is None:
            QMessageBox.information(self, "Print", "Select a printer from the list.")
            return
        self.accept()
