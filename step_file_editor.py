#!/usr/bin/env python3
"""NagaWorks STEP File Editor — text view and GPU-accelerated 3D solid preview."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path

# PyVista Qt backend (GPU viewport).
os.environ.setdefault("QT_API", "pyside6")

from app_info import (
    APP_DISPLAY_NAME,
    APP_LOGO_FILE,
    APP_TITLE_BRAND,
    APP_VENDOR,
    APP_VERSION,
    about_text,
    app_logo_path,
    window_title,
)
from app_logging import get_log_path, get_logger, log_exception, setup_logging

setup_logging()
log = get_logger()

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QFont,
    QIcon,
    QKeySequence,
    QShortcut,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSplitter,
    QStackedWidget,
    QSizePolicy,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

DEFAULT_DIR = Path(__file__).resolve().parent / "SarrusLiftingTableSTEP"

_VIEWER_AVAILABLE = True
_VIEWER_IMPORT_ERROR = ""
try:
    from step_mesh_viewer import SolidModel, load_solid_model
    from step_vtk_viewer import VIEWER_BACKGROUND, Vtk3DWidget
except ImportError as exc:
    _VIEWER_AVAILABLE = False
    _VIEWER_IMPORT_ERROR = str(exc)
    SolidModel = None  # type: ignore
    Vtk3DWidget = None  # type: ignore
    VIEWER_BACKGROUND = "#ffffff"

_BLUEPRINT_AVAILABLE = True
_BLUEPRINT_IMPORT_ERROR = ""
try:
    from step_blueprint import build_blueprint_snapshot, export_blueprint_snapshot
except ImportError as exc:
    _BLUEPRINT_AVAILABLE = False
    _BLUEPRINT_IMPORT_ERROR = str(exc)
    build_blueprint_snapshot = None  # type: ignore
    export_blueprint_snapshot = None  # type: ignore

_PENCIL_SKETCH_AVAILABLE = True
_PENCIL_SKETCH_IMPORT_ERROR = ""
try:
    from step_pencil_sketch import build_sketch_snapshot, export_pencil_sketch_snapshot
except ImportError as exc:
    _PENCIL_SKETCH_AVAILABLE = False
    _PENCIL_SKETCH_IMPORT_ERROR = str(exc)
    build_sketch_snapshot = None  # type: ignore
    export_pencil_sketch_snapshot = None  # type: ignore

_DPR_AVAILABLE = True
_DPR_IMPORT_ERROR = ""
try:
    from step_dpr_report import build_dpr_snapshot, export_dpr_snapshot
except ImportError as exc:
    _DPR_AVAILABLE = False
    _DPR_IMPORT_ERROR = str(exc)
    build_dpr_snapshot = None  # type: ignore
    export_dpr_snapshot = None  # type: ignore

_PRINT_AVAILABLE = True
_PRINT_IMPORT_ERROR = ""
try:
    from step_printer_dialog import PrinterSelectDialog, print_image
except ImportError as exc:
    _PRINT_AVAILABLE = False
    _PRINT_IMPORT_ERROR = str(exc)
    PrinterSelectDialog = None  # type: ignore
    print_image = None  # type: ignore


class ModelLoadBridge(QObject):
    """Thread-safe bridge: background thread emits, main thread receives."""

    finished = Signal(int, object, str)  # token, SolidModel | None, error message


class BlueprintBridge(QObject):
    finished = Signal(str, str)  # saved path, error message (empty if ok)


def _load_app_icon() -> QIcon | None:
    logo = app_logo_path()
    if logo is None:
        return None
    icon = QIcon(str(logo))
    return icon if not icon.isNull() else None


def _apply_app_icon(*, app: QApplication | None = None, window: QMainWindow | None = None) -> None:
    icon = _load_app_icon()
    if icon is None:
        log.warning("Application logo not found: %s", APP_LOGO_FILE)
        return
    if app is not None:
        app.setWindowIcon(icon)
    if window is not None:
        window.setWindowIcon(icon)


class StepFileEditor(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(window_title())
        _apply_app_icon(window=self)
        self.resize(1280, 800)

        self.current_dir = DEFAULT_DIR if DEFAULT_DIR.is_dir() else Path(__file__).resolve().parent
        self.current_file: Path | None = None
        self._dirty = False
        self._loading_text = False
        self._solid_model: SolidModel | None = None
        self._load_token = 0
        self._load_thread: threading.Thread | None = None
        self._load_bridge = ModelLoadBridge(self)
        self._load_bridge.finished.connect(self._apply_loaded_model)
        self._blueprint_bridge = BlueprintBridge(self)
        self._blueprint_bridge.finished.connect(self._on_blueprint_exported)
        self._sketch_bridge = BlueprintBridge(self)
        self._sketch_bridge.finished.connect(self._on_pencil_sketch_exported)
        self.vtk_view: Vtk3DWidget | None = None
        self._view3d_panel: QWidget | None = None
        self._view3d_tab_index = 1
        self._in_3d_fullscreen = False
        self._splitter: QSplitter | None = None
        self._left_panel: QWidget | None = None
        self._right_header: QWidget | None = None
        self._main_toolbar: QToolBar | None = None
        self._pending_opacity = 1.0
        self._opacity_timer = QTimer(self)
        self._opacity_timer.setSingleShot(True)
        self._opacity_timer.setInterval(250)
        self._opacity_timer.timeout.connect(self._apply_3d_opacity_debounced)
        self._viewport_fit_timer = QTimer(self)
        self._viewport_fit_timer.setSingleShot(True)
        self._viewport_fit_timer.setInterval(120)
        self._viewport_fit_timer.timeout.connect(self._fit_3d_viewport_safe)
        self._fullscreen_transition = False
        self._suppress_tab_3d_refresh = False
        self._3d_load_in_progress = False
        self._3d_load_path: Path | None = None
        self._splitter_sizes_before_fs: list[int] | None = None
        self._suppress_file_select = False
        self._viewport_host: QWidget | None = None
        self._viewport_layout: QVBoxLayout | None = None
        self._last_reflow_size: tuple[int, int] | None = None
        self._export_in_progress = False
        self._ruler_was_enabled_before_export = False

        self._build_ui()
        self._build_actions()
        self._refresh_file_list()

    def _build_ui(self) -> None:
        central = QWidget()
        central.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.setChildrenCollapsible(False)
        root.addWidget(self._splitter, stretch=1)

        # --- File list ---
        left = QWidget()
        left.setMinimumWidth(180)
        left.setMaximumWidth(420)
        self._left_panel = left
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("<b>Files</b>"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter…")
        self.filter_edit.textChanged.connect(self._refresh_file_list)
        left_layout.addWidget(self.filter_edit)
        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self._on_file_selected)
        left_layout.addWidget(self.file_list)
        self._splitter.addWidget(left)

        # --- Editor + 3D ---
        right = QWidget()
        right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        header_host = QWidget()
        self._right_header = header_host
        header = QHBoxLayout(header_host)
        header.setContentsMargins(0, 0, 0, 0)
        self.file_label = QLabel("No file open")
        self.file_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        header.addWidget(self.file_label)
        header.addStretch()
        self.model_info_label = QLabel("")
        self.model_info_label.setStyleSheet("color: #666;")
        header.addWidget(self.model_info_label)
        right_layout.addWidget(header_host)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.editor = QPlainTextEdit()
        self.editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.editor.setFont(QFont("Consolas", 10))
        self.editor.textChanged.connect(self._on_text_changed)
        self.tabs.addTab(self.editor, "Source Text")

        if _VIEWER_AVAILABLE:
            view3d = QWidget()
            view3d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            view3d_layout = QVBoxLayout(view3d)
            view3d_layout.setContentsMargins(2, 2, 2, 2)
            view3d_layout.setSpacing(2)
            controls = QHBoxLayout()
            refresh_btn = QPushButton("Refresh 3D")
            refresh_btn.clicked.connect(self._refresh_3d_view)
            controls.addWidget(refresh_btn)
            reset_btn = QPushButton("Iso")
            reset_btn.setToolTip("Isometric view")
            reset_btn.clicked.connect(lambda: self._set_3d_view("iso"))
            controls.addWidget(reset_btn)
            self.wireframe_check = QCheckBox("Wireframe")
            self.wireframe_check.toggled.connect(self._on_wireframe_toggled)
            controls.addWidget(self.wireframe_check)
            fullscreen_btn = QPushButton("Fullscreen (F12)")
            fullscreen_btn.clicked.connect(self._toggle_3d_fullscreen)
            controls.addWidget(fullscreen_btn)
            blueprint_btn = QPushButton("Blueprint…")
            blueprint_btn.setToolTip("Export orthographic blueprint sheet (PNG/PDF)")
            blueprint_btn.clicked.connect(self._export_blueprint)
            controls.addWidget(blueprint_btn)
            sketch_btn = QPushButton("Pencil Sketch…")
            sketch_btn.setToolTip("Save a hand-drawn pencil sketch image (PNG/JPEG)")
            sketch_btn.clicked.connect(self._export_pencil_sketch)
            controls.addWidget(sketch_btn)
            dpr_btn = QPushButton("DPR…")
            dpr_btn.setToolTip(
                "Detailed Print Report — save an image with dimensions (mm/cm/in/nm), "
                "volume, and 3D print checklist"
            )
            dpr_btn.clicked.connect(self._export_dpr)
            controls.addWidget(dpr_btn)
            if _PRINT_AVAILABLE and _DPR_AVAILABLE:
                print_btn = QPushButton("Print…")
                print_btn.setToolTip(
                    "Print Detailed Print Report — detects 3D printers "
                    "(Bambu Lab, Prusa, Creality, …) and other system printers"
                )
                print_btn.clicked.connect(self._print_dpr)
                controls.addWidget(print_btn)
            controls.addStretch()
            hint = QLabel("F12 = expand 3D • Esc = exit • Maximize window for more space")
            hint.setStyleSheet("color: #666;")
            controls.addWidget(hint)
            view3d_layout.addLayout(controls)

            view_row = QHBoxLayout()
            view_row.addWidget(QLabel("Views:"))
            for label, key in (
                ("Top", "top"),
                ("Bottom", "bottom"),
                ("Front", "front"),
                ("Back", "back"),
                ("Left", "left"),
                ("Right", "right"),
                ("Iso", "iso"),
            ):
                btn = QPushButton(label)
                btn.setFixedWidth(56)
                btn.clicked.connect(lambda _checked=False, v=key: self._set_3d_view(v))
                view_row.addWidget(btn)
            view_row.addStretch()
            view3d_layout.addLayout(view_row)

            appearance_row = QHBoxLayout()
            appearance_row.addWidget(QLabel("Model:"))
            self.color_swatch = QLabel()
            self.color_swatch.setFixedSize(28, 20)
            self.color_swatch.setStyleSheet("background: #8cb3e6; border: 1px solid #666;")
            appearance_row.addWidget(self.color_swatch)
            color_btn = QPushButton("Color…")
            color_btn.clicked.connect(self._pick_3d_color)
            appearance_row.addWidget(color_btn)
            appearance_row.addSpacing(12)
            appearance_row.addWidget(QLabel("Opacity:"))
            self.opacity_slider = QSlider(Qt.Horizontal)
            self.opacity_slider.setRange(5, 100)
            self.opacity_slider.setValue(100)
            self.opacity_slider.setFixedWidth(140)
            self.opacity_slider.setToolTip("Model transparency (5% = very transparent)")
            self.opacity_slider.valueChanged.connect(self._on_3d_opacity_changed)
            appearance_row.addWidget(self.opacity_slider)
            self.opacity_label = QLabel("100%")
            self.opacity_label.setFixedWidth(40)
            appearance_row.addWidget(self.opacity_label)
            appearance_row.addStretch()
            view3d_layout.addLayout(appearance_row)

            ruler_row = QHBoxLayout()
            self.ruler_check = QCheckBox("Ruler")
            self.ruler_check.setToolTip(
                "Show a futuristic scale bar in the corner (zoom-aware). "
                "STEP coordinates are interpreted as millimeters."
            )
            self.ruler_check.toggled.connect(self._on_ruler_toggled)
            ruler_row.addWidget(self.ruler_check)
            ruler_row.addSpacing(8)
            ruler_row.addWidget(QLabel("Units:"))
            self._ruler_unit_group = QButtonGroup(self)
            self._unit_radios: dict[str, QRadioButton] = {}
            for unit_key, unit_label in (
                ("nm", "nm"),
                ("mm", "mm"),
                ("cm", "cm"),
                ("m", "m"),
                ("in", "in"),
            ):
                radio = QRadioButton(unit_label)
                radio.setToolTip(f"Display scale in {unit_label}")
                self._ruler_unit_group.addButton(radio)
                self._unit_radios[unit_key] = radio
                ruler_row.addWidget(radio)
            self._unit_radios["mm"].setChecked(True)
            self._ruler_unit_group.buttonClicked.connect(self._on_ruler_unit_changed)
            ruler_row.addStretch()
            view3d_layout.addLayout(ruler_row)

            self.view3d_stack = QStackedWidget()
            self.view3d_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.view3d_stack.setStyleSheet(f"background-color: {VIEWER_BACKGROUND};")
            self.view3d_status = QLabel("Select a .stp file for 3D preview.")
            self.view3d_status.setAlignment(Qt.AlignCenter)
            self.view3d_status.setStyleSheet(
                f"color: #666; padding: 24px; background-color: {VIEWER_BACKGROUND};"
            )
            self._viewport_host = QWidget()
            self._viewport_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._viewport_host.setStyleSheet(f"background-color: {VIEWER_BACKGROUND};")
            self._viewport_layout = QVBoxLayout(self._viewport_host)
            self._viewport_layout.setContentsMargins(0, 0, 0, 0)
            self._viewport_layout.setSpacing(0)
            self.view3d_stack.addWidget(self.view3d_status)
            self.view3d_stack.addWidget(self._viewport_host)
            view3d_layout.addWidget(self.view3d_stack, stretch=1)
            view3d_layout.setStretch(view3d_layout.count() - 1, 1)
            self._view3d_panel = view3d
            self.tabs.addTab(view3d, "3D Solid View (GPU)")
        else:
            unavailable = QLabel(
                "3D preview requires GPU libraries.\n\n"
                "Run: pip install -r requirements.txt\n\n"
                f"{_VIEWER_IMPORT_ERROR}"
            )
            unavailable.setAlignment(Qt.AlignCenter)
            self.tabs.addTab(unavailable, "3D Solid View")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        right_layout.addWidget(self.tabs, stretch=1)
        self._splitter.addWidget(right)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([260, 940])
        self._splitter.splitterMoved.connect(lambda _pos, _index: self._schedule_viewport_fit())

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        log_path = get_log_path()
        log_link = QLabel(f'Log: <a href="file:///{log_path.as_posix()}">{log_path.name}</a>')
        log_link.setOpenExternalLinks(True)
        log_link.setToolTip(str(log_path))
        self.status_bar.addPermanentWidget(log_link)
        version_label = QLabel(f"{APP_TITLE_BRAND} V{APP_VERSION}")
        version_label.setStyleSheet("color: #888; padding-left: 12px;")
        version_label.setToolTip(APP_DISPLAY_NAME)
        self.status_bar.addPermanentWidget(version_label)

    def _build_actions(self) -> None:
        toolbar = QToolBar("Main")
        self._main_toolbar = toolbar
        self.addToolBar(toolbar)

        open_folder = QAction("Open Folder", self)
        open_folder.setShortcut(QKeySequence("Ctrl+Shift+O"))
        open_folder.triggered.connect(self._choose_folder)
        toolbar.addAction(open_folder)

        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._save_file)
        toolbar.addAction(save_action)
        toolbar.addAction("Reload", self._reload_file)

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        toolbar.addAction(about_action)

        if _VIEWER_AVAILABLE:
            preview = QAction("3D Preview", self)
            preview.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
            toolbar.addAction(preview)

        if _BLUEPRINT_AVAILABLE:
            blueprint_action = QAction("Blueprint", self)
            blueprint_action.setShortcut(QKeySequence("Ctrl+Shift+B"))
            blueprint_action.setToolTip("Export 2D blueprint sheet from the loaded 3D model")
            blueprint_action.triggered.connect(self._export_blueprint)
            toolbar.addAction(blueprint_action)
            self.addAction(blueprint_action)

        if _PENCIL_SKETCH_AVAILABLE:
            sketch_action = QAction("Pencil Sketch", self)
            sketch_action.setShortcut(QKeySequence("Ctrl+Shift+K"))
            sketch_action.setToolTip("Save a pencil sketch image of the loaded 3D model")
            sketch_action.triggered.connect(self._export_pencil_sketch)
            toolbar.addAction(sketch_action)
            self.addAction(sketch_action)

        if _DPR_AVAILABLE:
            dpr_action = QAction("DPR", self)
            dpr_action.setShortcut(QKeySequence("Ctrl+Shift+D"))
            dpr_action.setToolTip("Save Detailed Print Report image for 3D printing")
            dpr_action.triggered.connect(self._export_dpr)
            toolbar.addAction(dpr_action)
            self.addAction(dpr_action)

        if _PRINT_AVAILABLE and _DPR_AVAILABLE:
            print_action = QAction("Print", self)
            print_action.setShortcut(QKeySequence.Print)
            print_action.setToolTip(
                "Print Detailed Print Report to a 3D or system printer"
            )
            print_action.triggered.connect(self._print_dpr)
            toolbar.addAction(print_action)
            self.addAction(print_action)

        find_action = QAction("Find", self)
        find_action.setShortcut(QKeySequence.Find)
        find_action.triggered.connect(self._show_find_dialog)
        self.addAction(find_action)

        refresh_3d = QAction("Refresh 3D", self)
        refresh_3d.setShortcut(QKeySequence("F5"))
        refresh_3d.triggered.connect(self._refresh_3d_view)
        self.addAction(refresh_3d)

        if _VIEWER_AVAILABLE:
            fullscreen_3d = QAction("3D Fullscreen", self)
            fullscreen_3d.setShortcut(QKeySequence(Qt.Key_F12))
            fullscreen_3d.triggered.connect(self._toggle_3d_fullscreen)
            self.addAction(fullscreen_3d)
            fs_esc = QShortcut(QKeySequence(Qt.Key_Escape), self)
            fs_esc.setContext(Qt.ApplicationShortcut)
            fs_esc.activated.connect(self._exit_3d_fullscreen_if_active)

        folder_label = QLabel(f"  Folder: {self.current_dir}")
        folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        toolbar.addWidget(folder_label)
        self.folder_label = folder_label

    def _refresh_file_list(self) -> None:
        self.file_list.blockSignals(True)
        self.file_list.clear()
        if not self.current_dir.is_dir():
            self.file_list.blockSignals(False)
            return
        needle = self.filter_edit.text().strip().lower()
        for path in sorted(self.current_dir.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file():
                continue
            if needle and needle not in path.name.lower():
                continue
            self.file_list.addItem(path.name)
        self.file_list.blockSignals(False)

    def _ensure_vtk_view(self) -> Vtk3DWidget:
        if self.vtk_view is None:
            self.vtk_view = Vtk3DWidget()
            self.vtk_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            if self._viewport_layout is not None:
                self._viewport_layout.addWidget(self.vtk_view, stretch=1)
            self.vtk_view.set_fullscreen_toggle(self._toggle_3d_fullscreen)
            self._sync_ruler_controls_to_vtk()
        return self.vtk_view

    def _selected_ruler_unit(self) -> str:
        for unit_key, radio in self._unit_radios.items():
            if radio.isChecked():
                return unit_key
        return "mm"

    def _sync_ruler_controls_to_vtk(self) -> None:
        if self.vtk_view is None or not hasattr(self, "ruler_check"):
            return
        self.vtk_view.set_ruler_unit(self._selected_ruler_unit())
        self.vtk_view.set_ruler_visible(self.ruler_check.isChecked())

    def _on_ruler_toggled(self, checked: bool) -> None:
        if _VIEWER_AVAILABLE and self.vtk_view is not None:
            self.vtk_view.set_ruler_visible(checked)

    def _on_ruler_unit_changed(self, _button: QRadioButton) -> None:
        if _VIEWER_AVAILABLE and self.vtk_view is not None:
            self.vtk_view.set_ruler_unit(self._selected_ruler_unit())

    def _show_3d_viewport(self) -> None:
        if self._viewport_host is not None:
            self.view3d_stack.setCurrentWidget(self._viewport_host)
        elif self.vtk_view is not None:
            self.view3d_stack.setCurrentWidget(self.vtk_view)

    def _set_3d_view(self, view_name: str) -> None:
        if not _VIEWER_AVAILABLE:
            return
        if self.vtk_view is None and self._solid_model is None:
            return
        if self.vtk_view is None:
            self._ensure_vtk_view()
        if self._solid_model is not None:
            if not self._is_3d_fullscreen():
                idx = self.tabs.indexOf(self._view3d_panel) if self._view3d_panel else 1
                if idx >= 0:
                    self.tabs.setCurrentIndex(idx)
            self._show_3d_viewport()
            self.vtk_view.set_standard_view(view_name)

    def _is_3d_fullscreen(self) -> bool:
        return self._in_3d_fullscreen

    def _toggle_3d_fullscreen(self) -> None:
        if not _VIEWER_AVAILABLE:
            return
        if self.vtk_view is None or self._solid_model is None or self._view3d_panel is None:
            if not self._is_3d_fullscreen():
                QMessageBox.information(
                    self,
                    "3D Fullscreen",
                    "Open a STEP file and wait for the 3D model to load first.",
                )
            return
        # Defer so the F12 shortcut returns before any VTK/window work (avoids crashes).
        if self._is_3d_fullscreen():
            QTimer.singleShot(0, self._exit_3d_fullscreen)
        else:
            QTimer.singleShot(0, self._enter_3d_fullscreen)

    def _enter_3d_fullscreen(self) -> None:
        """Expand 3D to fill the window without changing Qt fullscreen (avoids VTK crashes)."""
        if self._view3d_panel is None or self._in_3d_fullscreen or self._fullscreen_transition:
            return

        log.info("Entering 3D expanded view")
        self._fullscreen_transition = True
        self._last_reflow_size = None
        self._viewport_fit_timer.stop()
        self._view3d_tab_index = self.tabs.indexOf(self._view3d_panel)

        if self._splitter is not None:
            self._splitter_sizes_before_fs = self._splitter.sizes()

        if self.vtk_view:
            self.vtk_view.set_render_paused(True)
            self._show_3d_viewport()

        self._suppress_tab_3d_refresh = True
        if self._view3d_tab_index >= 0:
            self.tabs.setCurrentIndex(self._view3d_tab_index)
        self._suppress_tab_3d_refresh = False

        self.tabs.tabBar().hide()

        if self._left_panel is not None:
            self._left_panel.hide()
        if self._right_header is not None:
            self._right_header.hide()
        if self._main_toolbar is not None:
            self._main_toolbar.hide()
        self.statusBar().hide()

        if self._splitter is not None:
            self._splitter.setSizes([0, max(self.width(), 1)])

        self._in_3d_fullscreen = True
        QTimer.singleShot(100, self._finish_enter_fullscreen)

    def _finish_enter_fullscreen(self) -> None:
        try:
            if self.vtk_view:
                self.vtk_view.set_render_paused(False)
                self.vtk_view.setFocus()
            if self._view3d_panel:
                self._view3d_panel.setFocus()
            QTimer.singleShot(200, self._reflow_3d_after_resize)
            log.info("3D expanded view ready")
        except Exception:
            log_exception("_finish_enter_fullscreen")
        finally:
            self._fullscreen_transition = False

    def _exit_3d_fullscreen_if_active(self) -> None:
        if self._in_3d_fullscreen:
            QTimer.singleShot(0, self._exit_3d_fullscreen)

    def _exit_3d_fullscreen(self) -> None:
        if not self._in_3d_fullscreen or self._fullscreen_transition:
            return

        log.info("Exiting 3D expanded view")
        self._fullscreen_transition = True
        self._in_3d_fullscreen = False
        self._last_reflow_size = None
        self._viewport_fit_timer.stop()

        if self.vtk_view:
            self.vtk_view.set_render_paused(True)
            self.vtk_view.discard_pending_fit()

        QTimer.singleShot(100, self._restore_chrome_after_fullscreen)

    def _restore_chrome_after_fullscreen(self) -> None:
        """Restore side panels and tab bar without touching VTK (layout-only)."""
        central = self.centralWidget()
        try:
            if central is not None:
                central.setUpdatesEnabled(False)

            self.tabs.tabBar().show()

            if self._left_panel is not None:
                self._left_panel.show()
            if self._right_header is not None:
                self._right_header.show()
            if self._main_toolbar is not None:
                self._main_toolbar.show()
            self.statusBar().show()

            if self._splitter is not None and self._splitter_sizes_before_fs is not None:
                self._splitter.setSizes(self._splitter_sizes_before_fs)
                self._splitter_sizes_before_fs = None
        except Exception:
            log_exception("_restore_chrome_after_fullscreen")
        finally:
            if central is not None:
                central.setUpdatesEnabled(True)

        QTimer.singleShot(350, self._resume_vtk_after_fullscreen)

    def _resume_vtk_after_fullscreen(self) -> None:
        try:
            if self.vtk_view:
                self.vtk_view.set_render_paused(False)
            QTimer.singleShot(150, self._reflow_3d_after_resize)
            log.info("Exited 3D expanded view")
        except Exception:
            log_exception("_resume_vtk_after_fullscreen")
        finally:
            self._fullscreen_transition = False

    def _is_3d_tab_active(self) -> bool:
        if not _VIEWER_AVAILABLE or self._view3d_panel is None:
            return False
        if self._is_3d_fullscreen():
            return True
        return self.tabs.currentWidget() is self._view3d_panel

    def _schedule_viewport_fit(self) -> None:
        if (
            not _VIEWER_AVAILABLE
            or not self.vtk_view
            or self._fullscreen_transition
            or self._export_in_progress
        ):
            return
        self._viewport_fit_timer.start()

    def _fit_3d_viewport_safe(self) -> None:
        if not self.vtk_view or not self._is_3d_tab_active():
            return
        if not self.vtk_view.isVisible():
            return
        try:
            self._reflow_3d_after_resize()
        except Exception:
            log_exception("fit_3d_viewport_safe")

    def _reflow_3d_after_resize(self) -> None:
        """Resize VTK to fill the stacked viewport area after layout changes."""
        if not self.vtk_view or not self._is_3d_tab_active() or self._fullscreen_transition:
            return
        try:
            self._show_3d_viewport()
            QApplication.processEvents()

            host = self._viewport_host or self.view3d_stack
            w, h = host.width(), host.height()
            if w < 32 or h < 32:
                w, h = self.view3d_stack.width(), self.view3d_stack.height()
            if w < 32 or h < 32:
                return

            if self._last_reflow_size == (w, h):
                return
            self._last_reflow_size = (w, h)

            log.debug("3D reflow %dx%d fullscreen=%s", w, h, self._in_3d_fullscreen)
            self.vtk_view.fit_viewport(w, h)
        except Exception:
            log_exception("_reflow_3d_after_resize")

    def _pick_3d_color(self) -> None:
        if not _VIEWER_AVAILABLE:
            return
        vtk = self.vtk_view
        if vtk is None and self._solid_model is None:
            return
        if vtk is None:
            vtk = self._ensure_vtk_view()
        current = QColor(vtk.get_display_color())
        chosen = QColorDialog.getColor(current, self, "Choose model color")
        if not chosen.isValid():
            return
        vtk.set_display_color(chosen.name())
        self.color_swatch.setStyleSheet(
            f"background: {chosen.name()}; border: 1px solid #666;"
        )

    def _on_3d_opacity_changed(self, value: int) -> None:
        self.opacity_label.setText(f"{value}%")
        self._pending_opacity = value / 100.0
        self._opacity_timer.start()

    def _apply_3d_opacity_debounced(self) -> None:
        if self.vtk_view is not None:
            self.vtk_view.set_opacity(self._pending_opacity)

    def _sync_3d_appearance_controls(self) -> None:
        if not _VIEWER_AVAILABLE or self.vtk_view is None:
            return
        color = self.vtk_view.get_display_color()
        self.color_swatch.setStyleSheet(f"background: {color}; border: 1px solid #666;")
        opacity_pct = int(round(self.vtk_view.get_opacity() * 100))
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(opacity_pct)
        self.opacity_slider.blockSignals(False)
        self.opacity_label.setText(f"{opacity_pct}%")

    def _choose_folder(self) -> None:
        if not self._confirm_discard():
            return
        chosen = QFileDialog.getExistingDirectory(self, "Select folder", str(self.current_dir))
        if not chosen:
            return
        self.current_dir = Path(chosen)
        self.folder_label.setText(f"  Folder: {self.current_dir}")
        self._clear_editor()
        self._refresh_file_list()

    def _on_file_selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None or self._suppress_file_select:
            return
        path = self.current_dir / current.text()
        if path == self.current_file:
            return
        if not self._confirm_discard():
            self._reselect_current_file()
            return
        self._load_file(path)

    def _reselect_current_file(self) -> None:
        if not self.current_file:
            return
        for row in range(self.file_list.count()):
            if self.file_list.item(row).text() == self.current_file.name:
                self.file_list.blockSignals(True)
                self.file_list.setCurrentRow(row)
                self.file_list.blockSignals(False)
                break

    def _load_file(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            QMessageBox.critical(self, "Open failed", f"Could not read file:\n{exc}")
            return

        self._loading_text = True
        self.editor.setPlainText(text)
        self.editor.document().setModified(False)
        self._loading_text = False

        self.current_file = path
        self._dirty = False
        size_kb = path.stat().st_size / 1024
        lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        self.file_label.setText(f"{path.name}  ({size_kb:.1f} KB, {lines:,} lines)")
        self.setWindowTitle(window_title(path.name))
        self._update_status()

        if _VIEWER_AVAILABLE and path.suffix.lower() in {".stp", ".step"}:
            self._solid_model = None
            self.model_info_label.setText("")
            self._refresh_3d_view()
        else:
            self._solid_model = None
            self.model_info_label.setText("")
            self._clear_3d_view("3D preview works with .stp / .step files only.")

    def _reload_file(self) -> None:
        if not self.current_file:
            return
        if self._dirty:
            answer = QMessageBox.question(
                self,
                "Reload",
                "Discard unsaved changes and reload?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self._load_file(self.current_file)

    def _clear_editor(self) -> None:
        self._loading_text = True
        self.editor.clear()
        self._loading_text = False
        self.current_file = None
        self._dirty = False
        self._solid_model = None
        self.model_info_label.setText("")
        self.file_label.setText("No file open")
        self.setWindowTitle(window_title())
        self._clear_3d_view("Select a .stp file for 3D preview.")
        self.status_bar.showMessage("Ready")

    def _on_text_changed(self) -> None:
        if self._loading_text:
            return
        self._dirty = True
        self._update_status()

    def _update_status(self) -> None:
        if not self.current_file:
            self.status_bar.showMessage("Ready")
            return
        cursor = self.editor.textCursor()
        line = cursor.blockNumber() + 1
        col = cursor.columnNumber() + 1
        dirty = " • modified" if self._dirty else ""
        self.status_bar.showMessage(f"{self.current_file.name}{dirty}  |  Line {line}, Col {col}")

    def _save_file(self) -> bool:
        if not self.current_file:
            return self._save_file_as()
        try:
            self.current_file.write_text(self.editor.toPlainText(), encoding="utf-8", newline="\n")
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save file:\n{exc}")
            return False
        self._dirty = False
        self._update_status()
        self.status_bar.showMessage(f"Saved {self.current_file.name}")
        return True

    def _save_file_as(self) -> bool:
        initial = str(self.current_dir / "untitled.stp")
        if self.current_file:
            initial = str(self.current_file)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save As",
            initial,
            "STEP files (*.stp *.step);;Text files (*.txt);;All files (*.*)",
        )
        if not path:
            return False
        self.current_file = Path(path)
        return self._save_file()

    def _confirm_discard(self) -> bool:
        if not self._dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "Discard unsaved changes?",
            QMessageBox.Yes | QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _show_find_dialog(self) -> None:
        query, ok = QInputDialog.getText(self, "Find", "Search for:")
        if not ok or not query:
            return
        found = self.editor.find(query)
        if not found:
            cursor = self.editor.textCursor()
            cursor.movePosition(QTextCursor.Start)
            self.editor.setTextCursor(cursor)
            found = self.editor.find(query)
        if not found:
            QMessageBox.information(self, "Find", "No matches found.")
            return
        fmt = QTextCharFormat()
        fmt.setBackground(Qt.yellow)
        cursor = self.editor.textCursor()
        cursor.mergeCharFormat(fmt)

    def _on_tab_changed(self, index: int) -> None:
        if not _VIEWER_AVAILABLE or self._suppress_tab_3d_refresh:
            return
        if self._view3d_panel is not None and self.tabs.widget(index) is self._view3d_panel:
            if self._solid_model is not None and self.vtk_view is not None:
                self._show_3d_viewport()
                self.vtk_view.setFocus()
                self._schedule_viewport_fit()
            elif (
                self.current_file
                and self.current_file.suffix.lower() in {".stp", ".step"}
                and self._solid_model is None
                and not self._3d_load_in_progress
            ):
                self._refresh_3d_view()

    def _on_wireframe_toggled(self, checked: bool) -> None:
        if _VIEWER_AVAILABLE and self.vtk_view is not None:
            self.vtk_view.set_wireframe(checked)

    def _clear_3d_view(self, message: str) -> None:
        if not _VIEWER_AVAILABLE:
            return
        if self.vtk_view is not None:
            self.vtk_view.set_ruler_visible(False)
            self.vtk_view.clear()
        self.view3d_status.setText(message)
        self.view3d_stack.setCurrentWidget(self.view3d_status)

    def _refresh_3d_view(self) -> None:
        if not _VIEWER_AVAILABLE:
            return
        if not self.current_file:
            self._clear_3d_view("Open a STEP file first.")
            return
        if self.current_file.suffix.lower() not in {".stp", ".step"}:
            self._clear_3d_view("3D preview works with .stp / .step files only.")
            return
        if self._3d_load_in_progress and self._3d_load_path == self.current_file.resolve():
            return
        if self._dirty:
            answer = QMessageBox.question(
                self,
                "Unsaved changes",
                "3D preview uses the saved file on disk.\n\nSave changes now?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if answer == QMessageBox.Cancel:
                return
            if answer == QMessageBox.Yes and not self._save_file():
                return

        path = self.current_file.resolve()
        self._3d_load_in_progress = True
        self._3d_load_path = path
        self._load_token += 1
        token = self._load_token

        tab_idx = self.tabs.indexOf(self._view3d_panel) if self._view3d_panel else 1
        self._suppress_tab_3d_refresh = True
        if tab_idx >= 0:
            self.tabs.setCurrentIndex(tab_idx)
        self._suppress_tab_3d_refresh = False

        self.view3d_status.setText(f"Loading 3D model from {path.name}…")
        self.view3d_stack.setCurrentWidget(self.view3d_status)
        self.status_bar.showMessage(f"Building GPU preview for {path.name}…")
        QApplication.processEvents()

        def worker() -> None:
            model = None
            error = ""
            try:
                log.info("Loading STEP mesh: %s", path.name)
                model = load_solid_model(path)
                log.info("Loaded STEP mesh: %s", path.name)
            except Exception as exc:
                error = str(exc)
                log.exception("STEP load failed: %s", path.name)
            self._load_bridge.finished.emit(token, model, error)

        load_thread = threading.Thread(target=worker, daemon=True, name="step-model-loader")
        load_thread.start()
        self._load_thread = load_thread

    def _apply_loaded_model(
        self,
        token: int,
        model: SolidModel | None,
        error: str,
    ) -> None:
        if token != self._load_token:
            return
        self._3d_load_in_progress = False
        self._3d_load_path = None
        if error:
            self._solid_model = None
            self.model_info_label.setText("")
            self._clear_3d_view(f"3D load failed:\n{error}")
            self.status_bar.showMessage("3D preview failed")
            return
        if model is None:
            self._clear_3d_view("No geometry found in this file.")
            return

        self._solid_model = model
        simplified = ""
        if model.original_faces > model.display_faces:
            simplified = (
                f" • simplified {model.display_faces:,}"
                f"/{model.original_faces:,} triangles"
            )
        self.model_info_label.setText(
            f"{model.part_count} part(s) • {model.vertex_count:,} vertices{simplified}"
        )

        vtk = self._ensure_vtk_view()
        self._show_3d_viewport()
        QApplication.processEvents()

        try:
            vtk.set_wireframe(self.wireframe_check.isChecked())
            vtk.show_model(model)
        except Exception as exc:
            log_exception("show_model in _apply_loaded_model")
            self._solid_model = None
            self._clear_3d_view(f"Could not display 3D model:\n{exc}")
            self.status_bar.showMessage("3D display failed — see step_editor.log")
            return

        self._show_3d_viewport()
        self._sync_3d_appearance_controls()

        tab_idx = self.tabs.indexOf(self._view3d_panel) if self._view3d_panel else 1
        self._suppress_tab_3d_refresh = True
        if tab_idx >= 0:
            self.tabs.setCurrentIndex(tab_idx)
        self._suppress_tab_3d_refresh = False

        vtk.setFocus()
        self._sync_ruler_controls_to_vtk()
        self.status_bar.showMessage(f"GPU 3D preview ready — {Path(model.source_path).name}")
        QTimer.singleShot(100, self._reflow_3d_after_resize)

    def _reset_3d_camera(self) -> None:
        self._set_3d_view("iso")

    def _begin_export_pause(self) -> None:
        """Pause VTK/ruler during matplotlib work (must run on the main thread)."""
        self._export_in_progress = True
        self._ruler_was_enabled_before_export = False
        if _VIEWER_AVAILABLE and self.vtk_view is not None and hasattr(self, "ruler_check"):
            self._ruler_was_enabled_before_export = self.ruler_check.isChecked()
            self.vtk_view.set_render_paused(True)
            if self._ruler_was_enabled_before_export:
                self.vtk_view.set_ruler_visible(False)
        QApplication.processEvents()

    def _end_export_pause(self) -> None:
        self._export_in_progress = False
        if _VIEWER_AVAILABLE and self.vtk_view is not None:
            if self._ruler_was_enabled_before_export:
                self.vtk_view.set_ruler_visible(True)
            self.vtk_view.set_render_paused(False)
            self._schedule_viewport_fit_after_export()

    def _schedule_viewport_fit_after_export(self) -> None:
        self._last_reflow_size = None
        QTimer.singleShot(150, self._reflow_3d_after_resize)

    def _show_about(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle(APP_DISPLAY_NAME)
        box.setIcon(QMessageBox.Icon.NoIcon)
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(about_text())
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        for label in box.findChildren(QLabel):
            if label.textFormat() == Qt.TextFormat.RichText:
                label.setOpenExternalLinks(True)
        box.exec()

    def _export_blueprint(self) -> None:
        if not _BLUEPRINT_AVAILABLE:
            QMessageBox.warning(
                self,
                "Blueprint",
                "Blueprint export requires matplotlib.\n\n"
                "Run: pip install -r requirements.txt\n\n"
                f"{_BLUEPRINT_IMPORT_ERROR}",
            )
            return
        if self._solid_model is None or self.current_file is None:
            QMessageBox.information(
                self,
                "Blueprint",
                "Open a STEP file and wait for the 3D model to load first.",
            )
            return

        default_name = f"{self.current_file.stem}_blueprint.png"
        default_path = self.current_file.parent / default_name
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Save Blueprint",
            str(default_path),
            "PNG image (*.png);;PDF document (*.pdf);;SVG vector (*.svg)",
        )
        if not chosen:
            return

        out_path = Path(chosen)
        model = self._solid_model
        self.status_bar.showMessage(f"Generating blueprint for {self.current_file.name}…")
        QApplication.processEvents()

        error = ""
        saved = str(out_path)
        self._begin_export_pause()
        try:
            snapshot = build_blueprint_snapshot(model)
            saved = str(export_blueprint_snapshot(snapshot, out_path))
            log.info("Blueprint saved: %s", saved)
        except Exception as exc:
            error = str(exc)
            log.exception("Blueprint export failed")
        finally:
            self._end_export_pause()

        self._on_blueprint_exported(saved, error)

    def _on_blueprint_exported(self, path: str, error: str) -> None:
        if error:
            QMessageBox.critical(self, "Blueprint failed", f"Could not create blueprint:\n{error}")
            self.status_bar.showMessage("Blueprint export failed")
            return

        self.status_bar.showMessage(f"Blueprint saved — {path}")
        answer = QMessageBox.question(
            self,
            "Blueprint saved",
            f"Blueprint saved to:\n{path}\n\nOpen the file now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _export_pencil_sketch(self) -> None:
        if not _PENCIL_SKETCH_AVAILABLE:
            QMessageBox.warning(
                self,
                "Pencil Sketch",
                "Pencil sketch export requires matplotlib.\n\n"
                "Run: pip install -r requirements.txt\n\n"
                f"{_PENCIL_SKETCH_IMPORT_ERROR}",
            )
            return
        if self._solid_model is None or self.current_file is None:
            QMessageBox.information(
                self,
                "Pencil Sketch",
                "Open a STEP file and wait for the 3D model to load first.",
            )
            return

        default_name = f"{self.current_file.stem}_sketch.png"
        default_path = self.current_file.parent / default_name
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Save Pencil Sketch",
            str(default_path),
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;PDF document (*.pdf)",
        )
        if not chosen:
            return

        out_path = Path(chosen)
        model = self._solid_model
        self.status_bar.showMessage(f"Drawing pencil sketch for {self.current_file.name}…")
        QApplication.processEvents()

        error = ""
        saved = str(out_path)
        self._begin_export_pause()
        try:
            snapshot = build_sketch_snapshot(model)
            saved = str(export_pencil_sketch_snapshot(snapshot, out_path))
            log.info("Pencil sketch saved: %s", saved)
        except Exception as exc:
            error = str(exc)
            log.exception("Pencil sketch export failed")
        finally:
            self._end_export_pause()

        self._on_pencil_sketch_exported(saved, error)

    def _on_pencil_sketch_exported(self, path: str, error: str) -> None:
        if error:
            QMessageBox.critical(
                self,
                "Pencil sketch failed",
                f"Could not create pencil sketch:\n{error}",
            )
            self.status_bar.showMessage("Pencil sketch export failed")
            return

        self.status_bar.showMessage(f"Pencil sketch saved — {path}")
        answer = QMessageBox.question(
            self,
            "Pencil sketch saved",
            f"Pencil sketch saved to:\n{path}\n\nOpen the image now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _export_dpr(self) -> None:
        if not _DPR_AVAILABLE:
            QMessageBox.warning(
                self,
                "DPR",
                "Detailed Print Report requires matplotlib.\n\n"
                "Run: pip install -r requirements.txt\n\n"
                f"{_DPR_IMPORT_ERROR}",
            )
            return
        if self._solid_model is None or self.current_file is None:
            QMessageBox.information(
                self,
                "DPR",
                "Open a STEP file and wait for the 3D model to load first.",
            )
            return

        default_name = f"{self.current_file.stem}_DPR.png"
        default_path = self.current_file.parent / default_name
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Save Detailed Print Report (DPR)",
            str(default_path),
            "PNG image (*.png);;PDF document (*.pdf);;JPEG image (*.jpg *.jpeg)",
        )
        if not chosen:
            return

        out_path = Path(chosen)
        model = self._solid_model
        self.status_bar.showMessage(f"Building DPR for {self.current_file.name}…")
        QApplication.processEvents()

        error = ""
        saved = str(out_path)
        self._begin_export_pause()
        try:
            snapshot = build_dpr_snapshot(model)
            saved = str(export_dpr_snapshot(snapshot, out_path))
            log.info("DPR saved: %s", saved)
        except Exception as exc:
            error = str(exc)
            log.exception("DPR export failed")
        finally:
            self._end_export_pause()

        if error:
            QMessageBox.critical(self, "DPR failed", f"Could not create print report:\n{error}")
            self.status_bar.showMessage("DPR export failed")
            return

        self.status_bar.showMessage(f"DPR saved — {saved}")
        answer = QMessageBox.question(
            self,
            "DPR saved",
            f"Detailed Print Report saved to:\n{saved}\n\nOpen the image now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            QDesktopServices.openUrl(QUrl.fromLocalFile(saved))

    def _print_dpr(self) -> None:
        if not _PRINT_AVAILABLE:
            QMessageBox.warning(
                self,
                "Print",
                "Print support is unavailable.\n\n"
                f"{_PRINT_IMPORT_ERROR}",
            )
            return
        if not _DPR_AVAILABLE:
            QMessageBox.warning(
                self,
                "Print",
                "Printing requires the Detailed Print Report module (matplotlib).\n\n"
                f"{_DPR_IMPORT_ERROR}",
            )
            return
        if self._solid_model is None or self.current_file is None:
            QMessageBox.information(
                self,
                "Print",
                "Open a STEP file and wait for the 3D model to load first.",
            )
            return

        dialog = PrinterSelectDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.selected_printer is None:
            return

        printer = dialog.selected_printer
        temp_dir = Path(tempfile.gettempdir()) / "nagaworks_step_editor"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{self.current_file.stem}_DPR_print.png"

        model = self._solid_model
        self.status_bar.showMessage(f"Preparing print report for {self.current_file.name}…")
        QApplication.processEvents()

        error = ""
        self._begin_export_pause()
        try:
            snapshot = build_dpr_snapshot(model)
            export_dpr_snapshot(snapshot, temp_path)
            log.info("DPR print image: %s", temp_path)
        except Exception as exc:
            error = str(exc)
            log.exception("DPR print preparation failed")
        finally:
            self._end_export_pause()

        if error:
            QMessageBox.critical(
                self,
                "Print failed",
                f"Could not prepare print report:\n{error}",
            )
            self.status_bar.showMessage("Print preparation failed")
            return

        self.status_bar.showMessage(f"Printing to {printer.name}…")
        QApplication.processEvents()

        try:
            printed = print_image(temp_path, printer.name, parent=self)
        except Exception as exc:
            log.exception("Print failed")
            QMessageBox.critical(self, "Print failed", f"Could not print:\n{exc}")
            self.status_bar.showMessage("Print failed")
            return

        if printed:
            label = printer.brand or printer.name
            self.status_bar.showMessage(f"Sent to printer — {label}")
        else:
            self.status_bar.showMessage("Print cancelled")

    def closeEvent(self, event) -> None:
        if not self._confirm_discard():
            event.ignore()
            return
        if self._is_3d_fullscreen():
            self._in_3d_fullscreen = False
            self._fullscreen_transition = False
            self.tabs.tabBar().show()
            if self._left_panel is not None:
                self._left_panel.show()
            if self._right_header is not None:
                self._right_header.show()
            if self._main_toolbar is not None:
                self._main_toolbar.show()
            self.statusBar().show()
            if self._splitter is not None and self._splitter_sizes_before_fs is not None:
                self._splitter.setSizes(self._splitter_sizes_before_fs)
        if self._load_thread and self._load_thread.is_alive():
            self._load_token += 1
        if _VIEWER_AVAILABLE and self.vtk_view is not None:
            self.vtk_view.close()
        event.accept()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if _VIEWER_AVAILABLE and self.vtk_view:
            QTimer.singleShot(0, self._schedule_viewport_fit)
            QTimer.singleShot(250, self._reflow_3d_after_resize)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if _VIEWER_AVAILABLE and self.vtk_view and not self._fullscreen_transition:
            self._schedule_viewport_fit()

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() != QEvent.Type.WindowStateChange:
            return
        if _VIEWER_AVAILABLE and self.vtk_view and not self._fullscreen_transition:
            QTimer.singleShot(0, self._schedule_viewport_fit)
            QTimer.singleShot(100, self._reflow_3d_after_resize)
            QTimer.singleShot(400, self._reflow_3d_after_resize)


def main() -> int:
    log.info("Starting application")
    try:
        app = QApplication(sys.argv)
        app.setApplicationName(APP_DISPLAY_NAME)
        # Empty so Windows does not append a second copy of the app name to the title bar.
        app.setApplicationDisplayName("")
        app.setApplicationVersion(APP_VERSION)
        app.setOrganizationName(APP_VENDOR)
        app.setStyle("Fusion")
        _apply_app_icon(app=app)
        window = StepFileEditor()
    except Exception:
        log_exception("Failed to create application window")
        raise

    if len(sys.argv) > 1:
        arg = Path(sys.argv[1]).resolve()
        if arg.is_dir():
            window.current_dir = arg
            window.folder_label.setText(f"  Folder: {arg}")
        elif arg.is_file():
            window.current_dir = arg.parent
            window.folder_label.setText(f"  Folder: {arg.parent}")
            window._suppress_file_select = True
            window._load_file(arg)
            for row in range(window.file_list.count()):
                if window.file_list.item(row).text() == arg.name:
                    window.file_list.blockSignals(True)
                    window.file_list.setCurrentRow(row)
                    window.file_list.blockSignals(False)
                    break
            window._suppress_file_select = False
            if _VIEWER_AVAILABLE and arg.suffix.lower() in {".stp", ".step"}:
                idx = window.tabs.indexOf(window._view3d_panel) if window._view3d_panel else 1
                if idx >= 0:
                    window.tabs.setCurrentIndex(idx)
        else:
            QMessageBox.warning(window, "Path not found", f"Could not open:\n{arg}")

    window._refresh_file_list()
    window.show()
    log.info("Main window shown")
    try:
        return app.exec()
    except Exception:
        log_exception("Application event loop crashed")
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        log_exception("Fatal error in main")
        raise
