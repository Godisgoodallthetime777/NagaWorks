"""NagaWorks STEP File Editor — application name and version."""

from __future__ import annotations

from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
APP_LOGO_FILE = APP_DIR / "logo.png"

APP_VENDOR = "NagaWorks"
APP_PRODUCT = "STEP File Editor"
APP_VERSION = "1.1"
APP_DISPLAY_NAME = f"{APP_VENDOR} {APP_PRODUCT} V{APP_VERSION}"
# Short name for title bar / taskbar (Windows appends this after the window title).
APP_TITLE_BRAND = APP_VENDOR
APP_DESCRIPTION = (
    "View and edit STEP (.stp) files with GPU 3D preview, "
    "technical blueprints, and pencil sketch export."
)
APP_COPYRIGHT = f"© {APP_VENDOR}"
APP_COMPANY = "Naga Soft Labs"
APP_WEBSITE = "https://nagasoftlabs.com/"
APP_WEBSITE_LABEL = "nagasoftlabs.com"


def app_logo_path() -> Path | None:
    """Path to logo.png beside the application, if present."""
    if APP_LOGO_FILE.is_file():
        return APP_LOGO_FILE
    return None


def window_title(file_name: str | None = None) -> str:
    """Single title-bar string (avoids Windows appending a duplicate app name)."""
    if file_name:
        return f"{file_name} – {APP_TITLE_BRAND}"
    return APP_TITLE_BRAND


def about_text() -> str:
    logo_html = ""
    logo = app_logo_path()
    if logo is not None:
        logo_html = (
            f'<p align="center"><img src="file:///{logo.as_posix()}" '
            f'width="80" alt="{APP_VENDOR}"></p>'
        )
    return (
        logo_html
        + f"<h3>{APP_DISPLAY_NAME}</h3>"
        f"<p>{APP_DESCRIPTION}</p>"
        f"<p>"
        f"<b>Version</b> {APP_VERSION}<br>"
        f"<b>Publisher</b> {APP_VENDOR}<br>"
        f"<b>Company</b> {APP_COMPANY}"
        f"</p>"
        f"<p>"
        f"Visit us at "
        f'<a href="{APP_WEBSITE}">{APP_WEBSITE}</a>'
        f"<br>"
        f"Digital technology solutions from {APP_COMPANY}."
        f"</p>"
        f"<p>{APP_COPYRIGHT}. All rights reserved.</p>"
    )
