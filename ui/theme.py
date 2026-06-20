"""
Application Theme
Centralized "refined dark" theme for the whole app.

Single source of truth for colors, fonts, and the Qt style sheet (QSS).
Apply it once via ``apply_theme(app)`` in ``main.py`` so every widget — including
dialogs, message boxes and the file picker — inherits a consistent look.
"""

from PyQt5.QtGui import QColor, QFont, QPalette


# --- Palette ---------------------------------------------------------------
# Deep neutral greys with a single cyan-blue accent.
BG = "#1a1c20"          # window background
PANEL = "#24272e"       # docks / panels
ELEVATED = "#2b2f37"    # inputs, hovered surfaces
BORDER = "#3a3f47"      # hairline borders / separators
BORDER_LIGHT = "#474d57"

TEXT = "#e6e8eb"        # primary text
TEXT_MUTED = "#9aa0a8"  # secondary text
TEXT_DIM = "#6b7178"    # tertiary / disabled text

ACCENT = "#4FC3F7"          # cyan-blue accent
ACCENT_HOVER = "#6fcef8"
ACCENT_PRESSED = "#3aa9de"

SUCCESS = "#43A047"     # Play
SUCCESS_HOVER = "#4cae50"
WARNING = "#FB8C00"     # Pause (checked)
WARNING_HOVER = "#ff9f29"
NEUTRAL = "#3a3f47"     # Reset / secondary buttons
NEUTRAL_HOVER = "#474d57"

# Fonts
FONT_FAMILY = '"Segoe UI", "Inter", "Roboto", system-ui, sans-serif'
MONO_FAMILY = '"Cascadia Code", "Consolas", "DejaVu Sans Mono", monospace'
FONT_SIZE = 10  # points


def base_font() -> QFont:
    """Return the application base font."""
    font = QFont("Segoe UI", FONT_SIZE)
    font.setStyleHint(QFont.SansSerif)
    return font


def build_palette() -> QPalette:
    """Build a QPalette so native bits (tooltips, selections) match the QSS."""
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(BG))
    pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.Base, QColor(ELEVATED))
    pal.setColor(QPalette.AlternateBase, QColor(PANEL))
    pal.setColor(QPalette.Text, QColor(TEXT))
    pal.setColor(QPalette.Button, QColor(PANEL))
    pal.setColor(QPalette.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.ToolTipBase, QColor(ELEVATED))
    pal.setColor(QPalette.ToolTipText, QColor(TEXT))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor(BG))
    pal.setColor(QPalette.Link, QColor(ACCENT))
    pal.setColor(QPalette.PlaceholderText, QColor(TEXT_DIM))
    # Disabled variants
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(TEXT_DIM))
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor(TEXT_DIM))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(TEXT_DIM))
    return pal


def build_stylesheet() -> str:
    """Return the comprehensive QSS for the whole application."""
    return f"""
    /* ---- Base ---- */
    QWidget {{
        background-color: {BG};
        color: {TEXT};
        font-family: {FONT_FAMILY};
        font-size: {FONT_SIZE}pt;
    }}
    QMainWindow, QDialog {{
        background-color: {BG};
    }}
    QToolTip {{
        background-color: {ELEVATED};
        color: {TEXT};
        border: 1px solid {BORDER_LIGHT};
        border-radius: 4px;
        padding: 4px 6px;
    }}

    /* ---- Menu bar / menus ---- */
    QMenuBar {{
        background-color: {PANEL};
        color: {TEXT};
        border-bottom: 1px solid {BORDER};
        padding: 2px;
    }}
    QMenuBar::item {{
        background: transparent;
        padding: 5px 10px;
        border-radius: 4px;
    }}
    QMenuBar::item:selected {{
        background-color: {ELEVATED};
        color: {ACCENT};
    }}
    QMenu {{
        background-color: {PANEL};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 4px;
    }}
    QMenu::item {{
        padding: 6px 22px 6px 18px;
        border-radius: 4px;
    }}
    QMenu::item:selected {{
        background-color: {ELEVATED};
        color: {ACCENT};
    }}
    QMenu::separator {{
        height: 1px;
        background: {BORDER};
        margin: 5px 8px;
    }}

    /* ---- Toolbar ---- */
    QToolBar {{
        background-color: {PANEL};
        border: none;
        border-bottom: 1px solid {BORDER};
        spacing: 4px;
        padding: 5px 6px;
    }}
    QToolBar::separator {{
        width: 1px;
        background: {BORDER};
        margin: 4px 6px;
    }}
    QToolButton {{
        background: transparent;
        color: {TEXT};
        border: 1px solid transparent;
        border-radius: 5px;
        padding: 5px 11px;
    }}
    QToolButton:hover {{
        background-color: {ELEVATED};
        border: 1px solid {BORDER_LIGHT};
    }}
    QToolButton:pressed {{
        background-color: {BORDER};
    }}
    QToolButton:checked {{
        background-color: {ELEVATED};
        color: {ACCENT};
        border: 1px solid {ACCENT};
    }}

    /* ---- Status bar ---- */
    QStatusBar {{
        background-color: {PANEL};
        color: {TEXT_MUTED};
        border-top: 1px solid {BORDER};
    }}
    QStatusBar::item {{
        border: none;
    }}
    QStatusBar QLabel {{
        background: transparent;
        color: {TEXT_MUTED};
        padding: 2px 4px;
    }}

    /* ---- Dock widgets ---- */
    QDockWidget {{
        color: {TEXT};
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
    }}
    QDockWidget::title {{
        background-color: {PANEL};
        color: {TEXT_MUTED};
        padding: 7px 10px;
        border-bottom: 1px solid {BORDER};
        text-align: left;
        font-weight: bold;
        letter-spacing: 1px;
    }}

    /* ---- Group boxes ---- */
    QGroupBox {{
        background-color: {PANEL};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 8px;
        margin-top: 14px;
        padding: 10px 10px 10px 10px;
        font-weight: bold;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        top: 1px;
        padding: 1px 6px;
        color: {ACCENT};
        background-color: {PANEL};
    }}

    /* ---- Labels ---- */
    QLabel {{
        background: transparent;
        color: {TEXT};
    }}
    QLabel#panelTitle {{
        font-size: 13pt;
        font-weight: bold;
        color: {TEXT};
        padding: 4px;
    }}
    QLabel#panelHint {{
        color: {TEXT_MUTED};
        font-size: 9pt;
    }}
    QLabel#statValue {{
        color: {TEXT_MUTED};
        font-family: {MONO_FAMILY};
        font-size: 9pt;
    }}

    /* ---- Push buttons (default) ---- */
    QPushButton {{
        background-color: {ELEVATED};
        color: {TEXT};
        border: 1px solid {BORDER_LIGHT};
        border-radius: 6px;
        padding: 7px 12px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background-color: {NEUTRAL_HOVER};
        border-color: {ACCENT};
    }}
    QPushButton:pressed {{
        background-color: {BORDER};
    }}
    QPushButton:disabled {{
        background-color: {PANEL};
        color: {TEXT_DIM};
        border-color: {BORDER};
    }}

    /* ---- Play / Reset accent buttons ---- */
    QPushButton#playButton {{
        background-color: {SUCCESS};
        color: white;
        border: none;
        padding: 9px;
    }}
    QPushButton#playButton:hover {{
        background-color: {SUCCESS_HOVER};
    }}
    QPushButton#playButton:checked {{
        background-color: {WARNING};
    }}
    QPushButton#playButton:checked:hover {{
        background-color: {WARNING_HOVER};
    }}
    QPushButton#playButton:disabled {{
        background-color: {NEUTRAL};
        color: {TEXT_DIM};
    }}
    QPushButton#resetButton {{
        background-color: {NEUTRAL};
        color: {TEXT};
        border: 1px solid {BORDER_LIGHT};
        padding: 7px;
    }}
    QPushButton#resetButton:hover {{
        background-color: {NEUTRAL_HOVER};
        border-color: {ACCENT};
    }}

    /* ---- Library object tiles ---- */
    QPushButton#objectTile {{
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        font-size: 11pt;
        padding: 0px;
    }}

    /* ---- Check boxes ---- */
    QCheckBox {{
        spacing: 8px;
        color: {TEXT};
        padding: 2px 0px;
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {BORDER_LIGHT};
        border-radius: 4px;
        background-color: {ELEVATED};
    }}
    QCheckBox::indicator:hover {{
        border-color: {ACCENT};
    }}
    QCheckBox::indicator:checked {{
        background-color: {ACCENT};
        border-color: {ACCENT};
        image: url(none);
    }}
    QCheckBox::indicator:checked:hover {{
        background-color: {ACCENT_HOVER};
    }}
    QCheckBox:disabled {{
        color: {TEXT_DIM};
    }}

    /* ---- Radio buttons ---- */
    QRadioButton {{
        spacing: 8px;
        color: {TEXT};
        padding: 2px 0px;
    }}
    QRadioButton::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {BORDER_LIGHT};
        border-radius: 9px;
        background-color: {ELEVATED};
    }}
    QRadioButton::indicator:hover {{
        border-color: {ACCENT};
    }}
    QRadioButton::indicator:checked {{
        background-color: {ACCENT};
        border: 4px solid {ELEVATED};
        border-radius: 9px;
    }}
    QRadioButton:disabled {{
        color: {TEXT_DIM};
    }}

    /* ---- Spin box ---- */
    QSpinBox {{
        background-color: {ELEVATED};
        color: {TEXT};
        border: 1px solid {BORDER_LIGHT};
        border-radius: 5px;
        padding: 3px 6px;
        min-height: 20px;
    }}
    QSpinBox:focus {{
        border-color: {ACCENT};
    }}
    QSpinBox::up-button, QSpinBox::down-button {{
        subcontrol-origin: border;
        width: 18px;
        background-color: {NEUTRAL};
        border-left: 1px solid {BORDER_LIGHT};
    }}
    QSpinBox::up-button {{
        subcontrol-position: top right;
        border-top-right-radius: 5px;
    }}
    QSpinBox::down-button {{
        subcontrol-position: bottom right;
        border-bottom-right-radius: 5px;
    }}
    QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
        background-color: {ACCENT};
    }}
    QSpinBox::up-arrow {{
        image: none;
        width: 0; height: 0;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-bottom: 5px solid {TEXT};
    }}
    QSpinBox::down-arrow {{
        image: none;
        width: 0; height: 0;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {TEXT};
    }}

    /* ---- Separators ---- */
    QFrame[frameShape="4"] {{   /* HLine */
        background-color: {BORDER};
        max-height: 1px;
        border: none;
    }}
    QFrame[frameShape="5"] {{   /* VLine */
        background-color: {BORDER};
        max-width: 1px;
        border: none;
    }}

    /* ---- Scrollbars ---- */
    QScrollBar:vertical {{
        background: {BG};
        width: 11px;
        margin: 0px;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER_LIGHT};
        min-height: 28px;
        border-radius: 5px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {ACCENT};
    }}
    QScrollBar:horizontal {{
        background: {BG};
        height: 11px;
        margin: 0px;
    }}
    QScrollBar::handle:horizontal {{
        background: {BORDER_LIGHT};
        min-width: 28px;
        border-radius: 5px;
        margin: 2px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {ACCENT};
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{
        height: 0px;
        width: 0px;
        background: none;
        border: none;
    }}
    QScrollBar::add-page, QScrollBar::sub-page {{
        background: none;
    }}

    /* ---- Combo boxes ---- */
    QComboBox {{
        background-color: {ELEVATED};
        color: {TEXT};
        border: 1px solid {BORDER_LIGHT};
        border-radius: 5px;
        padding: 2px 8px;
        min-height: 20px;
    }}
    QComboBox:hover {{
        border-color: {ACCENT};
    }}
    QComboBox:focus {{
        border-color: {ACCENT};
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: center right;
        width: 18px;
        border: none;
    }}
    QComboBox::down-arrow {{
        image: none;
        width: 0; height: 0;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {TEXT};
        margin-right: 6px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {PANEL};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        selection-background-color: {ELEVATED};
        selection-color: {ACCENT};
        outline: none;
        padding: 2px;
    }}

    /* ---- Viewport panes ---- */
    QWidget#viewportPane {{
        background-color: {BG};
        border: 2px solid transparent;
        border-radius: 4px;
    }}
    QWidget#viewportPane[active="true"] {{
        border: 2px solid {ACCENT};
    }}
    QFrame#viewportHeader {{
        background-color: {PANEL};
        border-bottom: 1px solid {BORDER};
    }}
    QLabel#viewportTitle {{
        color: {TEXT_MUTED};
        font-weight: bold;
        font-size: 9pt;
        letter-spacing: 1px;
        background: transparent;
    }}
    QComboBox#viewportPreset {{
        font-size: 9pt;
        padding: 1px 6px;
        min-height: 18px;
        max-height: 20px;
    }}

    /* ---- Message box / dialog buttons ---- */
    QMessageBox {{
        background-color: {PANEL};
    }}
    QMessageBox QPushButton {{
        min-width: 76px;
    }}
    """


def apply_theme(app):
    """Apply the refined-dark theme to a QApplication."""
    app.setStyle("Fusion")
    app.setFont(base_font())
    app.setPalette(build_palette())
    app.setStyleSheet(build_stylesheet())
