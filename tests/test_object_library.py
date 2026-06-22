"""Unit tests for ui.object_library (flag icon rendering, drag tile, library panel)."""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from ui.object_library import (
    FLAG_CLOTH_COLOR,
    FLAG_POLE_COLOR,
    ObjectButton,
    ObjectLibraryPanel,
    make_flag_pixmap,
)


def test_make_flag_pixmap_has_requested_size():
    pixmap = make_flag_pixmap(56, QColor(FLAG_CLOTH_COLOR), QColor(FLAG_POLE_COLOR))
    assert pixmap.width() == 56
    assert pixmap.height() == 56
    assert not pixmap.isNull()


def test_make_flag_pixmap_defaults_pole_color_to_cloth_color():
    cloth = QColor("#123456")
    pixmap = make_flag_pixmap(32, cloth)
    assert not pixmap.isNull()


def test_object_button_stores_object_type(qtbot):
    button = ObjectButton('flag')
    qtbot.addWidget(button)
    assert button.object_type == 'flag'


def test_object_button_has_flag_icon_and_no_text(qtbot):
    button = ObjectButton('flag')
    qtbot.addWidget(button)
    assert button.text() == ""
    assert not button.icon().isNull()


def test_object_button_initial_cursor_is_open_hand(qtbot):
    button = ObjectButton('flag')
    qtbot.addWidget(button)
    assert button.cursor().shape() == Qt.OpenHandCursor


def test_object_button_press_sets_closed_hand_cursor(qtbot):
    button = ObjectButton('flag')
    qtbot.addWidget(button)
    qtbot.mousePress(button, Qt.LeftButton)
    assert button.cursor().shape() == Qt.ClosedHandCursor
    qtbot.mouseRelease(button, Qt.LeftButton)
    assert button.cursor().shape() == Qt.OpenHandCursor


def test_lighten_and_darken_color_change_lightness():
    button = ObjectButton.__new__(ObjectButton)
    base = "#808080"
    lighter = button._lighten_color(base)
    darker = button._darken_color(base)
    assert QColor(lighter).lightness() > QColor(base).lightness()
    assert QColor(darker).lightness() < QColor(base).lightness()


def test_object_library_panel_has_one_flag_button(qtbot):
    panel = ObjectLibraryPanel()
    qtbot.addWidget(panel)
    assert panel._button.object_type == 'flag'


def test_object_library_panel_click_emits_object_selected(qtbot):
    panel = ObjectLibraryPanel()
    qtbot.addWidget(panel)
    with qtbot.waitSignal(panel.object_selected, timeout=1000) as blocker:
        qtbot.mouseClick(panel._button, Qt.LeftButton)
    assert blocker.args == ['flag']


def test_update_object_count_sets_label_text(qtbot):
    panel = ObjectLibraryPanel()
    qtbot.addWidget(panel)
    panel.update_object_count(3)
    assert panel.count_label.text() == "In scene: 3"


def test_update_object_count_default_is_zero(qtbot):
    panel = ObjectLibraryPanel()
    qtbot.addWidget(panel)
    assert panel.count_label.text() == "In scene: 0"
