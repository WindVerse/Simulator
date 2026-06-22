import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# UI tests run in CI/headless contexts without a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _qapp_instance(qapp):
    """Ensure a QApplication exists even for tests that build QPixmap/QPainter
    objects directly without requesting qtbot (Qt needs an app instance before
    any GUI object is constructed, or the platform plugin init hangs)."""
    return qapp
