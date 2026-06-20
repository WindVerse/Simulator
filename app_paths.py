"""
Central path resolution for both source runs and a future one-file PyInstaller
build.

Two concerns are separated here:

* **Read-only bundled assets** (model weights, .npy tables, OBJ meshes, the sample
  case) — located with :func:`resource_path`. Under PyInstaller one-file these are
  unpacked into a temporary ``sys._MEIPASS`` dir, so ``__file__``-relative lookups
  break; ``resource_path`` resolves against ``_MEIPASS`` when frozen and against the
  repo root otherwise.

* **The large, writable, persistent wind cache** (~32 GB) — located with
  :func:`user_cache_root`. It must NOT live inside ``_MEIPASS`` (that is wiped on
  exit) and must survive reboots. The chosen location is remembered in a small
  ``settings.json`` so it is stable across power cycles and inside the bundled exe.

Env overrides: ``WINDVERSE_CACHE_DIR`` (force a cache root), ``WINDVERSE_DATA_DIR``
(force the per-user data dir).
"""

import json
import os
import shutil
import string
import sys


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Default cache root on this machine. D: is roomy here; user_cache_root() falls
# back gracefully when D: is absent (e.g. the bundled app on another machine).
_DEFAULT_CACHE_ROOT = r"D:\windverse_cache"


def is_frozen() -> bool:
    """True when running inside a PyInstaller (or similar) frozen bundle."""
    return bool(getattr(sys, "frozen", False))


# --------------------------------------------------------------------------- #
# Read-only bundled assets
# --------------------------------------------------------------------------- #
def resource_base() -> str:
    """Base directory for read-only bundled resources."""
    if is_frozen():
        # PyInstaller one-file unpacks data here; one-dir sets it to the app dir.
        return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    return _REPO_ROOT


def resource_path(*parts: str) -> str:
    """Absolute path to a bundled read-only resource (e.g. ``"models", "x.npy"``)."""
    return os.path.join(resource_base(), *parts)


# --------------------------------------------------------------------------- #
# Per-user writable data dir + settings
# --------------------------------------------------------------------------- #
def user_data_dir() -> str:
    """Per-user writable dir for settings (and a fallback cache).

    ``%LOCALAPPDATA%\\WindVerse`` on Windows; overridable via ``WINDVERSE_DATA_DIR``.
    """
    env = os.environ.get("WINDVERSE_DATA_DIR")
    if env:
        return env
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "WindVerse")


def settings_path() -> str:
    return os.path.join(user_data_dir(), "settings.json")


def load_settings() -> dict:
    try:
        with open(settings_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_settings(data: dict) -> None:
    """Persist settings atomically (survives reboots / works inside the exe)."""
    os.makedirs(user_data_dir(), exist_ok=True)
    tmp = settings_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, settings_path())


# --------------------------------------------------------------------------- #
# Cache root resolution
# --------------------------------------------------------------------------- #
def _drive_root(path: str) -> str:
    drive = os.path.splitdrive(os.path.abspath(path))[0]
    return (drive + os.sep) if drive else path


def _drive_exists(path: str) -> bool:
    try:
        return os.path.exists(_drive_root(path))
    except OSError:
        return False


def free_space_gb(path: str) -> float:
    """Free space (GB) on the drive that would hold ``path`` (walk up if needed)."""
    probe = os.path.abspath(path)
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe).free / 1e9
    except OSError:
        return 0.0


def _roomiest_fixed_drive() -> str:
    """Drive root with the most free space (best-effort fallback)."""
    best, best_free = "", -1.0
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if not os.path.exists(root):
            continue
        try:
            free = shutil.disk_usage(root).free
        except OSError:
            continue
        if free > best_free:
            best, best_free = root, free
    return best


def user_cache_root() -> str:
    """Resolve the writable cache root.

    Order: ``WINDVERSE_CACHE_DIR`` env -> persisted ``settings["cache_dir"]`` (if its
    drive exists) -> ``D:\\windverse_cache`` if D: exists (the default; preserves an
    existing build) -> roomiest fixed drive -> ``<user_data_dir>/cache``.
    """
    env = os.environ.get("WINDVERSE_CACHE_DIR")
    if env:
        return env

    remembered = load_settings().get("cache_dir")
    if remembered and _drive_exists(remembered):
        return remembered

    if _drive_exists(_DEFAULT_CACHE_ROOT):
        return _DEFAULT_CACHE_ROOT

    roomiest = _roomiest_fixed_drive()
    if roomiest:
        return os.path.join(roomiest, "windverse_cache")

    return os.path.join(user_data_dir(), "cache")


def set_cache_root(path: str) -> None:
    """Remember an explicit cache root (persisted across reboots / in the exe)."""
    settings = load_settings()
    settings["cache_dir"] = os.path.abspath(path)
    save_settings(settings)
