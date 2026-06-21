# Building the Windows Installer

This produces a standalone installer for the **Wind Visualization System** that
installs and runs on another Windows machine **without a separate Python install**.
It bundles the app, the portable Python runtime (`python\`), the ML model files,
and the OpenFOAM sample dataset.

## How it works

Rather than freezing the app, the installer ships the existing **portable Python
runtime** that already powers `run.bat` (`python\python.exe main.py`). The on-disk
layout is copied intact, so the app behaves identically to a developer run and the
heavy ML stack (PyTorch + CUDA, `torch_geometric`/`torch_scatter`/`torch_sparse`/
`torch_cluster`/`torch_spline_conv`, PyQt5, PyOpenGL) just works.

## Prerequisites (one time, build machine only)

1. **Inno Setup 6** — the installer compiler:
   ```powershell
   winget install --id JRSoftware.InnoSetup -e
   ```
   (or download from https://jrsoftware.org/isdl.php)

2. A **populated `python\` folder** in the repo root (the portable runtime). It is
   gitignored (~5.8 GB), so it must exist locally. Confirm the app runs first:
   ```powershell
   .\run.bat
   ```

## Build a release

```powershell
powershell -NoProfile -File installer\build.ps1 -Version 1.0.0
```

- For a quicker (slightly larger) test build: add `-Fast`.
- Output lands in `installer\Output\`:
  - `WindVisualizationSystem-Setup-1.0.0.exe` plus `*.bin` slices.
  - The payload (~6.7 GB) exceeds the single-file limit, so the installer is
    **disk-spanned**. **Distribute the `.exe` together with all its `.bin` files.**

The compile compresses several GB and can take many minutes.

## Cutting a new release after code changes

Just bump the version and rebuild — this is the repeatable mechanism:

```powershell
pwsh installer\build.ps1 -Version 1.0.1
```

## What the installer does on the target machine

- Installs to `C:\Program Files\Wind Visualization System` (admin required).
- Creates Start Menu shortcuts:
  - **Wind Visualization System** — launches via `pythonw.exe` (no console).
  - **Wind Visualization System (Debug Console)** — launches via `python.exe`
    with a console for troubleshooting.
- Optional desktop shortcut.
- Provides a standard uninstaller.

## Regenerating the app icon (optional)

`installer\app.ico` is checked in. To regenerate it:

```powershell
.\python\python.exe installer\make_icon.py
```

## Files in `installer\`

| File | Purpose |
|---|---|
| `WindVisualizationSystem.iss` | Inno Setup script (what to bundle, shortcuts, metadata). |
| `build.ps1` | Build driver — locates ISCC, checks payload, compiles, reports output. |
| `make_icon.py` | Regenerates `app.ico` (pure numpy/zlib, no Pillow needed). |
| `app.ico` | Application / installer icon. |
| `Output\` | Build artifacts (gitignored). |
