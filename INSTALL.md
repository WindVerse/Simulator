# Installing the Wind Visualization System

This guide is for **installing and running** the app on a Windows machine. No
separate Python installation is required — everything (Python runtime, libraries,
ML models, and the OpenFOAM sample data) is bundled inside the installer.

> Building the installer from source is a different task — see [BUILD.md](BUILD.md).

## System requirements

- **Windows 10 or 11, 64-bit.**
- **~7 GB free disk space** on the drive you install to.
- A GPU is **optional** — the app uses an NVIDIA GPU (CUDA) if present and
  automatically falls back to the CPU otherwise.
- Administrator rights (the installer puts the app in `C:\Program Files`).

## What you received

The installer is split into three files that **must stay together in the same
folder**:

```
WindVisualizationSystem-Setup-1.0.0.exe     <- run this one
WindVisualizationSystem-Setup-1.0.0-1.bin
WindVisualizationSystem-Setup-1.0.0-2.bin
```

If you downloaded a `.zip`, extract all three first. The `.exe` will not work
without its `.bin` files next to it.

## Install steps

1. Double-click **`WindVisualizationSystem-Setup-1.0.0.exe`**.
2. If Windows SmartScreen shows "Windows protected your PC", click
   **More info -> Run anyway** (the installer is unsigned).
3. Approve the **User Account Control (admin)** prompt.
4. Accept the default install location (or choose your own) and click
   **Next / Install**.
5. Optionally tick **Create a desktop shortcut**.
6. Wait for the files to extract (several GB — this takes a few minutes), then
   click **Finish**. Leaving "Launch Wind Visualization System" checked starts
   the app immediately.

## Running the app

After installation you'll find these in the Start Menu under
**Wind Visualization System**:

- **Wind Visualization System** — the normal way to launch the app (no console
  window).
- **Wind Visualization System (Debug Console)** — launches with a console window
  that shows diagnostic messages and errors; use this if the app won't start or
  you need to report a problem.

On first launch the app automatically loads the bundled OpenFOAM sample dataset
and displays the wind field.

## Uninstalling

Use **Settings -> Apps -> Installed apps -> Wind Visualization System ->
Uninstall**, or the **Uninstall Wind Visualization System** entry in the Start
Menu group. This removes everything that was installed.

## Troubleshooting

| Symptom | What to do |
|---|---|
| "Setup is unable to find a file" / "disk 2" prompt during install | The `.bin` files aren't beside the `.exe`. Put all three files in one folder and re-run. |
| Double-clicking the shortcut does nothing | Launch via **(Debug Console)** to see the error message. |
| Black window / OpenGL error | Update your graphics drivers; the app needs working OpenGL support. |
| App is slow during simulation | Normal on machines without an NVIDIA GPU (it runs the ML model on the CPU). Output is identical, just slower. |
| SmartScreen / antivirus blocks it | The installer is unsigned; allow it via **More info -> Run anyway** or your AV's exclusions. |

## Notes

- The app does not require an internet connection.
- Your own scenes and any OpenFOAM cases you open are read from / saved to
  wherever you choose — the app does not modify the bundled sample data.
