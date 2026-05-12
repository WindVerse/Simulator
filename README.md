# Wind Visualization System

An interactive 3D wind visualization tool with ML-based mesh deformation.

## Features

- **OpenFOAM Sample Wind Auto-Loaded**: Bundled `wind_data/sample_wind_data` is loaded in the background on startup — no manual action required
- **Drag-and-Drop Object Placement**: Place trees, flags, cloth, and poles into the 3D scene
- **Real-Time Wind Visualization**: View animated wind velocity vectors in the 3D space
- **ML-Based Deformation**: PyTorch neural network predicts mesh vertex displacements
- **GPU Acceleration**: Automatic GPU usage when available
- **Interactive Camera**: Orbit, pan, and zoom controls (Z-up world)
- **Simulation Controls**: Play, pause, and reset simulation

## Project Structure

```
project/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── ui/                    # User interface components
│   ├── __init__.py
│   ├── main_window.py     # Main application window
│   ├── object_library.py  # Drag-and-drop object panel
│   └── simulation_controller.py  # Simulation loop management
├── renderer/              # 3D rendering components
│   ├── __init__.py
│   ├── scene.py           # Scene and camera management
│   └── opengl_widget.py   # OpenGL rendering widget
├── models/                # ML model components
│   ├── __init__.py
│   └── deformation_model.py  # PyTorch deformation network (optional)
├── objects/               # Mesh representations
│   ├── __init__.py
│   ├── object_mesh.py     # Mesh handling class
│   ├── tree.obj           # Sample tree mesh
│   ├── flag.obj           # Sample flag mesh
│   ├── cloth.obj          # Sample cloth mesh
│   └── pole.obj           # Sample pole mesh
├── wind_data/             # Wind field data
│   ├── __init__.py
│   ├── wind_field.py      # 5D wind field class (component, z, y, x, time)
│   ├── openfoam_loader.py # OpenFOAM .raw file parser
│   └── sample_wind_data/  # Bundled OpenFOAM sample (90 time steps × 3 Z-heights)
└── python/                # Portable Python installation (Windows)
```

## Installation

This project includes a **portable Python installation** (Windows) so no system-wide Python is needed.

### Quick Start (Windows - Portable Python Included)

Simply double-click `run.bat` or run:
```cmd
python\python.exe main.py
```

All dependencies are pre-installed in the `python\` folder.

### Alternative: Manual Setup with Virtual Environment

If you prefer to use your own Python installation:

1. **Navigate to the project directory**
   ```bash
   cd path/to/project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   
   **Windows (Command Prompt):**
   ```cmd
   .venv\Scripts\activate
   ```
   
   **Windows (PowerShell):**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   
   **Linux/macOS:**
   ```bash
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Note for PyTorch with GPU support:**
   If you have an NVIDIA GPU and want CUDA acceleration, install PyTorch with CUDA:
   ```bash
   # For CUDA 11.8
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

## Usage

### Controls

| Action | Control |
|--------|---------|
| Orbit Camera | Right-click + drag |
| Pan Camera | Middle-click + drag |
| Zoom | Mouse wheel |
| Play/Pause | Space or Play button |
| Reset Simulation | R key or Reset button |
| Toggle Grid | G key |
| Toggle Wind Vectors | W key |
| Reset Camera | C key |
| Lift object off ground | Hold **Shift** while dragging (mouse up = lift, mouse down = lower) |

### Placing Objects

1. **Drag and Drop**: Drag an object from the left panel onto the 3D viewport
2. **Click and Place**: Click an object in the library, then click in the viewport to place it
3. **Vertical placement**: After placing, drag while holding **Shift** to lift the object above the ground plane

### Object Types

- **Tree**: Trunk and foliage, base is fixed (rooted)
- **Flag**: Rectangular cloth attached to a pole edge
- **Cloth**: Square cloth with fixed corners
- **Pole**: Rigid cylinder (no deformation)

## Technical Details

### Coordinate Convention

The simulator uses a **right-handed, Z-up** world coordinate system:

| Axis | Direction |
|------|-----------|
| +X   | Horizontal (east) |
| +Y   | Horizontal (north) |
| +Z   | Vertical (up) — ground plane at `z = 0` |

OpenFOAM output is natively Z-up, so no axis remapping is applied when loading. Object meshes (`flag.obj`, etc.) are authored in the same Z-up convention.

### Wind Field

Wind data is stored as a 5D numpy array:
- Shape: `(component, grid_z, grid_y, grid_x, time_steps)`
- Component order is `(u_x, u_y, u_z)` in the world frame — `u_z` is vertical wind
- Trilinear interpolation for smooth velocity queries

### Deformation Model

The PyTorch neural network:
- **Input**: Current vertices + wind velocity + previous vertices
- **Architecture**: 4 hidden layers with LayerNorm and ReLU
- **Output**: Vertex displacement vectors

### Simulation Loop

At each timestep:
1. Advance wind field time
2. For each object:
   - Query wind velocity at object position
   - Feed mesh + wind to deformation model
   - Apply displacement with constraints
   - Recompute normals
3. Re-render scene

## Dependencies

- **PyQt5**: UI framework
- **PyOpenGL**: 3D rendering
- **NumPy**: Numerical computations
- **PyTorch**: ML model inference

## Performance Tips

1. **Enable GPU**: Ensure PyTorch CUDA is installed for faster inference
2. **Reduce Wind Vectors**: Toggle off wind visualization for better FPS
3. **Limit Objects**: Fewer objects = better performance
4. **Adjust FPS**: Lower target FPS in the control panel if needed

## Custom Objects

To add custom OBJ files:

1. Place the `.obj` file in the `objects/` folder
2. Author the mesh in **Z-up convention** (vertical extent along the Z axis, ground at Z = 0)
3. Load it programmatically:
   ```python
   mesh = ObjectMesh("custom", "objects/custom.obj", position=(0, 0, 0))
   scene.objects.append(mesh)
   ```

## Custom Wind Data

To load custom wind data programmatically:

```python
import numpy as np

# Shape: (component, grid_z, grid_y, grid_x, time_steps)
# Components: (u_x, u_y, u_z) in world Z-up frame
wind_data = np.random.randn(3, 10, 20, 20, 100).astype(np.float32)

# x_coords / y_coords = horizontal axes; z_coords = vertical (height) axis
np.savez_compressed(
   "wind_data/custom_wind.npz",
   wind_data=wind_data,
   x_coords=np.arange(20, dtype=np.float32),
   y_coords=np.arange(20, dtype=np.float32),
   z_coords=np.arange(10, dtype=np.float32),
   time_coords=np.arange(100, dtype=np.float32)
)

wind_field.load_from_file("wind_data/custom_wind.npz")
```

## OpenFOAM Wind Data

### Bundled sample (auto-loaded on startup)

The repository includes a sample dataset at `wind_data/sample_wind_data/` — 90 time steps at three height slices (2 m, 5 m, 10 m). It is parsed in a background thread immediately after the window opens. The status bar shows **"Loading sample wind…"** while parsing and **"OpenFOAM sample loaded"** when complete. The grid auto-fits to the sample's spatial bounds.

### Loading a different dataset

1. Open the **File** menu and choose **Load OpenFOAM Wind...**
2. Select a folder that contains time-step subdirectories, each holding `U_zNormal_*.raw` files
3. The loader reads the raw slices, builds the 5D wind array (Z-up, no axis remapping), and fits the grid

### Expected `.raw` file format

```
# U  POINT_DATA <n>
# x y z  U_x U_y U_z
80.0 290.0 2.0  16.2  -0.04  -0.008
...
```

Wind time advances based on real wall-clock time, updating every 0.1 seconds.

## Pre-trained Models

To use a pre-trained deformation model:

```python
model = DeformationModel(model_path="models/pretrained.pt")
```

## License

This project is provided for educational and research purposes.

## Troubleshooting

### OpenGL Errors

If you encounter OpenGL errors:
1. Update your graphics drivers
2. Ensure OpenGL 2.1+ is supported
3. Try running with software rendering: `set QT_OPENGL=software`

### PyTorch CUDA

If CUDA is not detected:
1. Verify CUDA installation: `nvidia-smi`
2. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reinstall PyTorch with correct CUDA version

### Import Errors

If modules are not found:
1. Ensure virtual environment is activated
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Check Python path includes project root
