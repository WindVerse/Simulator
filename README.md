# Wind Visualization System

An interactive 3D wind visualization tool with ML-based mesh deformation.

## Features

- **Drag-and-Drop Object Placement**: Place trees, flags, cloth, and poles into the 3D scene
- **Real-Time Wind Visualization**: View animated wind velocity vectors in the 3D space
- **ML-Based Deformation**: PyTorch neural network predicts mesh vertex displacements
- **GPU Acceleration**: Automatic GPU usage when available
- **Interactive Camera**: Orbit, pan, and zoom controls
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
│   └── wind_field.py      # 4D wind field class
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

### Placing Objects

1. **Drag and Drop**: Drag an object from the left panel onto the 3D viewport
2. **Click and Place**: Click an object in the library, then click in the viewport to place it

### Object Types

- **Tree**: Trunk and foliage, base is fixed (rooted)
- **Flag**: Rectangular cloth attached to a pole edge
- **Cloth**: Square cloth with fixed corners
- **Pole**: Rigid cylinder (no deformation)

## Technical Details

### Wind Field

Wind data is stored as a 5D numpy array:
- Shape: `(time_steps, grid_x, grid_y, grid_z, 3)`
- The last dimension contains velocity components `(u, v, w)`
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
2. Load it programmatically:
   ```python
   mesh = ObjectMesh("custom", "objects/custom.obj", position=(0, 0, 0))
   scene.objects.append(mesh)
   ```

## Custom Wind Data

To load custom wind data:

```python
import numpy as np

# Create or load your wind data
# Shape: (time_steps, grid_x, grid_y, grid_z, 3)
wind_data = np.random.randn(100, 20, 20, 10, 3).astype(np.float32)

# Save to file
np.savez_compressed("wind_data/custom_wind.npz", wind_data=wind_data)

# Load in application
wind_field.load_from_file("wind_data/custom_wind.npz")
```

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
