"""
SimulationController Class
Manages the simulation loop for wind deformation.
"""

import numpy as np
from typing import Optional, Callable, List
import time
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from renderer.scene import Scene
from models.deformation_model import DeformationModel
from models import config as cfg
from objects.object_mesh import ObjectMesh


class SimulationController(QObject):
    """
    Controls the simulation loop for wind-based mesh deformation.
    
    Manages:
    - Simulation timing and updates
    - Wind field time advancement
    - ML model predictions
    - Mesh deformation updates
    
    Signals:
        simulation_updated: Emitted after each simulation step
        simulation_started: Emitted when simulation starts
        simulation_stopped: Emitted when simulation stops
    """
    
    simulation_updated = pyqtSignal()
    simulation_started = pyqtSignal()
    simulation_stopped = pyqtSignal()
    
    def __init__(
        self,
        scene: Scene,
        deformation_model: Optional[DeformationModel] = None,
    ):
        """
        Initialize the simulation controller.

        The tick rate is locked to ``cfg.FPS`` — the rate the deformation
        model was trained at. Decoupling the simulation loop from render
        FPS would feed the Verlet integrator with mismatched dt and cause
        runaway vertex displacement.

        Args:
            scene: The scene to simulate
            deformation_model: ML model for deformation (creates default if None)
        """
        super().__init__()

        self.scene = scene
        self.deformation_model = deformation_model or DeformationModel()

        # ML Object tracking
        self.object_payloads = []

        # Timing — locked to the trained model rate.
        self.target_fps = cfg.FPS
        self._dt = cfg.DELTA_T
        self._last_update_time = 0.0
        
        # Simulation state
        self._is_running = False
        self._is_paused = False
        self._frame_count = 0
        self._simulation_time = 0.0
        
        # Timer for simulation loop
        self._timer = QTimer()
        self._timer.timeout.connect(self._simulation_step)
        
        # Performance tracking
        self._fps_samples: List[float] = []
        self._last_fps_update = 0.0
        self._current_fps = 0.0
        
        # Callbacks
        self._update_callbacks: List[Callable] = []

    def register_object_payload(self, payload: dict):
        """
        Record dropped object information for the ML pipeline.
        This provides the model with object-specific geometry facts like pole_center.
        """
        self.object_payloads.append(payload)
        if hasattr(self.deformation_model, 'register_object_payload'):
            self.deformation_model.register_object_payload(payload)

    def start(self):
        """Start the simulation."""
        if self._is_running:
            return
        
        self._is_running = True
        self._is_paused = False
        self._last_update_time = time.time()
        self._timer.start(int(1000 / self.target_fps))

        self.simulation_started.emit()
    
    def stop(self):
        """Stop the simulation."""
        self._is_running = False
        self._timer.stop()
        self.simulation_stopped.emit()
    
    def pause(self):
        """Pause the simulation."""
        if self._is_running:
            self._is_paused = True
            self._timer.stop()
    
    def resume(self):
        """Resume the simulation."""
        if self._is_running and self._is_paused:
            self._is_paused = False
            self._last_update_time = time.time()
            self._timer.start(int(1000 / self.target_fps))
    
    def toggle_pause(self):
        """Toggle pause state."""
        if self._is_paused:
            self.resume()
        else:
            self.pause()
    
    def reset(self):
        """Reset the simulation to initial state."""
        self.stop()
        
        # Reset all objects to original positions
        self.scene.reset_all_objects()
        
        # Reset wind field time
        self.scene.wind_field.reset_time()
        
        # Reset counters
        self._frame_count = 0
        self._simulation_time = 0.0

        self.simulation_updated.emit()
    
    def _simulation_step(self):
        """Perform one simulation step."""
        if not self._is_running or self._is_paused:
            return
        
        current_time = time.time()
        dt = current_time - self._last_update_time
        self._last_update_time = current_time

        # Tick rate is locked to cfg.FPS, so advance wind exactly one step
        # per tick — keeps wind dt aligned with the model's training dt.
        self.scene.wind_field.advance_time(1)

        # Update each object
        for obj in self.scene.objects:
            self._update_object(obj)
        
        # Update counters
        self._frame_count += 1
        self._simulation_time += dt
        
        # Update FPS tracking
        self._update_fps(dt)
        
        # Emit update signal
        self.simulation_updated.emit()
        
        # Call registered callbacks
        for callback in self._update_callbacks:
            callback()
    
    def _update_object(self, obj: ObjectMesh):
        """
        Update a single object's deformation using the MeshGraphNet model.

        Args:
            obj: The object to update
        """

        if not self.deformation_model.is_loaded:
            return

        # Get rest lengths of the edges
        rest_lengths = obj.rest_lengths

        # Get wind at object position (world frame, Z-up).
        # flag.obj vertices are Z-up (Y=0 everywhere), so world-frame wind is passed
        # directly to predict() without any axis remap.
        wind_velocity = self.scene.get_wind_at_object(obj)

        # Get current vertex data
        vertices = obj.current_vertices.copy()
        previous_vertices = obj.previous_vertices.copy()

        # Run MeshGraphNet inference — returns next-frame positions, or None
        # if the object's topology does not match the trained mesh.
        next_vertices = self.deformation_model.predict(
            vertices,
            wind_velocity,
            previous_vertices,
            rest_lengths,
        )
        if next_vertices is None:
            return

        # Apply object-specific constraints and commit
        new_vertices = self._apply_constraints(obj, next_vertices)
        obj.update_vertices(new_vertices)
    
    def _apply_constraints(
        self,
        obj: ObjectMesh,
        new_vertices: np.ndarray
    ) -> np.ndarray:
        """
        Apply constraints to vertex positions.
        
        Args:
            obj: The object mesh
            new_vertices: Proposed new vertex positions
            
        Returns:
            Constrained vertex positions
        """
        # Different constraints based on object type
        if obj.name.lower() == 'flag':
            # Fix the left edge (attached to pole)
            fixed_mask = obj.vertices[:, 0] < 0.1
            new_vertices[fixed_mask] = obj.vertices[fixed_mask]
        
        elif obj.name.lower() == 'cloth':
            # Fix corner vertices
            corners = [
                (obj.vertices[:, 0].min(), obj.vertices[:, 2].min()),
                (obj.vertices[:, 0].max(), obj.vertices[:, 2].min()),
            ]
            for cx, cz in corners:
                mask = (
                    (np.abs(obj.vertices[:, 0] - cx) < 0.2) &
                    (np.abs(obj.vertices[:, 2] - cz) < 0.2)
                )
                new_vertices[mask] = obj.vertices[mask]
        
        elif obj.name.lower() == 'tree':
            # Fix the base (roots)
            fixed_mask = obj.vertices[:, 1] < 0.5
            new_vertices[fixed_mask] = obj.vertices[fixed_mask]
        
        elif obj.name.lower() == 'pole':
            # Poles don't deform much
            new_vertices = obj.vertices.copy()
        
        return new_vertices
    
    def _update_fps(self, dt: float):
        """Update FPS tracking."""
        self._fps_samples.append(1.0 / max(dt, 0.001))
        
        # Keep last 30 samples
        if len(self._fps_samples) > 30:
            self._fps_samples.pop(0)
        
        # Update FPS every 0.5 seconds
        current_time = time.time()
        if current_time - self._last_fps_update > 0.5:
            self._current_fps = np.mean(self._fps_samples)
            self._last_fps_update = current_time
    
    def add_update_callback(self, callback: Callable):
        """
        Add a callback to be called after each update.
        
        Args:
            callback: Function to call
        """
        self._update_callbacks.append(callback)
    
    def remove_update_callback(self, callback: Callable):
        """
        Remove an update callback.
        
        Args:
            callback: Function to remove
        """
        if callback in self._update_callbacks:
            self._update_callbacks.remove(callback)
    
    @property
    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self._is_running
    
    @property
    def is_paused(self) -> bool:
        """Check if simulation is paused."""
        return self._is_paused
    
    @property
    def frame_count(self) -> int:
        """Get total frame count."""
        return self._frame_count
    
    @property
    def simulation_time(self) -> float:
        """Get total simulation time."""
        return self._simulation_time
    
    @property
    def current_fps(self) -> float:
        """Get current FPS."""
        return self._current_fps
    
    def get_stats(self) -> dict:
        """
        Get simulation statistics.
        
        Returns:
            Dictionary with simulation stats
        """
        return {
            'running': self._is_running,
            'paused': self._is_paused,
            'frame_count': self._frame_count,
            'simulation_time': self._simulation_time,
            'fps': self._current_fps,
            'object_count': len(self.scene.objects),
            'wind_time': self.scene.wind_field.current_time,
        }
