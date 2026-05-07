"""
DeformationModel Class
PyTorch-based ML model for predicting mesh vertex displacement.
Falls back to physics-based model if PyTorch is not available.
"""

from typing import Optional, Tuple
import os
import numpy as np
import os
import time
from .load_model import load_model

from . import config as cfg
# Try to import PyTorch, fall back gracefully if not available
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"PyTorch not available: {e}")
    print("Using physics-based deformation model instead.")
    torch = None
    nn = None


if not TORCH_AVAILABLE:
    # Dummy class when PyTorch is not available
    DeformationNetwork = None

class DeformationModel:
    """
    Manages the deformation prediction model.
    
    Handles model loading, inference, and GPU acceleration.
    Falls back to physics-based model if PyTorch is not available.
    
    Attributes:
        network: The neural network (None if PyTorch unavailable)
        device: Computation device (CPU/GPU)
        is_loaded: Whether a pretrained model is loaded
    """
    
    def __init__(
        self,
        model_ver=None,
        batch_pin_mask=None,
        edge_index=None,
        model_path: Optional[str] = None,
        use_gpu: bool = True,
    ):
        """
        Initialize the deformation model.
        
        Args:
            model_path: Path to pretrained model weights
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            use_gpu: Whether to use GPU if available
        """
        self.model = None
        self.device = None
        self.is_loaded = False
        
        # self.model_path = 
        
        if not TORCH_AVAILABLE:
            print("PyTorch not available - ML model disabled, using physics model")
            return
        
        # Set device
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        self.edge_index = torch.from_numpy(np.load(cfg.TOPOLOGY_PATH)).long().to(self.device)
        self.batch_pin_mask = cfg.PIN_MASK.to(self.device)
        
        # Track object payloads for ML inference context
        self.object_payloads = []

        # 2. Load Model
        if self.model is None:
            self.model = load_model(self.device)
            model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
    

        else:
            # Initialize with physics-inspired weights for demonstration
            # self._init_physics_inspired()
            # self.is_loaded = True
            print ("error")

    def register_object_payload(self, payload: dict):
        """Register object payload (e.g. pole center) for ML parsing."""
        self.object_payloads.append(payload)

    def integrate(pos, vel, accel, dt):
        """
        P_{t+1} = P_t + V_t*dt + 0.5*A*dt^2
        """
        new_pos = pos + (vel * dt) + (0.5 * accel * (dt ** 2))
        new_vel = vel + (accel * dt)
        return new_pos, new_vel

    def _init_physics_inspired(self):
        """
        Initialize model with physics-inspired behavior.
        This provides reasonable default behavior without training.
        """
        if not TORCH_AVAILABLE or self.network is None:
            return
        # The default initialization will give small random displacements
        # For demonstration, we'll use the network as-is
        self.model.eval()
    
    def predict(
        self,
        vertices: np.ndarray,
        curr_wind_raw: np.ndarray,        
        previous_vertices: np.ndarray,
        rest_lengths
    ) -> np.ndarray:
        """
        Predict vertex displacement based on wind and previous state.
        
        Args:
            vertices: Current vertex positions (N, 3)
            wind_velocity: Wind velocity vector (3,)
            previous_vertices: Previous vertex positions (N, 3)
            damping: Damping factor for displacement
            max_displacement: Maximum allowed displacement magnitude
            
        Returns:
            Displacement vectors (N, 3)
        """
        if not TORCH_AVAILABLE or self.network is None:
            print("no ml model")
            return
        

        def to_tensor(val):
            if torch.is_tensor(val): return val.to(device=self.device, dtype=torch.float32)
            return torch.tensor(val, device=self.device, dtype=torch.float32)
    
        with torch.no_grad():
            self.network.eval()

            # 1. Calculate Kinematic Velocity (matching train loop)
            curr_pos = vertices
            prev_pos = previous_vertices

            curr_vel = (curr_pos - prev_pos)
            curr_vel_scaled = curr_vel * cfg.VEL_UP  # scale up for stability (matches training)

            x = curr_pos[:, 0]
            y = curr_pos[:, 1]
            z = curr_pos[:, 2]
            
            ix = (x >= 0).long()
            iy = (y >= 0).long()
            iz = (z >= 0).long()
            
            cube_index = ix*4 + iy*2 + iz
            cube_index_expanded = cube_index.unsqueeze(-1).expand(-1, 3)
            
            # Gather local wind (Note: dim=0 here because raw is just [8, 3], not batched)
            wind_expanded = torch.gather(curr_wind_raw, 0, cube_index_expanded)
            wind_expanded_scaled = wind_expanded / cfg.WIND_DOWN  # scale down for stability (matches training)
            
            # 3. BUILD NODE FEATURES (Velocity + Wind + Pin Mask)
            node_features = torch.cat([curr_vel_scaled, wind_expanded_scaled, self.batch_pin_mask], dim=-1)

            # 4. BUILD EDGE FEATURES (Vector + Magnitude + Rel Vel + Rest Length)
            row, col = self.edge_index
            
            # A. Spatial Displacement
            x_ij = curr_pos[row] - curr_pos[col]
            x_ij_norm = torch.norm(x_ij, p=2, dim=-1, keepdim=True)
            
            # B. Relative Velocity (Damping)
            v_ij = curr_vel_scaled[row] - curr_vel_scaled[col]

            # C. Rest Lengths (Tension)
            # we must unsqueeze it to [E, 1] before concatenating.
            rest_lengths_expanded = rest_lengths.unsqueeze(-1)

                # Concatenate into 8D edge features
            edge_attr = torch.cat([x_ij, x_ij_norm, v_ij, rest_lengths_expanded], dim=-1)
            
            # --- B. Inference ---
            with torch.no_grad():
                pred_norm_acc = self.model(node_features, self.edge_index, edge_attr)
            
            # --- C. DENORMALIZE THE PREDICTION ---
            pred_real_acc = pred_norm_acc * std_acc + mean_acc
            
            # ==========================================
            # ENFORCE BOUNDARY CONDITIONS (The Secret!)
            # ==========================================
            # The network predicted garbage for the pinned nodes. Overwrite it to exactly 0.0.
            H, W = cfg.HEIGHT, cfg.WIDTH
            pinned_indices = [r * W for r in range(H)]
            
            # Force the real acceleration of pinned nodes to be zero
            # pred_real_acc shape is [N, 3]
            pred_real_acc[pinned_indices, :] = 0.0
            
            # --- C. Physics Integration ---
            # Calculate instantaneous kinematic velocity from the two most recent frames in the buffer
            kinematic_vel = (curr_pos - prev_pos) / cfg.DELTA_T
            
            if cfg.TARGET_TYPE in ["accelerations", "acc_new"]:
                next_pos, _ = self.integrate(curr_pos, kinematic_vel, pred_real_acc, cfg.DELTA_T)
            elif cfg.TARGET_TYPE == "acc":
                next_pos = (2 * curr_pos) - prev_pos + pred_real_acc
            elif cfg.TARGET_TYPE == "displacements":
                disp = pred_real_acc
                next_pos = curr_pos + disp

            
            # Convert back to numpy
            return next_pos
    
    def predict_batch(
        self,
        vertices_batch: np.ndarray,
        wind_velocities: np.ndarray,
        previous_vertices_batch: np.ndarray
    ) -> np.ndarray:
        """
        Predict displacements for multiple objects.
        
        Args:
            vertices_batch: Vertex positions (B, N, 3)
            wind_velocities: Wind velocities (B, 3)
            previous_vertices_batch: Previous vertices (B, N, 3)
            
        Returns:
            Displacement vectors (B, N, 3)
        """
        if not TORCH_AVAILABLE or self.network is None:
            return np.zeros_like(vertices_batch)
        
        with torch.no_grad():
            self.network.eval()
            
            verts_tensor = torch.from_numpy(vertices_batch).float().to(self.device)
            wind_tensor = torch.from_numpy(wind_velocities).float().to(self.device)
            prev_tensor = torch.from_numpy(previous_vertices_batch).float().to(self.device)
            
            displacement = self.network(verts_tensor, wind_tensor, prev_tensor)
            
            return displacement.cpu().numpy()
    
class SimplifiedPhysicsModel:
    """
    A simplified physics-based model for wind deformation.
    Used as fallback or for comparison with ML model.
    """
    
    def __init__(
        self,
        stiffness: float = 0.5,
        damping: float = 0.9,
        mass: float = 1.0
    ):
        """
        Initialize physics parameters.
        
        Args:
            stiffness: Spring stiffness coefficient
            damping: Velocity damping factor
            mass: Vertex mass
        """
        self.stiffness = stiffness
        self.damping = damping
        self.mass = mass
        self.velocities: Optional[np.ndarray] = None
    
    def reset(self, num_vertices: int):
        """Reset velocities for new simulation."""
        self.velocities = np.zeros((num_vertices, 3), dtype=np.float32)
    
    def compute_displacement(
        self,
        vertices: np.ndarray,
        original_vertices: np.ndarray,
        wind_velocity: np.ndarray,
        dt: float = 0.016
    ) -> np.ndarray:
        """
        Compute displacement using simplified physics.
        
        Args:
            vertices: Current vertex positions
            original_vertices: Rest positions
            wind_velocity: Wind velocity vector
            dt: Time step
            
        Returns:
            Displacement vectors
        """
        if self.velocities is None or len(self.velocities) != len(vertices):
            self.reset(len(vertices))
        
        # Wind force (simplified)
        wind_force = wind_velocity * 0.5
        
        # Spring force towards original position
        displacement = vertices - original_vertices
        spring_force = -self.stiffness * displacement
        
        # Total acceleration
        acceleration = (wind_force + spring_force) / self.mass
        
        # Update velocity with damping
        self.velocities = self.velocities * self.damping + acceleration * dt
        
        # Compute displacement
        return self.velocities * dt
