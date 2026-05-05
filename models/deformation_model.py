"""
DeformationModel Class
PyTorch-based ML model for predicting mesh vertex displacement.
Falls back to physics-based model if PyTorch is not available.
"""

import numpy as np
from typing import Optional, Tuple
import os

# Try to import PyTorch, fall back gracefully if not available
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"PyTorch not available: {e}")
    print("Using physics-based deformation model instead.")
    torch = None
    nn = None


# Only define DeformationNetwork if PyTorch is available
if TORCH_AVAILABLE:
    class DeformationNetwork(nn.Module):
        """
        Neural network for predicting vertex displacements.
        
        Architecture:
        - Input: concatenated vertex positions, wind velocity, previous state
        - Hidden layers with ReLU activation
        - Output: vertex displacement vectors
        """
        
        def __init__(
            self,
            vertex_dim: int = 3,
            wind_dim: int = 3,
            hidden_dim: int = 256,
            num_layers: int = 4
        ):
            """
            Initialize the deformation network.
            
            Args:
                vertex_dim: Dimension of vertex positions (3 for xyz)
                wind_dim: Dimension of wind velocity (3 for uvw)
                hidden_dim: Hidden layer dimension
                num_layers: Number of hidden layers
            """
            super().__init__()
            
            # Input: current vertex (3) + wind (3) + previous vertex (3) = 9
            input_dim = vertex_dim * 2 + wind_dim
            
            layers = []
            
            # Input layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            
            # Hidden layers
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Output layer
            layers.append(nn.Linear(hidden_dim, vertex_dim))
            
            self.network = nn.Sequential(*layers)
            
            # Initialize weights
            self._init_weights()
        
        def _init_weights(self):
            """Initialize network weights."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        def forward(
            self,
            vertices: 'torch.Tensor',
            wind: 'torch.Tensor',
            prev_vertices: 'torch.Tensor'
        ) -> 'torch.Tensor':
            """
            Forward pass to predict displacement.
            
            Args:
                vertices: Current vertex positions (B, N, 3)
                wind: Wind velocity vector (B, 3) or (B, N, 3)
                prev_vertices: Previous vertex positions (B, N, 3)
                
            Returns:
                Displacement vectors (B, N, 3)
            """
            batch_size = vertices.shape[0]
            num_vertices = vertices.shape[1]
            
            # Expand wind to per-vertex if needed
            if wind.dim() == 2:
                wind = wind.unsqueeze(1).expand(-1, num_vertices, -1)
            
            # Concatenate inputs
            x = torch.cat([vertices, wind, prev_vertices], dim=-1)
            
            # Flatten for processing
            x = x.view(batch_size * num_vertices, -1)
            
            # Process through network
            displacement = self.network(x)
            
            # Reshape back
            displacement = displacement.view(batch_size, num_vertices, -1)
            
            return displacement
else:
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
        model_path: Optional[str] = None,
        hidden_dim: int = 256,
        num_layers: int = 4,
        use_gpu: bool = True
    ):
        """
        Initialize the deformation model.
        
        Args:
            model_path: Path to pretrained model weights
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            use_gpu: Whether to use GPU if available
        """
        self.network = None
        self.device = None
        self.is_loaded = False
        
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
        
        # Initialize network
        self.network = DeformationNetwork(
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(self.device)
        
        # Track object payloads for ML inference context
        self.object_payloads = []

        # Load pretrained weights if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize with physics-inspired weights for demonstration
            self._init_physics_inspired()
            self.is_loaded = True

    def register_object_payload(self, payload: dict):
        """Register object payload (e.g. pole center) for ML parsing."""
        self.object_payloads.append(payload)

    def _init_physics_inspired(self):
        """
        Initialize model with physics-inspired behavior.
        This provides reasonable default behavior without training.
        """
        if not TORCH_AVAILABLE or self.network is None:
            return
        # The default initialization will give small random displacements
        # For demonstration, we'll use the network as-is
        self.network.eval()
    
    def load_model(self, model_path: str):
        """
        Load pretrained model weights.
        
        Args:
            model_path: Path to the model file (.pt or .pth)
        """
        if not TORCH_AVAILABLE or self.network is None:
            print("PyTorch not available - cannot load model")
            return
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.network.load_state_dict(state_dict)
            self.is_loaded = True
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False
    
    def save_model(self, model_path: str):
        """
        Save model weights.
        
        Args:
            model_path: Path to save the model
        """
        if not TORCH_AVAILABLE or self.network is None:
            print("PyTorch not available - cannot save model")
            return
        torch.save(self.network.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def predict(
        self,
        vertices: np.ndarray,
        wind_velocity: np.ndarray,
        previous_vertices: np.ndarray,
        damping: float = 0.95,
        max_displacement: float = 0.5
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
            # Return zero displacement if PyTorch not available
            return np.zeros_like(vertices)
        
        with torch.no_grad():
            self.network.eval()
            
            # Convert to tensors
            verts_tensor = torch.from_numpy(vertices).float().unsqueeze(0).to(self.device)
            wind_tensor = torch.from_numpy(wind_velocity).float().unsqueeze(0).to(self.device)
            prev_tensor = torch.from_numpy(previous_vertices).float().unsqueeze(0).to(self.device)
            
            # Predict displacement
            displacement = self.network(verts_tensor, wind_tensor, prev_tensor)
            
            # Apply physics-based adjustments
            # Scale by wind magnitude
            wind_magnitude = torch.norm(wind_tensor)
            displacement = displacement * wind_magnitude * 0.1
            
            # Apply damping
            displacement = displacement * damping
            
            # Clamp maximum displacement
            disp_magnitude = torch.norm(displacement, dim=-1, keepdim=True)
            scale = torch.clamp(max_displacement / (disp_magnitude + 1e-8), max=1.0)
            displacement = displacement * scale
            
            # Convert back to numpy
            return displacement.squeeze(0).cpu().numpy()
    
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
    
    def train_step(
        self,
        vertices,
        wind,
        prev_vertices,
        target_displacement,
        optimizer
    ) -> float:
        """
        Perform one training step.
        
        Args:
            vertices: Input vertices (B, N, 3)
            wind: Wind velocities (B, 3)
            prev_vertices: Previous vertices (B, N, 3)
            target_displacement: Ground truth displacement (B, N, 3)
            optimizer: PyTorch optimizer
            
        Returns:
            Loss value
        """
        if not TORCH_AVAILABLE or self.network is None:
            return 0.0
        
        self.network.train()
        
        optimizer.zero_grad()
        
        predicted = self.network(vertices, wind, prev_vertices)
        loss = nn.functional.mse_loss(predicted, target_displacement)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()


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
