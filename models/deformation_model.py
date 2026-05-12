"""
DeformationModel Class
PyTorch-based ML model (MeshGraphNet) for predicting mesh vertex deformation.
"""

from typing import Optional
import os
import numpy as np
import torch

from . import config as cfg
from .load_model import load_model


class DeformationModel:
    """
    Manages the MeshGraphNet deformation prediction model.

    Loads pretrained weights from best_model.pth and runs inference to
    produce next-frame vertex positions for the trained flag topology.
    """

    def __init__(self, use_gpu: bool = True):
        self.model = None
        self.is_loaded = False
        self.object_payloads = []

        # Device
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        # Trained topology + pin mask (matches the bundled flag.obj: HEIGHT*WIDTH vertices)
        self.edge_index = torch.from_numpy(np.load(cfg.TOPOLOGY_PATH)).long().to(self.device)
        self.batch_pin_mask = cfg.PIN_MASK.to(self.device)
        self.pin_mask_bool = self.batch_pin_mask.squeeze(-1).bool()
        self.expected_num_vertices = cfg.NUM_VERTICES

        # Strain projection (PBD) — clamp edges to within MAX_STRAIN × rest_length
        # after the unbounded model decoder, otherwise OOD wind can blow vertices
        # off the mesh. Jacobi-style updates need to be scaled by 1/degree so
        # simultaneous corrections from a vertex's ~8 neighbours don't overshoot.
        self.max_strain = 1.10
        self.projection_iters = 12
        row, _ = self.edge_index
        deg = torch.bincount(row, minlength=cfg.NUM_VERTICES).clamp_min(1).float()
        self.inv_degree = (1.0 / deg).unsqueeze(-1)

        # Load model
        self.model = load_model(self.device)
        model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        self.is_loaded = True

    def register_object_payload(self, payload: dict):
        """Register object payload (e.g. pole center) for ML parsing."""
        self.object_payloads.append(payload)

    @staticmethod
    def integrate(pos, vel, accel, dt):
        """P_{t+1} = P_t + V_t*dt + 0.5*A*dt^2"""
        new_pos = pos + (vel * dt) + (0.5 * accel * (dt ** 2))
        new_vel = vel + (accel * dt)
        return new_pos, new_vel

    def predict(
        self,
        vertices: np.ndarray,
        wind_velocity: np.ndarray,
        previous_vertices: np.ndarray,
        rest_lengths,
    ) -> Optional[np.ndarray]:
        """
        Predict next-frame vertex positions.

        Args:
            vertices:           Current vertex positions   (N, 3) numpy
            wind_velocity:      Wind vector(s) — either (3,) or (8, 3) numpy
            previous_vertices:  Previous vertex positions  (N, 3) numpy
            rest_lengths:       Edge rest lengths          (E,)   torch tensor

        Returns:
            next_pos: (N, 3) numpy array of next-frame positions, or None if
                      the input topology does not match the trained mesh.
        """
        if vertices.shape[0] != self.expected_num_vertices:
            return None

        with torch.no_grad():
            # ---- Convert inputs to tensors on device ----
            curr_pos = torch.as_tensor(vertices, device=self.device, dtype=torch.float32)
            prev_pos = torch.as_tensor(previous_vertices, device=self.device, dtype=torch.float32)

            wind_arr = np.asarray(wind_velocity, dtype=np.float32)
            if wind_arr.ndim == 1:
                # Broadcast a single uniform wind sample to the 8 octants the model expects
                wind_arr = np.tile(wind_arr.reshape(1, 3), (8, 1))
            curr_wind_raw = torch.as_tensor(wind_arr, device=self.device, dtype=torch.float32)

            if torch.is_tensor(rest_lengths):
                rest_lengths_t = rest_lengths.to(device=self.device, dtype=torch.float32)
            else:
                rest_lengths_t = torch.as_tensor(rest_lengths, device=self.device, dtype=torch.float32)

            # ---- Kinematic velocity (matches train loop) ----
            curr_vel = curr_pos - prev_pos
            curr_vel_scaled = curr_vel * cfg.VEL_UP

            # ---- Per-vertex wind via 8-octant lookup ----
            x = curr_pos[:, 0]
            y = curr_pos[:, 1]
            z = curr_pos[:, 2]

            ix = (x >= 0).long()
            iy = (y >= 0).long()
            iz = (z >= 0).long()

            cube_index = ix * 4 + iy * 2 + iz
            cube_index_expanded = cube_index.unsqueeze(-1).expand(-1, 3)

            wind_expanded = torch.gather(curr_wind_raw, 0, cube_index_expanded)
            wind_expanded_scaled = wind_expanded / cfg.WIND_DOWN

            # ---- Node features: [vel(3), wind(3), pin_mask(1)] = 7 ----
            node_features = torch.cat(
                [curr_vel_scaled, wind_expanded_scaled, self.batch_pin_mask], dim=-1
            )

            # ---- Edge features: [rel_pos(3), rel_pos_norm(1), rel_vel(3), rest_len(1)] = 8 ----
            row, col = self.edge_index
            x_ij = curr_pos[row] - curr_pos[col]
            x_ij_norm = torch.norm(x_ij, p=2, dim=-1, keepdim=True)
            v_ij = curr_vel_scaled[row] - curr_vel_scaled[col]
            rest_lengths_expanded = rest_lengths_t.unsqueeze(-1)

            edge_attr = torch.cat([x_ij, x_ij_norm, v_ij, rest_lengths_expanded], dim=-1)

            # ---- Inference ----
            # Encoder applies LayerNorm internally; decoder output is in physical units.
            pred = self.model(node_features, self.edge_index, edge_attr)

            # ---- Enforce pinned-node boundary condition ----
            H, W = cfg.HEIGHT, cfg.WIDTH
            pinned_indices = [r * W for r in range(H)]
            pred[pinned_indices, :] = 0.0

            # ---- Physics integration ----
            kinematic_vel = (curr_pos - prev_pos) / cfg.DELTA_T

            if cfg.TARGET_TYPE in ("accelerations", "acc_new"):
                next_pos, _ = self.integrate(curr_pos, kinematic_vel, pred, cfg.DELTA_T)
            elif cfg.TARGET_TYPE == "acc":
                next_pos = (2 * curr_pos) - prev_pos + pred
            elif cfg.TARGET_TYPE == "displacements":
                next_pos = curr_pos + pred
            else:
                raise ValueError(f"Unknown TARGET_TYPE in config: {cfg.TARGET_TYPE}")

            # Re-pin to be safe after integration
            next_pos[pinned_indices, :] = curr_pos[pinned_indices, :]

            # ---- Cap per-tick displacement ----
            # The decoder is unbounded; under OOD wind a single forward pass can
            # produce predictions that translate vertices by several metres in
            # one tick. Clamp the per-tick movement of each free vertex to a
            # fraction of the mean rest length so the mesh cannot blow up
            # before the strain projection has a chance to act.
            max_step = 0.5 * rest_lengths_t.mean()
            step = next_pos - curr_pos
            step_norm = torch.norm(step, dim=-1, keepdim=True).clamp_min(1e-8)
            scale = torch.minimum(torch.ones_like(step_norm), max_step / step_norm)
            next_pos = curr_pos + step * scale
            next_pos[pinned_indices, :] = curr_pos[pinned_indices, :]

            # ---- XPBD-style edge-length projection ----
            # The decoder is unbounded, so without this the mesh can drift
            # arbitrarily under OOD wind. Project any over-stretched edges back
            # to MAX_STRAIN × rest_length, splitting the correction between free
            # endpoints. Pinned vertices stay fixed.
            row, col = self.edge_index
            max_len = rest_lengths_t * self.max_strain
            free_mask = (~self.pin_mask_bool).float().unsqueeze(-1)
            w_row = free_mask[row]
            w_col = free_mask[col]
            w_sum = (w_row + w_col).clamp_min(1.0)

            for _ in range(self.projection_iters):
                delta = next_pos[row] - next_pos[col]
                length = torch.norm(delta, dim=-1).clamp_min(1e-8)
                excess = (length - max_len).clamp_min(0.0)
                if excess.max() == 0:
                    break
                direction = delta / length.unsqueeze(-1)
                # 0.5: edge_index is bidirectional, every undirected edge
                # appears as both (i,j) and (j,i).
                # inv_degree[v]: Jacobi damping — vertex v receives corrections
                # from all its incident edges in the same pass, so divide by
                # the vertex's degree to prevent overshoot.
                correction = 0.5 * excess.unsqueeze(-1) * direction
                next_pos.index_add_(0, row, -correction * (w_row / w_sum) * self.inv_degree[row])
                next_pos.index_add_(0, col,  correction * (w_col / w_sum) * self.inv_degree[col])

            next_pos[pinned_indices, :] = curr_pos[pinned_indices, :]

            return next_pos.detach().cpu().numpy()
