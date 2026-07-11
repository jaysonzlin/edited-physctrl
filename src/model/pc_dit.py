"""Diffusion transformer for single-object point-cloud trajectories."""

import torch
import torch.nn as nn

from model.spacetime import PointEmbed, SpaitalTemporalTransformer


class PCDiT(nn.Module):
    """Predict 48 future point-cloud frames from initial state conditions."""

    def __init__(self, n_points, n_frames, model_config):
        super().__init__()
        if model_config.latent_dim % 64:
            raise ValueError("model_config.latent_dim must be divisible by 64")

        self.n_points = n_points
        self.n_frames = n_frames
        self.latent_dim = model_config.latent_dim
        self.frame_cond = model_config.get("frame_cond", True)
        self.pred_offset = model_config.get("pred_offset", True)
        self.cond_frame = 1 if self.frame_cond else 0

        if model_config.get("point_embed", True):
            self.input_encoder = PointEmbed(dim=self.latent_dim)
        else:
            self.input_encoder = nn.Linear(3, self.latent_dim)

        self.linear_velocity_encoder = nn.Linear(3, self.latent_dim)
        self.angular_velocity_encoder = nn.Linear(3, self.latent_dim)
        self.dit = SpaitalTemporalTransformer(
            sample_points=n_points,
            sample_frames=n_frames + self.cond_frame,
            in_channels=3,
            num_layers=model_config.n_layers,
            num_attention_heads=self.latent_dim // 64,
            time_embed_dim=self.latent_dim,
            cond_seq_length=2,
            cond_seq_length_t=self.cond_frame,
            transformer_block=model_config.transformer_block,
        )

    def forward(
        self,
        x,
        timesteps,
        init_pc,
        initial_linear_velocity,
        initial_angular_velocity,
        null_emb=None,
    ):
        """Return x0 point positions with shape ``(B, F, 1, N, 3)``."""
        self._validate_inputs(x, init_pc, initial_linear_velocity, initial_angular_velocity)
        batch_size, n_frames, _, n_points, _ = x.shape
        x = x.squeeze(2)
        init_pc = init_pc.squeeze(1)
        linear_velocity = initial_linear_velocity.squeeze(1)
        angular_velocity = initial_angular_velocity.squeeze(1)

        condition_tokens = torch.stack(
            [
                self.linear_velocity_encoder(linear_velocity),
                self.angular_velocity_encoder(angular_velocity),
            ],
            dim=1,
        )
        if null_emb is not None:
            condition_tokens = condition_tokens * null_emb

        if self.frame_cond:
            x = torch.cat([init_pc.unsqueeze(1), x], dim=1)
        hidden_states = self.input_encoder(x.reshape(-1, n_points, 3)).reshape(
            batch_size, -1, n_points, self.latent_dim
        )
        output = self.dit(hidden_states, condition_tokens, timesteps).reshape(
            batch_size, -1, n_points, 3
        )
        output = output[:, self.cond_frame :]
        if self.pred_offset:
            output = output + init_pc.unsqueeze(1)
        return output.unsqueeze(2)

    def _validate_inputs(self, x, init_pc, initial_linear_velocity, initial_angular_velocity):
        if x.ndim != 5 or x.shape[2] != 1 or x.shape[-1] != 3:
            raise ValueError("x must have shape (B, F, 1, N, 3)")
        if x.shape[1] != self.n_frames or x.shape[3] != self.n_points:
            raise ValueError(f"x must have shape (B, {self.n_frames}, 1, {self.n_points}, 3)")
        if init_pc.shape != (x.shape[0], 1, self.n_points, 3):
            raise ValueError(f"init_pc must have shape (B, 1, {self.n_points}, 3)")
        for name, velocity in (
            ("initial_linear_velocity", initial_linear_velocity),
            ("initial_angular_velocity", initial_angular_velocity),
        ):
            if velocity.shape != (x.shape[0], 1, 3):
                raise ValueError(f"{name} must have shape (B, 1, 3)")
