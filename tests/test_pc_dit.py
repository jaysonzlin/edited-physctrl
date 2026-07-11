import torch
from diffusers import DDIMScheduler
from omegaconf import OmegaConf

from model.pc_dit import PCDiT
from pipeline_pc import PCPipeline


def make_model():
    return PCDiT(
        n_points=8,
        n_frames=48,
        model_config=OmegaConf.create(
            {
                "latent_dim": 64,
                "n_layers": 1,
                "frame_cond": True,
                "point_embed": False,
                "pred_offset": True,
                "transformer_block": "SpatialTemporalTransformerBlock",
            }
        ),
    )


def test_pc_dit_preserves_singleton_object_axis():
    model = make_model()

    output = model(
        torch.randn(2, 48, 1, 8, 3),
        torch.tensor([1, 2]),
        torch.randn(2, 1, 8, 3),
        torch.randn(2, 1, 3),
        torch.randn(2, 1, 3),
    )

    assert output.shape == (2, 48, 1, 8, 3)


def test_pipeline_returns_all_future_frames():
    model = make_model()
    pipeline = PCPipeline(model, DDIMScheduler(num_train_timesteps=10, prediction_type="sample"))

    output = pipeline(
        init_pc=torch.randn(1, 1, 8, 3),
        initial_linear_velocity=torch.randn(1, 1, 3),
        initial_angular_velocity=torch.randn(1, 1, 3),
        device="cpu",
        batch_size=1,
        num_inference_steps=2,
        guidance_scale=1.0,
        generator=torch.Generator().manual_seed(0),
    )

    assert output.shape == (1, 48, 1, 8, 3)
