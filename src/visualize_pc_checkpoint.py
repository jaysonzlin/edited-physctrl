"""Generate a predicted-versus-ground-truth point-cloud MP4 from a PC DiT checkpoint."""

import argparse
from pathlib import Path

import h5py
import torch
from diffusers import DDIMScheduler
from omegaconf import OmegaConf
from safetensors.torch import load_file

from model.pc_dit import PCDiT
from options import PCTrainingConfig
from pipeline_pc import PCPipeline
from visualize_pc import save_pointcloud_comparison_mp4


def resolve_sample_path(sample):
    """Accept either a sample directory or its ``pc.hdf5`` file."""
    sample_path = Path(sample)
    if sample_path.is_dir():
        sample_path = sample_path / "pc.hdf5"
    if not sample_path.is_file():
        raise FileNotFoundError(f"Sample HDF5 file does not exist: {sample_path}")
    return sample_path


def default_output_path(checkpoint_dir, sample_path):
    return Path(checkpoint_dir).parent / f"{Path(sample_path).parent.name}-comparison.mp4"


def combine_initial_and_future(init_pc, predicted_future):
    """Prepend the initial frame after aligning it with the sampled trajectory device."""
    initial_frame = init_pc.to(predicted_future.device).unsqueeze(1)
    return torch.cat([initial_frame, predicted_future], dim=1)


def load_checkpoint_model(checkpoint_dir, config_path):
    config = OmegaConf.merge(OmegaConf.structured(PCTrainingConfig), OmegaConf.load(config_path))
    model = PCDiT(config.pc_size, 48, config.model_config)

    checkpoint_dir = Path(checkpoint_dir)
    safetensors_path = checkpoint_dir / "model.safetensors"
    pytorch_path = checkpoint_dir / "pytorch_model.bin"
    if safetensors_path.is_file():
        state_dict = load_file(safetensors_path)
    elif pytorch_path.is_file():
        state_dict = torch.load(pytorch_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin found in checkpoint: {checkpoint_dir}"
        )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main(args):
    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    sample_path = resolve_sample_path(args.sample)
    config_path = Path(args.config) if args.config else checkpoint_dir.parent / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config does not exist: {config_path}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_path = Path(args.output) if args.output else default_output_path(checkpoint_dir, sample_path)
    model = load_checkpoint_model(checkpoint_dir, config_path)

    with h5py.File(sample_path, "r") as source:
        ground_truth = source["point_cloud"][:]
        init_pc = torch.from_numpy(ground_truth[0]).float().unsqueeze(0)
        linear_velocity = torch.from_numpy(source["initial_linear_velocity"][:]).float().unsqueeze(0)
        angular_velocity = torch.from_numpy(source["initial_angular_velocity"][:]).float().unsqueeze(0)

    pipeline = PCPipeline(
        model,
        DDIMScheduler(num_train_timesteps=1000, prediction_type="sample", clip_sample=False),
    )
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
    predicted_future = pipeline(
        init_pc=init_pc,
        initial_linear_velocity=linear_velocity,
        initial_angular_velocity=angular_velocity,
        device=device,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    predicted_sequence = combine_initial_and_future(init_pc, predicted_future).squeeze(0).cpu().numpy()
    save_pointcloud_comparison_mp4(predicted_sequence, ground_truth, output_path, fps=args.fps)
    print(f"Saved comparison video to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint-<step> directory")
    parser.add_argument("--sample", required=True, help="Path to sample directory or pc.hdf5 file")
    parser.add_argument("--output", help="MP4 output path; defaults next to the training output")
    parser.add_argument("--config", help="Run config path; defaults to the checkpoint parent config.yaml")
    parser.add_argument("--device", help="Torch device; defaults to CUDA when available")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=12)
    main(parser.parse_args())
