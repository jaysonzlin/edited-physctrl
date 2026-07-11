"""Train a point-cloud DiT on raw single-object trajectory samples."""

import argparse
import math
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import is_wandb_available
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from dataset.pc_dataset import PCDataset
from model.pc_dit import PCDiT
from options import PCTrainingConfig
from pipeline_pc import PCPipeline
from visualize_pc import save_pointcloud_comparison_mp4, save_pointcloud_mp4


def initialize_trackers(accelerator, args):
    """Initialize the configured experiment tracker on the main process."""
    if args.report_to is None:
        return
    if args.report_to == "wandb" and not is_wandb_available():
        raise ImportError("Install Weights & Biases with `pip install wandb` to use report_to: wandb.")

    tracker_config = OmegaConf.to_container(args, resolve=True) if OmegaConf.is_config(args) else vars(args)
    accelerator.init_trackers(args.tracker_project_name, config=tracker_config)


def should_save_visualization(epoch):
    """Return whether the zero-indexed epoch completes a 100-epoch interval."""
    return (epoch + 1) % 100 == 0


def load_resume_checkpoint(accelerator, checkpoint, output_dir):
    """Restore an Accelerate checkpoint and return its completed optimizer step."""
    if checkpoint is None:
        return 0

    if checkpoint == "latest":
        candidates = [path for path in Path(output_dir).glob("checkpoint-*") if path.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No checkpoint-* directories found in {output_dir}")
        checkpoint_path = max(candidates, key=lambda path: int(path.name.removeprefix("checkpoint-")))
    else:
        checkpoint_path = Path(checkpoint)

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")
    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint_path.name)
    if match is None:
        raise ValueError(f"Checkpoint directory must be named checkpoint-<step>, got {checkpoint_path.name}")

    accelerator.load_state(str(checkpoint_path))
    return int(match.group(1))


def main(args):
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / args.vis_dir
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=output_dir, logging_dir=output_dir / args.logging_dir),
    )
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(args, output_dir / "config.yaml")
    initialize_trackers(accelerator, args)

    model = PCDiT(args.pc_size, 48, args.model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    dataset = PCDataset("train", args.train_dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps or args.num_train_epochs * updates_per_epoch
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)
    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / updates_per_epoch)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="sample", clip_sample=False)
    global_step = load_resume_checkpoint(accelerator, args.resume_from_checkpoint, output_dir)
    if global_step > max_train_steps:
        raise ValueError(f"Checkpoint step {global_step} exceeds max_train_steps {max_train_steps}")
    first_epoch = global_step // updates_per_epoch
    resume_batch_step = (global_step % updates_per_epoch) * args.gradient_accumulation_steps
    progress_bar = tqdm(
        range(max_train_steps),
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        visualization_batch = None
        for batch_step, (batch, _) in enumerate(dataloader):
            if epoch == first_epoch and batch_step < resume_batch_step:
                continue
            if visualization_batch is None:
                visualization_batch = {key: value[:1].detach() for key, value in batch.items()}
            with accelerator.accumulate(model):
                latents = batch["points_tgt"]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, torch.randn_like(latents), timesteps)
                null_emb = None
                if args.condition_drop_rate > 0:
                    null_emb = (torch.rand(latents.shape[0], device=latents.device) > args.condition_drop_rate).float()
                    null_emb = null_emb[:, None, None]
                prediction = model(
                    noisy_latents,
                    timesteps,
                    batch["points_src"],
                    batch["initial_linear_velocity"],
                    batch["initial_angular_velocity"],
                    null_emb=null_emb,
                )
                loss = F.mse_loss(prediction.float(), latents.float())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                logs = {"train_loss": loss.detach().item(), "learning_rate": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                if args.report_to is not None:
                    accelerator.log(logs, step=global_step)
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    accelerator.save_state(output_dir / f"checkpoint-{global_step}")
            if global_step >= max_train_steps:
                break
        if should_save_visualization(epoch) and visualization_batch is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                model.eval()
                pipeline = PCPipeline(
                    model=accelerator.unwrap_model(model),
                    scheduler=DDIMScheduler.from_config(noise_scheduler.config),
                )
                prediction = pipeline(
                    visualization_batch["points_src"],
                    visualization_batch["initial_linear_velocity"],
                    visualization_batch["initial_angular_velocity"],
                    device=accelerator.device,
                    batch_size=1,
                )
                predicted_point_cloud = torch.cat(
                    [visualization_batch["points_src"].unsqueeze(1), prediction], dim=1
                ).squeeze(0).cpu().numpy()
                ground_truth_point_cloud = torch.cat(
                    [visualization_batch["points_src"].unsqueeze(1), visualization_batch["points_tgt"]], dim=1
                ).squeeze(0).cpu().numpy()
                save_pointcloud_mp4(predicted_point_cloud, vis_dir / f"epoch-{epoch + 1:04d}.mp4")
                save_pointcloud_comparison_mp4(
                    predicted_point_cloud,
                    ground_truth_point_cloud,
                    vis_dir / f"epoch-{epoch + 1:04d}-comparison.mp4",
                )
                model.train()
            accelerator.wait_for_everyone()
        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parsed_args = parser.parse_args()
    config = OmegaConf.merge(OmegaConf.structured(PCTrainingConfig), OmegaConf.load(parsed_args.config))
    main(config)
