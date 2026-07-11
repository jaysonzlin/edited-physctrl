"""Sampling pipeline for point-cloud trajectories."""

import torch
from diffusers import DiffusionPipeline


class PCPipeline(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        init_pc,
        initial_linear_velocity,
        initial_angular_velocity,
        device,
        batch_size=None,
        num_inference_steps=50,
        guidance_scale=1.0,
        generator=None,
    ):
        device = torch.device(device)
        batch_size = init_pc.shape[0] if batch_size is None else batch_size
        if init_pc.shape[0] != batch_size:
            raise ValueError("batch_size must match init_pc.shape[0]")

        init_pc = init_pc.to(device)
        initial_linear_velocity = initial_linear_velocity.to(device)
        initial_angular_velocity = initial_angular_velocity.to(device)
        sample = torch.randn(
            (batch_size, self.model.n_frames, 1, init_pc.shape[2], 3),
            generator=generator,
            device=device,
            dtype=init_pc.dtype,
        )
        self.model.to(device)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            init_pc = torch.cat([init_pc, init_pc])
            initial_linear_velocity = torch.cat([initial_linear_velocity, initial_linear_velocity])
            initial_angular_velocity = torch.cat([initial_angular_velocity, initial_angular_velocity])
            null_emb = torch.cat(
                [torch.zeros(batch_size, 1, 1, device=device), torch.ones(batch_size, 1, 1, device=device)]
            )
        else:
            null_emb = None

        for timestep in self.progress_bar(self.scheduler.timesteps):
            model_input = torch.cat([sample, sample]) if do_classifier_free_guidance else sample
            timesteps = torch.full((model_input.shape[0],), timestep, device=device, dtype=torch.long)
            model_output = self.model(
                model_input,
                timesteps,
                init_pc,
                initial_linear_velocity,
                initial_angular_velocity,
                null_emb=null_emb,
            )
            if do_classifier_free_guidance:
                unconditional, conditional = model_output.chunk(2)
                model_output = unconditional + guidance_scale * (conditional - unconditional)
            sample = self.scheduler.step(model_output, timestep, sample).prev_sample

        return sample
