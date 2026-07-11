import torch
from diffusers import DiffusionPipeline

class MCGPipeline(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, init_pc, v1, v2, w1, w2, rho1, rho2, friction1, friction2,
        generator, 
        device, 
        y=None,
        batch_size: int = 1, 
        num_inference_steps: int = 50, 
        guidance_scale=1.0, 
        n_frames=24,
        gt_traj=None
    ):
        # Sample gaussian noise to begin loop
        sample = torch.randn((batch_size, n_frames, init_pc.shape[2], 3), generator=generator).to(device)
        self.model.to(device)
        
        init_pc = init_pc.to(device)
        v1 = v1.to(device)
        v2 = v2.to(device)
        w1 = w1.to(device)
        w2 = w2.to(device)
        rho1 = rho1.to(device)
        rho2 = rho2.to(device)
        friction1 = friction1.to(device)
        friction2 = friction2.to(device)
        y = y.to(device) if y is not None else None
        
        # set step values
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        null_emb = None

        for t in self.progress_bar(self.scheduler.timesteps):
            if gt_traj is not None:
                sample[:, :, :2048, :] = gt_traj[:, :, :2048, :]
                
            t_batch = torch.tensor([t] * batch_size, device=device)
            
            # 1. predict noise model_output
            model_output = self.model(
                sample, t_batch, init_pc, 
                v1, v2, w1, w2, rho1, rho2, friction1, friction2, 
                y=y, null_emb=null_emb
            )
            
            # 2. predict previous mean of sample x_t-1
            sample = self.scheduler.step(model_output, t_batch[0], sample).prev_sample
            
        if gt_traj is not None:
            sample[:, :, :2048, :] = gt_traj[:, :, :2048, :]
            
        return sample