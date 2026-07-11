# PC DiT

`PCDiT` is a diffusion transformer that predicts the next 48 point-cloud
frames of one rigid object. It conditions on the initial point cloud, initial
linear velocity, and initial angular velocity. Coordinates and velocities stay
in the raw units stored in the HDF5 file; this pipeline does not normalize
them.

## Dataset layout

Set `train_dataset.dataset_path` in
[`src/configs/config_pc.yaml`](src/configs/config_pc.yaml) to a directory that
contains sample directories:

```text
DATASET_FOLDER/
  sample_0/
    pc.hdf5
  sample_1/
    pc.hdf5
  ...
```

Each `pc.hdf5` must contain float32 datasets with these exact shapes:

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `point_cloud` | `(49, 1, 2048, 3)` | Initial frame followed by 48 target frames. |
| `initial_linear_velocity` | `(1, 3)` | Linear velocity at the initial frame. |
| `initial_angular_velocity` | `(1, 3)` | Angular velocity at the initial frame. |

The singleton object dimension is preserved in the public dataset and pipeline
interfaces. A training batch therefore has these shapes:

```text
points_src:               (B, 1, 2048, 3)
points_tgt:               (B, 48, 1, 2048, 3)
initial_linear_velocity:  (B, 1, 3)
initial_angular_velocity: (B, 1, 3)
```

## Configure training

Edit `src/configs/config_pc.yaml` before training:

- Replace `DATASET_FOLDER` with the dataset root above.
- Set `output_dir` to the directory for checkpoints and the run config.
- Adjust `train_batch_size`, `dataloader_num_workers`, and `max_train_steps`
  for your hardware.
- Set `mixed_precision: "no"` for CPU-only runs. The provided `bf16` setting is
  intended for compatible accelerators.
- W&B logging is enabled by default with `report_to: wandb`. Authenticate once
  before training with `wandb login`, then use `tracker_project_name` to choose
  the destination project. Set `report_to: null` to disable external logging.

The default model uses 8 transformer layers, latent dimension 256, and 2,048
points. `latent_dim` must be divisible by 64.

## Train

From the repository root, run one process with:

```bash
accelerate launch --num_processes 1 src/train_pc.py --config src/configs/config_pc.yaml
```

For a simple local run, this also works:

```bash
python src/train_pc.py --config src/configs/config_pc.yaml
```

The trainer uses a DDPM objective that predicts x0 with MSE loss and saves an
Accelerate checkpoint every `checkpointing_steps` under:

```text
<output_dir>/checkpoint-<step>/
```

To resume, set `resume_from_checkpoint` in `src/configs/config_pc.yaml` to an
explicit checkpoint directory, such as
`./outputs/pc_dit_8layers/checkpoint-2500`, or to `latest` to select the
highest-numbered checkpoint under `output_dir`. The loader restores the model,
optimizer, scheduler, and training state, then skips the batches already
completed within the resumed epoch.

There is intentionally no validation split or validation loop. Set
`condition_drop_rate` above zero only when you want classifier-free guidance at
inference; its default is `0.0`.

At the end of every 100th epoch, the trainer samples one training example and
saves `<output_dir>/<vis_dir>/epoch-<epoch>-comparison.mp4`, which places
prediction and ground truth side by side and reports each frame's position
error (PE) and mean per-point error (ME).
The MP4 uses the same renderer as `src/visualize_pc.py`; it requires `imageio`
and `imageio-ffmpeg`, both listed in `requirements.txt`.

## Run inference

The following example reads `sample_0/pc.hdf5`, loads an Accelerate model
checkpoint, samples a 48-frame trajectory with DDIM, and writes it to
`pc_prediction.hdf5`.

```python
from pathlib import Path

import h5py
import torch
from diffusers import DDIMScheduler
from omegaconf import OmegaConf
from safetensors.torch import load_file

from model.pc_dit import PCDiT
from options import PCTrainingConfig
from pipeline_pc import PCPipeline


config_path = "src/configs/config_pc.yaml"
checkpoint_dir = Path("outputs/pc_dit_8layers/checkpoint-60000")
input_path = "sample_0/pc.hdf5"
output_path = "pc_prediction.hdf5"
device = "cuda" if torch.cuda.is_available() else "cpu"

config = OmegaConf.merge(OmegaConf.structured(PCTrainingConfig), OmegaConf.load(config_path))
model = PCDiT(config.pc_size, 48, config.model_config)

# Accelerate saves the model as model.safetensors by default. If safetensors
# was disabled for a run, use torch.load(checkpoint_dir / "pytorch_model.bin")
# instead.
model.load_state_dict(load_file(checkpoint_dir / "model.safetensors"))
model.eval()

with h5py.File(input_path, "r") as source:
    init_pc = torch.from_numpy(source["point_cloud"][0]).float().unsqueeze(0)
    linear_velocity = torch.from_numpy(source["initial_linear_velocity"][:]).float().unsqueeze(0)
    angular_velocity = torch.from_numpy(source["initial_angular_velocity"][:]).float().unsqueeze(0)

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    prediction_type="sample",
    clip_sample=False,
)
pipeline = PCPipeline(model, scheduler)
prediction = pipeline(
    init_pc=init_pc,
    initial_linear_velocity=linear_velocity,
    initial_angular_velocity=angular_velocity,
    device=device,
    num_inference_steps=50,
    guidance_scale=1.0,
    generator=torch.Generator(device=device).manual_seed(0),
)

with h5py.File(output_path, "w") as destination:
    destination.create_dataset("predicted_point_cloud", data=prediction.cpu().numpy()[0])
```

`prediction` has shape `(B, 48, 1, 2048, 3)`. The example removes only the
batch dimension before saving, so `predicted_point_cloud` has shape
`(48, 1, 2048, 3)`.

Use `guidance_scale > 1.0` only if the model was trained with a nonzero
`condition_drop_rate`. The default `guidance_scale=1.0` uses the conditional
model directly.

## Checkpoint comparison CLI

Use `visualize_pc_checkpoint.py` to load a saved checkpoint, sample a selected
HDF5 trajectory, and save the predicted-versus-ground-truth comparison MP4:

```bash
PYTHONPATH=src python src/visualize_pc_checkpoint.py \
  --checkpoint outputs/pc_dit_8layers/checkpoint-2500 \
  --sample training_dataset/sample_0
```

The script uses the `config.yaml` next to the checkpoint directory by default.
Use `--output`, `--num-inference-steps`, `--guidance-scale`, or `--seed` to
override the output path and sampling settings.
