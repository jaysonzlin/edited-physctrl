import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

import train_pc


class TinyDataset(Dataset):
    def __init__(self, split, cfg):
        self.split = split

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "points_src": torch.zeros(1, 8, 3),
            "points_tgt": torch.zeros(48, 1, 8, 3),
            "initial_linear_velocity": torch.zeros(1, 3),
            "initial_angular_velocity": torch.zeros(1, 3),
        }, {"path": "tiny"}


def test_train_pc_runs_one_step_without_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(train_pc, "PCDataset", TinyDataset)
    config = OmegaConf.create(
        {
            "output_dir": str(tmp_path / "output"),
            "logging_dir": "logs",
            "vis_dir": "vis",
            "report_to": None,
            "tracker_project_name": "pc-test",
            "seed": 0,
            "train_batch_size": 1,
            "num_train_epochs": 1,
            "max_train_steps": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "lr_warmup_steps": 0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_weight_decay": 0.0,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "mixed_precision": "no",
            "dataloader_num_workers": 0,
            "checkpointing_steps": 10,
            "resume_from_checkpoint": None,
            "condition_drop_rate": 0.0,
            "pc_size": 8,
            "model_config": {
                "latent_dim": 64,
                "n_layers": 1,
                "frame_cond": True,
                "point_embed": False,
                "pred_offset": True,
                "transformer_block": "SpatialTemporalTransformerBlock",
            },
            "train_dataset": {"dataset_path": "unused"},
        }
    )

    train_pc.main(config)

    assert (tmp_path / "output" / "config.yaml").exists()
