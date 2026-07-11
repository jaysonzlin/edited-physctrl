"""Dataset for single-object point-cloud trajectories stored per sample."""

from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class PCDataset(Dataset):
    """Load raw 49-frame point-cloud trajectories from ``sample_*/pc.hdf5``."""

    REQUIRED_KEYS = (
        "point_cloud",
        "initial_linear_velocity",
        "initial_angular_velocity",
    )
    POINT_CLOUD_SHAPE = (49, 1, 2048, 3)
    VELOCITY_SHAPE = (1, 3)

    def __init__(self, split, cfg):
        if split != "train":
            raise ValueError("PCDataset supports the training split only.")

        dataset_path = Path(cfg.dataset_path)
        self.samples = sorted(dataset_path.glob("sample_*/pc.hdf5"))
        if not self.samples:
            raise FileNotFoundError(f"No sample_*/pc.hdf5 files found in {dataset_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        with h5py.File(path, "r") as source:
            self._validate(source, path)
            point_cloud = torch.from_numpy(source["point_cloud"][:]).float()
            initial_linear_velocity = torch.from_numpy(source["initial_linear_velocity"][:]).float()
            initial_angular_velocity = torch.from_numpy(source["initial_angular_velocity"][:]).float()

        return {
            "points_src": point_cloud[0],
            "points_tgt": point_cloud[1:],
            "initial_linear_velocity": initial_linear_velocity,
            "initial_angular_velocity": initial_angular_velocity,
        }, {"path": str(path)}

    def _validate(self, source, path):
        missing_keys = [key for key in self.REQUIRED_KEYS if key not in source]
        if missing_keys:
            raise KeyError(f"{path} is missing required datasets: {', '.join(missing_keys)}")
        if source["point_cloud"].shape != self.POINT_CLOUD_SHAPE:
            raise ValueError(
                f"{path} point_cloud must have shape {self.POINT_CLOUD_SHAPE}, "
                f"got {source['point_cloud'].shape}"
            )
        for key in ("initial_linear_velocity", "initial_angular_velocity"):
            if source[key].shape != self.VELOCITY_SHAPE:
                raise ValueError(
                    f"{path} {key} must have shape {self.VELOCITY_SHAPE}, got {source[key].shape}"
                )
