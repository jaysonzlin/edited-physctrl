from types import SimpleNamespace

import h5py
import numpy as np

from dataset.pc_dataset import PCDataset


def make_cfg(dataset_path):
    return SimpleNamespace(dataset_path=str(dataset_path))


def write_sample(root, name="sample_0"):
    sample_dir = root / name
    sample_dir.mkdir()
    with h5py.File(sample_dir / "pc.hdf5", "w") as source:
        points = np.arange(49 * 1 * 2048 * 3, dtype=np.float32).reshape(49, 1, 2048, 3)
        source.create_dataset("point_cloud", data=points)
        source.create_dataset("initial_linear_velocity", data=np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        source.create_dataset("initial_angular_velocity", data=np.array([[4.0, 5.0, 6.0]], dtype=np.float32))
    return points


def test_dataset_reads_raw_hdf5_values(tmp_path):
    points = write_sample(tmp_path)

    dataset = PCDataset("train", make_cfg(tmp_path))
    sample, info = dataset[0]

    assert len(dataset) == 1
    assert sample["points_src"].shape == (1, 2048, 3)
    assert sample["points_tgt"].shape == (48, 1, 2048, 3)
    assert sample["points_src"].numpy()[0, 0, 0] == points[0, 0, 0, 0]
    assert sample["points_tgt"].numpy()[0, 0, 0, 0] == points[1, 0, 0, 0]
    assert sample["initial_linear_velocity"].tolist() == [[1.0, 2.0, 3.0]]
    assert sample["initial_angular_velocity"].tolist() == [[4.0, 5.0, 6.0]]
    assert info["path"].endswith("sample_0/pc.hdf5")
