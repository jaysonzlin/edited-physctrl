import torch

from visualize_pc_checkpoint import combine_initial_and_future, default_output_path, resolve_sample_path


def test_resolve_sample_path_accepts_sample_directory(tmp_path):
    sample_dir = tmp_path / "sample_0"
    sample_dir.mkdir()
    hdf5_path = sample_dir / "pc.hdf5"
    hdf5_path.touch()

    assert resolve_sample_path(sample_dir) == hdf5_path


def test_default_output_path_is_next_to_training_output(tmp_path):
    checkpoint_dir = tmp_path / "run" / "checkpoint-2500"
    sample_path = tmp_path / "training_dataset" / "sample_0" / "pc.hdf5"

    assert default_output_path(checkpoint_dir, sample_path) == tmp_path / "run" / "sample_0-comparison.mp4"


def test_combine_initial_and_future_uses_the_prediction_device():
    init_pc = torch.zeros(1, 1, 8, 3)
    predicted_future = torch.zeros(1, 48, 1, 8, 3, device="meta")

    sequence = combine_initial_and_future(init_pc, predicted_future)

    assert sequence.shape == (1, 49, 1, 8, 3)
    assert sequence.device.type == "meta"
