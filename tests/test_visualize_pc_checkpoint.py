from visualize_pc_checkpoint import default_output_path, resolve_sample_path


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
