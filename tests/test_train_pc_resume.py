from pathlib import Path

import pytest

from train_pc import load_resume_checkpoint


class FakeAccelerator:
    def __init__(self):
        self.loaded_path = None

    def load_state(self, path):
        self.loaded_path = path


def test_load_resume_checkpoint_restores_explicit_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "checkpoint-12"
    checkpoint_path.mkdir()
    accelerator = FakeAccelerator()

    global_step = load_resume_checkpoint(accelerator, str(checkpoint_path), tmp_path)

    assert global_step == 12
    assert accelerator.loaded_path == str(checkpoint_path)


def test_load_resume_checkpoint_rejects_path_without_step_number(tmp_path):
    checkpoint_path = tmp_path / "manual-save"
    checkpoint_path.mkdir()

    with pytest.raises(ValueError, match="checkpoint-<step>"):
        load_resume_checkpoint(FakeAccelerator(), str(checkpoint_path), tmp_path)
