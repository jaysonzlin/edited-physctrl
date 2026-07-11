from types import SimpleNamespace

import train_pc


class FakeAccelerator:
    def __init__(self):
        self.calls = []

    def init_trackers(self, project_name, config):
        self.calls.append((project_name, config))


def test_initialize_trackers_configures_wandb(monkeypatch):
    accelerator = FakeAccelerator()
    config = SimpleNamespace(
        report_to="wandb",
        tracker_project_name="pc-dit-test",
        train_batch_size=2,
    )
    monkeypatch.setattr(train_pc, "is_wandb_available", lambda: True)

    train_pc.initialize_trackers(accelerator, config)

    assert accelerator.calls == [
        ("pc-dit-test", {"report_to": "wandb", "tracker_project_name": "pc-dit-test", "train_batch_size": 2})
    ]
