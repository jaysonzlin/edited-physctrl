from train_pc import should_save_visualization


def test_visualization_is_saved_every_100_epochs():
    assert not should_save_visualization(0)
    assert not should_save_visualization(98)
    assert should_save_visualization(99)
    assert should_save_visualization(199)
