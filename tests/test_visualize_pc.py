import numpy as np

from visualize_pc import save_pointcloud_mp4


def test_save_pointcloud_mp4_writes_video(tmp_path):
    point_cloud = np.zeros((2, 1, 8, 3), dtype=np.float32)
    point_cloud[1, 0, :, 0] = 1.0
    output_path = tmp_path / "trajectory.mp4"

    save_pointcloud_mp4(point_cloud, output_path, fps=2)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
