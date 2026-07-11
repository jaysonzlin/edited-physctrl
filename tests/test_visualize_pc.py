import numpy as np

from visualize_pc import compute_trajectory_errors, save_pointcloud_comparison_mp4, save_pointcloud_mp4


def test_save_pointcloud_mp4_writes_video(tmp_path):
    point_cloud = np.zeros((2, 1, 8, 3), dtype=np.float32)
    point_cloud[1, 0, :, 0] = 1.0
    output_path = tmp_path / "trajectory.mp4"

    save_pointcloud_mp4(point_cloud, output_path, fps=2)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_compute_trajectory_errors_reports_centroid_and_point_errors():
    ground_truth = np.zeros((2, 1, 2, 3), dtype=np.float32)
    prediction = ground_truth.copy()
    prediction[1, 0, :, 0] = 2.0

    position_error, mean_error = compute_trajectory_errors(prediction, ground_truth)

    np.testing.assert_allclose(position_error, [0.0, 2.0])
    np.testing.assert_allclose(mean_error, [0.0, 2.0])


def test_save_pointcloud_comparison_mp4_writes_video(tmp_path):
    ground_truth = np.zeros((2, 1, 8, 3), dtype=np.float32)
    prediction = ground_truth.copy()
    prediction[1, 0, :, 0] = 1.0
    output_path = tmp_path / "comparison.mp4"

    save_pointcloud_comparison_mp4(prediction, ground_truth, output_path, fps=2)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
