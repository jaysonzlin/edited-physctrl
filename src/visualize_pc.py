import os
import sys

import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np


def compute_point_colors(pc_data):
    initial_heights = pc_data[0, :, :, 2]
    min_heights = initial_heights.min(axis=1, keepdims=True)
    height_ranges = np.ptp(initial_heights, axis=1, keepdims=True)
    normalized_heights = np.full(initial_heights.shape, 0.5, dtype=np.float64)
    np.divide(
        initial_heights - min_heights,
        height_ranges,
        out=normalized_heights,
        where=height_ranges > 0,
    )
    return plt.get_cmap("viridis")(normalized_heights)


def save_pointcloud_mp4(pc_data, output_video_path, fps=12):
    """Render a ``(frames, objects, points, 3)`` point-cloud sequence as MP4."""
    pc_data = np.asarray(pc_data)
    if pc_data.ndim != 4 or pc_data.shape[-1] != 3:
        raise ValueError("pc_data must have shape (frames, objects, points, 3)")

    output_video_path = os.fspath(output_video_path)
    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
    num_frames, num_objects, _, _ = pc_data.shape
    flat_pc = pc_data.reshape(-1, 3)
    min_coords = flat_pc.min(axis=0)
    max_coords = flat_pc.max(axis=0)
    mid_coords = (min_coords + max_coords) / 2
    max_range = max(max_coords[0] - min_coords[0], max_coords[1] - min_coords[1], max_coords[2])
    span = max_range + 1.0
    x_lim = (mid_coords[0] - span / 2, mid_coords[0] + span / 2)
    y_lim = (mid_coords[1] - span / 2, mid_coords[1] + span / 2)
    z_lim = (0.0, span)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    writer = imageio.get_writer(output_video_path, fps=fps)
    point_colors = compute_point_colors(pc_data)
    colorbar_mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0.0, vmax=1.0), cmap="viridis")
    colorbar_mappable.set_array([])
    colorbar = fig.colorbar(colorbar_mappable, ax=ax, pad=0.1, shrink=0.7)
    colorbar.set_label("Relative initial height")

    try:
        for frame_index in range(num_frames):
            ax.clear()
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)
            ax.set_box_aspect((1, 1, 1))
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(
                f"Point Cloud Trajectories - Frame {frame_index:03d} / {num_frames - 1:03d}", fontsize=14
            )
            ax.grid(True)
            for object_index in range(num_objects):
                points = pc_data[frame_index, object_index]
                ax.scatter(
                    points[:, 0], points[:, 1], points[:, 2],
                    c=point_colors[object_index], s=4, alpha=0.8,
                    edgecolors="none", label=f"Object {object_index} (Instance {object_index})",
                )
            ax.legend(loc="upper right")
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    finally:
        writer.close()
        plt.close(fig)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, default="output", help="Path to sample directory")
    args = parser.parse_known_args()[0]
    hdf5_path = os.path.join(args.sample_dir, "pc.hdf5")
    if not os.path.exists(hdf5_path):
        print(f"Error: {hdf5_path} not found. Please run generate_pc.py first.")
        sys.exit(1)

    with h5py.File(hdf5_path, "r") as source:
        point_cloud = source["point_cloud"][:]
    output_video_path = os.path.join(args.sample_dir, "pc_trajectory.mp4")
    save_pointcloud_mp4(point_cloud, output_video_path)
    print(f"Video successfully saved to {output_video_path}")


if __name__ == "__main__":
    main()
