"""Generate a single figure to verify point clouds (multiple azimuth views).

This script is self-contained and does NOT import `vipe` or any CUDA
extensions, so it can run in a lightweight environment.
"""

from pathlib import Path
import json
import zipfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import OpenEXR
import Imath
import imageio

matplotlib.use("Agg")
DEFAULT_AGG_WINDOW = 5
AZIMUTHS = (0, 90, 180, 270)
ELEV = 25
ELEVATIONS = (10, 25, 45)


def load_meta_sample(meta_json_path: Path):
    with open(meta_json_path, "r") as f:
        meta = json.load(f)
    samples = meta.get("test_datasets", [])
    if not samples:
        raise RuntimeError(f"No test_datasets found in {meta_json_path}")
    return samples[0]


def as_w2c_4x4(extr_3x4):
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :4] = np.asarray(extr_3x4, dtype=np.float32)
    return w2c


def camera_center_from_w2c(extr_3x4):
    c2w = np.linalg.inv(as_w2c_4x4(extr_3x4))
    return c2w[:3, 3]


def direction_to_elev_azim(dir_vec: np.ndarray) -> tuple[float, float]:
    """Approximate (elev, azim) for a given 3D direction.

    We follow Matplotlib's convention: elev 是相对 xy 平面的仰角，
    azim 是绕 z 轴的旋转角（atan2(x, y)).
    """
    x, y, z = dir_vec
    r_xy = float(np.hypot(x, y))
    elev = float(np.degrees(np.arctan2(z, r_xy)))
    azim = float(np.degrees(np.arctan2(x, y)))
    return elev, azim


def read_pose_npz(npz_file_path: Path):
    data = np.load(npz_file_path)
    # data["data"] are 4x4 cam2world matrices as saved in vipe
    return data["inds"], data["data"]


def read_intrinsics_npz(npz_file_path: Path):
    data = np.load(npz_file_path)
    return data["inds"], torch.from_numpy(data["data"])


def read_depth_zip(zip_file_path: Path):
    valid_width, valid_height = 0, 0
    with zipfile.ZipFile(zip_file_path, "r") as z:
        for file_name in sorted(z.namelist()):
            frame_idx = int(file_name.split(".")[0])
            with z.open(file_name) as f:
                try:
                    exr = OpenEXR.InputFile(f)
                except OSError:
                    assert valid_width > 0 and valid_height > 0
                    yield frame_idx, torch.full(
                        (valid_height, valid_width), float("nan"), dtype=torch.float32
                    )
                    continue
                header = exr.header()
                dw = header["dataWindow"]
                valid_width = dw.max.x - dw.min.x + 1
                valid_height = dw.max.y - dw.min.y + 1
                channels = exr.channels(["Z"])
                depth_data = np.frombuffer(channels[0], dtype=np.float16).reshape(
                    (valid_height, valid_width)
                )
                yield frame_idx, torch.from_numpy(depth_data.copy()).float()


def read_rgb_video(mp4_path: Path):
    reader = imageio.get_reader(str(mp4_path), "ffmpeg")
    for frame_idx, rgb in enumerate(reader):
        yield frame_idx, torch.from_numpy(rgb) / 255.0


def read_mask_zip(zip_file_path: Path):
    import io as _io
    with zipfile.ZipFile(zip_file_path, "r") as z:
        for file_name in sorted(z.namelist()):
            frame_idx = int(file_name.split(".")[0])
            with z.open(file_name) as f:
                data = imageio.imread(_io.BytesIO(f.read()))
            yield frame_idx, torch.from_numpy(data.astype(np.uint8))


def get_midframe_pointcloud(
    artifacts_root: Path,
    artifact_name: str,
    agg_window: int = DEFAULT_AGG_WINDOW,
    use_temporal_median: bool = True,
    use_depth_quantile_cutoff: bool = True,
):
    pose_path = artifacts_root / "pose" / f"{artifact_name}.npz"
    depth_path = artifacts_root / "depth" / f"{artifact_name}.zip"
    intr_path = artifacts_root / "intrinsics" / f"{artifact_name}.npz"
    camera_txt_path = artifacts_root / "intrinsics" / f"{artifact_name}_camera.txt"
    rgb_video_path = artifacts_root / "rgb" / f"{artifact_name}.mp4"
    mask_path = artifacts_root / "mask" / f"{artifact_name}.zip"

    pose_inds, poses_c2w = read_pose_npz(pose_path)
    n_frames = len(pose_inds)
    mid_idx = n_frames // 2
    half = max(0, agg_window // 2)
    start_idx = max(0, mid_idx - half)
    end_idx = min(n_frames, mid_idx + half + 1)
    use_ids = set(range(start_idx, end_idx))

    depth_frames = []
    depth_mid = None
    for i, (_, d) in enumerate(read_depth_zip(depth_path)):
        if i in use_ids:
            d_np = d.numpy()
            if i == mid_idx:
                depth_mid = d_np.copy()
            depth_frames.append(d_np)
    if not depth_frames:
        raise RuntimeError(
            f"No depth in aggregation window [{start_idx}, {end_idx}) for artifact {artifact_name}"
        )
    if use_temporal_median and agg_window > 1:
        depth_stack = np.stack(depth_frames, axis=0).astype(np.float32)
        depth_stack[depth_stack <= 0] = np.nan
        depth_np_full = np.nanmedian(depth_stack, axis=0)
        depth_np_full = np.nan_to_num(depth_np_full, nan=0.0)
    else:
        if depth_mid is None:
            depth_mid = depth_frames[len(depth_frames) // 2]
        depth_mid = depth_mid.astype(np.float32)
        depth_mid[depth_mid <= 0] = np.nan
        depth_np_full = np.nan_to_num(depth_mid, nan=0.0)

    rgb_frames = []
    rgb_mid = None
    for i, (_, r) in enumerate(read_rgb_video(rgb_video_path)):
        if i in use_ids:
            r_np = (r.numpy() * 255.0).astype(np.float32)
            if i == mid_idx:
                rgb_mid = r_np.copy()
            rgb_frames.append(r_np)
    if not rgb_frames:
        raise RuntimeError(
            f"No rgb in aggregation window [{start_idx}, {end_idx}) for artifact {artifact_name}"
        )
    if use_temporal_median and agg_window > 1:
        rgb_np_full = np.median(np.stack(rgb_frames, axis=0), axis=0).astype(np.uint8)
    else:
        if rgb_mid is None:
            rgb_mid = rgb_frames[len(rgb_frames) // 2]
        rgb_np_full = rgb_mid.astype(np.uint8)

    _, intrinsics = read_intrinsics_npz(intr_path)
    intrinsics = intrinsics.numpy()
    if mid_idx < intrinsics.shape[0]:
        intr = intrinsics[mid_idx]
    else:
        intr = intrinsics[0]

    fx, fy, cx, cy = intr

    h, w = rgb_np_full.shape[:2]
    step = 2
    vs = torch.arange(h).float()[::step]
    us = torch.arange(w).float()[::step]
    disp_v, disp_u = torch.meshgrid(vs, us, indexing="ij")

    X = (disp_u - cx) / fx
    Y = (disp_v - cy) / fy
    Z = torch.ones_like(X)
    rays = torch.stack([X, Y, Z], dim=-1).numpy()

    depth_np = depth_np_full[::step, ::step]
    rgb_np = rgb_np_full[::step, ::step]
    pts_cam = rays * depth_np[..., None]
    pts_flat = pts_cam.reshape(-1, 3)

    c2w = poses_c2w[mid_idx]
    pts_world = (c2w[:3, :3] @ pts_flat.T + c2w[:3, 3:4]).T

    depth_flat = depth_np.reshape(-1)
    valid = np.isfinite(pts_world).all(axis=1) & (depth_flat > 0)
    if use_depth_quantile_cutoff and valid.any():
        depth_valid = depth_flat[valid]
        near_cutoff = np.quantile(depth_valid, 0.9)
        valid = valid & (depth_flat <= near_cutoff)

    # Load mid-frame person mask if available.
    mask_np_full = None
    if mask_path.exists():
        for i, (_, m) in enumerate(read_mask_zip(mask_path)):
            if i == mid_idx:
                mask_np_full = m.numpy()
                break
    if mask_np_full is None:
        mask_np_full = np.zeros(depth_np_full.shape, dtype=np.uint8)

    mask_np = mask_np_full[::step, ::step]
    is_person_flat = mask_np.reshape(-1).astype(bool)

    pts_world = pts_world[valid]
    colors = rgb_np.reshape(-1, 3)[valid]
    is_person = is_person_flat[valid]
    if len(pts_world) > 120000:
        sel = np.random.choice(len(pts_world), 120000, replace=False)
        pts_world = pts_world[sel]
        colors = colors[sel]
        is_person = is_person[sel]

    return pts_world, colors, is_person, n_frames, mid_idx, start_idx, end_idx


def main():
    base = Path("/mnt/task_runtime/EgoX-EgoPriorRenderer/processed/h2o")
    artifacts_root = base / "vipe_results" / "subject1_h1_0_cam0"
    sample = load_meta_sample(base / "meta.json")

    ego_traj = np.asarray(
        [camera_center_from_w2c(extr_3x4) for extr_3x4 in sample["ego_extrinsics"]],
        dtype=np.float32,
    )
    exo_center = camera_center_from_w2c(sample["camera_extrinsics"])

    exo_pts, exo_colors, exo_is_person, n_frames, mid_idx, start_idx, end_idx = get_midframe_pointcloud(
        artifacts_root,
        "exo",
        agg_window=DEFAULT_AGG_WINDOW,
        use_temporal_median=True,
        use_depth_quantile_cutoff=True,
    )
    ego_pts, ego_colors, ego_is_person, _, _, _, _ = get_midframe_pointcloud(
        artifacts_root,
        "ego",
        agg_window=DEFAULT_AGG_WINDOW,
        use_temporal_median=True,
        use_depth_quantile_cutoff=True,
    )

    # Approximate Matplotlib (elev, azim) whose viewing direction is aligned
    # with the exo camera looking from its center towards the exo point cloud barycenter.
    exo_center_pts = exo_pts.mean(axis=0)
    exo_look_dir = exo_center_pts - exo_center
    elev_exo_like, azim_exo_like = direction_to_elev_azim(exo_look_dir)
    print(
        f"Approx exo-like view (looking from exo center to cloud center): "
        f"elev={elev_exo_like:.2f}°, azim={azim_exo_like:.2f}°"
    )

    # Also build a single-frame exo point cloud (no temporal median, no quantile cutoff)
    # to better visualize the person and transient content.
    exo_pts_single, exo_colors_single, exo_is_person_single, _, _, _, _ = get_midframe_pointcloud(
        artifacts_root,
        "exo",
        agg_window=DEFAULT_AGG_WINDOW,
        use_temporal_median=False,
        use_depth_quantile_cutoff=False,
    )

    fig = plt.figure(figsize=(16, 8))
    for row, (title, points, colors, cam_pts, cam_label) in enumerate(
        [
            ("exo point cloud", exo_pts, exo_colors, exo_center[None, :], "exo cam"),
            ("ego point cloud", ego_pts, ego_colors, ego_traj, "ego traj"),
        ]
    ):
        for col, azim in enumerate(AZIMUTHS):
            ax = fig.add_subplot(2, 4, row * 4 + col + 1, projection="3d")
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=colors / 255.0,
                s=0.5,
                alpha=0.85,
            )
            ax.scatter(
                cam_pts[:, 0],
                cam_pts[:, 1],
                cam_pts[:, 2],
                c="red",
                s=8,
                alpha=0.9,
                label=cam_label,
            )
            ax.view_init(elev=ELEV, azim=azim)
            ax.set_title(f"{title} | azim={azim}°")

    plt.suptitle(
        f"Point cloud verify (frames {start_idx}-{end_idx-1}, center {mid_idx}) | elev={ELEV}°",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = base / "viz_pointcloud_verify.png"
    fig.savefig(out_path, dpi=150)
    print(str(out_path))

    # Figure 2: exo single-frame point cloud from multiple azimuths.
    fig2 = plt.figure(figsize=(16, 4))
    for col, azim in enumerate(AZIMUTHS):
        ax = fig2.add_subplot(1, 4, col + 1, projection="3d")
        ax.scatter(
            exo_pts_single[:, 0],
            exo_pts_single[:, 1],
            exo_pts_single[:, 2],
            c=exo_colors_single / 255.0,
            s=0.5,
            alpha=0.85,
        )
        ax.scatter(
            [exo_center[0]],
            [exo_center[1]],
            [exo_center[2]],
            c="red",
            s=12,
            alpha=0.9,
        )
        ax.view_init(elev=ELEV, azim=azim)
        ax.set_title(f"exo single-frame | azim={azim}°")

    plt.suptitle(
        f"exo single-frame point cloud (frame {mid_idx}) | elev={ELEV}°",
        fontsize=11,
    )
    plt.tight_layout()
    out_path_single = base / "viz_pointcloud_singleframe_exo.png"
    fig2.savefig(out_path_single, dpi=150)
    print(str(out_path_single))

    # Figure 3: exo multi-frame point cloud with both azimuth and elevation sweeps.
    fig3 = plt.figure(figsize=(16, 9))
    for r, elev in enumerate(ELEVATIONS):
        for c, azim in enumerate(AZIMUTHS):
            ax = fig3.add_subplot(len(ELEVATIONS), len(AZIMUTHS), r * len(AZIMUTHS) + c + 1, projection="3d")
            ax.scatter(
                exo_pts[:, 0],
                exo_pts[:, 1],
                exo_pts[:, 2],
                c=exo_colors / 255.0,
                s=0.5,
                alpha=0.85,
            )
            ax.scatter(
                [exo_center[0]],
                [exo_center[1]],
                [exo_center[2]],
                c="red",
                s=12,
                alpha=0.9,
            )
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"exo multi-frame | elev={elev}°, azim={azim}°", fontsize=8)

    plt.suptitle(
        f"exo multi-frame point cloud (frames {start_idx}-{end_idx-1}, center {mid_idx}) "
        f"| elev∈{ELEVATIONS}, azim∈{AZIMUTHS}",
        fontsize=11,
    )
    plt.tight_layout()
    out_path_grid = base / "viz_pointcloud_exo_elev_azim_grid.png"
    fig3.savefig(out_path_grid, dpi=150)
    print(str(out_path_grid))

    # Figure 4: a single exo-like view using the approximate (elev, azim)
    # computed from exo camera -> cloud center direction.
    fig4 = plt.figure(figsize=(5, 4))
    ax4 = fig4.add_subplot(1, 1, 1, projection="3d")
    ax4.scatter(
        exo_pts[:, 0],
        exo_pts[:, 1],
        exo_pts[:, 2],
        c=exo_colors / 255.0,
        s=0.5,
        alpha=0.85,
    )
    ax4.scatter(
        [exo_center[0]],
        [exo_center[1]],
        [exo_center[2]],
        c="red",
        s=20,
        alpha=0.9,
    )
    ax4.view_init(elev=elev_exo_like, azim=azim_exo_like)
    ax4.set_title(
        f"exo multi-frame | approx exo-like view\n"
        f"elev={elev_exo_like:.2f}°, azim={azim_exo_like:.2f}°",
        fontsize=9,
    )
    plt.tight_layout()
    out_path_exo_like = base / "viz_pointcloud_exo_exo_like_view.png"
    fig4.savefig(out_path_exo_like, dpi=150)
    print(str(out_path_exo_like))

    # Figure 5: person highlighted (orange, large) vs background (gray, tiny)
    # Uses single-frame so person mask aligns with depth.
    bg_mask = ~exo_is_person_single
    person_pts_s = exo_pts_single[exo_is_person_single]
    person_colors_s = exo_colors_single[exo_is_person_single]
    bg_pts_s = exo_pts_single[bg_mask]
    print(f"Person pts: {len(person_pts_s)}, BG pts: {len(bg_pts_s)}")

    fig5 = plt.figure(figsize=(16, 4))
    for col, azim in enumerate(AZIMUTHS):
        ax = fig5.add_subplot(1, 4, col + 1, projection="3d")
        ax.scatter(
            bg_pts_s[:, 0], bg_pts_s[:, 1], bg_pts_s[:, 2],
            c="lightgray", s=0.2, alpha=0.15,
        )
        ax.scatter(
            person_pts_s[:, 0], person_pts_s[:, 1], person_pts_s[:, 2],
            c=person_colors_s / 255.0, s=4, alpha=1.0,
        )
        ax.scatter([exo_center[0]], [exo_center[1]], [exo_center[2]],
                   c="red", s=20, alpha=0.9)
        ax.view_init(elev=ELEV, azim=azim)
        ax.set_title(f"person highlighted | azim={azim}°", fontsize=8)

    plt.suptitle(
        f"Person (RGB color, s=4) vs background (gray, s=0.2) — single frame {mid_idx}",
        fontsize=11,
    )
    plt.tight_layout()
    out_path_person_hl = base / "viz_pointcloud_person_highlighted.png"
    fig5.savefig(out_path_person_hl, dpi=150)
    print(str(out_path_person_hl))

    # Figure 6: person-only point cloud, 4 azimuths
    fig6 = plt.figure(figsize=(16, 4))
    for col, azim in enumerate(AZIMUTHS):
        ax = fig6.add_subplot(1, 4, col + 1, projection="3d")
        ax.scatter(
            person_pts_s[:, 0], person_pts_s[:, 1], person_pts_s[:, 2],
            c=person_colors_s / 255.0, s=3, alpha=1.0,
        )
        ax.view_init(elev=ELEV, azim=azim)
        ax.set_title(f"person only | azim={azim}°", fontsize=8)

    plt.suptitle(
        f"Person-only point cloud — single frame {mid_idx} | {len(person_pts_s)} pts",
        fontsize=11,
    )
    plt.tight_layout()
    out_path_person_only = base / "viz_pointcloud_person_only.png"
    fig6.savefig(out_path_person_only, dpi=150)
    print(str(out_path_person_only))

    # Figure 7: background-only point cloud, 4 azimuths
    bg_colors_s = exo_colors_single[bg_mask]
    fig7 = plt.figure(figsize=(16, 4))
    for col, azim in enumerate(AZIMUTHS):
        ax = fig7.add_subplot(1, 4, col + 1, projection="3d")
        ax.scatter(
            bg_pts_s[:, 0], bg_pts_s[:, 1], bg_pts_s[:, 2],
            c=bg_colors_s / 255.0, s=0.5, alpha=0.85,
        )
        ax.scatter([exo_center[0]], [exo_center[1]], [exo_center[2]],
                   c="red", s=20, alpha=0.9)
        ax.view_init(elev=ELEV, azim=azim)
        ax.set_title(f"background only | azim={azim}°", fontsize=8)

    plt.suptitle(
        f"Background-only point cloud — single frame {mid_idx} | {len(bg_pts_s)} pts",
        fontsize=11,
    )
    plt.tight_layout()
    out_path_bg_only = base / "viz_pointcloud_background_only.png"
    fig7.savefig(out_path_bg_only, dpi=150)
    print(str(out_path_bg_only))


if __name__ == "__main__":
    main()
