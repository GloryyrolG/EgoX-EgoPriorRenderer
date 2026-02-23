from pathlib import Path
import json

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from vipe.utils.cameras import CameraType
from vipe.utils.io import (
    ArtifactPath,
    read_depth_artifacts,
    read_intrinsics_artifacts,
    read_pose_artifacts,
    read_rgb_artifacts,
)

matplotlib.use("Agg")
DEFAULT_AGG_WINDOW = 5


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


def load_video_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = int(np.clip(frame_idx, 0, max(0, total - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, total, frame_idx


def get_midframe_pointcloud(input_dir: Path, artifact_name: str, agg_window: int = DEFAULT_AGG_WINDOW):
    artifact = ArtifactPath(input_dir, artifact_name)
    pose_inds, poses = read_pose_artifacts(artifact.pose_path)
    poses = poses.matrix().numpy()
    n_frames = len(pose_inds)
    mid_idx = n_frames // 2
    half = max(0, agg_window // 2)
    start_idx = max(0, mid_idx - half)
    end_idx = min(n_frames, mid_idx + half + 1)
    use_ids = set(range(start_idx, end_idx))

    depth_frames = []
    for i, (_, d) in enumerate(read_depth_artifacts(artifact.depth_path)):
        if i in use_ids:
            depth_frames.append(d.numpy())
    if not depth_frames:
        raise RuntimeError(
            f"No depth in aggregation window [{start_idx}, {end_idx}) for artifact {artifact_name}"
        )
    depth_stack = np.stack(depth_frames, axis=0).astype(np.float32)
    depth_stack[depth_stack <= 0] = np.nan
    depth_np_full = np.nanmedian(depth_stack, axis=0)
    depth_np_full = np.nan_to_num(depth_np_full, nan=0.0)

    rgb_frames = []
    for i, (_, r) in enumerate(read_rgb_artifacts(artifact.rgb_path)):
        if i in use_ids:
            rgb_frames.append((r.cpu().numpy() * 255.0).astype(np.float32))
    if not rgb_frames:
        raise RuntimeError(
            f"No rgb in aggregation window [{start_idx}, {end_idx}) for artifact {artifact_name}"
        )
    rgb_np_full = np.median(np.stack(rgb_frames, axis=0), axis=0).astype(np.uint8)

    intr = None
    camera_type = None
    for i, (k, ct) in enumerate(
        zip(*read_intrinsics_artifacts(artifact.intrinsics_path, artifact.camera_type_path)[1:3])
    ):
        if i == mid_idx:
            intr = k
            camera_type = ct
            break
    if intr is None:
        intr, camera_type = next(
            iter(zip(*read_intrinsics_artifacts(artifact.intrinsics_path, artifact.camera_type_path)[1:3]))
        )

    h, w = rgb_np_full.shape[:2]
    step = 2
    model = camera_type.build_camera_model(intr)
    disp_v, disp_u = torch.meshgrid(
        torch.arange(h).float()[::step],
        torch.arange(w).float()[::step],
        indexing="ij",
    )
    if camera_type == CameraType.PANORAMA:
        disp_v = disp_v / (h - 1)
        disp_u = disp_u / (w - 1)
    disp = torch.ones_like(disp_v)
    pts, _, _ = model.iproj_disp(disp, disp_u, disp_v)
    rays = pts[..., :3].numpy()
    if camera_type != CameraType.PANORAMA:
        rays /= np.clip(rays[..., 2:3], 1e-6, None)

    depth_np = depth_np_full[::step, ::step]
    rgb_np = rgb_np_full[::step, ::step]
    pts_cam = rays * depth_np[..., None]
    pts_flat = pts_cam.reshape(-1, 3)
    c2w = poses[mid_idx]
    pts_world = (c2w[:3, :3] @ pts_flat.T + c2w[:3, 3:4]).T

    depth_flat = depth_np.reshape(-1)
    valid = np.isfinite(pts_world).all(axis=1) & (depth_flat > 0)
    if valid.any():
        depth_valid = depth_flat[valid]
        near_cutoff = np.quantile(depth_valid, 0.9)
        valid = valid & (depth_flat <= near_cutoff)

    pts_world = pts_world[valid]
    colors = rgb_np.reshape(-1, 3)[valid]
    if len(pts_world) > 120000:
        sel = np.random.choice(len(pts_world), 120000, replace=False)
        pts_world = pts_world[sel]
        colors = colors[sel]

    return pts_world, colors, n_frames, mid_idx, start_idx, end_idx


def main():
    base = Path("/mnt/task_runtime/EgoX-EgoPriorRenderer/processed/h2o")
    input_dir = base / "vipe_results" / "subject1_h1_0_cam0"
    videos_dir = base / "videos" / "subject1_h1_0_cam0"
    sample = load_meta_sample(base / "meta.json")

    ego_traj = np.asarray(
        [camera_center_from_w2c(extr_3x4) for extr_3x4 in sample["ego_extrinsics"]],
        dtype=np.float32,
    )
    exo_center = camera_center_from_w2c(sample["camera_extrinsics"])[None, :]

    rows = [
        {
            "title": "exo->ego",
            "pc_artifact": "exo",
            "traj_name": "ego",
            "traj": ego_traj,
            "prior_video": videos_dir / "ego_Prior.mp4",
        },
        {
            "title": "ego->exo",
            "pc_artifact": "ego",
            "traj_name": "exo",
            "traj": exo_center,
            "prior_video": videos_dir / "exo_Prior.mp4",
        },
    ]

    fig = plt.figure(figsize=(18, 10))
    for r, cfg in enumerate(rows):
        points, colors, n_frames, mid_idx, start_idx, end_idx = get_midframe_pointcloud(
            input_dir, cfg["pc_artifact"], agg_window=DEFAULT_AGG_WINDOW
        )
        traj = cfg["traj"]
        traj_mid = min(mid_idx, len(traj) - 1)
        prior_img, prior_total, prior_mid = load_video_frame(cfg["prior_video"], mid_idx)

        ax1 = fig.add_subplot(2, 3, r * 3 + 1, projection="3d")
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors / 255.0, s=0.5, alpha=0.85)
        ax1.set_title(
            f"{cfg['title']} | {cfg['pc_artifact']} point cloud (frames {start_idx}-{end_idx-1}, center {mid_idx})"
        )

        ax2 = fig.add_subplot(2, 3, r * 3 + 2, projection="3d")
        if len(traj) > 1:
            ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], "-o", markersize=2, linewidth=1)
        ax2.scatter([traj[traj_mid, 0]], [traj[traj_mid, 1]], [traj[traj_mid, 2]], c="r", s=30)
        traj_len_text = f"N={len(traj)}"
        if cfg["traj_name"] == "exo":
            traj_len_text += " (fixed cam)"
        ax2.set_title(f"{cfg['title']} | {cfg['traj_name']} traj ({traj_len_text})")

        ax3 = fig.add_subplot(2, 3, r * 3 + 3)
        ax3.imshow(prior_img)
        mean_val = float(prior_img.mean())
        ax3.set_title(
            f"{cfg['title']} | {cfg['traj_name']} prior frame {prior_mid}/{prior_total} | mean={mean_val:.2f}"
        )
        ax3.axis("off")

    plt.tight_layout()
    out_path = base / "viz_exo2ego_vs_ego2exo_midframe.png"
    fig.savefig(out_path, dpi=180)
    print(str(out_path))


if __name__ == "__main__":
    main()
