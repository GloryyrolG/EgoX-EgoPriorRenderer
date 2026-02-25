from pathlib import Path
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio.v2 as imageio
from PIL import Image

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
    reader = imageio.get_reader(str(video_path), "ffmpeg")
    total = reader.get_length()
    frame_idx = int(np.clip(frame_idx, 0, max(0, total - 1)))
    frame = reader.get_data(frame_idx)
    reader.close()
    return frame, total, frame_idx


PREPROC_MAX_FRAMES = 49
PREPROC_HEIGHT = 448
PREPROC_WIDTH = 1232


def load_preprocessed_midframe(video_path: Path, video_type: str, frame_hint: int):
    # Approximate dataset preprocessing: resize ego_gt / ego_prior frames
    # to (PREPROC_HEIGHT, PREPROC_HEIGHT) without keeping aspect ratio.
    if isinstance(video_path, Path):
        video_path = video_path.as_posix()
    reader = imageio.get_reader(str(video_path), "ffmpeg")
    try:
        total_raw = reader.get_length()
    except Exception:
        total_raw = None

    frame_idx = frame_hint
    if total_raw is not None and total_raw > 0:
        frame_idx = int(np.clip(frame_hint, 0, total_raw - 1))
    frame = reader.get_data(frame_idx)
    reader.close()

    target_h = PREPROC_HEIGHT
    target_w = PREPROC_HEIGHT  # ego branch uses square width=height
    frame_resized = np.array(
        Image.fromarray(frame).resize((target_w, target_h), resample=Image.BILINEAR)
    )

    total_effective = total_raw if total_raw is not None else PREPROC_MAX_FRAMES
    return frame_resized.astype(np.uint8), total_effective, frame_idx


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
            "gt_video": videos_dir / "ego.mp4",
        },
        {
            "title": "ego->exo",
            "pc_artifact": "ego",
            "traj_name": "exo",
            "traj": exo_center,
            "prior_video": videos_dir / "exo_Prior.mp4",
            "gt_video": videos_dir / "exo.mp4",
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
        gt_img, gt_total, gt_mid = load_video_frame(cfg["gt_video"], mid_idx)

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

    # Additional figure: GT vs prior comparison (single figure)
    fig2 = plt.figure(figsize=(18, 10))
    for r, cfg in enumerate(rows):
        _, _, n_frames, mid_idx, _, _ = get_midframe_pointcloud(
            input_dir, cfg["pc_artifact"], agg_window=DEFAULT_AGG_WINDOW
        )
        # Use same preprocessing as dataset/infer for fair comparison
        prior_img, prior_total, prior_mid = load_preprocessed_midframe(
            cfg["prior_video"], "ego_prior", mid_idx
        )
        gt_img, gt_total, gt_mid = load_preprocessed_midframe(
            cfg["gt_video"], "ego_gt", mid_idx
        )

        # Ensure same spatial size for visualization, but keep prior aspect ratio.
        h, w = gt_img.shape[:2]
        ph, pw = prior_img.shape[:2]
        if (ph, pw) != (h, w):
            scale = min(h / ph, w / pw)
            new_h = max(1, int(round(ph * scale)))
            new_w = max(1, int(round(pw * scale)))
            prior_resized_small = np.array(
                Image.fromarray(prior_img).resize((new_w, new_h), resample=Image.BILINEAR)
            )
            prior_img_resized = np.zeros_like(gt_img)
            top = (h - new_h) // 2
            left = (w - new_w) // 2
            prior_img_resized[top : top + new_h, left : left + new_w] = prior_resized_small
        else:
            prior_img_resized = prior_img

        overlay = (
            0.5 * gt_img.astype(np.float32) + 0.5 * prior_img_resized.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

        # Column 1: GT
        ax_gt = fig2.add_subplot(2, 3, r * 3 + 1)
        ax_gt.imshow(gt_img)
        ax_gt.set_title(
            f"{cfg['title']} | GT frame {gt_mid}/{gt_total}"
        )
        ax_gt.axis("off")

        # Column 2: prior
        ax_prior = fig2.add_subplot(2, 3, r * 3 + 2)
        ax_prior.imshow(prior_img_resized)
        ax_prior.set_title(
            f"{cfg['title']} | prior frame {prior_mid}/{prior_total}"
        )
        ax_prior.axis("off")

        # Column 3: overlay
        ax_ov = fig2.add_subplot(2, 3, r * 3 + 3)
        ax_ov.imshow(overlay)
        diff_mean = float(np.abs(gt_img.astype(np.float32) - prior_img_resized.astype(np.float32)).mean())
        ax_ov.set_title(
            f"{cfg['title']} | overlay (mean diff={diff_mean:.2f})"
        )
        ax_ov.axis("off")

    plt.tight_layout()
    out_path2 = base / "viz_prior_vs_gt_midframe.png"
    fig2.savefig(out_path2, dpi=180)
    print(str(out_path2))

    # GT vs gtdepth-prior overlay (exo->ego only); prefer GT-intrinsics version when present
    gtdepth_prior = videos_dir / "ego_Prior_gtdepth_gtint.mp4"
    if not gtdepth_prior.exists():
        gtdepth_prior = videos_dir / "ego_Prior_gtdepth.mp4"
    if gtdepth_prior.exists():
        _, _, n_frames, mid_idx, _, _ = get_midframe_pointcloud(
            input_dir, "exo", agg_window=DEFAULT_AGG_WINDOW
        )
        # gtdepth video may have fewer frames (e.g. single-frame render); clamp to its length
        try:
            _r = imageio.get_reader(str(gtdepth_prior), "ffmpeg")
            gtdepth_len = _r.get_length()
            _r.close()
        except Exception:
            gtdepth_len = 1
        if not np.isfinite(gtdepth_len) or gtdepth_len <= 0:
            gtdepth_len = 1
        frame_for_gtdepth = int(np.clip(mid_idx, 0, gtdepth_len - 1))
        prior_img, prior_total, prior_mid = load_preprocessed_midframe(
            gtdepth_prior, "ego_prior", frame_for_gtdepth
        )
        gt_img, gt_total, gt_mid = load_preprocessed_midframe(
            videos_dir / "ego.mp4", "ego_gt", mid_idx
        )
        h, w = gt_img.shape[:2]
        ph, pw = prior_img.shape[:2]
        if (ph, pw) != (h, w):
            scale = min(h / ph, w / pw)
            new_h = max(1, int(round(ph * scale)))
            new_w = max(1, int(round(pw * scale)))
            prior_resized_small = np.array(
                Image.fromarray(prior_img).resize((new_w, new_h), resample=Image.BILINEAR)
            )
            prior_img_resized = np.zeros_like(gt_img)
            top = (h - new_h) // 2
            left = (w - new_w) // 2
            prior_img_resized[top : top + new_h, left : left + new_w] = prior_resized_small
        else:
            prior_img_resized = prior_img
        overlay = (
            0.5 * gt_img.astype(np.float32) + 0.5 * prior_img_resized.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)
        fig3 = plt.figure(figsize=(18, 6))
        ax_gt = fig3.add_subplot(1, 3, 1)
        ax_gt.imshow(gt_img)
        ax_gt.set_title(f"GT frame {gt_mid}/{gt_total}")
        ax_gt.axis("off")
        ax_prior = fig3.add_subplot(1, 3, 2)
        ax_prior.imshow(prior_img_resized)
        ax_prior.set_title(f"gtdepth prior frame {prior_mid}/{prior_total}")
        ax_prior.axis("off")
        ax_ov = fig3.add_subplot(1, 3, 3)
        ax_ov.imshow(overlay)
        diff_mean = float(np.abs(gt_img.astype(np.float32) - prior_img_resized.astype(np.float32)).mean())
        ax_ov.set_title(f"overlay (mean diff={diff_mean:.2f})")
        ax_ov.axis("off")
        plt.tight_layout()
        out_path3 = base / "viz_prior_gtdepth_vs_gt_midframe.png"
        fig3.savefig(out_path3, dpi=180)
        print(str(out_path3))
    else:
        print(f"Skip gtdepth vs GT figure: {gtdepth_prior} not found.")


if __name__ == "__main__":
    main()
