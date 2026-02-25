from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
from PIL import Image


matplotlib.use("Agg")

PREPROC_HEIGHT = 448


def load_preprocessed_midframe(video_path: Path, frame_hint: int):
    """Load one frame and resize to training-style ego resolution (square 448x448)."""
    reader = imageio.get_reader(str(video_path), "ffmpeg")
    try:
        total_raw = reader.get_length()
        if not np.isfinite(total_raw) or total_raw <= 0:
            total_raw = None
    except Exception:
        total_raw = None

    frame_idx = frame_hint
    if total_raw is not None:
        frame_idx = int(np.clip(frame_hint, 0, int(total_raw) - 1))
    frame = reader.get_data(frame_idx)
    reader.close()

    target_h = PREPROC_HEIGHT
    target_w = PREPROC_HEIGHT  # ego branch uses square width=height
    frame_resized = np.array(
        Image.fromarray(frame).resize((target_w, target_h), resample=Image.BILINEAR)
    )

    total_effective = int(total_raw) if total_raw is not None else 49
    return frame_resized.astype(np.uint8), total_effective, frame_idx


def main():
    base = Path("/mnt/task_runtime/EgoX-EgoPriorRenderer/processed/h2o")
    videos_dir = base / "videos" / "subject1_h1_0_cam0"

    gt_video = videos_dir / "ego.mp4"
    prior_orig = videos_dir / "ego_Prior.mp4"
    # New prior rendered with (approximately) GT intrinsics and same training-style resize.
    prior_gtint = videos_dir / "subject1_h1_0_cam0" / "ego_Prior_gtint.mp4"

    if not gt_video.exists():
        raise FileNotFoundError(gt_video)
    if not prior_orig.exists():
        raise FileNotFoundError(prior_orig)
    if not prior_gtint.exists():
        raise FileNotFoundError(prior_gtint)

    # Use midframe index based on GT length.
    reader = imageio.get_reader(str(gt_video), "ffmpeg")
    try:
        total_gt_raw = reader.get_length()
        if not np.isfinite(total_gt_raw) or total_gt_raw <= 0:
            total_gt_raw = 49
    except Exception:
        total_gt_raw = 49
    reader.close()
    total_gt = int(total_gt_raw)
    mid_idx = total_gt // 2

    # Row 0: original prior vs GT
    gt_img_orig, gt_total, gt_mid = load_preprocessed_midframe(gt_video, mid_idx)
    prior_img_orig, prior_total_orig, prior_mid_orig = load_preprocessed_midframe(
        prior_orig, mid_idx
    )
    overlay_orig = (
        0.5 * gt_img_orig.astype(np.float32)
        + 0.5 * prior_img_orig.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    # Row 1: GT-intrinsics prior vs GT
    gt_img_gtint, gt_total2, gt_mid2 = load_preprocessed_midframe(gt_video, mid_idx)
    prior_img_gtint, prior_total_gtint, prior_mid_gtint = load_preprocessed_midframe(
        prior_gtint, mid_idx
    )
    overlay_gtint = (
        0.5 * gt_img_gtint.astype(np.float32)
        + 0.5 * prior_img_gtint.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    fig = plt.figure(figsize=(18, 10))
    rows = [
        ("orig prior", prior_img_orig, overlay_orig, prior_mid_orig, prior_total_orig),
        ("gt-int prior", prior_img_gtint, overlay_gtint, prior_mid_gtint, prior_total_gtint),
    ]

    for r, (row_name, prior_img, overlay, prior_mid, prior_total) in enumerate(rows):
        # Column 1: GT
        ax_gt = fig.add_subplot(2, 3, r * 3 + 1)
        ax_gt.imshow(gt_img_orig)
        ax_gt.set_title(f"GT (frame {gt_mid}/{gt_total})")
        ax_gt.axis("off")

        # Column 2: prior
        ax_prior = fig.add_subplot(2, 3, r * 3 + 2)
        ax_prior.imshow(prior_img)
        ax_prior.set_title(f"{row_name} (frame {prior_mid}/{prior_total})")
        ax_prior.axis("off")

        # Column 3: overlay
        ax_ov = fig.add_subplot(2, 3, r * 3 + 3)
        ax_ov.imshow(overlay)
        diff_mean = float(
            np.abs(gt_img_orig.astype(np.float32) - prior_img.astype(np.float32)).mean()
        )
        ax_ov.set_title(f"{row_name} overlay (mean diff={diff_mean:.2f})")
        ax_ov.axis("off")

    plt.tight_layout()
    out_path = base / "viz_prior_orig_vs_gtint_vs_gt_midframe.png"
    fig.savefig(out_path, dpi=180)
    print(str(out_path))


if __name__ == "__main__":
    main()

