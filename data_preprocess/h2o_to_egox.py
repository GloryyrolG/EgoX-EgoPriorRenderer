#!/usr/bin/env python3
"""
H2O Dataset to EgoX Format Converter
Converts H2O dataset structure to EgoX-EgoPriorRenderer compatible format.

Usage:
    python data_preprocess/h2o_to_egox.py --subject subject1 --scene h1 --sequence 0 --exo_cam cam0 --start_frame 0
"""

import argparse
import json
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import subprocess


def load_h2o_intrinsics(intrinsics_path):
    """
    Load H2O intrinsics: fx fy cx cy width height
    Convert to 3x3 matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    """
    data = np.loadtxt(intrinsics_path)
    fx, fy, cx, cy, width, height = data

    intrinsics_3x3 = [
        [float(fx), 0.0, float(cx)],
        [0.0, float(fy), float(cy)],
        [0.0, 0.0, 1.0]
    ]

    return intrinsics_3x3, int(width), int(height)


def load_h2o_extrinsics(pose_dir, num_frames, start_frame=0):
    """
    Load H2O camera poses (camera-to-world 4x4)
    Convert to world-to-camera 3x4 for EgoX

    Args:
        pose_dir: Directory containing pose txt files
        num_frames: Number of frames to load
        start_frame: Starting frame index

    Returns:
        List of 3x4 matrices (world-to-camera)
    """
    extrinsics_list = []

    for i in range(num_frames):
        pose_file = pose_dir / f"{start_frame + i:06d}.txt"
        if not pose_file.exists():
            raise FileNotFoundError(f"Pose file not found: {pose_file}")

        # Load 4x4 camera-to-world matrix
        c2w = np.loadtxt(pose_file).reshape(4, 4)

        # Convert to world-to-camera
        w2c = np.linalg.inv(c2w)

        # Take 3x4 (exclude last row [0,0,0,1])
        w2c_3x4 = w2c[:3, :].tolist()

        extrinsics_list.append(w2c_3x4)

    return extrinsics_list


def load_h2o_text(h2o_root, scene, sequence, start_frame, end_frame):
    """
    Load H2O text description closest to middle frame.
    Text files exist every 5 frames at taein/database/h2o/text/.../cam2/rgb/.
    """
    text_dir = h2o_root.parent / 'taein' / 'database' / 'h2o' / 'text' / h2o_root.name / scene / str(sequence) / 'cam2' / 'rgb'
    mid_frame = (start_frame + end_frame) // 2
    nearest = round(mid_frame / 5) * 5
    text_file = text_dir / f"{nearest:06d}.txt"

    if not text_file.exists():
        raise FileNotFoundError(f"H2O text not found: {text_file}")

    return text_file.read_text().strip()


def format_h2o_prompt(text):
    """
    Format H2O text into EgoX prompt structure.
    Matches training format: [Exo/Ego view] **Scene Overview:** ... **Action Analysis:** ...
    H2O text describes hand-object interaction, placed under Action Analysis.
    """
    return (
        f"[Exo view] "
        f"**Scene Overview:** An indoor scene for hand-object interaction. "
        f"**Action Analysis:** {text} "
        f"[Ego view] "
        f"**Scene Overview:** First-person view of a hand-object interaction scene. "
        f"**Action Analysis:** {text}"
    )


def create_video_from_images(image_dir, output_video, start_frame=0, num_frames=49, fps=30):
    """
    Create MP4 video from PNG image sequence using ffmpeg.

    Args:
        image_dir: Directory containing *.png images (000000.png, 000001.png, ...)
        output_video: Output MP4 file path
        start_frame: Starting frame index
        num_frames: Number of frames to encode
        fps: Frame rate (default: 30)
    """
    print(f"Creating video from {image_dir} (frames {start_frame}-{start_frame + num_frames - 1})...")

    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-start_number', str(start_frame),
        '-i', str(image_dir / '%06d.png'),
        '-frames:v', str(num_frames),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        str(output_video)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    print(f"✓ Video created: {output_video}")


def process_h2o_sequence(h2o_root, subject, scene, sequence, exo_cam, output_dir, fps=30, start_frame=0, end_frame=None):
    """
    Process a single H2O sequence for EgoX.

    Args:
        h2o_root: Root directory of H2O dataset
        subject: Subject name (e.g., 'subject1', 'subject2', ...)
        scene: Scene name (e.g., 'h1', 'h2', 'k1', ...)
        sequence: Sequence number (e.g., 0, 1, 2, ...)
        exo_cam: Exocentric camera to use (e.g., 'cam0', 'cam1', ...)
        output_dir: Output directory for processed data
        fps: Video frame rate
        start_frame: Starting frame index
        end_frame: Ending frame index (None = start_frame + 48)
    """
    h2o_root = Path(h2o_root) / subject
    output_dir = Path(output_dir)

    # Paths
    seq_dir = h2o_root / scene / str(sequence)
    exo_dir = seq_dir / exo_cam
    ego_dir = seq_dir / 'cam4'

    # Validate
    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence not found: {seq_dir}")
    if not exo_dir.exists():
        raise FileNotFoundError(f"Exocentric camera not found: {exo_dir}")
    if not ego_dir.exists():
        raise FileNotFoundError(f"Egocentric camera not found: {ego_dir}")

    # Count frames
    rgb_files = sorted((exo_dir / 'rgb').glob('*.png'))
    total_frames = len(rgb_files)

    if end_frame is None:
        end_frame = start_frame + 48

    num_frames = end_frame - start_frame + 1

    print(f"\n{'='*60}")
    print(f"Processing: {scene}/{sequence} ({exo_cam} as exo view)")
    print(f"Frames: {start_frame} to {end_frame} ({num_frames} frames)")
    print(f"{'='*60}\n")

    # Create output structure
    take_name = f"{subject}_{scene}_{sequence}_{exo_cam}"
    videos_dir = output_dir / 'videos' / take_name
    videos_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create exo video
    exo_video = videos_dir / 'exo.mp4'
    if not exo_video.exists():
        create_video_from_images(exo_dir / 'rgb', exo_video, start_frame, num_frames, fps)
    else:
        print(f"✓ Exo video exists: {exo_video}")

    # 2. Load exo intrinsics
    exo_intrinsics, exo_w, exo_h = load_h2o_intrinsics(exo_dir / 'cam_intrinsics.txt')
    print(f"✓ Exo intrinsics loaded: {exo_w}x{exo_h}")

    # 3. Load exo extrinsics and verify fixed camera
    print(f"Loading exo poses ({num_frames} frames)...")
    exo_extrinsics = load_h2o_extrinsics(exo_dir / 'cam_pose', num_frames, start_frame)

    # Verify fixed camera assumption
    exo_array = np.array(exo_extrinsics)  # Shape: (num_frames, 3, 4)
    exo_std = np.std(exo_array, axis=0)
    max_exo_std = np.max(exo_std)

    if max_exo_std > 1e-4:
        print(f"⚠️  WARNING: {exo_cam} may NOT be fixed (max std: {max_exo_std:.6f})")
        print(f"   EgoX is trained on fixed exo cameras. Results may be suboptimal.")
    else:
        print(f"✓ Verified: {exo_cam} is fixed (max std: {max_exo_std:.10f})")

    # Use first frame for EgoX (fixed camera expects single pose)
    exo_w2c = exo_extrinsics[0]
    print(f"✓ Exo extrinsics: using frame {start_frame:06d}")

    # 4. Ego intrinsics: use standard virtual wide-FOV camera for ego_Prior rendering
    # (same as in_the_wild and egoexo4d; real cam4 intrinsics are for 1280x720,
    #  but ego_Prior is rendered at 448x448 with fish-eye distortion)
    # 为什么不能用真实 ego 相机的 intrinsics？
    # - 真实 ego 相机（如 Aria 眼镜）有鱼眼畸变、不同分辨率（如 1280×720），参数各异
    # - 但 ego_Prior 是用点云重新渲染的，渲染过程用的是统一的虚拟相机
    # - 如果填真实相机参数，会和实际渲染的 ego_Prior 视频不匹配，导致几何关系错误
    ego_intrinsics = [
        [150.0, 0.0, 255.5],
        [0.0, 150.0, 255.5],
        [0.0, 0.0, 1.0]
    ]
    print(f"✓ Ego intrinsics: virtual wide-FOV camera (f=150, cx=cy=255.5)")

    # 5. Load ego extrinsics (per-frame, should be moving)
    print(f"Loading ego poses ({num_frames} frames)...")
    ego_extrinsics = load_h2o_extrinsics(ego_dir / 'cam_pose', num_frames, start_frame)
    print(f"✓ Ego extrinsics loaded: {len(ego_extrinsics)} frames")

    # 6. Generate meta.json
    # Paths in meta.json should be relative to EgoX project root (one level above EgoX-EgoPriorRenderer)
    meta_path_prefix = Path("./EgoX-EgoPriorRenderer") / output_dir
    meta_data = {
        "test_datasets": [
            {
                "exo_path": './' + str(meta_path_prefix / exo_video.relative_to(output_dir)),
                "ego_prior_path": './' + str(meta_path_prefix / (videos_dir / 'ego_Prior.mp4').relative_to(output_dir)),
                "prompt": format_h2o_prompt(load_h2o_text(h2o_root, scene, sequence, start_frame, end_frame)),
                "camera_intrinsics": exo_intrinsics,
                "camera_extrinsics": exo_w2c,  # 3x4 world-to-camera
                "ego_intrinsics": ego_intrinsics,
                "ego_extrinsics": ego_extrinsics  # List of 3x4 per frame
            }
        ]
    }

    meta_json = output_dir / 'meta.json'
    with open(meta_json, 'w') as f:
        json.dump(meta_data, f, indent=2)

    print(f"✓ Meta JSON created: {meta_json}")

    print(f"\n{'='*60}")
    print(f"✅ Processing complete!")
    print(f"{'='*60}\n")
    print(f"Next steps:")
    print(f"1. Run ViPE inference:")
    print(f"   cd /data/rongyu_chen/projs/EgoX/EgoX-EgoPriorRenderer")
    print(f"   vipe infer {exo_video} \\")
    print(f"        --assume_fixed_camera_pose --pipeline lyra \\")
    print(f"        --use_exo_intrinsic_gt '[[{exo_intrinsics[0][0]},0,{exo_intrinsics[0][2]}],[0,{exo_intrinsics[1][1]},{exo_intrinsics[1][2]}],[0,0,1]]'")
    print(f"\n2. Render ego_Prior video:")
    print(f"   python scripts/render_vipe_pointcloud.py \\")
    print(f"        --input_dir vipe_results/{take_name} \\")
    print(f"        --out_dir {videos_dir.parent} \\")
    print(f"        --meta_json_path {meta_json} \\")
    print(f"        --point_size 5.0 \\")
    print(f"        --start_frame 0 \\")
    print(f"        --end_frame {num_frames - 1} \\")
    print(f"        --fish_eye_rendering \\")
    print(f"        --use_mean_bg \\")
    print(f"        --no_aria")
    print(f"\n3. Run EgoX inference with the generated meta.json")

    return meta_json


def main():
    parser = argparse.ArgumentParser(description='Convert H2O dataset to EgoX format')
    parser.add_argument('--h2o_root', type=str, default='/data448/rongyu_chen/dses/h2o',
                        help='Root directory of H2O dataset')
    parser.add_argument('--subject', type=str, required=True,
                        help='Subject name')
    parser.add_argument('--scene', type=str, required=True,
                        help='Scene name (h1, h2, k1, k2, o1, o2)')
    parser.add_argument('--sequence', type=int, required=True,
                        help='Sequence number (0, 1, 2, ...)')
    parser.add_argument('--exo_cam', type=str, required=True,
                        choices=['cam0', 'cam1', 'cam2', 'cam3'],
                        help='Exocentric camera to use')
    parser.add_argument('--output_dir', type=str, default='./processed/h2o',
                        help='Output directory for processed data')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video frame rate (default: 30)')
    parser.add_argument('--start_frame', type=int, required=True,
                        help='Starting frame index')
    parser.add_argument('--end_frame', type=int, default=None,
                        help='Ending frame index (default: start_frame + 48)')

    args = parser.parse_args()

    process_h2o_sequence(
        h2o_root=args.h2o_root,
        subject=args.subject,
        scene=args.scene,
        sequence=args.sequence,
        exo_cam=args.exo_cam,
        output_dir=args.output_dir,
        fps=args.fps,
        start_frame=args.start_frame,
        end_frame=args.end_frame
    )


if __name__ == '__main__':
    main()
