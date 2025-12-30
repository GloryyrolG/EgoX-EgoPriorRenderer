#!/usr/bin/env python3
import argparse
import glob
import os
from typing import Tuple

import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


def load_image_grayscale(image_path: str) -> np.ndarray:
    if PIL_AVAILABLE:
        img = Image.open(image_path).convert("L")
        return np.asarray(img, dtype=np.float32)
    import imageio.v3 as iio
    arr = iio.imread(image_path)
    if arr.ndim == 3:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        arr = 0.299 * r + 0.587 * g + 0.114 * b
    return arr.astype(np.float32)


def make_mask(gray: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    logical = (gray >= threshold).astype(np.uint8)
    vis = (logical * 255).astype(np.uint8)
    return logical, vis


def main():
    parser = argparse.ArgumentParser(description="Create masks for all frames in a render directory and compute white-pixel metrics.")
    parser.add_argument("--dir", required=True, help="Render directory containing frame_*.png")
    parser.add_argument("--threshold", type=float, default=30.0, help="Threshold in [0,255]")
    args = parser.parse_args()

    # Collect only original frames, skip already-generated mask images
    frame_paths = sorted(glob.glob(os.path.join(args.dir, "frame_*.png")))
    frame_paths = [p for p in frame_paths if not p.endswith("_mask.png")]
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in: {args.dir}")

    total_white_pixels = 0
    frames_with_any_white = 0

    for frame_path in frame_paths:
        gray = load_image_grayscale(frame_path)
        logical, vis = make_mask(gray, args.threshold)

        mask_path = os.path.splitext(frame_path)[0] + "_mask.png"
        if PIL_AVAILABLE:
            Image.fromarray(vis).save(mask_path)
        else:
            import imageio.v3 as iio
            iio.imwrite(mask_path, vis)

        ones = int(logical.sum())
        total_white_pixels += ones
        if ones > 0:
            frames_with_any_white += 1

    print(f"total_white_pixels={total_white_pixels}")
    print(f"frames_with_any_white={frames_with_any_white}")
    print(f"num_frames={len(frame_paths)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import glob
import os
from typing import Tuple

import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


def load_image_grayscale(image_path: str) -> np.ndarray:
    if PIL_AVAILABLE:
        img = Image.open(image_path).convert("L")
        return np.asarray(img, dtype=np.float32)
    import imageio.v3 as iio
    arr = iio.imread(image_path)
    if arr.ndim == 3:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        arr = 0.299 * r + 0.587 * g + 0.114 * b
    return arr.astype(np.float32)


def make_mask(gray: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    logical = (gray >= threshold).astype(np.uint8)
    vis = (logical * 255).astype(np.uint8)
    return logical, vis


def main():
    parser = argparse.ArgumentParser(description="Create masks for all frames in a render directory and compute white-pixel metrics.")
    parser.add_argument("--dir", required=True, help="Render directory containing frame_*.png")
    parser.add_argument("--threshold", type=float, default=30.0, help="Threshold in [0,255]")
    args = parser.parse_args()

    frame_paths = sorted(glob.glob(os.path.join(args.dir, "frame_*.png")))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in: {args.dir}")

    total_white_pixels = 0
    frames_with_any_white = 0

    for frame_path in frame_paths:
        gray = load_image_grayscale(frame_path)
        logical, vis = make_mask(gray, args.threshold)

        mask_path = os.path.splitext(frame_path)[0] + "_mask.png"
        if PIL_AVAILABLE:
            Image.fromarray(vis).save(mask_path)
        else:
            import imageio.v3 as iio
            iio.imwrite(mask_path, vis)

        ones = int(logical.sum())
        total_white_pixels += ones
        if ones > 0:
            frames_with_any_white += 1

    # Print simple key=value pairs for easy parsing in bash
    print(f"total_white_pixels={total_white_pixels}")
    print(f"frames_with_any_white={frames_with_any_white}")
    print(f"num_frames={len(frame_paths)}")


if __name__ == "__main__":
    main()


