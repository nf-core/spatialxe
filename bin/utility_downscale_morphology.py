#!/usr/bin/env python3
"""
Pre-downscale a morphology image for Cellpose.

Reduces image dimensions by a scale factor so that Cellpose's internal
rescaling (diam_mean / diameter) does not exceed GPU/CPU memory. The
scale factor defaults to diameter / diam_mean (e.g., 9 / 30 = 0.3).
After downscaling, Cellpose should run with --diameter equal to
diam_mean (no further internal rescaling).

Outputs:
    {prefix}/downscaled.tif - Downscaled image at the same dtype as input.
    {prefix}/scale_info.json - Scale factor and original/new dimensions.
"""

import argparse
import json
from pathlib import Path

import tifffile
from skimage.transform import resize

# Cellpose network requires a minimum spatial size of 256 px.
MIN_DIM = 256


def downscale_image(
    image_path: str, diameter: float, diam_mean: float, prefix: str
) -> None:
    """
    Downscale image so Cellpose can run with diameter == diam_mean.

    Args:
        image_path: Path to morphology TIFF (2D, 3D, or 4D).
        diameter: Target object diameter (used to compute scale).
        diam_mean: Cellpose model's mean diameter assumption.
        prefix: Output directory.
    """
    scale = min(diameter / diam_mean, 1.0)  # clamp to prevent upscaling

    img = tifffile.imread(image_path)
    print(f"Original: {img.shape}, dtype={img.dtype}, ndim={img.ndim}")

    # Handle multichannel OME-TIFFs: shape can be (H, W), (C, H, W), or (Z, C, H, W)
    output_shape: tuple[int, ...]
    if img.ndim == 2:
        orig_h, orig_w = img.shape
        new_h = max(int(orig_h * scale), MIN_DIM)
        new_w = max(int(orig_w * scale), MIN_DIM)
        output_shape = (new_h, new_w)
    elif img.ndim == 3:
        orig_h, orig_w = img.shape[1], img.shape[2]
        new_h = max(int(orig_h * scale), MIN_DIM)
        new_w = max(int(orig_w * scale), MIN_DIM)
        output_shape = (img.shape[0], new_h, new_w)
    else:
        orig_h, orig_w = img.shape[-2], img.shape[-1]
        new_h = max(int(orig_h * scale), MIN_DIM)
        new_w = max(int(orig_w * scale), MIN_DIM)
        output_shape = img.shape[:-2] + (new_h, new_w)

    print(f"Downscaling by {scale:.3f}: ({orig_h}, {orig_w}) -> ({new_h}, {new_w})")

    img_ds = resize(img, output_shape, order=3, preserve_range=True, anti_aliasing=True)
    img_ds = img_ds.astype(img.dtype)

    out_dir = Path(prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_dir / "downscaled.tif"), img_ds, compression="zlib")

    info = {
        "scale": scale,
        "orig_h": orig_h,
        "orig_w": orig_w,
        "new_h": new_h,
        "new_w": new_w,
        "diameter": diameter,
        "diam_mean": diam_mean,
    }
    with open(out_dir / "scale_info.json", "w") as f:
        json.dump(info, f)
    print(f"Done: downscaled.tif written, shape={img_ds.shape}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pre-downscale a morphology image for Cellpose."
    )
    parser.add_argument("--image", required=True, help="Morphology TIFF input")
    parser.add_argument(
        "--diameter", type=float, required=True, help="Target object diameter"
    )
    parser.add_argument(
        "--diam-mean", type=float, required=True, help="Cellpose model diam_mean"
    )
    parser.add_argument("--prefix", required=True, help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    downscale_image(
        image_path=args.image,
        diameter=args.diameter,
        diam_mean=args.diam_mean,
        prefix=args.prefix,
    )
