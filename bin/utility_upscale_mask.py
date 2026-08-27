#!/usr/bin/env python3
"""
Restore Cellpose masks to original image resolution.

Uses nearest-neighbor interpolation to upscale segmentation masks back
to the original image dimensions recorded in scale_info.json (produced
by downscale_morphology.py).

Output: {prefix}/upscaled_{mask_basename}.tif
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def upscale_mask(mask_path: str, scale_info_path: str, prefix: str) -> None:
    """
    Read a downscaled mask and upscale it to original dimensions.

    Args:
        mask_path: Path to downscaled segmentation mask TIFF.
        scale_info_path: Path to scale_info.json from downscale_morphology.
        prefix: Output directory.
    """
    with open(scale_info_path) as f:
        info = json.load(f)
    orig_h, orig_w = info["orig_h"], info["orig_w"]

    mask = tifffile.imread(mask_path)
    print(
        f"Mask: {mask.shape}, dtype={mask.dtype}, "
        f"unique cells: {len(np.unique(mask)) - 1}"
    )
    print(f"Upscaling to ({orig_h}, {orig_w})")

    pil_mask = Image.fromarray(mask)
    pil_mask = pil_mask.resize((orig_w, orig_h), Image.Resampling.NEAREST)
    mask_up = np.array(pil_mask, dtype=mask.dtype)

    out_dir = Path(prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(mask_path).stem
    out_name = out_dir / f"upscaled_{base}.tif"
    tifffile.imwrite(str(out_name), mask_up, compression="zlib")
    print(f"Done: {out_name}, unique cells: {len(np.unique(mask_up)) - 1}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Upscale a Cellpose mask back to original resolution."
    )
    parser.add_argument("--mask", required=True, help="Downscaled mask TIFF")
    parser.add_argument(
        "--scale-info", required=True, help="scale_info.json from downscale step"
    )
    parser.add_argument("--prefix", required=True, help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    upscale_mask(
        mask_path=args.mask,
        scale_info_path=args.scale_info,
        prefix=args.prefix,
    )
