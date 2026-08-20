#!/usr/bin/env python3
"""Write spatialdata object from segmentation format."""

import argparse
import sys

import pandas as pd
from spatialdata_io import xenium

# Fix zarr v3 + anndata + numcodecs incompatibility:
# anndata's string writer passes numcodecs.VLenUTF8 to zarr.Group.create_array,
# but zarr v3 only accepts ArrayArrayCodec types. OME-Zarr 0.5 requires zarr v3
# for images, so we can't downgrade the store format. Instead, we intercept
# create_array to strip numcodecs codecs and let zarr v3 handle strings natively.
import numcodecs
import zarr.core.group as _zarr_group

_orig_create_array = _zarr_group.Group.create_array


def _v3_compat_create_array(self, *args, **kwargs):
    """Strip numcodecs VLenUTF8 from codec params for zarr v3 compatibility."""
    for param in ("filters", "compressor", "object_codec"):
        val = kwargs.get(param)
        if val is None:
            continue
        if isinstance(val, numcodecs.vlen.VLenUTF8):
            del kwargs[param]
        elif isinstance(val, (list, tuple)):
            cleaned = [v for v in val if not isinstance(v, numcodecs.vlen.VLenUTF8)]
            if len(cleaned) != len(val):
                if cleaned:
                    kwargs[param] = cleaned
                else:
                    del kwargs[param]
    return _orig_create_array(self, *args, **kwargs)


_zarr_group.Group.create_array = _v3_compat_create_array


def _is_arrow_backed(dtype):
    """Check if a pandas dtype is backed by PyArrow."""
    return (
        isinstance(dtype, pd.ArrowDtype)
        or (hasattr(dtype, "storage") and getattr(dtype, "storage", None) == "pyarrow")
        or "pyarrow" in str(dtype)
    )


def _convert_df_arrow_to_numpy(df):
    """Convert Arrow-backed dtypes in a DataFrame to numpy object dtype.

    Handles three cases:
    1. Regular columns with Arrow-backed dtypes
    2. Categorical columns whose categories are Arrow-backed
    3. Index with Arrow-backed dtype
    """
    for col in df.columns:
        dtype = df[col].dtype
        if _is_arrow_backed(dtype):
            df[col] = df[col].astype("object")
        elif isinstance(dtype, pd.CategoricalDtype):
            cats = dtype.categories
            if cats is not None and _is_arrow_backed(cats.dtype):
                df[col] = df[col].cat.rename_categories(cats.astype("object"))
    if _is_arrow_backed(df.index.dtype):
        df.index = pd.Index(df.index.astype("object"))


def convert_arrow_to_numpy(sdata):
    """Convert Arrow-backed dtypes to numpy for anndata zarr write compatibility."""
    for table_key in list(sdata.tables.keys()):
        adata = sdata.tables[table_key]
        _convert_df_arrow_to_numpy(adata.obs)
        _convert_df_arrow_to_numpy(adata.var)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Write spatialdata object from segmentation format")
    parser.add_argument("--bundle", required=True, help="Path to input bundle")
    parser.add_argument("--prefix", required=True, help="Output prefix (sample ID)")
    parser.add_argument("--output-folder", required=True, help="Output folder name")
    parser.add_argument("--segmented-object", required=True, help="Segmented object type (cells, nuclei, cells_and_nuclei)")
    parser.add_argument("--coordinate-space", required=True, help="Coordinate space (pixels, microns)")
    parser.add_argument("--format", required=True, help="Input format (xenium)")
    return parser.parse_args()


def main():
    """Run spatialdata write."""
    args = parse_args()
    print("[START]")

    cells_as_circles = False
    cells_boundaries = False
    nucleus_boundaries = False
    cells_labels = False
    nucleus_labels = False

    if args.segmented_object == "cells":
        cells_boundaries = True
        cells_labels = True
    elif args.segmented_object == "nuclei":
        nucleus_boundaries = True
        nucleus_labels = True
    elif args.segmented_object == "cells_and_nuclei":
        cells_boundaries = True
        nucleus_boundaries = True
        cells_labels = True
        nucleus_labels = True
    else:
        cells_as_circles = False

    # set sd variables based on the coordinate space
    if args.coordinate_space == "pixels":
        cells_labels = True
        nucleus_labels = True
        # Labels are sufficient in pixel space; boundaries can contain
        # degenerate polygons (< 4 vertices) from XeniumRanger that
        # crash spatialdata_io's shapely LinearRing parser.
        cells_boundaries = False
        nucleus_boundaries = False

    if args.coordinate_space == "microns":
        cells_labels = False
        cells_boundaries = True
        nucleus_boundaries = False
        nucleus_labels = False
        cells_as_circles = False

    # set sd variables based on the coordinate space
    if args.coordinate_space == "all":
        cells_labels = True
        nucleus_labels = True
        cells_boundaries = True
        nucleus_boundaries = True
        cells_as_circles = True

    print(f"[NOTE] Apply cells_boundaries {cells_boundaries}")
    print(f"[NOTE] Apply nucleus_boundaries {nucleus_boundaries}")

    if args.format == "xenium":
        sd_xenium_obj = xenium(
            args.bundle,
            cells_as_circles=cells_as_circles,
            cells_boundaries=cells_boundaries,
            nucleus_boundaries=nucleus_boundaries,
            cells_labels=cells_labels,
            nucleus_labels=nucleus_labels,
            transcripts=True,
            morphology_mip=True,
            morphology_focus=True,
        )
        print(sd_xenium_obj)
        convert_arrow_to_numpy(sd_xenium_obj)
        sd_xenium_obj.write(f"spatialdata/{args.prefix}/{args.output_folder}")
    else:
        sys.exit("[ERROR] Format not found")

    print("[FINISH]")


if __name__ == "__main__":
    main()
