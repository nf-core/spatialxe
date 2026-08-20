//
// Run xeniumranger import-segmentation
//

include { XENIUMRANGER_IMPORTSEGMENTATION as IMP_SEG_COUNT_MATRIX_EXP_DISTANCE } from '../../../modules/nf-core/xeniumranger/importsegmentation/main'
include { XENIUMRANGER_IMPORTSEGMENTATION as IMP_SEG_POLYGON_GEOJSON_INPUT     } from '../../../modules/nf-core/xeniumranger/importsegmentation/main'
include { XENIUMRANGER_IMPORTSEGMENTATION as IMP_SEG_TRANS_MATRIX_INPUT        } from '../../../modules/nf-core/xeniumranger/importsegmentation/main'


workflow XENIUMRANGER_IMPORT_SEGMENTATION_REDEFINE_BUNDLE {
    take:
    ch_bundle_path             // channel: [ val(meta), [ "path-to-xenium-bundle" ] ]
    alignment_csv              // value: path to alignment csv (or null)
    expansion_distance         // value: nuclear expansion distance
    nucleus_segmentation_only  // value: bool
    qupath_polygons            // value: path to qupath polygons dir (or null)

    main:

    ch_versions = channel.empty()
    ch_redefined_bundle = channel.empty()
    ch_coordinate_space = channel.empty()

    cells = ch_bundle_path.map { meta, bundle ->
        return [meta, bundle + "/cells.zarr.zip"]
    }

    // scenario - 1 change nuclear expansion distance / create a nucleus-only count matrix(--expansion_distance=0)
    if (expansion_distance == 0 || expansion_distance != 5) {
        ch_coordinate_space = "microns"
        ch_imp_seg_inputs = ch_bundle_path
            .combine(cells, by: 0)
            .map { meta, bundle, cells_zarr ->
                tuple(
                    meta,
                    bundle,
                    [],
                    [],
                    cells_zarr,
                    [],
                    [],
                    ch_coordinate_space.val,
                    expansion_distance,
                )
            }

        IMP_SEG_COUNT_MATRIX_EXP_DISTANCE(
            ch_imp_seg_inputs
        )
        ch_redefined_bundle = IMP_SEG_COUNT_MATRIX_EXP_DISTANCE.out.outs
    }

    // scenario - 2 polygon input - geojson format (from QuPath)
    if (qupath_polygons && nucleus_segmentation_only) {

        ch_coordinate_space = "microns"
        ch_imp_seg_inputs = ch_bundle_path
            .combine(qupath_polygons)
            .map { meta, bundle, polygons_geojson ->
                tuple(
                    meta,
                    bundle,
                    [],
                    [],
                    polygons_geojson,
                    [],
                    [],
                    ch_coordinate_space.val,
                    expansion_distance,
                )
            }

        IMP_SEG_POLYGON_GEOJSON_INPUT(
            ch_imp_seg_inputs
        )
        ch_redefined_bundle = IMP_SEG_POLYGON_GEOJSON_INPUT.out.outs
    }
    else if (qupath_polygons) {

        ch_coordinate_space = "microns"
        ch_imp_seg_inputs = ch_bundle_path
            .combine(qupath_polygons)
            .map { meta, bundle, polygons_geojson ->
                tuple(
                    meta,
                    bundle,
                    [],
                    [],
                    polygons_geojson,
                    polygons_geojson,
                    [],
                    ch_coordinate_space.val,
                    expansion_distance,
                )
            }

        IMP_SEG_POLYGON_GEOJSON_INPUT(
            ch_imp_seg_inputs
        )
        ch_redefined_bundle = IMP_SEG_POLYGON_GEOJSON_INPUT.out.outs
    }

    // scenario 3 - mask input - included in the cellpose subworkflow

    // scenario 4 - transcript assignment input - included in the baysor & proseg subworkflows

    // scenario 5 - transformation matrix input
    if (qupath_polygons && alignment_csv) {

        ch_imp_seg_inputs = ch_bundle_path
            .combine(qupath_polygons)
            .combine(alignment_csv)
            .map { meta, bundle, polygons_geojson, alignment_csv_file ->
                tuple(
                    meta,
                    bundle,
                    [],
                    [],
                    polygons_geojson,
                    polygons_geojson,
                    alignment_csv_file,
                    ch_coordinate_space.val,
                    expansion_distance,
                )
            }

        IMP_SEG_TRANS_MATRIX_INPUT(
            ch_imp_seg_inputs
        )
        ch_redefined_bundle = IMP_SEG_TRANS_MATRIX_INPUT.out.outs
    }

    emit:
    redefined_bundle = ch_redefined_bundle // channel: [ val(meta), ["redefined-xenium-bundle"] ]
    coordinate_space = ch_coordinate_space // channel: [ ["pixels"] ]
    versions         = ch_versions // channel: [ versions.yml ]
}
