//
// generate spatialdata object from the spatialaxe layers
//

include { SPATIALDATA_META                                        } from '../../../modules/local/spatialdata/meta/main'
include { SPATIALDATA_WRITE as SPATIALDATA_WRITE_RAW_BUNDLE       } from '../../../modules/local/spatialdata/write/main'
include { SPATIALDATA_MERGE as SPATIALDATA_MERGE_RAW_REDEFINED    } from '../../../modules/local/spatialdata/merge/main'
include { SPATIALDATA_WRITE as SPATIALDATA_WRITE_REDEFINED_BUNDLE } from '../../../modules/local/spatialdata/write/main'

workflow SPATIALDATA_WRITE_META_MERGE {
    take:
    ch_bundle_path             // channel: [ val(meta), [ "path-to-xenium-bundle" ] ]
    ch_redefined_bundle        // channel: [ val(meta), [ "redefined-xenium-bundle" ] ]
    ch_coordinate_space        // channel: [ "pixels" or "microns" ]
    cell_segmentation_only     // value: bool
    mode                       // value: pipeline mode (image/coordinate/...)
    nucleus_segmentation_only  // value: bool
    run_qc                     // value: bool

    main:

    ch_segmented_object = channel.empty()
    sd_redefined_bundle = channel.empty()
    sd_merged_bundle    = channel.empty()
    sd_metadata         = channel.empty()
    ch_coordinate_space_raw = channel.empty()

    // check segmentation - only nuclei, cells or both cells & nuclei
    if (mode == 'image') {

        if (nucleus_segmentation_only && cell_segmentation_only) {
            ch_segmented_object = channel.value('cells_and_nuclei')
        }
        else if (nucleus_segmentation_only) {
            ch_segmented_object = channel.value('nuclei')
        }
        else if (cell_segmentation_only) {
            ch_segmented_object = channel.value('cells')
        }
        else {
            ch_segmented_object = channel.value([])
        }
    }

    // set all boundaries as false - default
    if (mode == 'coordinate' || mode == 'qc') {
        ch_segmented_object = channel.value([])
    }

    if (mode == 'qc' || run_qc) {
        ch_coordinate_space_raw = channel.value("all")
    } else {
        ch_coordinate_space_raw = channel.value("cells_and_nuclei")
    }

    // write spatialdata object from the raw xenium bundle
    SPATIALDATA_WRITE_RAW_BUNDLE(
        ch_bundle_path,
        'raw_bundle',
        ch_segmented_object,
        ch_coordinate_space_raw,
    )

    if (mode != 'qc') {

        // write spatialdata object after running IMP_SEG
        SPATIALDATA_WRITE_REDEFINED_BUNDLE(
            ch_redefined_bundle,
            'redefined_bundle',
            ch_segmented_object,
            ch_coordinate_space,
        )
        sd_redefined_bundle = SPATIALDATA_WRITE_REDEFINED_BUNDLE.out.spatialdata


        // merge raw & redefined spatialdata objects
        SPATIALDATA_MERGE_RAW_REDEFINED(
            SPATIALDATA_WRITE_RAW_BUNDLE.out.spatialdata.combine(ch_redefined_bundle, by: 0),
            'merged_bundle'
        )
        sd_merged_bundle = SPATIALDATA_MERGE_RAW_REDEFINED.out.merged_bundle


        // write metadata with spatialdata object
        SPATIALDATA_META(
            SPATIALDATA_MERGE_RAW_REDEFINED.out.merged_bundle.combine(ch_bundle_path, by: 0),
            'metadata'
        )
        sd_metadata = SPATIALDATA_META.out.metadata

    }

    emit:
    sd_raw_bundle       = SPATIALDATA_WRITE_RAW_BUNDLE.out.spatialdata // channel: [ val(meta), "spatialdata_raw" ]
    sd_redefined_bundle = sd_redefined_bundle // channel: [ val(meta), "spatialdata_redefined" ]
    sd_merged_bundle    = sd_merged_bundle    // channel: [ val(meta), "spatialdata_merged" ]
    sd_metadata         = sd_metadata         // channel: [ val(meta), "spatialdata_meta" ]
}
