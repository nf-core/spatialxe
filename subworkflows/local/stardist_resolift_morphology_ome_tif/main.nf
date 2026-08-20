//
// Run stardist nuclei segmentation on the morphology tiff
//

include { RESOLIFT                         } from '../../../modules/local/resolift/main'
include { EXTRACT_DAPI                     } from '../../../modules/local/utility/extract_dapi/main'
include { STARDIST as STARDIST_NUCLEI      } from '../../../modules/nf-core/stardist/main'
include { CONVERT_MASK_UINT32              } from '../../../modules/local/utility/convert_mask_uint32/main'
include { XENIUMRANGER_IMPORTSEGMENTATION  } from '../../../modules/nf-core/xeniumranger/importsegmentation/main'

workflow STARDIST_RESOLIFT_MORPHOLOGY_OME_TIF {
    take:
    ch_morphology_image    // channel: [ val(meta), ["path-to-morphology.ome.tiff"] ]
    ch_bundle_path         // channel: [ val(meta), ["path-to-xenium-bundle"] ]
    sharpen_tiff           // value: bool
    stardist_nuclei_model  // value: stardist pretrained model name
    expansion_distance     // value: nuclear expansion distance

    main:

    ch_imp_seg_inputs = channel.empty()
    ch_coordinate_space = channel.value("pixels")

    // Use default model when no model is provided
    stardist_model = stardist_nuclei_model ?: '2D_versatile_fluo'

    // sharpen morphology tiff if param - sharpen_tiff is true
    if (sharpen_tiff) {

        RESOLIFT(ch_morphology_image)

        ch_image = RESOLIFT.out.enhanced_tiff
    }
    else {

        ch_image = ch_morphology_image
    }

    // Extract DAPI channel for StarDist (expects single-channel input)
    EXTRACT_DAPI(ch_image)

    // Run StarDist nuclei segmentation on DAPI channel
    STARDIST_NUCLEI(EXTRACT_DAPI.out.dapi, [stardist_model, []])

    // Convert mask to uint32 for XeniumRanger compatibility
    CONVERT_MASK_UINT32(STARDIST_NUCLEI.out.mask)

    // Run import-segmentation with nuclei only
    // XeniumRanger expands nuclei by expansion_distance to create cell boundaries
    ch_imp_seg_inputs = ch_bundle_path
        .combine(CONVERT_MASK_UINT32.out.mask, by: 0)
        .map { meta, bundle, nuclei_seg ->
            tuple(
                meta,
                bundle,
                [],
                [],
                nuclei_seg,
                [],
                [],
                ch_coordinate_space.val,
                expansion_distance,
            )
        }
    XENIUMRANGER_IMPORTSEGMENTATION(
        ch_imp_seg_inputs
    )

    emit:
    coordinate_space = ch_coordinate_space // channel: [ ["pixels"] ]
    redefined_bundle = XENIUMRANGER_IMPORTSEGMENTATION.out.outs // channel: [ val(meta), ["redefined-xenium-bundle"] ]
}
