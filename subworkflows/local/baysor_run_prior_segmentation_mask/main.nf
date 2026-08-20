//
// Run baysor run & import-segmentation
//

include { BAYSOR_PREPROCESS_TRANSCRIPTS    } from '../../../modules/local/baysor/preprocess/main'
include { BAYSOR_RUN                       } from '../../../modules/local/baysor/run/main'
include { XENIUMRANGER_IMPORTSEGMENTATION  } from '../../../modules/nf-core/xeniumranger/importsegmentation/main'

include { BAYSOR_PREPROCESS_TRANSCRIPTS    } from '../../../modules/local/utility/preprocess/main'
include { BAYSOR_ESTIMATE_SCALE_FACTOR     } from '../../../modules/local/utility/estimatescalefactor/main'

workflow BAYSOR_RUN_PRIOR_SEGMENTATION_MASK {

    take:
    ch_bundle_path         // channel: [ val(meta), ["path-to-xenium-bundle"] ]
    ch_transcripts_file // channel: [ val(meta), ["path-to-transcripts.parquet"] ]
    ch_segmentation_mask   // channel: [ ["path-to-prior-segmentation-mask"] ]
    ch_config              // channel: [ "path-to-xenium.toml" ]
    max_x                  // value: spatial filter upper x bound
    max_y                  // value: spatial filter upper y bound
    min_qv                 // value: minimum transcript QV
    min_x                  // value: spatial filter lower x bound
    min_y                  // value: spatial filter lower y bound
    expansion_distance     // value: nuclear expansion distance

    main:

    ch_transcripts      = channel.empty()
    ch_redefined_bundle = channel.empty()
    ch_coordinate_space = channel.value("pixels")
    ch_x_column         = channel.value("x_location")
    ch_y_column         = channel.value("y_location")
    ch_polygon_format   = channel.value("GeometryCollectionLegacy")

    // Always preprocess transcripts.parquet to CSV for Baysor 0.7.1 compatibility.
    // Baysor's Julia Parquet.jl cannot read zstd-compressed parquet files from Xenium bundles.
    // Also applies optional spatial/QV filtering when filter_transcripts is true.
    BAYSOR_PREPROCESS_TRANSCRIPTS(
        ch_transcripts_parquet,
        min_qv,
        max_x,
        min_x,
        max_y,
        min_y,
    )
    ch_transcripts = BAYSOR_PREPROCESS_TRANSCRIPTS.out.transcripts_csv


    // estimated scale factor cell radius
    BAYSOR_ESTIMATE_SCALE_FACTOR (
        ch_transcripts_parquet,
        ch_prior_column,
        ch_x_column,
        ch_y_column,
        ch_transcripts_per_cell
    )
    ch_scale_factor = BAYSOR_ESTIMATE_SCALE_FACTOR.out.scale_factor


    // run baysor with prior segmentation mask with the estimated scale factor
    ch_baysor_input = BAYSOR_PREPROCESS_TRANSCRIPTS.out.transcripts_csv
        .combine(ch_segmentation_mask)
        .combine(ch_config)
        .combine(ch_scale_factor)
        .map { meta, transcripts, mask, config, scale_factor ->
            tuple(
                meta,
                transcripts,
                mask,
                config,
                scale_factor,
            )
        }
    BAYSOR_RUN(ch_baysor_input, ch_prior_column, ch_prior_confidence, ch_polygon_format)


    // run import-segmentation with baysor outs
    ch_imp_seg_inputs = ch_bundle_path
        .combine(BAYSOR_RUN.out.segmentation, by: 0)
        .map { meta, bundle, _segmentation_csv, polygons2d ->
            tuple(
                meta,
                bundle,
                [],
                [],
                polygons2d,
                polygons2d,
                [],
                ch_coordinate_space.val,
                expansion_distance,
            )
        }
    XENIUMRANGER_IMPORTSEGMENTATION(
        ch_imp_seg_inputs
    )

    ch_redefined_bundle = XENIUMRANGER_IMPORTSEGMENTATION.out.outs

    emit:

    coordinate_space = ch_coordinate_space // channel: [ "pixels" ]
    redefined_bundle = ch_redefined_bundle // channel: [ val(meta), ["redefined-xenium-bundle"] ]
}
