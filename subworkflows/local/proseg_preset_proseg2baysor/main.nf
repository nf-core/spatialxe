//
// Runs proseg for the xenium format and proseg2baysor to generate cell ploygons
//

include { PROSEG                           } from '../../../modules/local/proseg/preset/main'
include { PROSEG2BAYSOR                    } from '../../../modules/local/proseg/proseg2baysor/main'
include { XENIUMRANGER_IMPORTSEGMENTATION  } from '../../../modules/nf-core/xeniumranger/importsegmentation/main'

workflow PROSEG_PRESET_PROSEG2BAYSOR {
    take:
    ch_bundle_path         // channel: [ val(meta), ["path-to-xenium-bundle"] ]
    ch_transcripts_file // channel: [ val(meta), [ "transcripts.parquet" ] ]
    expansion_distance      // value: nuclear expansion distance

    main:

    ch_coordinate_space = channel.value("microns")
    ch_proseg_mode = channel.value("xenium")
    ch_expected_fmt = channel.value(["csv.gz", "csv.gz", "csv.gz"])

    // run proseg with the xenium format
    PROSEG(ch_transcripts_parquet, ch_proseg_mode, ch_expected_fmt)

    // run proseg-to-baysor on the zarr output from proseg v3
    PROSEG2BAYSOR(PROSEG.out.zarr)


    // run xeniumranger import-segmentation
    ch_imp_seg_inputs = ch_bundle_path
        .combine(PROSEG2BAYSOR.out.transcript_metadata, by: 0)
        .combine(PROSEG2BAYSOR.out.cell_polygons, by: 0)
        .map { meta, bundle, metadata, polygons2d ->
            tuple(
                meta,
                bundle,
                metadata,
                polygons2d,
                [],
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
    coordinate_space = ch_coordinate_space // channel: [ "microns" ]
    redefined_bundle = XENIUMRANGER_IMPORTSEGMENTATION.out.outs // channel: [ val(meta), ["redefined-xenium-bundle"] ]
}
