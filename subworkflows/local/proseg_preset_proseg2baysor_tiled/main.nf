//
// Runs proseg with tiling: divide transcripts -> proseg per patch -> proseg2baysor -> stitch -> xeniumranger
//

include { XENIUM_PATCH_DIVIDE              } from '../../../modules/local/xenium_patch/divide/main'
include { PROSEG                           } from '../../../modules/local/proseg/preset/main'
include { PROSEG2BAYSOR                    } from '../../../modules/local/proseg/proseg2baysor/main'
include { XENIUM_PATCH_STITCH              } from '../../../modules/local/xenium_patch/stitch/main'
include { XENIUMRANGER_IMPORTSEGMENTATION  } from '../../../modules/nf-core/xeniumranger/importsegmentation/main'

include { XENIUM_PATCH_STITCH              } from '../../../modules/local/xenium_patch/stitch/main'
include { XENIUM_PATCH_DIVIDE              } from '../../../modules/local/xenium_patch/divide/main'

workflow PROSEG_PRESET_PROSEG2BAYSOR_TILED {

    take:
    ch_bundle_path         // channel: [ val(meta), ["path-to-xenium-bundle"] ]
    ch_transcripts_file // channel: [ val(meta), [ "transcripts.parquet" ] ]
    expansion_distance      // value: nuclear expansion distance

    main:

    ch_coordinate_space = channel.value("microns")
    ch_proseg_mode = channel.value("xenium")
    ch_expected_fmt = channel.value(["csv.gz", "csv.gz", "csv.gz"])

    // Step 1: Divide transcripts into overlapping patches
    XENIUM_PATCH_DIVIDE ( ch_transcripts_parquet )

    // Step 2: Fan out patches for parallel processing
    // transpose() emits one item per patch file: [meta, parquet_path]
    ch_patches = XENIUM_PATCH_DIVIDE.out.patch_transcripts
        .transpose()
        .map { meta, parquet_file ->
            def patch_id = parquet_file.parent.name
            def patch_meta = meta.clone()
            patch_meta.sample_id = meta.id
            patch_meta.patch_id = patch_id
            patch_meta.id = "${meta.id}_${patch_id}"
            tuple(patch_meta, parquet_file)
        }

    // Step 3: Run proseg on each patch independently
    PROSEG(ch_patches, ch_proseg_mode, ch_expected_fmt)

    // Step 4: Convert proseg output to baysor format per patch
    PROSEG2BAYSOR ( PROSEG.out.zarr )

    // Step 5: Gather patch results per sample for stitching
    ch_for_stitch = PROSEG2BAYSOR.out.cell_polygons
        .join(PROSEG2BAYSOR.out.transcript_metadata, by: 0)
        .map { patch_meta, geojson, csv ->
            tuple(patch_meta.sample_id, [patch_meta.patch_id, csv, geojson])
        }
        .groupTuple(by: 0)
        .map { sample_id, patch_data ->
            def sorted = patch_data.sort { it -> it[0] }
            def patch_ids = sorted.collect { it -> it[0] }
            def csvs = sorted.collect { it -> it[1] }
            def geojsons = sorted.collect { it -> it[2] }
            tuple(sample_id, patch_ids, csvs, geojsons)
        }

    // Combine with grid metadata from DIVIDE
    ch_stitch_input = ch_for_stitch
        .join(
            XENIUM_PATCH_DIVIDE.out.grid
                .map { meta, grid -> tuple(meta.id, grid) }
        )
        .map { sample_id, patch_ids, csvs, geojsons, grid_json ->
            def meta = [id: sample_id]
            tuple(meta, grid_json, patch_ids, csvs, geojsons)
        }

    // Step 6: Stitch patch results into unified segmentation output
    XENIUM_PATCH_STITCH ( ch_stitch_input )

    // Step 7: Run xeniumranger import-segmentation
    // Note: Cell size filtering is handled inline by STITCH via --filter-method
    ch_xr = ch_bundle_path
        .combine(XENIUM_PATCH_STITCH.out.xr_polygons_transcript, by: 0)
        .combine(ch_coordinate_space)
        .map { meta, bundle, geojson, csv, coord_space ->
            tuple(meta, bundle, csv, geojson, [], [], [], coord_space, expansion_distance)
        }

    XENIUMRANGER_IMPORTSEGMENTATION ( ch_xr )

    emit:

    coordinate_space = ch_coordinate_space                          // channel: [ "microns" ]
    redefined_bundle = XENIUMRANGER_IMPORTSEGMENTATION.out.outs    // channel: [ val(meta), ["redefined-xenium-bundle"] ]
}
