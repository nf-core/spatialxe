//
// Runs baysor with tiling: divide transcripts -> preprocess per patch -> baysor per patch -> stitch -> xeniumranger
//

include { BAYSOR_RUN                       } from '../../../modules/nf-core/baysor/run/main'
include { XENIUMRANGER_IMPORT_SEGMENTATION } from '../../../modules/nf-core/xeniumranger/import-segmentation/main'

include { XENIUM_PATCH_DIVIDE              } from '../../../modules/local/xenium_patch/divide/main'
include { BAYSOR_PREPROCESS_TRANSCRIPTS    } from '../../../modules/local/utility/preprocess/main'
include { XENIUM_PATCH_STITCH              } from '../../../modules/local/xenium_patch/stitch/main'
include { XENIUMRANGER_IMPORTSEGMENTATION  } from '../../../modules/nf-core/xeniumranger/importsegmentation/main'

workflow BAYSOR_RUN_TRANSCRIPTS_PARQUET_TILED {

    take:
    ch_bundle_path         // channel: [ val(meta), ["xenium-bundle"] ]
    ch_transcripts_file // channel: [ val(meta), ["transcripts.parquet"] ]
    ch_config              // channel: ["path-to-xenium.toml"]
    max_x                  // value: spatial filter upper x bound
    max_y                  // value: spatial filter upper y bound
    min_qv                 // value: minimum transcript QV
    min_x                  // value: spatial filter lower x bound
    min_y                  // value: spatial filter lower y bound
    expansion_distance     // value: nuclear expansion distance

    main:

    ch_coordinate_space = channel.value("microns")
    ch_polygon_format   = channel.value("GeometryCollectionLegacy")

    // Step 1: Divide transcripts into overlapping patches
    XENIUM_PATCH_DIVIDE ( ch_transcripts_parquet )

    // Step 2: Fan out patches for parallel processing
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

    // Step 3: Preprocess each patch's parquet to CSV for Baysor 0.7.1 compatibility
    // Baysor's Julia Parquet.jl cannot read zstd-compressed parquet files
    BAYSOR_PREPROCESS_TRANSCRIPTS (
        ch_patches,
        min_qv,
        max_x,
        min_x,
        max_y,
        min_y
    )

    // Step 4: // estimate scale factor which specifed the cell radius for each patch
    BAYSOR_ESTIMATE_SCALE_FACTOR (
        ch_transcripts_parquet,
        ch_prior_column,
        ch_x_column,
        ch_y_column,
        ch_transcripts_per_cell
    )
    ch_scale_factor = BAYSOR_ESTIMATE_SCALE_FACTOR.out.scale_factor

    // Step 4: Run Baysor on each patch independently
    ch_baysor_input = BAYSOR_PREPROCESS_TRANSCRIPTS.out.transcripts_csv
        .combine(ch_config)
        .combine(ch_scale_factor)
        .map { meta, transcripts, config, scale_factor ->
            tuple(meta, transcripts, [], config, scale_factor)
        }

    BAYSOR_RUN ( ch_baysor_input, [], [], ch_polygon_format )

    // Step 5: Gather patch results per sample for stitching
    ch_for_stitch = BAYSOR_RUN.out.segmentation
        .map { patch_meta, csv, polygons ->
            tuple(patch_meta.sample_id, [patch_meta.patch_id, csv, polygons])
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
