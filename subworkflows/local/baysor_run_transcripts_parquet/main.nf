//
// Unified Baysor subworkflow: handles both tiled and non-tiled paths.
//
// When baysor_tiling=true:  divide → per-patch Baysor → stitch → xeniumranger
// When baysor_tiling=false: preprocess → Baysor → xeniumranger
//
// Prior segmentation support:
//   Column-based (cells): works with both tiled and non-tiled
//   Image-based (cellpose): non-tiled only (mask passed to Baysor)
//

include { BAYSOR_RUN                       } from '../../../modules/nf-core/baysor/run/main'
include { XENIUMRANGER_IMPORT_SEGMENTATION } from '../../../modules/nf-core/xeniumranger/import-segmentation/main'

include { XENIUM_PATCH_DIVIDE              } from '../../../modules/local/xenium_patch/divide/main'
include { XENIUM_PATCH_STITCH              } from '../../../modules/local/xenium_patch/stitch/main'
include { PARQUET2CSV                      } from '../../../modules/local/utility/parquet2csv/main'
include { BAYSOR_PREPROCESS_TRANSCRIPTS    } from '../../../modules/local/utility/preprocess/main'
include { RECONSTRUCT_PATCHES              } from '../../../modules/local/utility/reconstruct_patches/main'
include { XENIUMRANGER_IMPORTSEGMENTATION  } from '../../../modules/nf-core/xeniumranger/importsegmentation/main'


workflow BAYSOR_RUN_TRANSCRIPTS_PARQUET {

    take:
    ch_bundle_path         // channel: [ val(meta), ["path-to-xenium-bundle"] ]
    ch_transcripts_file // channel: [ val(meta), ["transcripts.parquet"] ]
    ch_morphology_image    // channel: [ val(meta), ["morphology_focus.ome.tif"] ]
    ch_config              // channel: ["path-to-xenium.toml"]
    ch_prior_mask          // channel: [ val(meta), ["resized_mask.tif"] ] or empty (cellpose)
    baysor_config          // value: path to baysor config TOML (or null)
    baysor_scale           // value: Baysor --scale for non-tiled runs
    baysor_tiling          // value: bool — enable tiling
    baysor_tiling_scale    // value: Baysor --scale for tiled runs
    max_x                  // value: spatial filter upper x bound
    max_y                  // value: spatial filter upper y bound
    min_qv                 // value: minimum transcript QV
    min_x                  // value: spatial filter lower x bound
    min_y                  // value: spatial filter lower y bound
    expansion_distance     // value: nuclear expansion distance

    main:

    ch_x_column = channel.value("x_location")
    ch_y_column = channel.value("y_location")
    ch_coordinate_space = channel.value("microns")
    ch_polygon_format   = channel.value("GeometryCollectionLegacy")

    // Estimate scale factor which specified the cell radius for each patch
    BAYSOR_ESTIMATE_SCALE_FACTOR (
        ch_transcripts_parquet,
        ch_prior_column,
        ch_x_column,
        ch_y_column,
        ch_transcripts_per_cell
    )

    // keep meta-keyed for non-tiled join
    ch_scale_factor_by_meta = BAYSOR_ESTIMATE_SCALE_FACTOR.out.scale_factor

    // keyed by sample_id (String) for the tiled/patch path
    ch_scale_factor_by_sample = ch_scale_factor_by_meta
        .map { meta, scale_factor -> tuple(meta.id, scale_factor) }

    // ── TILED PATH ───────────────────────
    if ( baysor_tiling ) {

        // Step 1: Divide transcripts into overlapping patches
        ch_divide_input = ch_transcripts_parquet
            .join(ch_morphology_image, by: 0)

        XENIUM_PATCH_DIVIDE ( ch_divide_input )

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

        // Step 2b: Convert parquet to CSV (Baysor Julia Parquet.jl incompatibility)
        PARQUET2CSV ( ch_patches )

        // Step 3: Run Baysor on each patch independently
        // Use baysor_tiling_scale (larger than baysor_scale) to compensate for EM
        // convergence producing smaller cells on tile-sized datasets.
        ch_baysor_input = PARQUET2CSV.out.transcripts_csv
            .map { meta, csv -> tuple(meta.sample_id, meta, csv) }
            .combine(ch_config)
            .combine(ch_scale_factor_by_sample, by: 0)
            .map { _sample_id, meta, transcripts, config, scale_factor ->
                tuple(meta, transcripts, [], config ? file(config) : [], scale_factor)
            }
        BAYSOR_RUN (ch_baysor_input, [], [], ch_polygon_format)

        // Step 4: Gather patch results per sample and reconstruct patches directory
        ch_baysor_results = BAYSOR_RUN.out.segmentation
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

        ch_stitch_input = ch_baysor_results
            .join(
                XENIUM_PATCH_DIVIDE.out.grid
                    .map { meta, grid -> tuple(meta.id, grid) }
            )
            .map { sample_id, patch_ids, csvs, geojsons, grid_json ->
                def meta = [id: sample_id]
                tuple(meta, grid_json, patch_ids, csvs, geojsons)
            }

        // Step 5: Stitch patch results
        RECONSTRUCT_PATCHES ( ch_stitch_input )
        XENIUM_PATCH_STITCH ( RECONSTRUCT_PATCHES.out.patches_dir )

        // Step 6: xeniumranger import-segmentation (tiled)
        // spatialaxe signature: meta, bundle, transcript_assignment, viz_polygons, nuclei, cells, coordinate_transform, units
        ch_xr = ch_bundle_path
            .combine(XENIUM_PATCH_STITCH.out.xr_polygons_transcript, by: 0)
            .map {
                meta, bundle, xr_cell_polygons, xr_transcript_metadata -> tuple(
                    meta, bundle,
                    xr_transcript_metadata,
                    xr_cell_polygons,
                    [], [], [],
                    "microns",
                    expansion_distance,
                )
            }

        XENIUMRANGER_IMPORTSEGMENTATION (ch_xr)

    } else {

        // ── NON-TILED PATH ──────────────────────────────────────────────

        // Preprocess: parquet → CSV with optional spatial/QV filtering
        BAYSOR_PREPROCESS_TRANSCRIPTS(
            ch_transcripts_parquet,
            min_qv,
            max_x,
            min_x,
            max_y,
            min_y,
        )

        // Run Baysor on full transcripts (with optional image-based prior mask)
        ch_csv_with_mask = BAYSOR_PREPROCESS_TRANSCRIPTS.out.transcripts_csv
            .join(ch_prior_mask, by: 0, remainder: true)
            .map { meta, transcripts, mask ->
                tuple(meta, transcripts, mask ?: [])
            }
        ch_baysor_input = ch_csv_with_mask
            .combine(ch_config)
            .combine(ch_scale_factor_by_meta, by: 0)
            .map { meta, transcripts, mask, config, scale_factor ->
                tuple(meta, transcripts, mask, config, scale_factor)
            }
        BAYSOR_RUN(ch_baysor_input, [], [], ch_polygon_format)

        // xeniumranger import-segmentation (non-tiled)
        // spatialaxe signature: meta, bundle, transcript_assignment, viz_polygons, nuclei, cells, coordinate_transform, units
        ch_xr = ch_bundle_path
            .combine(BAYSOR_RUN.out.segmentation, by: 0)
            .map { meta, bundle, segmentation_csv, polygons2d ->
                tuple(meta, bundle,
                    segmentation_csv,
                    polygons2d,
                    [], [], [],
                    ch_coordinate_space.val,
                    expansion_distance)
            }

        XENIUMRANGER_IMPORTSEGMENTATION(ch_xr)
    }

    emit:
    redefined_bundle = XENIUMRANGER_IMPORTSEGMENTATION.out.outs
    coordinate_space = ch_coordinate_space
}
