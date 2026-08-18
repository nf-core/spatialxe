/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT MODULES / SUBWORKFLOWS / FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

// multiqc
include { MULTIQC                                          } from '../modules/nf-core/multiqc/main'
include { MULTIQC as MULTIQC_PRE_XR_RUN                    } from '../modules/nf-core/multiqc/main'
include { MULTIQC as MULTIQC_POST_XR_RUN                   } from '../modules/nf-core/multiqc/main'
include { paramsSummaryMultiqc                             } from '../subworkflows/nf-core/utils_nfcore_pipeline'

// nf-core functionality
include { softwareVersionsToYAML                           } from '../subworkflows/nf-core/utils_nfcore_pipeline'
include { methodsDescriptionText                           } from '../subworkflows/local/utils_nfcore_spatialaxe_pipeline'
include { paramsSummaryMap                                 } from 'plugin/nf-schema'

// nf-core modules
include { UNTAR                                            } from '../modules/nf-core/untar/main'
include { UNZIP                                            } from '../modules/nf-core/unzip/main'

// coordinate-based segmentation subworklfows
include { SEGGER_CREATE_TRAIN_PREDICT                      } from '../subworkflows/local/segger_create_train_predict/main'
include { PROSEG_PRESET_PROSEG2BAYSOR                      } from '../subworkflows/local/proseg_preset_proseg2baysor/main'
include { PROSEG_PRESET_PROSEG2BAYSOR_TILED                } from '../subworkflows/local/proseg_preset_proseg2baysor_tiled/main'
include { BAYSOR_GENERATE_PREVIEW                          } from '../subworkflows/local/baysor_generate_preview/main'
include { BAYSOR_RUN_TRANSCRIPTS_PARQUET                   } from '../subworkflows/local/baysor_run_transcripts_parquet/main'

// image-based segmentation subworklfows
include { BAYSOR_RUN_PRIOR_SEGMENTATION_MASK               } from '../subworkflows/local/baysor_run_prior_segmentation_mask/main'
include { CELLPOSE_RESOLIFT_MORPHOLOGY_OME_TIF             } from '../subworkflows/local/cellpose_resolift_morphology_ome_tif/main'
include { CELLPOSE_BAYSOR_IMPORT_SEGMENTATION              } from '../subworkflows/local/cellpose_baysor_import_segmentation/main'
include { STARDIST_RESOLIFT_MORPHOLOGY_OME_TIF             } from '../subworkflows/local/stardist_resolift_morphology_ome_tif/main'
include { XENIUMRANGER_RESEGMENT_MORPHOLOGY_OME_TIF        } from '../subworkflows/local/xeniumranger_resegment_morphology_ome_tif/main'

// segmentation-free subworkflows
include { BAYSOR_GENERATE_SEGFREE                          } from '../subworkflows/local/baysor_generate_segfree/main'
include { FICTURE_PREPROCESS_MODEL                         } from '../subworkflows/local/ficture_preprocess_model/main'

// xeniumranger subworkflows
include { XENIUMRANGER_RELABEL_RESEGMENT                   } from '../subworkflows/local/xeniumranger_relabel_resegment/main'
include { XENIUMRANGER_IMPORT_SEGMENTATION_REDEFINE_BUNDLE } from '../subworkflows/local/xeniumranger_import_segmentation_redefine_bundle/main'

// spatialdata subworkflows
include { SPATIALDATA_WRITE_META_MERGE                     } from '../subworkflows/local/spatialdata_write_meta_merge/main'

// qc layer subworkflows
include { OPT_FLIP_TRACK_STAT                              } from '../subworkflows/local/opt_flip_track_stat/main'
include { SPOQC                                            } from '../subworkflows/local/spoqc/main'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow SPATIALAXE {
    take:
    ch_samplesheet                       // channel: samplesheet read in from --input
    alignment_csv
    baysor_config
    baysor_prior
    baysor_scale
    baysor_tiling
    baysor_tiling_scale
    buffer_samples
    buffer_size
    cell_segmentation_only
    cellpose_downscale
    cellpose_model
    expansion_distance
    features
    gene_panel
    gene_synonyms
    max_x
    max_y
    method
    min_qv
    min_x
    min_y
    mode
    multiqc_config
    multiqc_logo
    multiqc_methods_description
    nucleus_segmentation_only
    offtarget_probe_tracking
    outdir
    probes_fasta
    qupath_polygons
    reference_annotations
    relabel_genes
    run_qc
    segger_model
    segmentation_mask
    sharpen_tiff
    stardist_nuclei_model
    tiling
    xeniumranger_only
    spoqc

    main:

    /*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        SPATIALAXE - GENERATE INPUT CHANNELS
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    */

    ch_versions = channel.empty()

    ch_input = channel.empty()
    ch_config = channel.empty()
    ch_features = channel.value([])
    ch_raw_bundle = channel.empty()
    ch_gene_panel = channel.empty()
    ch_qc_reports = channel.empty()
    ch_bundle_path = channel.empty()
    ch_preview_html = channel.empty()
    ch_exp_metadata = channel.empty()
    ch_gene_synonyms = channel.empty()
    ch_multiqc_files = channel.empty()
    ch_multiqc_report = channel.empty()
    ch_qupath_polygons = channel.empty()
    ch_morphology_image = channel.empty()
    ch_redefined_bundle = channel.empty()
    ch_coordinate_space = channel.empty()
    ch_panel_probes_fasta = channel.empty()
    ch_transcripts_file = channel.empty()
    ch_reference_annotations = channel.empty()
    ch_multiqc_pre_xr_report = channel.empty()
    ch_multiqc_post_xr_report = channel.empty()


    /*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        SPATIALAXE - DATA STAGING
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    */

    if (workflow.profile.contains('test')) {

        // get sample, xenium bundle and image path
        ch_input_compressed = ch_samplesheet.map { meta, bundle, _image, _annotation, _stainings ->
            return [meta, bundle]
        }

        if (workflow.profile.contains('test_full')) {

            // get testdata
            UNZIP(ch_input_compressed)

            ch_unzip_outs = UNZIP.out.unzipped_archive.map { meta, bundle ->
                // use toUriString() (not toString()) so the URI scheme (e.g. s3://)
                // is preserved when the work dir is on object storage
                return [meta, bundle.toUriString()]
            }

            ch_samplesheet
                .combine(ch_unzip_outs, by: 0)
                .map { meta, _url, image, annotation, stainings, test_bundle ->
                    return [meta, test_bundle, image, annotation, stainings]
                }
                .set { ch_input }

        } else {

            // get testdata
            UNTAR(ch_input_compressed)

            ch_untar_outs = UNTAR.out.untar.map { meta, bundle ->
                // use toUriString() (not toString()) so the URI scheme (e.g. s3://)
                // is preserved when the work dir is on object storage
                return [meta, bundle.toUriString()]
            }

            ch_samplesheet
                .combine(ch_untar_outs, by: 0)
                .map { meta, _url, image, annotation, stainings, test_bundle ->
                    return [meta, test_bundle, image, annotation, stainings]
                }
                .set { ch_input }
        }
    }
    else {
        // for all other profile runs

        // check if samples are buffered
        if (buffer_samples) {
            ch_input = ch_samplesheet.buffer(size: buffer_size).map
            { buffered_sample ->
                def (meta, bundle, tif, ann, sta) = buffered_sample[0]
                tuple(meta, bundle, tif, ann, sta)
            }
        }
        else {
            ch_input = ch_samplesheet
        }
    }

    // validate xenium bundle POST-staging — works uniformly for production
    // directories and for tarball inputs that the UNTAR step has just extracted
    def bundle_required_files = [
        "cell_boundaries.csv.gz",
        "cell_boundaries.parquet",
        "cell_feature_matrix.h5",
        "cell_feature_matrix.zarr.zip",
        "cells.csv.gz",
        "cells.parquet",
        "cells.zarr.zip",
        "experiment.xenium",
        "gene_panel.json",
        "metrics_summary.csv",
        "morphology.ome.tif",
        "nucleus_boundaries.csv.gz",
        "nucleus_boundaries.parquet",
        "transcripts.parquet",
        "transcripts.zarr.zip",
    ]
    def bundle_optional_files = [
        "analysis.tar.gz",
        "analysis.zarr.zip",
        "analysis_summary.html",
        "morphology_focus/",
    ]

    // path to bundle input
    ch_bundle_path = ch_input.map { meta, bundle, _image, _annotation, _stainings ->

        def bundle_path = file(bundle)
        if( !bundle_path.exists() ) {
            error("❌ Xenium bundle does not exist: ${bundle}")
        }

        def missing_required = bundle_required_files.findAll { check -> !bundle_path.resolve(check).exists() }
        if (missing_required) {
            error("❌ Missing required file(s) in xenium bundle '${bundle}': ${missing_required}")
        }

        def missing_optional = bundle_optional_files.findAll { check -> !bundle_path.resolve(check).exists() }
        if (missing_optional) {
            log.warn("⚠️ Missing optional file(s) in xenium bundle '${bundle}': ${missing_optional}")
        }

        log.info("✅ Xenium bundle validated: ${bundle}")
        return [meta, bundle]
    }

    // get transcript.parquet from the xenium bundle
    ch_transcripts_file = ch_input.map { meta, bundle, _image, _annotation, _stainings ->
        def transcripts_parquet = file(
            file(bundle).toUriString().replaceFirst(/\/$/, '') + "/transcripts.parquet",
            checkIfExists: true
        )
        return [meta, transcripts_parquet]
    }

    // get morphology focus image from the xenium bundle (single 2D plane)
    // supports all Xenium versions:
    //   v2/v3: morphology_focus/morphology_focus_0000.ome.tif
    //   v4+:   morphology_focus/ch0000_dapi.ome.tif
    //   v1.x:  morphology_focus.ome.tif (single file at bundle root)
    //   fallback: morphology.ome.tif (multi-Z stack, not ideal for Cellpose)
    ch_morphology_image = ch_input.map { meta, bundle, image, _annotation, _stainings ->
        def morphology_img
        if (image) {
            morphology_img = file(image)
        } else {
            def bundle_path = file(bundle).toUriString().replaceFirst(/\/$/, '')
            def focus_v3 = file("${bundle_path}/morphology_focus/morphology_focus_0000.ome.tif")
            def focus_v4 = file("${bundle_path}/morphology_focus/ch0000_dapi.ome.tif")
            def focus_v1 = file("${bundle_path}/morphology_focus.ome.tif")
            if (focus_v3.exists()) {
                morphology_img = focus_v3
            } else if (focus_v4.exists()) {
                morphology_img = focus_v4
            } else if (focus_v1.exists()) {
                morphology_img = focus_v1
            } else {
                morphology_img = file("${bundle_path}/morphology.ome.tif", checkIfExists: true)
            }
        }
        return [meta, morphology_img]
    }

    // get experiment metdata - experiment.xenium
    ch_exp_metadata = ch_input.map { meta, bundle, _image, _annotation, _stainings ->
        def exp_metadata = file(
            file(bundle).toUriString().replaceFirst(/\/$/, '') + "/experiment.xenium",
            checkIfExists: true
        )
        return [meta, exp_metadata]
    }

    // get baysor xenium config
    ch_config = channel.fromPath(
            "${projectDir}/assets/config/xenium.toml",
            checkIfExists: true
        )
        .flatten()

    // get segmentation mask if provided with --segmentation_mask for the baysor method
    if (segmentation_mask) {
        ch_segmentation_mask = channel.fromPath(
                segmentation_mask,
                checkIfExists: true
            )
            .flatten()
    }

    // get a list of features if provided with the --features for the ficture method
    ch_features = features
        ? channel.fromPath(features, checkIfExists: true).flatten()
        : channel.value([])

    // get custom cellpose model if provided with the --cellpose_model for the cellpose method
    if (cellpose_model) {
        cellpose_model = channel.fromPath(
                cellpose_model,
                checkIfExists: true
            )
            .flatten()
    }

    // get panel probes fasta for off-target-probe tracking
    if (probes_fasta) {
        ch_panel_probes_fasta = channel.fromPath(
                probes_fasta,
                checkIfExists: true
            )
            .flatten()
    }

    // get reference annotation files (gff,fa) for off-target-probe tracking
    if (reference_annotations) {
        ch_reference_annotations = channel.fromPath(
                "${reference_annotations}/*.{fa,gff}".toString(),
                checkIfExists: true
            )
            .flatten()
    }

    // get gene synonyms for off-target-probe tracking
    if (gene_synonyms) {
        ch_gene_synonyms = channel.fromPath(
                gene_synonyms,
                checkIfExists: true
            )
            .flatten()
    }

    // get qupath ploygons
    if (qupath_polygons) {
        ch_qupath_polygons = channel.fromPath(
                "${qupath_polygons}/*.geojson",
                checkIfExists: true
            )
            .flatten()
    }

    // get gene_panel.json if provided with --gene_panel, sets relabel_genes to true
    def do_relabel = gene_panel ? true : relabel_genes
    if (gene_panel) {

        def gene_panel_file = file(gene_panel, checkIfExists: true)
        ch_gene_panel = ch_input.map { meta, _bundle, _image, _annotation, _stainings ->
            return [meta, gene_panel_file]
        }
    }
    else {

        // gene panel to use if only --relabel_genes is provided
        ch_gene_panel = ch_input.map { meta, bundle, _image, _annotation, _stainings ->
            def gene_panel_file = file(
                file(bundle).toUriString().replaceFirst(/\/$/, '') + "/gene_panel.json",
                checkIfExists: true
            )
            return [meta, gene_panel_file]
        }
    }

    // get stainings, keyed by meta so per-sample staining lists never mix across samples
    ch_stainings = ch_input.flatMap { meta, _bundle, _image, _annotation, stainings ->
        def staining_list = (stainings instanceof List)
            ? stainings
            : stainings.tokenize(';')*.toInteger()
        staining_list.collect { staining -> [meta, staining] }
    }

    // get annotation
    ch_annotation = ch_input.map { meta, _bundle, _image, annotation, _stainings ->
        [meta, annotation]
    }

    /*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        SPATIALAXE - RELABEL GENES
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    */

    // run xr relabel if relabel_genes is true, check if gene_panel.json is provided
    if (do_relabel) {

        XENIUMRANGER_RELABEL_RESEGMENT(
            ch_bundle_path,
            ch_gene_panel,
        )
        ch_raw_bundle = XENIUMRANGER_RELABEL_RESEGMENT.out.redefined_bundle
    }
    else {
        ch_raw_bundle = ch_bundle_path
    }

    /*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        SPATIALAXE - DATA PREVIEW
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    */
    // run baysor preview if `generate_preview ` is true
    if (mode == 'preview') {

        BAYSOR_GENERATE_PREVIEW(
            ch_transcripts_file,
            ch_config,
        )
        ch_preview_html = BAYSOR_GENERATE_PREVIEW.out.preview_html
    }

    /*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        SPATIALAXE - XENIUMRANGER LAYER
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    */
    // run only xeniumranger import segmentation with changes xr specific params
    if (mode == 'image' && xeniumranger_only) {

        XENIUMRANGER_IMPORT_SEGMENTATION_REDEFINE_BUNDLE(
            ch_bundle_path,
            alignment_csv,
            expansion_distance,
            nucleus_segmentation_only,
            qupath_polygons,
        )
        ch_redefined_bundle = XENIUMRANGER_IMPORT_SEGMENTATION_REDEFINE_BUNDLE.out.redefined_bundle
        ch_coordinate_space = XENIUMRANGER_IMPORT_SEGMENTATION_REDEFINE_BUNDLE.out.coordinate_space
    }


    /*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        SPATIALAXE - IMAGE-BASED SEGMENTATION LAYER
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    */
    if (mode == 'image') {

        // trigger the default image-based workflow if no method is specified
        if (!method) {

            CELLPOSE_BAYSOR_IMPORT_SEGMENTATION(
                ch_morphology_image,
                ch_bundle_path,
                ch_transcripts_file,
                ch_exp_metadata,
                ch_config,
                cell_segmentation_only,
                cellpose_model,
                max_x,
                max_y,
                min_qv,
                min_x,
                min_y,
                nucleus_segmentation_only,
                sharpen_tiff,
                stardist_nuclei_model,
                expansion_distance,
            )
            ch_redefined_bundle = CELLPOSE_BAYSOR_IMPORT_SEGMENTATION.out.redefined_bundle
            ch_coordinate_space = CELLPOSE_BAYSOR_IMPORT_SEGMENTATION.out.coordinate_space
        }

        // run xeniumranger resegment with morphology_ome.tif
        if (method == 'xeniumranger') {

            XENIUMRANGER_RESEGMENT_MORPHOLOGY_OME_TIF(
                ch_bundle_path,
                nucleus_segmentation_only,
                expansion_distance,
            )
            ch_redefined_bundle = XENIUMRANGER_RESEGMENT_MORPHOLOGY_OME_TIF.out.redefined_bundle
            ch_coordinate_space = XENIUMRANGER_RESEGMENT_MORPHOLOGY_OME_TIF.out.coordinate_space
        }

        // run baysor run with morphology_ome.tif
        if (method == 'baysor') {

            if (segmentation_mask) {
                BAYSOR_RUN_PRIOR_SEGMENTATION_MASK(
                    ch_bundle_path,
                    ch_transcripts_file,
                    ch_segmentation_mask,
                    ch_config,
                    max_x,
                    max_y,
                    min_qv,
                    min_x,
                    min_y,
                    expansion_distance,
                )
            }
            ch_redefined_bundle = BAYSOR_RUN_PRIOR_SEGMENTATION_MASK.out.redefined_bundle
            ch_coordinate_space = BAYSOR_RUN_PRIOR_SEGMENTATION_MASK.out.coordinate_space
        }

        // run cellpose on the morphology_ome.tif
        if (method == 'cellpose') {

            CELLPOSE_RESOLIFT_MORPHOLOGY_OME_TIF(
                ch_morphology_image,
                ch_bundle_path,
                cellpose_downscale,
                cellpose_model,
                nucleus_segmentation_only,
                sharpen_tiff,
                stardist_nuclei_model,
                expansion_distance,
            )
            ch_redefined_bundle = CELLPOSE_RESOLIFT_MORPHOLOGY_OME_TIF.out.redefined_bundle
            ch_coordinate_space = CELLPOSE_RESOLIFT_MORPHOLOGY_OME_TIF.out.coordinate_space
        }

        // run stardist on the morphology_ome.tif
        if (method == 'stardist') {

            STARDIST_RESOLIFT_MORPHOLOGY_OME_TIF(
                ch_morphology_image,
                ch_bundle_path,
                sharpen_tiff,
                stardist_nuclei_model,
                expansion_distance,
            )
            ch_redefined_bundle = STARDIST_RESOLIFT_MORPHOLOGY_OME_TIF.out.redefined_bundle
            ch_coordinate_space = STARDIST_RESOLIFT_MORPHOLOGY_OME_TIF.out.coordinate_space
        }
    }

    /*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        SPATIALAXE - TRANSCRIPT-BASED SEGMENTATION LAYER
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    */
    if (mode == 'coordinate') {

        // run proseg with transcripts.parquet if method = proseg or is not provided (default workflow)
        if (!method || method == 'proseg') {

            if (tiling) {
                PROSEG_PRESET_PROSEG2BAYSOR_TILED(
                    ch_bundle_path,
                    ch_transcripts_file,
                    expansion_distance,
                )
                ch_redefined_bundle = PROSEG_PRESET_PROSEG2BAYSOR_TILED.out.redefined_bundle
                ch_coordinate_space = PROSEG_PRESET_PROSEG2BAYSOR_TILED.out.coordinate_space
            } else {
                PROSEG_PRESET_PROSEG2BAYSOR(
                    ch_bundle_path,
                    ch_transcripts_file,
                    expansion_distance,
                )
                ch_redefined_bundle = PROSEG_PRESET_PROSEG2BAYSOR.out.redefined_bundle
                ch_coordinate_space = PROSEG_PRESET_PROSEG2BAYSOR.out.coordinate_space
            }
        }

        // run segger with transcripts.parquet
        if (method == 'segger') {

            SEGGER_CREATE_TRAIN_PREDICT(
                ch_bundle_path,
                ch_transcripts_file,
                segger_model,
                expansion_distance,
            )
            ch_redefined_bundle = SEGGER_CREATE_TRAIN_PREDICT.out.redefined_bundle
            ch_coordinate_space = SEGGER_CREATE_TRAIN_PREDICT.out.coordinate_space
        }

        // run baysor with transcripts.parquet (unified tiled/non-tiled subworkflow)
        if (method == 'baysor') {

            // Image-based prior (cellpose mask) requires non-tiled Baysor
            if ( baysor_tiling && baysor_prior == 'cellpose' ) {
                error "ERROR: baysor_prior='cellpose' (image-based) requires baysor_tiling=false. " +
                      "For tiled Baysor, use baysor_prior='cells' (column-based)."
            }

            ch_prior_mask = channel.empty()

            BAYSOR_RUN_TRANSCRIPTS_PARQUET(
                ch_bundle_path,
                ch_transcripts_file,
                ch_morphology_image,
                ch_config,
                ch_prior_mask,
                baysor_config,
                baysor_scale,
                baysor_tiling,
                baysor_tiling_scale,
                max_x,
                max_y,
                min_qv,
                min_x,
                min_y,
                expansion_distance,
            )
            ch_redefined_bundle = BAYSOR_RUN_TRANSCRIPTS_PARQUET.out.redefined_bundle
            ch_coordinate_space = BAYSOR_RUN_TRANSCRIPTS_PARQUET.out.coordinate_space
        }
    }



    /*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        SPATIALAXE - SPATIALDATA / METADATA LAYER
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    */

    // run spatialdata modules to generate sd objects in image or coordinate mode
    if (mode == 'image' || mode == 'coordinate' || mode == 'qc') {

        SPATIALDATA_WRITE_META_MERGE(
            ch_bundle_path,
            ch_redefined_bundle,
            ch_coordinate_space,
            cell_segmentation_only,
            mode,
            nucleus_segmentation_only,
            run_qc,
        )
    }

    /*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        SPATIALAXE - QC LAYER
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    */

    // check to run the qc layer
    if (mode == 'qc' || run_qc) {

        if (offtarget_probe_tracking) {

            // run off-target probe tracking
            OPT_FLIP_TRACK_STAT(
                ch_panel_probes_fasta,
                ch_reference_annotations,
                ch_gene_synonyms,
            )
        }


        if (spoqc){
            SPOQC (
                SPATIALDATA_WRITE_META_MERGE.out.sd_raw_bundle,
                ch_annotation,
                ch_stainings,
            )
        }

    }


    /*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        SPATIALAXE - SEGMENTATION-FREE LAYER
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    */
    if (mode == 'segfree') {

        // trigger the default segfree workflow if no method or if the method is baysor
        if (!method || method == 'baysor') {

            BAYSOR_GENERATE_SEGFREE(
                ch_transcripts_file,
                ch_config,
                max_x,
                max_y,
                min_qv,
                min_x,
                min_y,
            )
        }

        // run ficture with transcripts.parquet
        if (method == 'ficture') {

            FICTURE_PREPROCESS_MODEL(
                ch_transcripts_file,
                ch_features,
                features,
            )
        }
    }



    /*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        SPATIALAXE - COLLATE & SAVE SOFTWARE VERSIONS
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    */
    // Collect versions published via topic channels (local modules)
    ch_topic_versions = channel.topic('versions')
        .map { process, tool, version ->
            "\"${process}\":\n    ${tool}: ${version}"
        }

    softwareVersionsToYAML(ch_versions.mix(ch_topic_versions))
        .collectFile(
            storeDir: "${outdir}/pipeline_info",
            name: 'nf_core_' + 'spatialaxe_software_' + 'mqc_' + 'versions.yml',
            sort: true,
            newLine: true,
        )
        .set { ch_collated_versions }

    /*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        SPATIALAXE - MultiQC
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    */
    ch_multiqc_config = channel.fromPath(
        "${projectDir}/assets/multiqc_config.yml",
        checkIfExists: true
    )

    ch_multiqc_custom_config = multiqc_config
        ? channel.fromPath(multiqc_config, checkIfExists: true)
        : channel.empty()

    ch_multiqc_logo = multiqc_logo
        ? channel.fromPath(multiqc_logo, checkIfExists: true)
        : channel.empty()

    // Combine default and custom configs into a single list for the tuple-based MULTIQC input
    ch_multiqc_configs = ch_multiqc_config.mix(ch_multiqc_custom_config).collect()

    summary_params = paramsSummaryMap(
        workflow,
        parameters_schema: "nextflow_schema.json"
    )

    ch_workflow_summary = channel.value(paramsSummaryMultiqc(summary_params))

    ch_multiqc_files = ch_multiqc_files.mix(
        ch_workflow_summary.collectFile(name: 'workflow_summary_mqc.yaml')
    )

    ch_multiqc_custom_methods_description = multiqc_methods_description
        ? file(multiqc_methods_description, checkIfExists: true)
        : file("${projectDir}/assets/methods_description_template.yml", checkIfExists: true)

    ch_methods_description = channel.value(
        methodsDescriptionText(ch_multiqc_custom_methods_description)
    )

    ch_multiqc_files = ch_multiqc_files.mix(ch_collated_versions)

    ch_multiqc_files = ch_multiqc_files.mix(
        ch_methods_description.collectFile(
            name: 'methods_description_mqc.yaml',
            sort: true,
        )
    )

    if (mode == 'qc' || run_qc) {

        // get path to the raw bundle
        ch_multiqc_files = ch_multiqc_files.mix(
            ch_bundle_path.map { _meta, bundle -> file(bundle) }.collect().ifEmpty([])
        )

        MULTIQC_PRE_XR_RUN (
            ch_multiqc_files.collect().map { mqc_files -> [mqc_files] }
                .combine(ch_multiqc_configs.map { mqc_configs -> [mqc_configs] })
                .combine(ch_multiqc_logo.toList().map { mqc_logo -> [mqc_logo] })
                .map { files, configs, logo ->
                    [ [id: 'multiqc_pre_xr'], files, configs, logo ? logo[0] : [], [], [] ]
                }
        )
        ch_multiqc_pre_xr_report = MULTIQC_PRE_XR_RUN.out.report.map { _meta, report -> report }.toList()

        // get path to the redefined bundle
        ch_multiqc_files = ch_multiqc_files.mix(
            ch_redefined_bundle.map { _meta, bundle -> file(bundle) }.collect().ifEmpty([])
        )

        MULTIQC_POST_XR_RUN (
            ch_multiqc_files.collect().map { mqc_files -> [mqc_files] }
                .combine(ch_multiqc_configs.map { mqc_configs -> [mqc_configs] })
                .combine(ch_multiqc_logo.toList().map { mqc_logo -> [mqc_logo] })
                .map { files, configs, logo ->
                    [ [id: 'multiqc_post_xr'], files, configs, logo ? logo[0] : [], [], [] ]
                }
        )
        ch_multiqc_post_xr_report = MULTIQC_POST_XR_RUN.out.report.map { _meta, report -> report }.toList()

    } else {

        // get path to the raw bundle
        ch_multiqc_files = ch_multiqc_files.mix(
            ch_bundle_path.map { _meta, bundle -> file(bundle) }.collect().ifEmpty([])
        )


        // get the qc htmls if qc mode is run
        if (mode == 'qc' || run_qc) {

            ch_multiqc_files = ch_multiqc_files.mix(
                ch_qc_reports.map { _meta, qc_reports -> qc_reports }.collect().ifEmpty([])
            )

        }


        // get the preview html if preview mode is run
        if (mode == 'preview') {

            ch_multiqc_files = ch_multiqc_files.mix(
                ch_preview_html.map { _meta, preview_html -> preview_html }.collect().ifEmpty([])
            )

        }


        MULTIQC (
            ch_multiqc_files.collect().map { mqc_files -> [mqc_files] }
                .combine(ch_multiqc_configs.map { mqc_configs -> [mqc_configs] })
                .combine(ch_multiqc_logo.toList().map { mqc_logo -> [mqc_logo] })
                .map { files, configs, logo ->
                    [ [id: 'multiqc'], files, configs, logo ? logo[0] : [], [], [] ]
                }
        )
        ch_multiqc_report = MULTIQC.out.report.map { _meta, report -> report }.toList()

    }

    emit:
    multiqc_pre_xr_report  = ch_multiqc_pre_xr_report  // channel: /path/to/multiqc_report.html
    multiqc_post_xr_report = ch_multiqc_post_xr_report // channel: /path/to/multiqc_report.html
    multiqc_report         = ch_multiqc_report         // channel: /path/to/multiqc_report.html
    versions               = ch_versions               // channel: [ path(versions.yml) ]
}
