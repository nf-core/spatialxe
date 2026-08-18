#!/usr/bin/env nextflow
/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    nf-core/spatialaxe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Github : https://github.com/nf-core/spatialaxe
    Website: https://nf-co.re/spatialaxe
    Slack  : https://nfcore.slack.com/channels/spatialaxe
----------------------------------------------------------------------------------------
*/

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT FUNCTIONS / MODULES / SUBWORKFLOWS / WORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { SPATIALAXE               } from './workflows/spatialaxe.nf'
include { PIPELINE_INITIALISATION } from './subworkflows/local/utils_nfcore_spatialaxe_pipeline'
include { PIPELINE_COMPLETION     } from './subworkflows/local/utils_nfcore_spatialaxe_pipeline'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NAMED WORKFLOWS FOR PIPELINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

//
// WORKFLOW: Run main analysis pipeline depending on type of input
//
workflow NFCORE_SPATIALAXE {

    take:
    samplesheet // channel: samplesheet read in from --input

    main:

    //
    // WORKFLOW: Run pipeline
    //
    SPATIALAXE (
        samplesheet,
        params.alignment_csv,
        params.baysor_config,
        params.baysor_prior,
        params.baysor_scale,
        params.baysor_tiling,
        params.baysor_tiling_scale,
        params.buffer_samples,
        params.buffer_size,
        params.cell_segmentation_only,
        params.cellpose_downscale,
        params.cellpose_model,
        params.expansion_distance,
        params.features,
        params.gene_panel,
        params.gene_synonyms,
        params.max_x,
        params.max_y,
        params.method,
        params.min_qv,
        params.min_x,
        params.min_y,
        params.mode,
        params.multiqc_config,
        params.multiqc_logo,
        params.multiqc_methods_description,
        params.nucleus_segmentation_only,
        params.offtarget_probe_tracking,
        params.outdir,
        params.probes_fasta,
        params.qupath_polygons,
        params.reference_annotations,
        params.relabel_genes,
        params.run_qc,
        params.segger_model,
        params.segmentation_mask,
        params.sharpen_tiff,
        params.stardist_nuclei_model,
        params.tiling,
        params.xeniumranger_only,
        params.spoqc,
    )
    emit:
    multiqc_report = SPATIALAXE.out.multiqc_report // channel: /path/to/multiqc_report.html
}
/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow {

    main:
    //
    // SUBWORKFLOW: Run initialisation tasks
    //
    PIPELINE_INITIALISATION (
        params.version,
        params.validate_params,
        params.monochrome_logs,
        args,
        params.outdir,
        params.input,
        params.help,
        params.help_full,
        params.show_hidden,
        params.format,
        params.gene_panel,
        params.gene_synonyms,
        params.image_seg_methods,
        params.method,
        params.mode,
        params.nucleus_segmentation_only,
        params.offtarget_probe_tracking,
        params.probes_fasta,
        params.reference_annotations,
        params.relabel_genes,
        params.segmentation_mask,
        params.transcript_seg_methods,
    )

    //
    // WORKFLOW: Run main workflow
    //
    NFCORE_SPATIALAXE (
        PIPELINE_INITIALISATION.out.samplesheet
    )
    //
    // SUBWORKFLOW: Run completion tasks
    //
    PIPELINE_COMPLETION (
        params.email,
        params.email_on_fail,
        params.plaintext_email,
        params.outdir,
        params.monochrome_logs,
        NFCORE_SPATIALAXE.out.multiqc_report
    )
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
