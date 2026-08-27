//
// QC: image and transcript quality control for a Xenium bundle.
//
// Thin wrapper that runs the two QC subworkflows — image QC and transcript QC —
// on one bundle and emits their analysis directories and HTML reports. It
// deliberately holds no pre/post-segmentation logic and no MultiQC: when and on
// which bundle QC runs is decided by the caller (workflows/spatialaxe.nf).
//
// Following the pipeline convention, only the entry workflow reads `params.*`;
// everything this subworkflow needs arrives through `take:`.
//

include { IMAGE_QC      } from '../image_qc/main'
include { TRANSCRIPT_QC } from '../transcript_qc/main'

workflow QC {
    take:
    ch_bundle                 // channel: [ val(meta), path(xenium_bundle) ]
    ch_image_thresholds       // channel: path(roi_image_qc_thresholds.yaml)
    ch_transcript_thresholds  // channel: path(transcript_qc_thresholds.yaml)
    image_notebook            // path:    image QC report .qmd
    transcript_notebook       // path:    transcript QC report .qmd
    image_outdir_label        // val:     publish subdirectory for image QC
    transcript_outdir_label   // val:     publish subdirectory for transcript QC

    main:
    // Shared per-sample meta for both reports. The bundle path is carried in
    // meta so the notebooks can state which bundle they describe.
    ch_qc_meta = ch_bundle.map { meta, bundle ->
        [
            [
                id: meta.id,
                samplesheet_xenium_bundle: bundle.toString(),
                xenium_bundle_source: meta.xenium_bundle_source ?: '',
                cropped: meta.cropped ?: false,
            ],
            bundle,
        ]
    }

    // Per-sample values reach the analysis modules through the `parameters`
    // list; module-wide settings come from `task.ext.*` in conf/modules.config,
    // so the modules never read `params.*` themselves.
    ch_qc_input = ch_qc_meta.map { meta, bundle -> tuple(meta, [], [bundle]) }

    IMAGE_QC(
        ch_qc_input,
        ch_image_thresholds,
        image_notebook,
        image_outdir_label,
    )

    TRANSCRIPT_QC(
        ch_qc_input,
        ch_transcript_thresholds,
        transcript_notebook,
        transcript_outdir_label,
    )

    emit:
    image_qc_outdir      = IMAGE_QC.out.outdir      // channel: [ val(meta), path(outdir) ]
    image_qc_report      = IMAGE_QC.out.report      // channel: [ val(meta), path(html) ]
    transcript_qc_outdir = TRANSCRIPT_QC.out.outdir // channel: [ val(meta), path(outdir) ]
    transcript_qc_report = TRANSCRIPT_QC.out.report // channel: [ val(meta), path(html) ]
}
