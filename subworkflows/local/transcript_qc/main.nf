//
// TRANSCRIPT_QC: transcript-level quality control for a Xenium bundle.
//
// Runs the transcript QC analysis (per-transcript, per-cell and per-FoV metrics
// + figures) and renders an HTML report from the analysis outputs with the
// nf-core QUARTO_NOTEBOOK module. Versions are reported via the `versions` topic
// channel by each module.
//

include { TRANSCRIPT_QC_PROCESSING as ANALYSIS } from '../../../modules/local/transcript_qc/main'
include { QUARTO_NOTEBOOK as REPORT            } from '../../../modules/nf-core/quarto/notebook/main'

workflow TRANSCRIPT_QC {
    take:
    ch_input      // channel: [ val(meta), val(parameters), path(input_files) ]
    ch_thresholds // channel: path(transcript_qc_thresholds.yaml)
    notebook      // path: the transcript QC report .qmd
    outdir_label  // val: publish subdirectory name, used for the report's self-reference

    main:
    ANALYSIS(ch_input)

    // Render with QUARTONOTEBOOK — see the note in subworkflows/local/image_qc
    // on why every per-sample channel derives from one upstream channel.
    ch_report = ANALYSIS.out.outdir.combine(ch_thresholds.first()).map { meta, analysis_outdir, thresholds ->
        def parameters = [
            INDIR                  : analysis_outdir.name,
            SAMPLE_NAME            : meta.id,
            XENIUM_BUNDLE          : meta.samplesheet_xenium_bundle ?: '',
            SAMPLE_PUBLISHED_OUTDIR: outdir_label,
            // The transcript QC notebook reads the staged thresholds file under
            // the same ROI_THRESHOLDS_YAML name the image QC notebook uses.
            ROI_THRESHOLDS_YAML    : thresholds.name,
        ]
        [meta, parameters, [analysis_outdir, thresholds]]
    }
    REPORT(
        ch_report.map { meta, _parameters, _input_files -> [meta, notebook] },
        ch_report.map { _meta, parameters, _input_files -> parameters },
        ch_report.map { _meta, _parameters, input_files -> input_files },
        [],
    )

    emit:
    outdir = ANALYSIS.out.outdir // channel: [ val(meta), path(outdir) ]
    report = REPORT.out.html     // channel: [ val(meta), path(html) ]
}
