//
// IMAGE_QC: image-based quality control for a Xenium bundle.
//
// Runs the image QC analysis (focus / SNR / morphology metrics + figures) and
// renders an HTML report from the analysis outputs with the nf-core
// QUARTO_NOTEBOOK module. Versions are reported via the `versions` topic channel
// by each module, so this subworkflow does not thread versions through emit.
//

include { IMAGE_QC_ANALYSIS as ANALYSIS } from '../../../modules/local/image_qc/main'
include { QUARTO_NOTEBOOK as REPORT     } from '../../../modules/nf-core/quarto/notebook/main'

workflow IMAGE_QC {
    take:
    ch_input      // channel: [ val(meta), val(parameters), path(input_files) ]
    ch_thresholds // channel: path(roi_image_qc_thresholds.yaml)
    notebook      // path: the image QC report .qmd
    outdir_label  // val: publish subdirectory name, used for the report's self-reference

    main:
    ANALYSIS(ch_input, ch_thresholds.first())

    // Render with QUARTONOTEBOOK. Its four inputs are separate channels paired
    // by emission order, so every per-sample channel is derived from one
    // upstream channel to guarantee alignment. The analysis output directory and
    // the thresholds YAML are staged as input files; the notebook's parameters
    // cell receives their staged names through params.yml.
    ch_report = ANALYSIS.out.outdir.combine(ch_thresholds.first()).map { meta, analysis_outdir, thresholds ->
        def parameters = [
            INDIR                  : analysis_outdir.name,
            SAMPLE_NAME            : meta.id,
            XENIUM_BUNDLE          : meta.samplesheet_xenium_bundle ?: '',
            SAMPLE_PUBLISHED_OUTDIR: outdir_label,
            ROI_THRESHOLDS_YAML    : thresholds.name,
            PIPELINE_VERSION       : workflow.manifest.version,
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
