//
// Subworkflow with functionality specific to the nf-core/spatialaxe pipeline
//

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT FUNCTIONS / MODULES / SUBWORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { UTILS_NFSCHEMA_PLUGIN     } from '../../nf-core/utils_nfschema_plugin'
include { paramsSummaryMap          } from 'plugin/nf-schema'
include { samplesheetToList         } from 'plugin/nf-schema'
include { paramsHelp                } from 'plugin/nf-schema'
include { completionEmail           } from '../../nf-core/utils_nfcore_pipeline'
include { completionSummary         } from '../../nf-core/utils_nfcore_pipeline'
include { UTILS_NFCORE_PIPELINE     } from '../../nf-core/utils_nfcore_pipeline'
include { UTILS_NEXTFLOW_PIPELINE   } from '../../nf-core/utils_nextflow_pipeline'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SUBWORKFLOW TO INITIALISE PIPELINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow PIPELINE_INITIALISATION {
    take:
    version                    // boolean: Display version and exit
    validate_params            // boolean: Boolean whether to validate parameters against the schema at runtime
    monochrome_logs            // boolean: Do not use coloured log outputs
    nextflow_cli_args          // array: List of positional nextflow CLI args
    outdir                     // string: The output directory where the results will be saved
    input                      // string: Path to input samplesheet
    help                       // boolean: Display help message and exit
    help_full                  // boolean: Show the full help message
    show_hidden                // boolean: Show hidden parameters in the help message
    format                     // string: input data platform (xenium | cosmx | merscope)
    gene_panel                 // string: path to gene panel
    gene_synonyms              // string: path to gene synonyms
    image_seg_methods          // list: valid image-mode segmentation methods
    method                     // string: chosen segmentation method
    mode                       // string: pipeline mode
    nucleus_segmentation_only  // boolean
    offtarget_probe_tracking   // boolean
    probes_fasta               // string: path to probes fasta
    reference_annotations      // string: path to reference annotations
    relabel_genes              // boolean
    segmentation_mask          // string: path to segmentation mask
    transcript_seg_methods     // list: valid coordinate-mode segmentation methods

    main:

    //
    // Print version and exit if required and dump pipeline parameters to JSON file
    //
    UTILS_NEXTFLOW_PIPELINE(
        version,
        true,
        outdir,
        workflow.profile.tokenize(',').intersect(['conda', 'mamba']).size() >= 1,
    )

    //
    // Validate parameters and generate parameter summary to stdout
    //

    def before_text = ""
    def after_text = ""
    before_text = """
-\033[2m----------------------------------------------------\033[0m-
                                        \033[0;32m,--.\033[0;30m/\033[0;32m,-.\033[0m
\033[0;34m        ___     __   __   __   ___     \033[0;32m/,-._.--~\'\033[0m
\033[0;34m  |\\ | |__  __ /  ` /  \\ |__) |__         \033[0;33m}  {\033[0m
\033[0;34m  | \\| |       \\__, \\__/ |  \\ |___     \033[0;32m\\`-._,-`-,\033[0m
                                        \033[0;32m`._,._,\'\033[0m
\033[0;35m  nf-core/spatialaxe ${workflow.manifest.version}\033[0m
-\033[2m----------------------------------------------------\033[0m-
"""
    after_text = """${workflow.manifest.doi ? "\n* The pipeline\n" : ""}${workflow.manifest.doi.tokenize(",").collect { doi -> "    https://doi.org/${doi.trim().replace('https://doi.org/','')}"}.join("\n")}${workflow.manifest.doi ? "\n" : ""}
* The nf-core framework
    https://doi.org/10.1038/s41587-020-0439-x

* Software dependencies
    https://github.com/nf-core/spatialaxe/blob/master/CITATIONS.md
"""
    if (monochrome_logs) {
        before_text = before_text.replaceAll(/\033\[[0-9;]*m/, '')
    }

    command = "nextflow run ${workflow.manifest.name} -profile <docker/singularity/.../institute> --input samplesheet.csv --outdir <OUTDIR>"

    UTILS_NFSCHEMA_PLUGIN (
        workflow,
        validate_params,
        null,
        help,
        help_full,
        show_hidden,
        before_text,
        after_text,
        command,
        null
    )

    //
    // Check config provided to the pipeline
    //
    UTILS_NFCORE_PIPELINE (
        nextflow_cli_args
    )

    //
    // Custom validation for pipeline parameters
    //
    validateInputParameters(
        input,
        mode,
        method,
        format,
        image_seg_methods,
        transcript_seg_methods,
        relabel_genes,
        gene_panel,
        nucleus_segmentation_only,
        segmentation_mask,
        offtarget_probe_tracking,
        probes_fasta,
        reference_annotations,
        gene_synonyms,
    )
    log.info("✅ Pipeline parameters validated.")

    //
    // Create channel from input file provided through --input
    //
    try {

        channel.fromList(samplesheetToList(input, "${projectDir}/assets/schema_input.json"))
            .map { meta, bundle, image, annotation, stainings ->
                return [[id: meta.id], bundle, image, annotation, stainings]
            }
            .set { ch_samplesheet }

        log.info("✅ Samplesheet validated.")
    }
    catch (Exception e) {

        error("❌ Samplesheet validation failed: ${e.message}")
    }


    // Xenium bundle file-presence validation now runs in the main workflow
    // (workflows/spatialaxe.nf) AFTER UNTAR staging, so it works uniformly for
    // both directory inputs and tarball inputs.

    emit:
    samplesheet = ch_samplesheet
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SUBWORKFLOW FOR PIPELINE COMPLETION
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow PIPELINE_COMPLETION {
    take:
    email           //  string: email address
    email_on_fail   //  string: email address sent on pipeline failure
    plaintext_email // boolean: Send plain-text email instead of HTML
    outdir          //    path: Path to output directory where results will be published
    monochrome_logs // boolean: Disable ANSI colour codes in log output
    multiqc_report  //  string: Path to MultiQC report

    main:
    summary_params = paramsSummaryMap(workflow, parameters_schema: "nextflow_schema.json")
    def multiqc_reports = multiqc_report.toList()

    //
    // Completion email and summary
    //
    workflow.onComplete {
        if (email || email_on_fail) {
            completionEmail(
                summary_params,
                email,
                email_on_fail,
                plaintext_email,
                outdir,
                monochrome_logs,
                multiqc_reports.getVal(),
            )
        }

        completionSummary(monochrome_logs)

    }

    workflow.onError {
        log.error "Pipeline failed. Please refer to troubleshooting docs for common issues: https://nf-co.re/docs/running/troubleshooting"
    }
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
//
// Check and validate pipeline parameters
//
def validateInputParameters(
    input,
    mode,
    method,
    format,
    image_seg_methods,
    transcript_seg_methods,
    relabel_genes,
    gene_panel,
    nucleus_segmentation_only,
    segmentation_mask,
    offtarget_probe_tracking,
    probes_fasta,
    reference_annotations,
    gene_synonyms
) {

    // check if conda profile is provided
    if (workflow.profile.contains('conda')) {
        error("❌ Error: `nf-core/spatialaxe` does not support running the pipeline with profile: conda ")
    }

    // check if the samplesheet provided with the test config is assets/samplesheet.csv
    if (workflow.profile.contains('test') && !"${input}".endsWith("assets/samplesheet.csv") && !workflow.profile.contains('test_full')) {
        error("❌ Error: Use the samplesheet at: ${projectDir}/assets/samplesheet.csv with `--input` when running the pipeline in test profile.")
    }

    // check if the segmentation method provided is valid for a mode
    if (mode == 'image' && method) {
        if (!image_seg_methods.contains(method)) {
            error("❌ Error: Invalid segmentation method: ${method} provided for the `image` based mode. Options: ${image_seg_methods}")
        }
    }

    if (mode == 'coordinate' && method) {
        if (!transcript_seg_methods.contains(method)) {
            error("❌ Error: Invalid segmentation method: `${method}` provided for the `coordinate` based mode. Options: ${transcript_seg_methods}")
        }
    }

    // check method-format compatibility (schema enum constrains the universe; this enforces the method-specific subset)
    def valid_segger_formats = ['xenium']
    def valid_proseg_formats = ['xenium', 'cosmx', 'merscope']
    if (method == 'segger' && !(format in valid_segger_formats)) {
        error("❌ Error: Invalid --format '${format}' for segger. Valid: ${valid_segger_formats}")
    }
    if (method == 'proseg' && !(format in valid_proseg_formats)) {
        error("❌ Error: Invalid --format '${format}' for proseg. Valid: ${valid_proseg_formats}")
    }

    // check if --relabel_genes is true but --gene_panel is not provided
    if (relabel_genes && !gene_panel) {
        log.warn("⚠️  Relabel genes is enabled, but gene panel is not provided with the `--gene_panel`. Using `gene_panel.json` in the xenium bundle.")
    }

    // check if --relabel_genes is true but --gene_panel is not provided
    if (gene_panel && !relabel_genes) {
        log.warn("⚠️  Gene panel provided, but relabel genes is disabled. Using `gene_panel.json` only to generate metadata.")
    }

    // check if segmentation method is xeniumranger and nucleus_ony_segmentation is enabled
    if (method == 'xeniumranger' && !nucleus_segmentation_only) {
        log.warn("⚠️  Nucleus segmentation is disabled. Running xeniumranger resegment module to redefine xenium bundle without nucleus segmentation.")
        log.warn("⚠️  Use --nucleus_segmentation_only to enable nucleus segmentation to redefine xenium bundle with import-segmentation module.")
    }

    // check if segmentation mask is provided in image mode and baysor method
    if (mode == 'image' && method == 'baysor') {
        if (!segmentation_mask) {
            log.warn("⚠️  Missing segmentation mask with `--segmentation_mask` when pipeline is run in ${mode} and with the ${method}. Running in coordinate mode.")
        }
    }

    // check if required arguments are provided for off-target probe tracking
    if (!mode && offtarget_probe_tracking) {
        if(!probes_fasta || !reference_annotations || !gene_synonyms) {
            error("❌ Error: Missing required param(s) for off-target-proebe detection.")
        }
        error("❌ Error: Use --mode qc and --offtraget_probe_tracking to run off-target probe tracking.")
    }
}

//
// Generate methods description for MultiQC
//
def toolCitationText() {
    // TODO nf-core: Optionally add in-text citation tools to this list.
    // Can use ternary operators to dynamically construct based conditions, e.g. params["run_xyz"] ? "Tool (Foo et al. 2023)" : "",
    // Uncomment function in methodsDescriptionText to render in MultiQC report
    def citation_text = [
            "Tools used in the workflow included:",
            "FastQC (Andrews 2010),",
            "MultiQC (Ewels et al. 2016)",
            "."
        ].join(' ').trim()

    return citation_text
}

def toolBibliographyText() {
    // TODO nf-core: Optionally add bibliographic entries to this list.
    // Can use ternary operators to dynamically construct based conditions, e.g. params["run_xyz"] ? "<li>Author (2023) Pub name, Journal, DOI</li>" : "",
    // Uncomment function in methodsDescriptionText to render in MultiQC report
    def reference_text = [
            "<li>Andrews S, (2010) FastQC, URL: https://www.bioinformatics.babraham.ac.uk/projects/fastqc/).</li>",
            "<li>Ewels, P., Magnusson, M., Lundin, S., & Käller, M. (2016). MultiQC: summarize analysis results for multiple tools and samples in a single report. Bioinformatics , 32(19), 3047–3048. doi: /10.1093/bioinformatics/btw354</li>"
        ].join(' ').trim()

    return reference_text
}

def methodsDescriptionText(mqc_methods_yaml) {
    // Convert  to a named map so can be used as with familiar NXF ${workflow} variable syntax in the MultiQC YML file
    def meta = [:]
    meta.workflow = workflow.toMap()
    meta["manifest_map"] = workflow.manifest.toMap()

    // Pipeline DOI
    if (meta.manifest_map.doi) {
        // Using a loop to handle multiple DOIs
        // Removing `https://doi.org/` to handle pipelines using DOIs vs DOI resolvers
        // Removing ` ` since the manifest.doi is a string and not a proper list
        def temp_doi_ref = ""
        def manifest_doi = meta.manifest_map.doi.tokenize(",")
        manifest_doi.each { doi_ref ->
            temp_doi_ref += "(doi: <a href=\'https://doi.org/${doi_ref.replace("https://doi.org/", "").replace(" ", "")}\'>${doi_ref.replace("https://doi.org/", "").replace(" ", "")}</a>), "
        }
        meta["doi_text"] = temp_doi_ref.substring(0, temp_doi_ref.length() - 2)
    } else meta["doi_text"] = ""
    meta["nodoi_text"] = meta.manifest_map.doi ? "" : "<li>If available, make sure to update the text to include the Zenodo DOI of version of the pipeline used. </li>"

    // Tool references
    meta["tool_citations"] = ""
    meta["tool_bibliography"] = ""

    // Only uncomment below if logic in toolCitationText/toolBibliographyText has been filled!
    // meta["tool_citations"] = toolCitationText().replaceAll(", \\.", ".").replaceAll("\\. \\.", ".").replaceAll(", \\.", ".")
    // meta["tool_bibliography"] = toolBibliographyText()


    def methods_text = mqc_methods_yaml.text

    def engine =  new groovy.text.SimpleTemplateEngine()
    def description_html = engine.createTemplate(methods_text).make(meta)

    return description_html.toString()
}
