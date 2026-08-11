process SPOQC_FINALREPORT {
    tag "${meta.id}"
    label 'process_tiny_cpus'
    label 'process_low_mem'
    label 'process_tiny_time'
    label 'spoqc'


    container "heylf/spoqc:0.0.1"

    input:
    tuple val(meta), path(spatialdata, stageAs: "*")
    val(step)
    path(report_general, stageAs: "report/generalqc")
    path(report_doublet, stageAs: "report/doubletqc")
    path(report_void, stageAs: "report/voidqc")
    path(report_cell, stageAs: "report/cellqc")
    path(report_hqcr_ident, stageAs: "report/hqcr/hqcr_ident")
    path(report_hqcr_celltype, stageAs: "report/hqcr/hqcr_celltype")
    path(report_hqpr_metrices, stageAs: "report/hqpr/hqpr_metrices/*")
    path(report_hqpr_clustering, stageAs: "report/hqpr/hqpr_clustering/*")
    path(report_hqpr_refinement, stageAs: "report/hqpr/hqpr_refinement/*")
    path(report_hqpr_bounding_box, stageAs: "report/hqpr/hqpr_bounding_box/*")
    path(report_hqpr_celltype, stageAs: "report/hqpr/hqpr_celltype/*")
    path(report_hqtr_metrices, stageAs: "report/hqtr/hqtr_metrices")
    path(report_hqtr_ac, stageAs: "report/hqtr/hqtr_ac")
    path(report_hqtr_qv, stageAs: "report/hqtr/hqtr_qv")
    path(report_hqtr_clustering, stageAs: "report/hqtr/hqtr_clustering")
    path(report_hqtr_refinement, stageAs: "report/hqtr/hqtr_refinement")
    path(report_hqtr_bounding_box, stageAs: "report/hqtr/hqtr_bounding_box")
    path(report_hqtr_celltype, stageAs: "report/hqtr/hqtr_celltype")
    path(report_combine_masks, stageAs: "report/combine_masks/*")
    path(report_transcript, stageAs: "report/transcriptqc")
    path(report_cellcycle, stageAs: "report/cellcycleqc")
    path(report_model, stageAs: "report/modelqc")
    path(report_analysis_overview, stageAs: "report/analysis/overview")
    path(report_analysis_category, stageAs: "report/analysis/category")
    path(report_analysis_cluster, stageAs: "report/analysis/cluster")

    output:
    tuple val(meta), path("report/report.html")                      , emit: report
    tuple val("${task.process}"), val('spoqc'), eval("spoqc --version 2>&1 | grep -oP '\\d+\\.\\d+\\.\\d+' || echo unknown"), topic: versions, emit: versions_spoqc

    when:
    task.ext.when == null || task.ext.when

    script:
    // Exit if running this module with -profile conda / -profile mamba
    if (workflow.profile.tokenize(',').intersect(['conda', 'mamba']).size() >= 1) {
        error("SPOQC_FINALREPORT module does not support Conda. Please use Docker / Singularity / Podman instead.")
    }

    def args = task.ext.args ?: ''

    """
    python3 -m spoqc \\
        -i ${spatialdata} \\
        -o ./ \\
        -t ./spoQC_tmp/ \\
        -n ${task.cpus} \\
        -s ${step}  \\
        --dataset ${meta.id} \\
        ${args}
    """

    stub:
    // Exit if running this module with -profile conda / -profile mamba
    if (workflow.profile.tokenize(',').intersect(['conda', 'mamba']).size() >= 1) {
        error("SPOQC_FINALREPORT module does not support Conda. Please use Docker / Singularity / Podman instead.")
    }

    """
    mkdir -p report/
    touch report/report.html
    """
}
