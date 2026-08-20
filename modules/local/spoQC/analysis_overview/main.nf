process SPOQC_ANALYSIS_OVERVIEW {
    tag "${meta.id}"
    label 'process_high_cpus'
    label 'process_xl_mem'
    label 'process_mid_time'
    label 'spoqc'


    container "heylf/spoqc:0.0.1"

    input:
    tuple val(meta), path(spatialdata, stageAs: "*")
    path(annotation, stageAs: "*")
    val(step)
    path(tmp_general, stageAs: "spoQC_tmp/generalqc_output_hqcr.parquet")
    path(tmp_bubble, stageAs: "spoQC_tmp/bubbleqc_output_hqcr.parquet")
    path(tmp_doublet, stageAs: "spoQC_tmp/doubletqc_output_hqcr.parquet")
    path(tmp_void, stageAs: "spoQC_tmp/voidqc_output_hqcr.parquet")
    path(tmp_cell, stageAs: "spoQC_tmp/cellqc_output_hqcr.parquet")
    path(mask_hqcr, stageAs: "spoQC_tmp/hqcr_output_mask_raw.parquet")
    path(mask_smoothed_hqcr, stageAs: "spoQC_tmp/hqcr_output_mask_smoothed_raw.parquet")
    path(qv, stageAs: "spoQC_tmp/hqtr_output_qv_prob")
    path(ac, stageAs: "spoQC_tmp/hqtr_output_ac_prob")
    path(metrices_hqtr, stageAs: "spoQC_tmp/metrices/hqtr")
    path(mask_smoothed_hqtr, stageAs: "spoQC_tmp/hqtr_output_mask_smoothed_raw")
    path(mask_hqtr, stageAs: "spoQC_tmp/hqtr_output_mask_raw")
    path(metrices_hqpr, stageAs: "spoQC_tmp/metrices/hqpr/*")
    path(mask_smoothed_hqpr, stageAs: "spoQC_tmp/*")
    path(mask_hqpr, stageAs: "spoQC_tmp/*")

    output:
    tuple val(meta), path("report/analysis/overview")                , emit: report
    tuple val(meta), path("report/analysis/rna_qc_annotated.h5ad")   , emit: h5ad
    tuple val("${task.process}"), val('spoqc'), eval("spoqc --version 2>&1 | grep -oP '\\d+\\.\\d+\\.\\d+' || echo unknown"), topic: versions, emit: versions_spoqc

    when:
    task.ext.when == null || task.ext.when

    script:
    // Exit if running this module with -profile conda / -profile mamba
    if (workflow.profile.tokenize(',').intersect(['conda', 'mamba']).size() >= 1) {
        error("SPOQC_ANALYSIS_OVERVIEW module does not support Conda. Please use Docker / Singularity / Podman instead.")
    }

    def args = task.ext.args ?: ''
    def annotation_arg = annotation ? "-a ${annotation}" : ""

    """
    python3 -m spoqc \\
        -i ${spatialdata} \\
        -o ./ \\
        -t ./spoQC_tmp/ \\
        -n ${task.cpus} \\
        ${annotation_arg} \\
        -s ${step}  \\
        --dataset ${meta.id} \\
        ${args}
    """

    stub:
    // Exit if running this module with -profile conda / -profile mamba
    if (workflow.profile.tokenize(',').intersect(['conda', 'mamba']).size() >= 1) {
        error("SPOQC_ANALYSIS_OVERVIEW module does not support Conda. Please use Docker / Singularity / Podman instead.")
    }

    """
    mkdir -p report/analysis/overview
    touch report/analysis/rna_qc_annotated.h5ad
    """
}
