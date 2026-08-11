
process SPOQC_HQCR_IDENT {
    tag "$meta.id"
    label 'process_tiny_cpus'
    label 'process_high_mem'
    label 'process_low_time'
    label 'spoqc'


    container "heylf/spoqc:0.0.1"

    input:
    tuple val(meta), path(spatialdata, stageAs: "*")
    val(step)
    path(tmp_general, stageAs: "./spoQC_tmp/generalqc_output_hqcr.parquet")
    path(tmp_bubble, stageAs: "./spoQC_tmp/bubbleqc_output_hqcr.parquet")
    path(tmp_doublet, stageAs: "./spoQC_tmp/doubletqc_output_hqcr.parquet")
    path(tmp_void, stageAs: "./spoQC_tmp/voidqc_output_hqcr.parquet")
    path(tmp_cell, stageAs: "./spoQC_tmp/cellqc_output_hqcr.parquet")

    output:
    tuple val(meta), path("./report/hqcr/hqcr_ident")                             , emit: report
    tuple val(meta), path("./spoQC_tmp/hqcr_output_mask_raw.parquet")             , emit: mask
    tuple val(meta), path("./spoQC_tmp/hqcr_output_mask_smoothed_raw.parquet")    , emit: mask_smoothed
    tuple val("${task.process}"), val('spoqc'), eval("spoqc --version 2>&1 | grep -oP '\\d+\\.\\d+\\.\\d+' || echo unknown"), topic: versions, emit: versions_spoqc

    when:
    task.ext.when == null || task.ext.when

    script:
    // Exit if running this module with -profile conda / -profile mamba
    if (workflow.profile.tokenize(',').intersect(['conda', 'mamba']).size() >= 1) {
        error("SPOQC_HQCR_IDENT module does not support Conda. Please use Docker / Singularity / Podman instead.")
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
        error("SPOQC_HQCR_IDENT module does not support Conda. Please use Docker / Singularity / Podman instead.")
    }

    """
    mkdir -p ./report/hqcr/hqcr_ident
    mkdir -p ./spoQC_tmp
    touch ./spoQC_tmp/hqcr_output_mask_raw.parquet
    touch ./spoQC_tmp/hqcr_output_mask_smoothed_raw.parquet
    """
}
