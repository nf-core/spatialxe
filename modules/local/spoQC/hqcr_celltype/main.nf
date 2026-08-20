
process SPOQC_HQCR_CELLTYPE {
    tag "$meta.id"
    label 'process_tiny_cpus'
    label 'process_high_mem'
    label 'process_tiny_time'
    label 'spoqc'


    container "heylf/spoqc:0.0.1"

    input:
    tuple val(meta), path(spatialdata, stageAs: "*")
    path(annotation, stageAs: "*")
    val(step)
    path(tmp_general, stageAs: "./spoQC_tmp/generalqc_output_hqcr.parquet")
    path(tmp_bubble, stageAs: "./spoQC_tmp/bubbleqc_output_hqcr.parquet")
    path(tmp_doublet, stageAs: "./spoQC_tmp/doubletqc_output_hqcr.parquet")
    path(tmp_void, stageAs: "./spoQC_tmp/voidqc_output_hqcr.parquet")
    path(tmp_cell, stageAs: "./spoQC_tmp/cellqc_output_hqcr.parquet")

    output:
    tuple val(meta), path("./report/hqcr/hqcr_celltype")                                       , emit: report
    tuple val(meta), path("./spoQC_tmp/hqcr_output_mask_smoothed_celltype_refined.parquet")    , emit: mask
    tuple val("${task.process}"), val('spoqc'), eval("spoqc --version 2>&1 | grep -oP '\\d+\\.\\d+\\.\\d+' || echo unknown"), topic: versions, emit: versions_spoqc

    when:
    task.ext.when == null || task.ext.when

    script:
    // Exit if running this module with -profile conda / -profile mamba
    if (workflow.profile.tokenize(',').intersect(['conda', 'mamba']).size() >= 1) {
        error("SPOQC_HQCR_CELLTYPE module does not support Conda. Please use Docker / Singularity / Podman instead.")
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
        error("SPOQC_HQCR_CELLTYPE module does not support Conda. Please use Docker / Singularity / Podman instead.")
    }

    """
    mkdir -p ./report/hqcr/hqcr_celltype
    mkdir -p ./spoQC_tmp
    touch ./spoQC_tmp/hqcr_output_mask_smoothed_celltype_refined.parquet
    """

}
