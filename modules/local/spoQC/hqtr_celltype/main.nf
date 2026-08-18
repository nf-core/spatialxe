
process SPOQC_HQTR_CELLTYPE {
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
    path(mask, stageAs: "./spoQC_tmp/hqtr_output_mask_raw")
    path(mask_smoothed, stageAs: "./spoQC_tmp/hqtr_output_mask_smoothed_raw")

    output:
    tuple val(meta), path("./report/hqtr/hqtr_celltype")                 , emit: report
    tuple val("${task.process}"), val('spoqc'), eval("spoqc --version 2>&1 | grep -oP '\\d+\\.\\d+\\.\\d+' || echo unknown"), topic: versions, emit: versions_spoqc

    when:
    task.ext.when == null || task.ext.when

    script:
    // Exit if running this module with -profile conda / -profile mamba
    if (workflow.profile.tokenize(',').intersect(['conda', 'mamba']).size() >= 1) {
        error("SPOQC_HQTR_CELLTYPE module does not support Conda. Please use Docker / Singularity / Podman instead.")
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
        error("SPOQC_HQTR_CELLTYPE module does not support Conda. Please use Docker / Singularity / Podman instead.")
    }

    """
    mkdir -p ./report/hqtr/hqtr_celltype
    """

}
