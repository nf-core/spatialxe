
process SPOQC_HQPR_BOUNDING_BOX {
    tag "${meta.id}_${staining}"
    label 'process_high_cpus'
    label 'process_high_mem'
    label 'process_low_time'
    label 'spoqc'


    container "heylf/spoqc:0.0.1"

    input:
    tuple val(meta), path(spatialdata, stageAs: "*"), val(_stain_sd)
    val(step)
    tuple val(staining), path(mask, stageAs: "spoQC_tmp/*")

    output:
    tuple val(meta), val(staining), path("report/hqpr/hqpr_bounding_box/${staining}")    , emit: report
    tuple val("${task.process}"), val('spoqc'), eval("spoqc --version 2>&1 | grep -oP '\\d+\\.\\d+\\.\\d+' || echo unknown"), topic: versions, emit: versions_spoqc

    when:
    task.ext.when == null || task.ext.when

    script:
    // Exit if running this module with -profile conda / -profile mamba
    if (workflow.profile.tokenize(',').intersect(['conda', 'mamba']).size() >= 1) {
        error("SPOQC_HQPR_BOUNDING_BOX module does not support Conda. Please use Docker / Singularity / Podman instead.")
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
        --staining ${staining} \\
        ${args}
    """

    stub:
    // Exit if running this module with -profile conda / -profile mamba
    if (workflow.profile.tokenize(',').intersect(['conda', 'mamba']).size() >= 1) {
        error("SPOQC_HQPR_BOUNDING_BOX module does not support Conda. Please use Docker / Singularity / Podman instead.")
    }

    """
    mkdir -p report/hqpr/hqpr_bounding_box/${staining}
    """

}
