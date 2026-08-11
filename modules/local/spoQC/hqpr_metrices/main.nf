
process SPOQC_HQPR_METRICES {
    tag "${meta.id}_${staining}"
    label 'process_xl_cpus'
    label 'process_xl_mem'
    label 'process_mid_time'
    label 'spoqc'


    container "heylf/spoqc:0.0.1"

    input:
    tuple val(meta), path(spatialdata, stageAs: "*")
    each staining
    val(step)

    output:
    tuple val(meta), val(staining), path("report/hqpr/hqpr_metrices/${staining}")                       , emit: report
    tuple val(meta), val(staining), path("spoQC_tmp/metrices/hqpr/${staining}")                         , emit: metrices
    tuple val(meta), path("report/staining_log.txt")                                                    , emit: staininglog
    tuple val("${task.process}"), val('spoqc'), eval("spoqc --version 2>&1 | grep -oP '\\d+\\.\\d+\\.\\d+' || echo unknown"), topic: versions, emit: versions_spoqc

    when:
    task.ext.when == null || task.ext.when

    script:
    // Exit if running this module with -profile conda / -profile mamba
    if (workflow.profile.tokenize(',').intersect(['conda', 'mamba']).size() >= 1) {
        error("SPOQC_HQPR_METRICES module does not support Conda. Please use Docker / Singularity / Podman instead.")
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
        error("SPOQC_HQPR_METRICES module does not support Conda. Please use Docker / Singularity / Podman instead.")
    }

    """
    mkdir -p report/hqpr/hqpr_metrices/${staining}
    mkdir -p spoQC_tmp/metrices/hqpr/${staining}
    mkdir -p report
    touch report/staining_log.txt
    """

}
