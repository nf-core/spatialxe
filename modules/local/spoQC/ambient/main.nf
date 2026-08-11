
process SPOQC_AMBIENT {
    tag "$meta.id"
    label 'process_high_cpus'
    label 'process_xl_mem'
    label 'process_mid_time'
    label 'spoqc'

    container "heylf/spoqc:0.0.1"

    input:
    tuple val(meta), path(spatialdata, stageAs: "*")
    val(step)

    output:
    tuple val(meta), path("./report/ambientqc")                          , emit: report
    tuple val(meta), path("./spoQC_tmp/ambient_output_genes.parquet")    , emit: tmp
    tuple val("${task.process}"), val('spoqc'), eval("spoqc --version 2>&1 | grep -oP '\\d+\\.\\d+\\.\\d+' || echo unknown"), topic: versions, emit: versions_spoqc

    when:
    task.ext.when == null || task.ext.when

    script:
    // Exit if running this module with -profile conda / -profile mamba
    if (workflow.profile.tokenize(',').intersect(['conda', 'mamba']).size() >= 1) {
        error("SPOQC_AMBIENT module does not support Conda. Please use Docker / Singularity / Podman instead.")
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
        error("SPOQC_AMBIENT module does not support Conda. Please use Docker / Singularity / Podman instead.")
    }

    """
    mkdir -p ./report/ambientqc
    mkdir -p ./spoQC_tmp
    touch ./spoQC_tmp/ambient_output_genes.parquet
    """

}
