
process SPOQC_HQTR_AC {
    tag "$meta.id"
    label 'process_xl_cpus'
    label 'process_xl_mem'
    label 'process_mid_time'
    label 'spoqc'


    container "heylf/spoqc:0.0.1"

    input:
    tuple val(meta), path(spatialdata, stageAs: "*")
    val(step)
    path(ambient, stageAs: "./spoQC_tmp/ambient_output_genes.parquet")

    output:
    tuple val(meta), path("./report/hqtr/hqtr_ac")                       , emit: report
    tuple val(meta), path("./spoQC_tmp/hqtr_output_ac_prob")             , emit: tmp
    tuple val("${task.process}"), val('spoqc'), eval("spoqc --version 2>&1 | grep -oP '\\d+\\.\\d+\\.\\d+' || echo unknown"), topic: versions, emit: versions_spoqc

    when:
    task.ext.when == null || task.ext.when

    script:
    // Exit if running this module with -profile conda / -profile mamba
    if (workflow.profile.tokenize(',').intersect(['conda', 'mamba']).size() >= 1) {
        error("SPOQC_HQTR_AC module does not support Conda. Please use Docker / Singularity / Podman instead.")
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
        error("SPOQC_HQTR_AC module does not support Conda. Please use Docker / Singularity / Podman instead.")
    }

    """
    mkdir -p ./report/hqtr/hqtr_ac
    mkdir -p ./spoQC_tmp/hqtr_output_ac_prob
    """

}
