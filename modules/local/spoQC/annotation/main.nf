
process SPOQC_ANNOTATION {
    tag "$meta.id"
    label 'process_xl_cpus'
    label 'process_mid_mem'
    label 'process_high_time'
    label 'spoqc'


    container "heylf/spoqc:0.0.1"

    input:
    tuple val(meta), path(spatialdata, stageAs: "*")
    path(annotation, stageAs: "*")
    val(step)

    output:
    tuple val(meta), path("./report/annotation")                                                      , emit: report
    tuple val(meta), path("./report/annotation/unsupervised_cell_annotation.tsv")                     , emit: annotation
    tuple val("${task.process}"), val('spoqc'), eval("spoqc --version 2>&1 | grep -oP '\\d+\\.\\d+\\.\\d+' || echo unknown"), topic: versions, emit: versions_spoqc

    when:
    !annotation || (annotation instanceof List && annotation.isEmpty())

    script:
    // Exit if running this module with -profile conda / -profile mamba
    if (workflow.profile.tokenize(',').intersect(['conda', 'mamba']).size() >= 1) {
        error("SPOQC_ANNOTATION module does not support Conda. Please use Docker / Singularity / Podman instead.")
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
        error("SPOQC_ANNOTATION module does not support Conda. Please use Docker / Singularity / Podman instead.")
    }

    """
    mkdir -p ./report/annotation
    touch ./report/annotation/unsupervised_cell_annotation.tsv
    """
}
