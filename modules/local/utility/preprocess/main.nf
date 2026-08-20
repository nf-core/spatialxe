process BAYSOR_PREPROCESS_TRANSCRIPTS {
    tag "${meta.id}"
    label 'process_medium'

    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'https://community-cr-prod.seqera.io/docker/registry/v2/blobs/sha256/94/9409ce399922a5746bea1b7df5668c3d1d79b9af49a15950d9818c4fe45ac749/data' :
        'community.wave.seqera.io/library/pandas_procs_pyarrow:d8f882b65dfea451' }"

    input:
    tuple val(meta), path(transcripts)
    val min_qv
    val max_x
    val min_x
    val max_y
    val min_y

    output:
    tuple val(meta), path("${prefix}/filtered_transcripts.csv"), emit: transcripts_csv
    tuple val("${task.process}"), val('python'), eval("python3 --version | sed 's/Python //'"), topic: versions, emit: versions_python

    when:
    task.ext.when == null || task.ext.when

    script:
    // Exit if running this module with -profile conda / -profile mamba
    if (workflow.profile.tokenize(',').intersect(['conda', 'mamba']).size() >= 1) {
        error("BAYSOR_PREPROCESS_TRANSCRIPTS module does not support Conda. Please use Docker / Singularity / Podman instead.")
    }

    prefix = task.ext.prefix ?: "${meta.id}"

    """
    baysor_preprocess_transcripts.py \\
        --transcripts ${transcripts} \\
        --prefix ${prefix} \\
        --min-qv ${min_qv} \\
        --min-x ${min_x} \\
        --max-x ${max_x} \\
        --min-y ${min_y} \\
        --max-y ${max_y}
    """

    stub:
    // Exit if running this module with -profile conda / -profile mamba
    if (workflow.profile.tokenize(',').intersect(['conda', 'mamba']).size() >= 1) {
        error("BAYSOR_PREPROCESS_TRANSCRIPTS module does not support Conda. Please use Docker / Singularity / Podman instead.")
    }

    prefix = task.ext.prefix ?: "${meta.id}"

    """
    mkdir -p ${prefix}
    touch ${prefix}/filtered_transcripts.csv
    """
}
