process BAYSOR_ESTIMATE_SCALE_FACTOR {
    tag "$meta.id"
    label 'process_single'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine in ['singularity', 'apptainer'] && !task.ext.singularity_pull_docker_container ?
        'https://community-cr-prod.seqera.io/docker/registry/v2/blobs/sha256/52/52405e3fa5becaa3347716386a108ca08fc67a416aa0a86115d8e6102ec1700d/data':
        'community.wave.seqera.io/library/numpy_pandas_scipy:0e1d34285644fc85' }"

    input:
    tuple val(meta), path(transcripts)
    val(cell_col)
    val(x_col)
    val(y_col)
    val(min_transcripts_per_cell)

    output:
    tuple val(meta), path("${prefix}_scale_factor.txt"), emit: scale_factor
    tuple val("${task.process}"), val('numpy'), eval("python3 -c 'import numpy; print(numpy.__version__)'"), topic: versions, emit: versions_numpy
    tuple val("${task.process}"), val('pandas'), eval("python3 -c 'import pandas; print(pandas.__version__)'"), topic: versions, emit: versions_pandas
    tuple val("${task.process}"), val('python3'), eval("python3 -c 'import platform; print(platform.python_version())'"), topic: versions, emit: versions_python

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def cell_column = cell_col ? "--cell-column ${cell_col}" : ''
    prefix = task.ext.prefix ?: "${meta.id}"

    """
    baysor_estimate_scale_factor.py \\
        --transcripts ${transcripts} \\
        ${cell_column} \\
        --x-column ${x_col} \\
        --y-column ${y_col} \\
        --percentile 90.0 \\
        --min-transcripts ${min_transcripts_per_cell} \\
        --prefix ${prefix} \\
        --verbose \\
        ${args}
    """

    stub:
    prefix = task.ext.prefix ?: "${meta.id}"

    """
    touch ${prefix}_scale_factor.txt
    """
}
