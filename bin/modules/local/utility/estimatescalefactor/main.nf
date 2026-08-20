process BAYSOR_ESTIMATE_SCALE_FACTOR {
    tag "$meta.id"
    label 'process_single'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine in ['singularity', 'apptainer'] && !task.ext.singularity_pull_docker_container ?
        'https://community-cr-prod.seqera.io/docker/registry/v2/blobs/sha256/35/351697a9ef131734b4aa14d71c7f6263075294013a107685a90a28573f1b3721/data':
        'community.wave.seqera.io/library/pandas_numpy:278960cc5cd28666' }"

    input:
    tuple val(meta), path(transcripts)
    tuple val(cell_col), val(x_col), val(y_col), val(min_transcripts_per_cell)

    output:
    tuple val(meta), stdout, emit: scale_factor
    tuple val("${task.process}"), val('python3'), eval("python3 --version | awk '{print $2}'"), topic: versions, emit: versions_python
    tuple val("${task.process}"), val('numpy'), eval("python3 -c 'import numpy; print(numpy.__version__)'"), topic: versions, emit: versions_numpy
    tuple val("${task.process}"), val('pandas'), eval("python3 -c 'import pandas; print(pandas.__version__)'"), topic: versions, emit: versions_pandas

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''

    """
    baysor_estimate_scale_factor.py \\
        --transcripts ${transcripts} \\
        --cell-column ${cell_col} \\
        --x-column ${x_col} \\
        --y-column ${y_col} \\
        --percentile 90.0 \\
        --min-transcripts ${min_transcripts_per_cell} \\
        --verbose \\
        ${args}
    """

    stub:
    """
    echo "7.00"
    """
}
