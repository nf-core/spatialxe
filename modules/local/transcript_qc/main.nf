process TRANSCRIPT_QC_PROCESSING {
    tag "${meta.id}"
    label 'process_high'

    conda "${moduleDir}/environment.yml"
    // Built from environment.yml in this directory (see the module Dockerfile).
    // Hosted on the author's quay.io namespace for now; to be migrated to the
    // nf-core org before release.
    container "quay.io/dongzehe/transcript_qc:1.0.0"

    input:
    tuple val(meta), val(parameters), path(input_files)

    output:
    tuple val(meta), path(outdir), emit: outdir
    tuple val("${task.process}"), val('python'), eval("python3 --version | sed 's/Python //'"), topic: versions, emit: versions_python
    tuple val("${task.process}"), val('numpy'), eval("python3 -c 'import numpy; print(numpy.__version__)'"), topic: versions, emit: versions_numpy
    tuple val("${task.process}"), val('pandas'), eval("python3 -c 'import pandas; print(pandas.__version__)'"), topic: versions, emit: versions_pandas
    tuple val("${task.process}"), val('pyarrow'), eval("python3 -c 'import pyarrow; print(pyarrow.__version__)'"), topic: versions, emit: versions_pyarrow
    tuple val("${task.process}"), val('scanpy'), eval("python3 -c 'import scanpy; print(scanpy.__version__)'"), topic: versions, emit: versions_scanpy
    tuple val("${task.process}"), val('anndata'), eval("python3 -c 'import anndata; print(anndata.__version__)'"), topic: versions, emit: versions_anndata
    tuple val("${task.process}"), val('scipy'), eval("python3 -c 'import scipy; print(scipy.__version__)'"), topic: versions, emit: versions_scipy
    tuple val("${task.process}"), val('matplotlib'), eval("python3 -c 'import matplotlib; print(matplotlib.__version__)'"), topic: versions, emit: versions_matplotlib
    tuple val("${task.process}"), val('seaborn'), eval("python3 -c 'import seaborn; print(seaborn.__version__)'"), topic: versions, emit: versions_seaborn

    when:
    task.ext.when == null || task.ext.when

    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    outdir = prefix

    // Convert parameters to script arguments
    def args = []

    // Required parameters
    args << "--xenium-bundle-dir '${input_files[0]}'"
    args << "--outdir '${outdir}'"
    args << "--threads ${task.cpus}"
    args << "--task-process '${task.process}'"

    // Parse optional parameters from the parameters list
    def param_map = [:]
    if (parameters) {
        parameters.collate(2).each { key, value ->
            if (value != null && value != '') {
                param_map[key] = value
            }
        }
    }

    // Add optional parameters. Per-sample `parameters` entries win; the
    // module-level `task.ext.*` keys are the fallback, so the process never
    // reads `params.*` directly (nf-core reviewer requirement).
    def non_gene_prefix = param_map['NON_GENE_PREFIX'] ?: task.ext.non_gene_prefix
    if (non_gene_prefix) {
        def prefixes = non_gene_prefix.toString().split(';').collect { "'${it.trim()}'" }.join(' ')
        args << "--non-gene-prefix ${prefixes}"
    }

    def stain_names = param_map['STAIN_NAMES'] ?: task.ext.stain_names
    if (stain_names) {
        def stains = stain_names.toString().split(';').collect { "'${it.trim()}'" }.join(' ')
        args << "--stain-names ${stains}"
    }

    def num_row_groups = param_map['NUM_ROW_GROUPS'] ?: task.ext.num_row_groups
    if (num_row_groups) {
        args << "--num-row-groups ${num_row_groups}"
    }

    def pipeline_segmentation = param_map['PIPELINE_SEGMENTATION'] ?: task.ext.pipeline_segmentation
    if (pipeline_segmentation) {
        args << "--pipeline-segmentation '${pipeline_segmentation}'"
    }

    // Boolean flag — emit only when truthy (never '--is-resegmented false')
    def is_resegmented = param_map.containsKey('IS_RESEGMENTED') ? param_map['IS_RESEGMENTED'] : task.ext.is_resegmented
    if (is_resegmented != null && is_resegmented.toString() == 'true') {
        args << "--is-resegmented"
    }

    // Run-global segmentation versions.yml (staged as input_files[1] on
    // post-seg runs; input_files[0] is always the bundle dir). Used to label
    // the pipeline tool version.
    if (input_files.size() > 1) {
        args << "--seg-versions-file '${input_files[1]}'"
    }

    """
    # print() is block-buffered when stdout is not a tty, so an OOM SIGKILL
    # discards every buffered progress line and the task log shows only the kill.
    # That is what made this module's OOM undiagnosable.
    export PYTHONUNBUFFERED=1
    export MKL_NUM_THREADS="$task.cpus"
    export OPENBLAS_NUM_THREADS="$task.cpus"
    export OMP_NUM_THREADS="$task.cpus"
    export NUMBA_NUM_THREADS="$task.cpus"


    transcript_qc_processing.py \\
        ${args.join(' \\\n        ')}
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    outdir = prefix
    """
    mkdir -p "${outdir}/figures"
    touch "${outdir}/transcript_qc_metrics.json"
    """
}
