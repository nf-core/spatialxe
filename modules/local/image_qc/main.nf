process IMAGE_QC_ANALYSIS {
    tag "${meta.id}"
    // process_gpu_qc carries only the GPU request; cpus/memory/time come from
    // process_xl (30 cpus / 240 GB / 24 h), which matches what a 5.5 gigapixel
    // production bundle actually needed.
    label 'process_gpu_qc'
    label 'process_xl'

    conda "${moduleDir}/environment.yml"
    // Built from environment.yml in this directory (see the module Dockerfile).
    // Hosted on the author's quay.io namespace for now; to be migrated to the
    // nf-core org before release.
    container "quay.io/dongzehe/image_qc:1.0.0"

    input:
    tuple val(meta), val(parameters), path(input_files)
    path(roi_thresholds_yaml)

    output:
    tuple val(meta), path(outdir), emit: outdir
    tuple val("${task.process}"), val('python'), eval("python3 --version | sed 's/Python //'"), topic: versions, emit: versions_python
    tuple val("${task.process}"), val('numpy'), eval("python3 -c 'import numpy; print(numpy.__version__)'"), topic: versions, emit: versions_numpy
    tuple val("${task.process}"), val('scikit-image'), eval("python3 -c 'import skimage; print(skimage.__version__)'"), topic: versions, emit: versions_skimage

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
    args << "--sample-id '${meta.id}'"

    // ROI thresholds YAML (staged file)
    if (roi_thresholds_yaml.name != 'NO_FILE') {
        args << "--roi-thresholds-yaml '${roi_thresholds_yaml}'"
    }
    // Parse optional parameters from the parameters list
    def param_map = [:]
    if (parameters) {
        parameters.collate(2).each { key, value ->
            if (value != null && value != '') {
                param_map[key] = value
            }
        }
    }

    // --stain-names is deliberately not passed: the script parses it and then
    // discards it (every figure title and metric key is hard-coded, and those
    // keys are the contract the threshold YAML and the report read), so the
    // pipeline parameter was dropped rather than shipping a knob with no effect.

    // Tile size parameter (a.k.a. ROI size internally). Per-sample
    // `parameters` map can override via ROI_SIZE; otherwise the module-level
    // `task.ext.tile_size` (wired from the Seqera-visible pipeline param) is
    // used. Default 35 px.
    if (param_map.containsKey('ROI_SIZE') && param_map['ROI_SIZE']) {
        args << "--roi-size ${param_map['ROI_SIZE']}"
    } else {
        args << "--roi-size ${task.ext.tile_size ?: 35}"
    }

    // Legacy focus score mode (CPU for-loop instead of GPU convolution)
    if (task.ext.legacy_focus) {
        args << "--legacy-focus"
    }

    if (task.ext.no_snr) {
        args << '--no-snr'
    }
    if (task.ext.snr_no_roi_tx_table) {
        args << '--snr-no-roi-tx-table'
    }
    if (task.ext.snr_otsu_max_rois != null) {
        args << "--snr-otsu-max-rois ${task.ext.snr_otsu_max_rois}"
    }
    // snr_no_moran true (default): Moran off — no flag. false: opt in to Moran.
    if (!task.ext.snr_no_moran) {
        args << '--snr-with-moran'
    }
    if (task.ext.save_dapi_maps_tiff) {
        args << '--save-dapi-maps-tiff'
    }
    // Streaming is the script default: each tile is reduced and dropped, so no
    // full-resolution pixel plane is written. The planes cost ~154 GB of scratch on
    // a 5.5 gigapixel sample and their writeback to S3 is what pushed run
    // 3nkeHOEV1ONlbK past the 4 h wall. Opt out only to reproduce the old path.
    if (!task.ext.stream_tiles) {
        args << '--no-stream-tiles'
    }
    // Per-figure figures_source/*.csv source-data exports are unused downstream
    // (the QMD embeds only the PNGs) and the big ROI-table dumps cost ~80 s each.
    // Off by default; opt in to write them.
    if (task.ext.figure_source_tables) {
        args << '--figure-source-tables'
    }
    // Figure generation master toggle. Off skips ALL figure rendering for a
    // metrics-only fast run; metrics/JSON/parquet are always produced.
    if (!task.ext.figures) {
        args << '--no-figures'
    }
    // Cap the devices the script uses. `accelerator` only tells AWS Batch how many
    // GPUs to request -- it does not restrict what CUDA can see, so a task asking
    // for 1 GPU that lands on a 4-GPU instance would otherwise detect and use all
    // four. Observed on run 3nkeHOEV1ONlbK: image_qc_gpus=1 placed on a
    // g6e.12xlarge and the script reported "4 GPU(s)".
    args << "--max-gpus ${task.ext.max_gpus}"
    if (task.ext.lap_sigma != null) {
        args << "--lap-sigma ${task.ext.lap_sigma}"
    }

    if (param_map.containsKey('PIPELINE_SEGMENTATION') && param_map['PIPELINE_SEGMENTATION']) {
        args << "--pipeline-segmentation '${param_map['PIPELINE_SEGMENTATION']}'"
    }

    // Boolean flag — emit only when truthy (never '--is-resegmented false')
    if (param_map.containsKey('IS_RESEGMENTED') && param_map['IS_RESEGMENTED'].toString() == 'true') {
        args << "--is-resegmented"
    }

    """
    export MKL_NUM_THREADS="$task.cpus"
    export OPENBLAS_NUM_THREADS="$task.cpus"
    export OMP_NUM_THREADS="$task.cpus"
    export NUMBA_NUM_THREADS="$task.cpus"

    # Capture the analysis exit code without aborting (Nextflow runs with set -e).
    # A very dim sample can make image_qc.py exit 1 (e.g. no tissue tiles clear the
    # intensity gate); we still want a report, so on exit 1 we record a failed
    # status and exit 0. The Quarto report reads image_qc_status.json and renders a
    # QC-FAILED banner. Signal / OOM / preemption codes (104, 130-145) are re-raised
    # so Nextflow's retry errorStrategy still fires.
    rc=0
    image_qc.py \\
        ${args.join(' \\\n        ')} || rc=\$?

    mkdir -p "${outdir}"

    if [ "\$rc" -eq 0 ]; then
        echo '{"status": "ok", "sample_id": "${meta.id}"}' > "${outdir}/image_qc_status.json"
    elif [ "\$rc" -eq 1 ]; then
        echo "WARNING: image_qc.py exited 1 (Python error); writing failed status for report" >&2
        echo '{"status": "failed", "exit_code": 1, "sample_id": "${meta.id}"}' > "${outdir}/image_qc_status.json"
    else
        echo "image_qc.py exited \$rc (signal/OOM); propagating for errorStrategy" >&2
        exit \$rc
    fi
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    outdir = prefix
    """
    mkdir -p "${outdir}/figures"
    touch "${outdir}/image_qc_metrics.json"
    touch "${outdir}/image_qc_metrics.csv"
    """
}
