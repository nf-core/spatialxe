process SPOQC_COMBINE_MASKS {
    tag "${meta.id}_${staining}"
    label 'process_mid_cpus'
    label 'process_xl_mem'
    label 'process_low_time'
    label 'spoqc'


    container "heylf/spoqc:0.0.1"

    input:
    tuple val(meta), path(spatialdata, stageAs: "*"), val(_stain_sd)
    val(step)
    tuple path(mask_hqcr, stageAs: "spoQC_tmp/hqcr_output_mask_raw.parquet"), val(_stain_hqcr)
    tuple val(staining), path(mask_hqpr, stageAs: "spoQC_tmp/*")
    tuple path(mask_hqtr, stageAs: "spoQC_tmp/hqtr_output_mask_raw"), val(_stain_hqtr)
    tuple path(mask_smoothed_hqcr, stageAs: "spoQC_tmp/hqcr_output_mask_smoothed_raw.parquet"), val(_stain_smoothed_hqcr)
    tuple val(_s), path(mask_smoothed_hqpr, stageAs: "spoQC_tmp/*")
    tuple path(mask_smoothed_hqtr, stageAs: "spoQC_tmp/hqtr_output_mask_smoothed_raw"), val(_stain_smoothed_hqtr)

    output:
    tuple val(meta), path("report/combine_masks/${staining}")    , emit: report
    tuple val("${task.process}"), val('spoqc'), eval("spoqc --version 2>&1 | grep -oP '\\d+\\.\\d+\\.\\d+' || echo unknown"), topic: versions, emit: versions_spoqc

    when:
    task.ext.when == null || task.ext.when

    script:
    // Exit if running this module with -profile conda / -profile mamba
    if (workflow.profile.tokenize(',').intersect(['conda', 'mamba']).size() >= 1) {
        error("SPOQC_COMBINE_MASKS module does not support Conda. Please use Docker / Singularity / Podman instead.")
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
        error("SPOQC_COMBINE_MASKS module does not support Conda. Please use Docker / Singularity / Podman instead.")
    }

    """
    mkdir -p report/combine_masks/${staining}
    """

}
