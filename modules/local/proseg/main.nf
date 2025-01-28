process PROSEG {
    tag "$meta.id"
    label 'process_high'

    container "nf-core/proseg:1.1.8"

    input:
    tuple val(meta), path(transcripts)

    output:
    tuple val(meta), path("cell-polygons.geojson.gz"), emit: cell_polygons_2d
    path("expected-counts.csv.gz"), emit: expected_counts
    path("cell-metadata.csv.gz"), emit: cell_metadata
    path("transcript-metadata.csv.gz"), emit: transcript_metadata
    path("gene-metadata.csv.gz"), emit: gene_metadata
    path("rates.csv.gz"), emit: rates
    path("cell-polygons-layers.geojson.gz"), emit:  cell_polygons_layers
    path("cell-hulls.geojson.gz"), emit: cell_hulls
    path("versions.yml"), emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    // Exit if running this module with -profile conda / -profile mamba
    if (workflow.profile.tokenize(',').intersect(['conda', 'mamba']).size() >= 1) {
        error "PROSEG module does not support Conda. Please use Docker / Singularity / Podman instead."
    }
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def platform = preset ? "${params.preset}" : ""

    // check for preset values
    if (!(platform in ['xenium', 'cosmx', 'merscope'])) {
        error "${platform} is an invalid platform (preset) type. Please specify xenium, cosmx, or merscope"
    }

    """
    proseg \\
        --${preset} \\
        ${transcripts} \\
        --nthreads ${task.cpus} \\
        --output-expected-counts expected-counts.csv.gz \\
        --output-cell-metadata cell-metadata.csv.gz \\
        --output-transcript-metadata transcript-metadata.csv.gz \\
        --output-gene-metadata gene-metadata.csv.gz \\
        --output-rates rates.csv.gz \\
        --output-cell-polygons cell-polygons.geojson.gz \\
        --output-cell-polygon-layers cell-polygons-layers.geojson.gz \\
        --output-cell-hulls cell-hulls.geojson.gz \\
        ${args}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        proseg: \$(proseg --version | sed 's/proseg //')
    END_VERSIONS
    """

    stub:
    // Exit if running this module with -profile conda / -profile mamba
    if (workflow.profile.tokenize(',').intersect(['conda', 'mamba']).size() >= 1) {
        error "PROSEG module does not support Conda. Please use Docker / Singularity / Podman instead."
    }
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"

    """
    touch expected-counts.csv.gz
    touch cell-metadata.csv.gz
    touch transcript-metadata.csv.gz
    touch gene-metadata.csv.gz
    touch rates.csv.gz
    touch cell-polygons.geojson.gz
    touch cell-polygons-layers.geojson.gz
    touch cell-hulls.geojson.gz

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        proseg: \$(proseg --version | sed 's/proseg //')
    END_VERSIONS
    """
}
