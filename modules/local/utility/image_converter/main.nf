process IMAGE_COVERTER{
    
    tag "IMAGE_COVERTER"
    
    publishDir "${params.outdir}/images", mode: 'copy'
    
    container "docker://labsyspharm/bftools:latest"
    
    memory = '50.GB'
    cpus = 10

    input:
    path (omi_tif_image)
    val (extension)

    output:
    path ("bfconvert_output"), emit: converted_images
    path ("versions.yml"), emit: versions
    
    when:
    task.ext.when == null || task.ext.when
    
    script:
    def args = task.ext.args ?: ''
    """
    # Increase heap memory used by JVM inside the container to avoid OutOfMemoryError
    # Should be set according to the amount of memonry allocated to the module
    export BF_MAX_MEM=50g
    
    # Create output directory for bfconvert results
    mkdir bfconvert_output
    
    # Run bfconvert with the specified parameters
    # The command for now only works by converting the .ome.tiff image into several .png files.
    bfconvert \\
        $args \\
        ${omi_tif_image} bfconvert_output/output_series_%s_Z%z.${extension}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bftools: \$(echo \$(bfconvert -version 2>&1) | sed 's/^.*Version: //; s/ Build.*\$//')
    END_VERSIONS
    """

    stub:
    """
    mkdir bfconvert_output
    touch bfconvert_output/output_series_0_Z0.${extension}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bftools: \$(echo \$(bfconvert -version 2>&1) | sed 's/^.*Version: //; s/ Build.*\$//')
    END_VERSIONS
    """
}