process UNZIP {
    tag "$archive"
    label 'process_single'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine in ['singularity', 'apptainer'] && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/p7zip:16.02' :
        'quay.io/biocontainers/p7zip:16.02' }"

    input:
    tuple val(meta), path(archive)

    output:
    tuple val(meta), path("${prefix}/"), emit: unzipped_archive
    tuple val("${task.process}"), val('7za'), eval("7za 2>&1 | sed -n '2s/^.* \\([0-9.]*\\) .*/\\1/p'"), topic: versions, emit: versions_7za

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    if ( archive instanceof List && archive.name.size > 1 ) { error "[UNZIP] error: 7za only accepts a single archive as input. Please check module input." }
    prefix = task.ext.prefix ?: ( meta.id ? "${meta.id}" : archive.baseName)
    """
    7za \\
        x \\
        -o"${prefix}"/ \\
        $args \\
        $archive
    """

    stub:
    if ( archive instanceof List && archive.name.size > 1 ) { error "[UNZIP] error: 7za only accepts a single archive as input. Please check module input." }
    prefix = task.ext.prefix ?: ( meta.id ? "${meta.id}" : archive.baseName)
    """
    mkdir "${prefix}"

    ## Emit a valid empty file. nf-test's snapshot md5 path decompresses .gz files,
    ## so a 0-byte .gz (from `touch`) throws EOF in GZIPInputStream and the snapshot
    ## falls back to dumping non-deterministic File metadata (freeSpace, etc).
    ## `: | gzip -n` produces a 20-byte deterministic empty gzip that md5s consistently.
    emit_stub() {
        local f="\$1"
        if [[ "\$f" == *.gz ]]; then
            : | gzip -n > "\$f"
        else
            touch "\$f"
        fi
    }

    ## Dry-run listing the archive contents (7za is the only archiver guaranteed to be
    ## present) and mirror its structure under prefix, matching the unstripped layout
    ## that `7za x` produces in the real script block.
    7za l -slt "${archive}" | awk '
        BEGIN{RS=""; FS="\\n"}
        \$1=="----------"{started=1}
        started{
            path=""; isdir=0
            for(i=1;i<=NF;i++){
                if (\$i ~ /^Path = /) path=substr(\$i,8)
                else if (\$i ~ /^Folder = \\+/) isdir=1
            }
            if (path!="") print (isdir?"D":"F") "\\t" path
        }
    ' | while IFS=\$'\\t' read -r type path; do
        [ -z "\${path}" ] && continue
        if [ "\${type}" = "D" ]; then
            mkdir -p "${prefix}/\${path}"
        else
            mkdir -p "${prefix}/\$(dirname "\${path}")"
            emit_stub "${prefix}/\${path}"
        fi
    done
    """
}
