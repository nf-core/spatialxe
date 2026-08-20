//
// Run baysor segfree
//

include { BAYSOR_SEGFREE                } from '../../../modules/nf-core/baysor/segfree/main'

include { BAYSOR_PREPROCESS_TRANSCRIPTS } from '../../../modules/local/utility/preprocess/main'
// TODO - should we include a module to process the output loom file with scapny or anndata?

workflow BAYSOR_GENERATE_SEGFREE {

    take:
    ch_transcripts_file // channel: [ val(meta), ["transcripts.parquet"] ]
    ch_config           // channel: [ ["path-to-xenium.toml"] ]
    max_x               // value: spatial filter upper x bound
    max_y               // value: spatial filter upper y bound
    min_qv              // value: minimum transcript QV
    min_x               // value: spatial filter lower x bound
    min_y               // value: spatial filter lower y bound

    main:

    ch_transcripts = channel.empty()

    // Always preprocess transcripts.parquet to CSV for Baysor 0.7.1 compatibility.
    // Baysor's Julia Parquet.jl cannot read zstd-compressed parquet files from Xenium bundles.
    // Also applies optional spatial/QV filtering when filter_transcripts is true.
    BAYSOR_PREPROCESS_TRANSCRIPTS(
        ch_transcripts_file,
        min_qv,
        max_x,
        min_x,
        max_y,
        min_y,
    )
    ch_transcripts = BAYSOR_PREPROCESS_TRANSCRIPTS.out.transcripts_csv

    // run baysor segfree
    ch_baysor_segfree_input = ch_transcripts
                                .combine(ch_config)
                                .map { meta, transcripts, config ->
                                    tuple(
                                        meta,
                                        transcripts,
                                        config
                                    )
                                }
    BAYSOR_SEGFREE(
        ch_baysor_segfree_input
    )

    emit:

    ncvs     = BAYSOR_SEGFREE.out.ncvs // channel: [ val(meta), ["ncvs.loom"] ]
}
