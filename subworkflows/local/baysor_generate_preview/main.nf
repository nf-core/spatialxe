//
// Run baysor create_dataset & preview
//

include { BAYSOR_PREVIEW        } from '../../../modules/nf-core/baysor/preview/main'

include { BAYSOR_CREATE_DATASET } from '../../../modules/local/utility/create_dataset/main'
include { EXTRACT_PREVIEW_DATA  } from '../../../modules/local/utility/extract_preview_data/main'
include { PARQUET2CSV           } from '../../../modules/local/utility/parquet2csv/main'

workflow BAYSOR_GENERATE_PREVIEW {

    take:
    ch_transcripts_file // channel: [ val(meta), ["path-to-transcripts.parquet"] ]
    ch_config           // channel: ["path-to-xenium.toml"]

    main:

    ch_preview_mqc_html = channel.empty()
    ch_preview_mqc_png  = channel.empty()

    // run parquet to csv
    PARQUET2CSV(ch_transcripts_file)

    // generate randomised sample data
    BAYSOR_CREATE_DATASET(PARQUET2CSV.out.transcripts_csv, 0.3)

    // run baysor preview if param - generate_preview is true
    ch_sampled_transcripts = BAYSOR_CREATE_DATASET.out.sampled_transcripts
    ch_baysor_preview_input = ch_sampled_transcripts
                                .combine(ch_config)
                                .map { meta, transcripts, config ->
                                    tuple(
                                        meta,
                                        transcripts,
                                        config
                                    )
                                }
    BAYSOR_PREVIEW(ch_baysor_preview_input)

    // clean the preview html file generated
    EXTRACT_PREVIEW_DATA(BAYSOR_PREVIEW.out.html)

    ch_preview_mqc_html = EXTRACT_PREVIEW_DATA.out.mqc_data
    ch_preview_mqc_png  = EXTRACT_PREVIEW_DATA.out.mqc_img

    emit:

    preview_html = ch_preview_mqc_html // channel: [ val(meta), ["*_mqc.tsv"] ]
    preview_img  = ch_preview_mqc_png  // channel: [ val(meta), ["*_mqc.png"] ]
}
