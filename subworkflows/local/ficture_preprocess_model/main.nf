//
// Run ficture preprocess and model modules
//

include { FICTURE_PREPROCESS } from '../../../modules/local/ficture/preprocess/main'
include { FICTURE            } from '../../../modules/local/ficture/model/main'
include { PARQUET2CSV        } from '../../../modules/local/utility/parquet2csv/main'



workflow FICTURE_PREPROCESS_MODEL {
    take:
    ch_transcripts_parquet // channel: [ val(meta), [ "transcripts.parquet" ] ]
    ch_features            // channel: [ ["features"] ]
    features               // value: path to features list (or null)

    main:

    // convert parquet to csv
    PARQUET2CSV(ch_transcripts_parquet)

    // run ficture preprocessing
    ch_transcripts = PARQUET2CSV.out.transcripts_csv

    FICTURE_PREPROCESS(ch_transcripts, ch_features)

    // run the ficture wrapper pipeline
    ch_features_clean = features ? FICTURE_PREPROCESS.out.features : channel.value([])
    FICTURE(
        FICTURE_PREPROCESS.out.transcripts,
        FICTURE_PREPROCESS.out.coordinate_minmax,
        ch_features_clean,
    )
    emit:
    transcripts       = FICTURE_PREPROCESS.out.transcripts       // channel: [ val(meta), [ "*processed_transcripts.tsv.gz" ] ]
    coordinate_minmax = FICTURE_PREPROCESS.out.coordinate_minmax // channel: [ "*coordinate_minmax.tsv" ]
    features          = FICTURE_PREPROCESS.out.features          // channel: [ "*feature.clean.tsv.gz" ]
    results           = FICTURE.out.results                      // channel: [ val(meta), [ "results/** ] ]
}
