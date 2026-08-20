//
// Run segger create_dataset, train and predict modules
//

include { SEGGER2XR                        } from '../../../modules/local/utility/segger2xr/main'
include { SEGGER_TRAIN                     } from '../../../modules/local/segger/train/main'
include { SEGGER_PREDICT                   } from '../../../modules/local/segger/predict/main'
include { SEGGER_CREATE_DATASET            } from '../../../modules/local/segger/create_dataset/main'
include { XENIUMRANGER_IMPORTSEGMENTATION  } from '../../../modules/nf-core/xeniumranger/importsegmentation/main'

workflow SEGGER_CREATE_TRAIN_PREDICT {
    take:
    ch_bundle              // channel: [ val(meta), ["path-to-xenium-bundle"] ]
    ch_transcripts_file // channel: [ val(meta), [bundle + "/transcripts.parquet"]]
    segger_model           // value: path to a pre-trained segger model checkpoint (or null)
    expansion_distance      // value: nuclear expansion distance

    main:

    // Note: spatialaxe uses "pixels" but per 10x docs, transcript-based segmentation
    // (like Baysor/Segger) must use "microns" since Xenium coordinates are in microns
    ch_coordinate_space = channel.value("microns")

    // create dataset (always needed for predict step)
    SEGGER_CREATE_DATASET(ch_bundle)

    // Determine model source and join all PREDICT inputs by meta.
    // Without meta-based join, queue channels align by emission order,
    // which is non-deterministic and causes cross-sample input mispairing.
    if (segger_model) {
        // Use pre-trained model - skip training
        def model_path = file(segger_model)
        ch_predict_paired = SEGGER_CREATE_DATASET.out.datasetdir
            .join(ch_transcripts_file)
            .map { meta, dataset, tx -> [meta, dataset, model_path, tx] }
    } else {
        // Train a new model per sample, join all inputs by meta
        SEGGER_TRAIN(SEGGER_CREATE_DATASET.out.datasetdir)
        ch_predict_paired = SEGGER_CREATE_DATASET.out.datasetdir
            .join(SEGGER_TRAIN.out.trained_models)
            .join(ch_transcripts_file)
    }
    // ch_predict_paired: [meta, dataset_dir, models_dir, transcripts]

    SEGGER_PREDICT(
        ch_predict_paired.map { meta, dataset, _m, _tx -> [meta, dataset] },
        ch_predict_paired.map { _meta, _dataset, models, _tx -> models },
        ch_predict_paired.map { _meta, _dataset, _m, tx -> [tx] },
    )
    // convert parquet to XR compatible form
    SEGGER2XR(SEGGER_PREDICT.out.transcripts)

    // run xeniumranger import-segmentation with Baysor-format CSV + viz polygons
    // xeniumranger 4.0 expects Baysor CSV (with is_noise column) for --transcript-assignment
    ch_imp_seg_inputs = ch_bundle
        .combine(SEGGER2XR.out.segmentation_csv, by: 0)
        .combine(SEGGER2XR.out.viz_polygons, by: 0)
        .map { meta, bundle, segmentation_csv, polygons ->
            tuple(
                meta,
                bundle,
                segmentation_csv,  // transcript_assignment (Baysor-format CSV)
                polygons,  // viz_polygons (GeoJSON cell boundaries)
                [],  // nuclei
                [],  // cells
                [],  // coordinate_transform
                ch_coordinate_space.val,
                expansion_distance,
            )
        }

    XENIUMRANGER_IMPORTSEGMENTATION(
        ch_imp_seg_inputs
    )

    emit:
    coordinate_space = ch_coordinate_space // channel: [ "microns" ]
    redefined_bundle = XENIUMRANGER_IMPORTSEGMENTATION.out.outs // channel: [ val(meta), ["redefined-xenium-bundle"] ]
}
