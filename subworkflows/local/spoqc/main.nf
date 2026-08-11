//
// generate spatialdata object from the spatialxe layers
//

include { SPATIALDATA_WRITE as SPATIALDATA_WRITE_RAW_BUNDLE       } from '../../../modules/local/spatialdata/write/main'
// spoQC general stuff
include { SPOQC_ANNOTATION       } from '../../../modules/local/spoQC/annotation/main'
include { SPOQC_WHOLE_SLIDE       } from '../../../modules/local/spoQC/whole_slide/main'
include { SPOQC_GENERAL       } from '../../../modules/local/spoQC/general/main'
include { SPOQC_BUBBLE       } from '../../../modules/local/spoQC/bubble/main'
include { SPOQC_DOUBLET       } from '../../../modules/local/spoQC/doublet/main'
include { SPOQC_VOID       } from '../../../modules/local/spoQC/void/main'
include { SPOQC_CELL       } from '../../../modules/local/spoQC/cell/main'
include { SPOQC_AMBIENT    } from '../../../modules/local/spoQC/ambient/main'
// spoQC hqcr
include { SPOQC_HQCR_IDENT       } from '../../../modules/local/spoQC/hqcr_ident/main'
include { SPOQC_HQCR_CELLTYPE       } from '../../../modules/local/spoQC/hqcr_celltype/main'
// spoQC hqpr
include { SPOQC_HQPR_METRICES       } from '../../../modules/local/spoQC/hqpr_metrices/main'
include { SPOQC_HQPR_CLUSTERING       } from '../../../modules/local/spoQC/hqpr_clustering/main'
include { SPOQC_HQPR_REFINEMENT       } from '../../../modules/local/spoQC/hqpr_refinement/main'
include { SPOQC_HQPR_BOUNDING_BOX       } from '../../../modules/local/spoQC/hqpr_bounding_box/main'
include { SPOQC_HQPR_CELLTYPE       } from '../../../modules/local/spoQC/hqpr_celltype/main'
// spoQC hqtr
include { SPOQC_HQTR_METRICES       } from '../../../modules/local/spoQC/hqtr_metrices/main'
include { SPOQC_HQTR_AC       } from '../../../modules/local/spoQC/hqtr_ac/main'
include { SPOQC_HQTR_QV       } from '../../../modules/local/spoQC/hqtr_qv/main'
include { SPOQC_HQTR_CLUSTERING       } from '../../../modules/local/spoQC/hqtr_clustering/main'
include { SPOQC_HQTR_REFINEMENT       } from '../../../modules/local/spoQC/hqtr_refinement/main'
include { SPOQC_HQTR_BOUNDING_BOX       } from '../../../modules/local/spoQC/hqtr_bounding_box/main'
include { SPOQC_HQTR_CELLTYPE       } from '../../../modules/local/spoQC/hqtr_celltype/main'
// spoQC downstream
include { SPOQC_COMBINE_MASKS       } from '../../../modules/local/spoQC/combine_masks/main'
include { SPOQC_TRANSCRIPT       } from '../../../modules/local/spoQC/transcript/main'
include { SPOQC_CELLCYCLE       } from '../../../modules/local/spoQC/cellcycle/main'
include { SPOQC_MODEL       } from '../../../modules/local/spoQC/model/main'
include { SPOQC_MARKER       } from '../../../modules/local/spoQC/marker/main'
// spoQC final analysis
include { SPOQC_ANALYSIS_OVERVIEW       } from '../../../modules/local/spoQC/analysis_overview/main'
include { SPOQC_ANALYSIS_CATEGORY       } from '../../../modules/local/spoQC/analysis_category/main'
include { SPOQC_ANALYSIS_CLUSTER       } from '../../../modules/local/spoQC/analysis_cluster/main'
// spoQC final report
include { SPOQC_FINALREPORT       } from '../../../modules/local/spoQC/finalreport/main'

workflow SPOQC {

    take:
    ch_sd                   // channel: [ val(meta), [ "path-to-spatialdata-bundle" ] ]
    ch_annotation_src       // channel: [ val(meta), "path-to-annotation-file" ]
    ch_stainings            // channel: [ [ 1,2,.... ] ]

    main:

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // General
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // keep only actual, usable annotations (non-null and not an empty list)
    ch_annotation_present = ch_annotation_src.filter { _meta, a -> a && (!(a instanceof List) || !a.isEmpty()) }

    // pair each sample's spatialdata bundle with its own annotation by meta.id
    ch_sd
        .join(ch_annotation_src, by: 0)
        .multiMap { meta, spatialdata, annotation ->
            sd:         [meta, spatialdata]
            annotation: annotation
        }
        .set { ch_sd_annotation_src }

    SPOQC_ANNOTATION(
        ch_sd_annotation_src.sd,
        ch_sd_annotation_src.annotation,
        "annotation",
    )
    ch_annotation_path = ch_annotation_present.mix(
        SPOQC_ANNOTATION.out.annotation
    )

    // shared join of ch_sd + ch_annotation_path by meta.id, split back into
    // the two positional args each downstream module expects - multiMap keeps the
    // two outputs in lockstep so the pairing established by join() is preserved
    ch_sd
        .join(ch_annotation_path, by: 0, remainder: true)
        .multiMap { meta, spatialdata, annotation ->
            sd:         [meta, spatialdata]
            annotation: annotation ?: []
        }
        .set { ch_sd_annotation }

    SPOQC_GENERAL(
        ch_sd_annotation.sd,
        ch_sd_annotation.annotation,
        "generalqc",
    )

    SPOQC_WHOLE_SLIDE(
        ch_sd,
        "whole_slide_qc",
    )

    SPOQC_BUBBLE(
        ch_sd,
        "bubbleqc",
    )

    SPOQC_DOUBLET(
        ch_sd_annotation.sd,
        ch_sd_annotation.annotation,
        "doubletqc",
    )

    SPOQC_VOID(
        ch_sd,
        "voidqc",
    )

    SPOQC_CELL(
        ch_sd,
        "cellqc",
    )

    SPOQC_AMBIENT(
        ch_sd,
        "ambientqc",
    )

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // HQCR
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    SPOQC_HQCR_IDENT(
        ch_sd,
        "hqcr_ident",
        SPOQC_GENERAL.out.tmp.map { _meta, f -> f },
        SPOQC_BUBBLE.out.tmp.map { _meta, f -> f },
        SPOQC_DOUBLET.out.tmp.map { _meta, f -> f },
        SPOQC_VOID.out.tmp.map { _meta, f -> f },
        SPOQC_CELL.out.tmp.map { _meta, f -> f },
    )

    SPOQC_HQCR_CELLTYPE(
        ch_sd_annotation.sd,
        ch_sd_annotation.annotation,
        "hqcr_celltype",
        SPOQC_GENERAL.out.tmp.map { _meta, f -> f },
        SPOQC_BUBBLE.out.tmp.map { _meta, f -> f },
        SPOQC_DOUBLET.out.tmp.map { _meta, f -> f },
        SPOQC_VOID.out.tmp.map { _meta, f -> f },
        SPOQC_CELL.out.tmp.map { _meta, f -> f },
    )

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // HQPR
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    SPOQC_HQPR_METRICES(
        ch_sd,
        ch_stainings,
        "hqpr_metrices",
    )

    ch_spatialdata_stainings = ch_sd.combine(ch_stainings)

    SPOQC_HQPR_CLUSTERING(
        ch_spatialdata_stainings,
        "hqpr_clustering",
        SPOQC_HQPR_METRICES.out.metrices.map { _meta, staining, f -> return [staining, f] },
    )

    ch_spoqc_hpq_clustering = SPOQC_HQPR_CLUSTERING.out.mask.map { _meta, staining, f -> return [staining, f] }

    SPOQC_HQPR_REFINEMENT(
        ch_spatialdata_stainings,
        "hqpr_refinement",
        ch_spoqc_hpq_clustering,
    )

    ch_spoqc_hqpr_refinement = SPOQC_HQPR_REFINEMENT.out.mask_smoothed.map { _meta, staining, f -> return [staining, f] }

    SPOQC_HQPR_BOUNDING_BOX(
        ch_spatialdata_stainings,
        "hqpr_bounding_box",
        ch_spoqc_hqpr_refinement,
    )

    // same meta.id-keyed pairing as ch_sd_annotation, extended with staining -
    // combine() is applied after the join so both outputs stay aligned per sample+staining
    ch_sd
        .join(ch_annotation_path, by: 0, remainder: true)
        .map { meta, spatialdata, annotation -> [meta, spatialdata, annotation ?: []] }
        .combine(ch_stainings)
        .multiMap { meta, spatialdata, annotation, staining ->
            sd:         [meta, spatialdata, staining]
            annotation: [annotation, staining]
        }
        .set { ch_sd_annotation_stainings }

    // ch_masks_joined emits: tuple(val(staining), path(mask), path(mask))
    ch_masks_joined = ch_spoqc_hpq_clustering.join( ch_spoqc_hqpr_refinement )

    SPOQC_HQPR_CELLTYPE(
        ch_sd_annotation_stainings.sd,
        ch_sd_annotation_stainings.annotation,
        "hqpr_celltype",
        SPOQC_GENERAL.out.tmp.map { _meta, f -> f }.combine(ch_stainings),
        SPOQC_BUBBLE.out.tmp.map { _meta, f -> f }.combine(ch_stainings),
        SPOQC_DOUBLET.out.tmp.map { _meta, f -> f }.combine(ch_stainings),
        SPOQC_VOID.out.tmp.map { _meta, f -> f }.combine(ch_stainings),
        SPOQC_CELL.out.tmp.map { _meta, f -> f }.combine(ch_stainings),
        ch_masks_joined,
    )

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // HQTR
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SPOQC_HQTR_METRICES(
        ch_sd,
        "hqtr_metrices",
    )

    SPOQC_HQTR_AC(
        ch_sd,
        "hqtr_ac",
        SPOQC_AMBIENT.out.tmp.map { _meta, f -> f },
    )

    SPOQC_HQTR_QV(
        ch_sd,
        "hqtr_qv",
    )

    SPOQC_HQTR_CLUSTERING(
        ch_sd,
        "hqtr_clustering",
        SPOQC_HQTR_METRICES.out.metrices.map { _meta, f -> f },
        SPOQC_HQTR_QV.out.tmp.map { _meta, f -> f },
        SPOQC_HQTR_AC.out.tmp.map { _meta, f -> f },
    )

    SPOQC_HQTR_REFINEMENT(
        ch_sd,
        "hqtr_refinement",
        SPOQC_HQTR_CLUSTERING.out.mask.map { _meta, f -> f },
    )

    SPOQC_HQTR_BOUNDING_BOX(
        ch_sd,
        "hqtr_bounding_box",
        SPOQC_HQTR_REFINEMENT.out.mask_smoothed.map { _meta, f -> f },
    )

    SPOQC_HQTR_CELLTYPE(
        ch_sd_annotation.sd,
        ch_sd_annotation.annotation,
        "hqtr_celltype",
        SPOQC_GENERAL.out.tmp.map { _meta, f -> f },
        SPOQC_BUBBLE.out.tmp.map { _meta, f -> f },
        SPOQC_DOUBLET.out.tmp.map { _meta, f -> f },
        SPOQC_VOID.out.tmp.map { _meta, f -> f },
        SPOQC_CELL.out.tmp.map { _meta, f -> f },
        SPOQC_HQTR_CLUSTERING.out.mask.map { _meta, f -> f },
        SPOQC_HQTR_REFINEMENT.out.mask_smoothed.map { _meta, f -> f },
    )

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Downstream
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    SPOQC_COMBINE_MASKS(
        ch_spatialdata_stainings,
        "combine_masks",
        SPOQC_HQCR_IDENT.out.mask.map { _meta, f -> f }.combine(ch_stainings),
        ch_spoqc_hpq_clustering,
        SPOQC_HQTR_CLUSTERING.out.mask.map { _meta, f -> f }.combine(ch_stainings),
        SPOQC_HQCR_IDENT.out.mask_smoothed.map { _meta, f -> f }.combine(ch_stainings),
        ch_spoqc_hqpr_refinement,
        SPOQC_HQTR_REFINEMENT.out.mask_smoothed.map { _meta, f -> f }.combine(ch_stainings),
    )

    SPOQC_TRANSCRIPT(
        ch_sd_annotation.sd,
        ch_sd_annotation.annotation,
        "transcriptqc",
    )

    SPOQC_CELLCYCLE(
        ch_sd,
        "cellcycleqc",
    )

    SPOQC_MODEL(
        ch_sd,
        "modelqc",
    )

    // SPOQC_MARKER(
    //     ch_sd,
    //     ch_annotation_path,
    //     "markerqc"
    // )

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Analysis
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ch_files_hqpr_metrics = SPOQC_HQPR_METRICES.out.metrices
        .map { _meta, _staining, p -> p }
        .collect()

    ch_files_hqpr_masks = SPOQC_HQPR_CLUSTERING.out.mask
        .map { _meta, _staining, p -> p }
        .collect()

    ch_files_hqpr_masks_smoothed = SPOQC_HQPR_REFINEMENT.out.mask_smoothed
        .map { _meta, _staining, p -> p }
        .collect()

    SPOQC_ANALYSIS_OVERVIEW(
        ch_sd_annotation.sd,
        ch_sd_annotation.annotation,
        "analysis_overview",
        SPOQC_GENERAL.out.tmp.map { _meta, f -> f },
        SPOQC_BUBBLE.out.tmp.map { _meta, f -> f },
        SPOQC_DOUBLET.out.tmp.map { _meta, f -> f },
        SPOQC_VOID.out.tmp.map { _meta, f -> f },
        SPOQC_CELL.out.tmp.map { _meta, f -> f },
        SPOQC_HQCR_IDENT.out.mask.map { _meta, f -> f },
        SPOQC_HQCR_IDENT.out.mask_smoothed.map { _meta, f -> f },
        SPOQC_HQTR_QV.out.tmp.map { _meta, f -> f },
        SPOQC_HQTR_AC.out.tmp.map { _meta, f -> f },
        SPOQC_HQTR_METRICES.out.metrices.map { _meta, f -> f },
        SPOQC_HQTR_REFINEMENT.out.mask_smoothed.map { _meta, f -> f },
        SPOQC_HQTR_CLUSTERING.out.mask.map { _meta, f -> f },
        ch_files_hqpr_metrics,
        ch_files_hqpr_masks_smoothed,
        ch_files_hqpr_masks,
    )

    SPOQC_ANALYSIS_CATEGORY(
        ch_sd_annotation.sd,
        ch_sd_annotation.annotation,
        "analysis_category",
        SPOQC_GENERAL.out.tmp.map { _meta, f -> f },
        SPOQC_BUBBLE.out.tmp.map { _meta, f -> f },
        SPOQC_DOUBLET.out.tmp.map { _meta, f -> f },
        SPOQC_VOID.out.tmp.map { _meta, f -> f },
        SPOQC_CELL.out.tmp.map { _meta, f -> f },
        SPOQC_HQCR_IDENT.out.mask.map { _meta, f -> f },
        SPOQC_HQCR_IDENT.out.mask_smoothed.map { _meta, f -> f },
        SPOQC_HQTR_QV.out.tmp.map { _meta, f -> f },
        SPOQC_HQTR_AC.out.tmp.map { _meta, f -> f },
        SPOQC_HQTR_METRICES.out.metrices.map { _meta, f -> f },
        SPOQC_HQTR_REFINEMENT.out.mask_smoothed.map { _meta, f -> f },
        SPOQC_HQTR_CLUSTERING.out.mask.map { _meta, f -> f },
        ch_files_hqpr_metrics,
        ch_files_hqpr_masks_smoothed,
        ch_files_hqpr_masks,
    )

    SPOQC_ANALYSIS_CLUSTER(
        ch_sd_annotation.sd,
        ch_sd_annotation.annotation,
        "analysis_cluster",
        SPOQC_GENERAL.out.tmp.map { _meta, f -> f },
        SPOQC_BUBBLE.out.tmp.map { _meta, f -> f },
        SPOQC_DOUBLET.out.tmp.map { _meta, f -> f },
        SPOQC_VOID.out.tmp.map { _meta, f -> f },
        SPOQC_CELL.out.tmp.map { _meta, f -> f },
        SPOQC_HQCR_IDENT.out.mask.map { _meta, f -> f },
        SPOQC_HQCR_IDENT.out.mask_smoothed.map { _meta, f -> f },
        SPOQC_HQTR_QV.out.tmp.map { _meta, f -> f },
        SPOQC_HQTR_AC.out.tmp.map { _meta, f -> f },
        SPOQC_HQTR_METRICES.out.metrices.map { _meta, f -> f },
        SPOQC_HQTR_REFINEMENT.out.mask_smoothed.map { _meta, f -> f },
        SPOQC_HQTR_CLUSTERING.out.mask.map { _meta, f -> f },
        ch_files_hqpr_metrics,
        ch_files_hqpr_masks_smoothed,
        ch_files_hqpr_masks,
    )

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Final Report
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // collect the per-staining report fragments into a single list per sample
    ch_report_hqpr_metrices     = SPOQC_HQPR_METRICES.out.report
        .map { _meta, _staining, p -> return p }
        .collect()
    ch_report_hqpr_clustering   = SPOQC_HQPR_CLUSTERING.out.report
        .map { _meta, _staining, p -> p }
        .collect()
    ch_report_hqpr_refinement   = SPOQC_HQPR_REFINEMENT.out.report
        .map { _meta, _staining, p -> p }
        .collect()
    ch_report_hqpr_bounding_box = SPOQC_HQPR_BOUNDING_BOX.out.report
        .map { _meta, _staining, p -> p }
        .collect()
    ch_report_combine_masks     = SPOQC_COMBINE_MASKS.out.report.map { _meta, f -> f }.collect()

    SPOQC_FINALREPORT(
        ch_sd,
        "final_report",
        SPOQC_GENERAL.out.report.map { _meta, f -> f },
        SPOQC_DOUBLET.out.report.map { _meta, f -> f },
        SPOQC_VOID.out.report.map { _meta, f -> f },
        SPOQC_CELL.out.report.map { _meta, f -> f },
        SPOQC_HQCR_IDENT.out.report.map { _meta, f -> f },
        SPOQC_HQCR_CELLTYPE.out.report.map { _meta, f -> f },
        ch_report_hqpr_metrices,
        ch_report_hqpr_clustering,
        ch_report_hqpr_refinement,
        ch_report_hqpr_bounding_box,
        SPOQC_HQPR_CELLTYPE.out.report.map { _meta, f -> f },
        SPOQC_HQTR_METRICES.out.report.map { _meta, f -> f },
        SPOQC_HQTR_AC.out.report.map { _meta, f -> f },
        SPOQC_HQTR_QV.out.report.map { _meta, f -> f },
        SPOQC_HQTR_CLUSTERING.out.report.map { _meta, f -> f },
        SPOQC_HQTR_REFINEMENT.out.report.map { _meta, f -> f },
        SPOQC_HQTR_BOUNDING_BOX.out.report.map { _meta, f -> f },
        SPOQC_HQTR_CELLTYPE.out.report.map { _meta, f -> f },
        ch_report_combine_masks,
        SPOQC_TRANSCRIPT.out.report.map { _meta, f -> f },
        SPOQC_CELLCYCLE.out.report.map { _meta, f -> f },
        SPOQC_MODEL.out.report.map { _meta, f -> f },
        SPOQC_ANALYSIS_OVERVIEW.out.report.map { _meta, f -> f },
        SPOQC_ANALYSIS_CATEGORY.out.report.map { _meta, f -> f },
        SPOQC_ANALYSIS_CLUSTER.out.report.map { _meta, f -> f },
    )

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Output
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    emit:

    ch_sd_raw       = ch_sd         // channel: [ val(meta), "spatialdata_raw" ]
}
