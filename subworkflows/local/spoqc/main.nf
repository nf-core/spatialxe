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
    ch_stainings            // channel: [ val(meta), val(staining) ] - one item per sample x its own staining

    main:

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // General
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // Keep only actual, usable annotations (non-null and not an empty list).
    ch_annotation_present = ch_annotation_src.filter { _meta, a -> a && (!(a instanceof List) || !a.isEmpty()) }

    // Pair each sample's spatialdata bundle with its own annotation by meta.id.
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

    // Shared join of ch_sd + ch_annotation_path by meta.id, split back into
    // the two positional args each downstream module expects.
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

    // Combine(by: 0) scopes the cross product to matching meta, so each sample only
    // ever gets crossed with its own declared stainings.
    ch_spatialdata_stainings = ch_sd.combine(ch_stainings, by: 0)

    SPOQC_HQPR_METRICES(
        ch_spatialdata_stainings,
        "hqpr_metrices",
    )

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

    // Same meta.id-keyed pairing as ch_sd_annotation, extended with staining.
    // combine() is applied after the join so both outputs stay aligned per sample+staining.
    ch_sd
        .join(ch_annotation_path, by: 0, remainder: true)
        .map { meta, spatialdata, annotation -> [meta, spatialdata, annotation ?: []] }
        .combine(ch_stainings, by: 0)
        .multiMap { meta, spatialdata, annotation, staining ->
            sd:         [meta, spatialdata, staining]
            annotation: [annotation, staining]
        }
        .set { ch_sd_annotation_stainings }

    // Join on the composite [meta, staining] key.
    ch_masks_joined = SPOQC_HQPR_CLUSTERING.out.mask
        .map { meta, staining, f -> [[meta, staining], f] }
        .join(
            SPOQC_HQPR_REFINEMENT.out.mask_smoothed.map { meta, staining, f -> [[meta, staining], f] }
        )
        .map { key, m1, m2 -> [key[1], m1, m2] }

    SPOQC_HQPR_CELLTYPE(
        ch_sd_annotation_stainings.sd,
        ch_sd_annotation_stainings.annotation,
        "hqpr_celltype",
        SPOQC_GENERAL.out.tmp.combine(ch_stainings, by: 0).map { _meta, f, staining -> [f, staining] },
        SPOQC_BUBBLE.out.tmp.combine(ch_stainings, by: 0).map { _meta, f, staining -> [f, staining] },
        SPOQC_DOUBLET.out.tmp.combine(ch_stainings, by: 0).map { _meta, f, staining -> [f, staining] },
        SPOQC_VOID.out.tmp.combine(ch_stainings, by: 0).map { _meta, f, staining -> [f, staining] },
        SPOQC_CELL.out.tmp.combine(ch_stainings, by: 0).map { _meta, f, staining -> [f, staining] },
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
        SPOQC_HQCR_IDENT.out.mask.combine(ch_stainings, by: 0).map { _meta, f, staining -> [f, staining] },
        ch_spoqc_hpq_clustering,
        SPOQC_HQTR_CLUSTERING.out.mask.combine(ch_stainings, by: 0).map { _meta, f, staining -> [f, staining] },
        SPOQC_HQCR_IDENT.out.mask_smoothed.combine(ch_stainings, by: 0).map { _meta, f, staining -> [f, staining] },
        ch_spoqc_hqpr_refinement,
        SPOQC_HQTR_REFINEMENT.out.mask_smoothed.combine(ch_stainings, by: 0).map { _meta, f, staining -> [f, staining] },
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

    // Group each sample's per-staining HQPR files by meta, so a sample's ANALYSIS_* task only 
    // ever receives its own HQPR files.
    ch_files_hqpr_metrics = SPOQC_HQPR_METRICES.out.metrices
        .map { meta, _staining, p -> [meta, p] }
        .groupTuple(by: 0)

    ch_files_hqpr_masks = SPOQC_HQPR_CLUSTERING.out.mask
        .map { meta, _staining, p -> [meta, p] }
        .groupTuple(by: 0)

    ch_files_hqpr_masks_smoothed = SPOQC_HQPR_REFINEMENT.out.mask_smoothed
        .map { meta, _staining, p -> [meta, p] }
        .groupTuple(by: 0)

    ch_sd
        .join(ch_annotation_path, by: 0, remainder: true)
        .join(SPOQC_GENERAL.out.tmp, by: 0)
        .join(SPOQC_BUBBLE.out.tmp, by: 0)
        .join(SPOQC_DOUBLET.out.tmp, by: 0)
        .join(SPOQC_VOID.out.tmp, by: 0)
        .join(SPOQC_CELL.out.tmp, by: 0)
        .join(SPOQC_HQCR_IDENT.out.mask, by: 0)
        .join(SPOQC_HQCR_IDENT.out.mask_smoothed, by: 0)
        .join(SPOQC_HQTR_QV.out.tmp, by: 0)
        .join(SPOQC_HQTR_AC.out.tmp, by: 0)
        .join(SPOQC_HQTR_METRICES.out.metrices, by: 0)
        .join(SPOQC_HQTR_REFINEMENT.out.mask_smoothed, by: 0)
        .join(SPOQC_HQTR_CLUSTERING.out.mask, by: 0)
        .join(ch_files_hqpr_metrics, by: 0, remainder: true)
        .join(ch_files_hqpr_masks_smoothed, by: 0, remainder: true)
        .join(ch_files_hqpr_masks, by: 0, remainder: true)
        .multiMap { meta, spatialdata, annotation,
                    general, bubble, doublet, void_qc, cell,
                    hqcr_mask, hqcr_mask_smoothed,
                    hqtr_qv, hqtr_ac, hqtr_metrices, hqtr_mask_smoothed, hqtr_mask,
                    hqpr_metrics, hqpr_masks_smoothed, hqpr_masks ->
            sd:                  [meta, spatialdata]
            annotation:          annotation ?: []
            general_tmp:         general
            bubble_tmp:          bubble
            doublet_tmp:         doublet
            void_tmp:            void_qc
            cell_tmp:            cell
            hqcr_mask:           hqcr_mask
            hqcr_mask_smoothed:  hqcr_mask_smoothed
            hqtr_qv:             hqtr_qv
            hqtr_ac:             hqtr_ac
            hqtr_metrices:       hqtr_metrices
            hqtr_mask_smoothed:  hqtr_mask_smoothed
            hqtr_mask:           hqtr_mask
            hqpr_metrics:        hqpr_metrics ?: []
            hqpr_masks_smoothed: hqpr_masks_smoothed ?: []
            hqpr_masks:          hqpr_masks ?: []
        }
        .set { ch_analysis_inputs }

    SPOQC_ANALYSIS_OVERVIEW(
        ch_analysis_inputs.sd,
        ch_analysis_inputs.annotation,
        "analysis_overview",
        ch_analysis_inputs.general_tmp,
        ch_analysis_inputs.bubble_tmp,
        ch_analysis_inputs.doublet_tmp,
        ch_analysis_inputs.void_tmp,
        ch_analysis_inputs.cell_tmp,
        ch_analysis_inputs.hqcr_mask,
        ch_analysis_inputs.hqcr_mask_smoothed,
        ch_analysis_inputs.hqtr_qv,
        ch_analysis_inputs.hqtr_ac,
        ch_analysis_inputs.hqtr_metrices,
        ch_analysis_inputs.hqtr_mask_smoothed,
        ch_analysis_inputs.hqtr_mask,
        ch_analysis_inputs.hqpr_metrics,
        ch_analysis_inputs.hqpr_masks_smoothed,
        ch_analysis_inputs.hqpr_masks,
    )

    SPOQC_ANALYSIS_CATEGORY(
        ch_analysis_inputs.sd,
        ch_analysis_inputs.annotation,
        "analysis_category",
        ch_analysis_inputs.general_tmp,
        ch_analysis_inputs.bubble_tmp,
        ch_analysis_inputs.doublet_tmp,
        ch_analysis_inputs.void_tmp,
        ch_analysis_inputs.cell_tmp,
        ch_analysis_inputs.hqcr_mask,
        ch_analysis_inputs.hqcr_mask_smoothed,
        ch_analysis_inputs.hqtr_qv,
        ch_analysis_inputs.hqtr_ac,
        ch_analysis_inputs.hqtr_metrices,
        ch_analysis_inputs.hqtr_mask_smoothed,
        ch_analysis_inputs.hqtr_mask,
        ch_analysis_inputs.hqpr_metrics,
        ch_analysis_inputs.hqpr_masks_smoothed,
        ch_analysis_inputs.hqpr_masks,
    )

    SPOQC_ANALYSIS_CLUSTER(
        ch_analysis_inputs.sd,
        ch_analysis_inputs.annotation,
        "analysis_cluster",
        ch_analysis_inputs.general_tmp,
        ch_analysis_inputs.bubble_tmp,
        ch_analysis_inputs.doublet_tmp,
        ch_analysis_inputs.void_tmp,
        ch_analysis_inputs.cell_tmp,
        ch_analysis_inputs.hqcr_mask,
        ch_analysis_inputs.hqcr_mask_smoothed,
        ch_analysis_inputs.hqtr_qv,
        ch_analysis_inputs.hqtr_ac,
        ch_analysis_inputs.hqtr_metrices,
        ch_analysis_inputs.hqtr_mask_smoothed,
        ch_analysis_inputs.hqtr_mask,
        ch_analysis_inputs.hqpr_metrics,
        ch_analysis_inputs.hqpr_masks_smoothed,
        ch_analysis_inputs.hqpr_masks,
    )

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Final Report
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // Group the per-staining report fragments by meta, so a
    // sample's final report only ever bundles its own report fragments.
    ch_report_hqpr_metrices = SPOQC_HQPR_METRICES.out.report
        .map { meta, _staining, p -> [meta, p] }
        .groupTuple(by: 0)
    ch_report_hqpr_clustering = SPOQC_HQPR_CLUSTERING.out.report
        .map { meta, _staining, p -> [meta, p] }
        .groupTuple(by: 0)
    ch_report_hqpr_refinement = SPOQC_HQPR_REFINEMENT.out.report
        .map { meta, _staining, p -> [meta, p] }
        .groupTuple(by: 0)
    ch_report_hqpr_bounding_box = SPOQC_HQPR_BOUNDING_BOX.out.report
        .map { meta, _staining, p -> [meta, p] }
        .groupTuple(by: 0)
    ch_report_combine_masks = SPOQC_COMBINE_MASKS.out.report
        .groupTuple(by: 0)

    ch_sd
        .join(SPOQC_GENERAL.out.report, by: 0)
        .join(SPOQC_DOUBLET.out.report, by: 0)
        .join(SPOQC_VOID.out.report, by: 0)
        .join(SPOQC_CELL.out.report, by: 0)
        .join(SPOQC_HQCR_IDENT.out.report, by: 0)
        .join(SPOQC_HQCR_CELLTYPE.out.report, by: 0)
        .join(ch_report_hqpr_metrices, by: 0, remainder: true)
        .join(ch_report_hqpr_clustering, by: 0, remainder: true)
        .join(ch_report_hqpr_refinement, by: 0, remainder: true)
        .join(ch_report_hqpr_bounding_box, by: 0, remainder: true)
        .join(SPOQC_HQPR_CELLTYPE.out.report, by: 0)
        .join(SPOQC_HQTR_METRICES.out.report, by: 0)
        .join(SPOQC_HQTR_AC.out.report, by: 0)
        .join(SPOQC_HQTR_QV.out.report, by: 0)
        .join(SPOQC_HQTR_CLUSTERING.out.report, by: 0)
        .join(SPOQC_HQTR_REFINEMENT.out.report, by: 0)
        .join(SPOQC_HQTR_BOUNDING_BOX.out.report, by: 0)
        .join(SPOQC_HQTR_CELLTYPE.out.report, by: 0)
        .join(ch_report_combine_masks, by: 0, remainder: true)
        .join(SPOQC_TRANSCRIPT.out.report, by: 0)
        .join(SPOQC_CELLCYCLE.out.report, by: 0)
        .join(SPOQC_MODEL.out.report, by: 0)
        .join(SPOQC_ANALYSIS_OVERVIEW.out.report, by: 0)
        .join(SPOQC_ANALYSIS_CATEGORY.out.report, by: 0)
        .join(SPOQC_ANALYSIS_CLUSTER.out.report, by: 0)
        .multiMap {
            meta, spatialdata,
            report_general, report_doublet, report_void, report_cell,
            report_hqcr_ident, report_hqcr_celltype,
            report_hqpr_metrices, report_hqpr_clustering, report_hqpr_refinement, report_hqpr_bounding_box,
            report_hqpr_celltype,
            report_hqtr_metrices, report_hqtr_ac, report_hqtr_qv, report_hqtr_clustering,
            report_hqtr_refinement, report_hqtr_bounding_box, report_hqtr_celltype,
            report_combine_masks, report_transcript, report_cellcycle, report_model,
            report_analysis_overview, report_analysis_category, report_analysis_cluster ->
            sd:                       [meta, spatialdata]
            report_general:           report_general
            report_doublet:           report_doublet
            report_void:              report_void
            report_cell:              report_cell
            report_hqcr_ident:        report_hqcr_ident
            report_hqcr_celltype:     report_hqcr_celltype
            report_hqpr_metrices:     report_hqpr_metrices ?: []
            report_hqpr_clustering:   report_hqpr_clustering ?: []
            report_hqpr_refinement:   report_hqpr_refinement ?: []
            report_hqpr_bounding_box: report_hqpr_bounding_box ?: []
            report_hqpr_celltype:     report_hqpr_celltype
            report_hqtr_metrices:     report_hqtr_metrices
            report_hqtr_ac:           report_hqtr_ac
            report_hqtr_qv:           report_hqtr_qv
            report_hqtr_clustering:   report_hqtr_clustering
            report_hqtr_refinement:   report_hqtr_refinement
            report_hqtr_bounding_box: report_hqtr_bounding_box
            report_hqtr_celltype:     report_hqtr_celltype
            report_combine_masks:     report_combine_masks ?: []
            report_transcript:        report_transcript
            report_cellcycle:         report_cellcycle
            report_model:             report_model
            report_analysis_overview: report_analysis_overview
            report_analysis_category: report_analysis_category
            report_analysis_cluster:  report_analysis_cluster
        }
        .set { ch_finalreport_inputs }

    SPOQC_FINALREPORT(
        ch_finalreport_inputs.sd,
        "final_report",
        ch_finalreport_inputs.report_general,
        ch_finalreport_inputs.report_doublet,
        ch_finalreport_inputs.report_void,
        ch_finalreport_inputs.report_cell,
        ch_finalreport_inputs.report_hqcr_ident,
        ch_finalreport_inputs.report_hqcr_celltype,
        ch_finalreport_inputs.report_hqpr_metrices,
        ch_finalreport_inputs.report_hqpr_clustering,
        ch_finalreport_inputs.report_hqpr_refinement,
        ch_finalreport_inputs.report_hqpr_bounding_box,
        ch_finalreport_inputs.report_hqpr_celltype,
        ch_finalreport_inputs.report_hqtr_metrices,
        ch_finalreport_inputs.report_hqtr_ac,
        ch_finalreport_inputs.report_hqtr_qv,
        ch_finalreport_inputs.report_hqtr_clustering,
        ch_finalreport_inputs.report_hqtr_refinement,
        ch_finalreport_inputs.report_hqtr_bounding_box,
        ch_finalreport_inputs.report_hqtr_celltype,
        ch_finalreport_inputs.report_combine_masks,
        ch_finalreport_inputs.report_transcript,
        ch_finalreport_inputs.report_cellcycle,
        ch_finalreport_inputs.report_model,
        ch_finalreport_inputs.report_analysis_overview,
        ch_finalreport_inputs.report_analysis_category,
        ch_finalreport_inputs.report_analysis_cluster,
    )

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Output
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    emit:

    report       = SPOQC_FINALREPORT.out.report
}
