# nf-core/spatialaxe: Output

## Introduction

This document describes the output produced by the pipeline.

The directories listed below will be created in the results directory after the pipeline has finished. All paths are relative to the top-level results directory.

## Pipeline overview

The pipeline is built using [Nextflow](https://www.nextflow.io/) and processes data using the following steps:

- Mode specific output:
  - [image mode](#image-mode)
  - [cooridnate mode](#coordinate-mode)
  - [segfree mode](#segfree-mode)
  - [qc mode](#qc-mode) (or using `--run_qc`)
  - [preview mode](#preview-mode)
- [Additional functionality of spatialaxe](#additional-functionality):
  - [SpatialData](#spatialdata)
  - [Xenium Ranger import segmentation](#xenium-ranger-import-segmentation)
  - [MultiQC](#multiqc) - Aggregate report describing results and QC from the whole pipeline
  - [Pipeline information](#pipeline-information) - Report metrics generated during the workflow execution

## Quality Control

[Quality Control](qc.md): An overview of the different QC reports.

## Image mode

<details markdown="1">
<summary>Output files</summary>

- `image/`
  - `xeniumranger/`
    - `resegment/`
      - `${meta.id}/` Directory containing the output xenium bundle of Xenium
  - `baysor/`
    - `preprocess/`
      - `*.csv` filtered transcripts CSV (for Baysor 0.7.1 Parquet.jl compatibility)
    - `run/`
      - `*segmentation.csv` results of segmentation
      - `*.json` file with outlines of segmentation
      - `segmentation_params.dump.toml` file with full list of parameters used for the model
      - `segmentation_log.log` output file with metadata of running the workflow
      - `segmentation_counts.loom` loom file with metadata
      - `segmentation_cell_stats.csv` statistics of segmented cells
  - `cellpose_cells/`
    - `*masks.tif` labelled mask output from cellpose in tif format
    - `*flows.tif` cell flow output from cellpose
    - `*seg.npy` numpy array with cell segmentation data
  - `stardist_nuclei/`
    - `*.{tiff,tif}` labelled mask output from stardist in tif format
  - `resolift/`
    - `*.tiff` path to save the upscaled TIFF file

</details>

## Coordinate mode

<details markdown="1">
<summary>Output files</summary>

- `coordinate/`
  - `xenium_patch/`
    - `patches/patch_grid.json` patch_grid.json metadata file
    - `patches/patch_*/transcripts.parquet` per-patch transcripts.parquet files (one per patch)
    - `output/xr-cell-polygons.geojson` stitched cell polygons
    - `output/xr-transcript-metadata.csv` transcript metadata
  - `proseg/`
    - `preset/`
      - `cell-polygons.geojson.gz` 2D polygons for each cell in GeoJSON format. These are flattened from 3D
      - `expected-counts.csv.gz` cell-by-gene count matrix
      - `cell-metadata.csv.gz` cell centroids, volume, and other information
      - `transcript-metadata.csv.gz` transcript ids, genes, revised positions, assignment probability
      - `gene-metadata.csv.gz` per-gene summary statistics
      - `rates.csv.gz` cell-by-gene Poisson rate parameters
      - `cell-polygons-layers.geojson.gz` a separate, non-overlapping cell polygon for each z-layer, preserving 3D segmentation
      - `cell-hulls.geojson.gz` convex hulls around assigned transcripts
    - `proseg2baysor/`
      - `xr-cell-polygons.geojson` 2D polygons for each cell in GeoJSON format. These are flattened from 3D
      - `xr-transcript-metadata.csv` transcript ids, genes, revised positions, assignment probability
  - `segger/`
    - `create_dataset/`
      - `${meta.id}/` directory to save the processed Segger dataset (in PyTorch Geometric format)
    - `train/`
      - `${meta.id}/` directory to save the trained model and checkpoints
    - `predict/`
      - `${meta.id}/` directory to save the segmentation results, including cell boundaries and associations
  - `baysor/`
    - `run/`
      - `*segmentation.csv` results of segmentation
      - `*.json` file with outlines of segmentation
      - `segmentation_params.dump.toml` file with full list of parameters used for the model
      - `segmentation_log.log` output file with metadata of running the workflow
      - `segmentation_counts.loom` loom file with metadata
      - `segmentation_cell_stats.csv` statistics of segmented cells

</details>

## Segfree mode

<details markdown="1">
<summary>Output files</summary>

- `segfree/`
  - `baysor/`
    - `preprocess/`
      - `*.csv` filtered transcripts CSV (for Baysor 0.7.1 Parquet.jl compatibility)
    - `segfree/`
      - `ncvs.loom` loom file with neighborhood results
      - `ncvs_segfree_log.log` Log file with summary statistics
  - `ficture/`
    - `preprocess/`
      - `processed_transcripts.tsv.gz` transcirpt file used for FICTURE
      - `coordinate_minmax.tsv` listing the min and max of the coordinates used for FICTURE
      - `feature.clean.tsv.gz` another file contains the (unique) names of genes that should be used for FICUTRE
    - `${meta.id}/results/` files containing the results of FICTURE

</details>

## QC mode

<details markdown="1">
<summary>Output files</summary>

- `opt/`
  - `flip/`
    - `*.fa` the forward oriented fasta file
  - `track/`
    - `*.tsv` TSV file containing the gene and transcript information to which each probe aligns
  - `stat/`
    - `*.tsv` TSV file containing the summary stats
- `spoqc/`
  - `report/`
    - `annotation/`
      - `unsupervised_cell_annotation.tsv` unsupervised cell-type annotation, generated only when no annotation file was supplied as input
    - `whole_slide_qc/` whole-slide QC overview report
    - `generalqc/` general-purpose QC report
    - `bubbleqc/` bubble-artifact QC report
    - `doubletqc/` doublet-detection QC report
    - `voidqc/` void/empty-region QC report
    - `cellqc/` cell-level QC report
    - `ambientqc/` ambient-RNA (background gene) QC report
    - `hqcr/`
      - `hqcr_ident/` high-quality cell region (HQCR) identification report
      - `hqcr_celltype/` cell-type-refined HQCR report
    - `hqpr/`
      - `hqpr_metrices/${staining}/` per-staining high-quality pixel region (HQPR) metrics report
      - `hqpr_clustering/${staining}/` per-staining HQPR clustering report
      - `hqpr_refinement/${staining}/` per-staining HQPR mask refinement report
      - `hqpr_bounding_box/${staining}/` per-staining HQPR bounding-box report
    - `hqtr/`
      - `hqtr_metrices/` high-quality transcript region (HQTR) metrics report
      - `hqtr_ac/` HQTR ambient-contamination probability report
      - `hqtr_qv/` HQTR quality-value probability report
      - `hqtr_clustering/` HQTR clustering report
      - `hqtr_refinement/` HQTR mask refinement report
      - `hqtr_bounding_box/` HQTR bounding-box report
    - `combine_masks/${staining}/` report combining the HQCR, HQPR and HQTR masks per staining
    - `transcriptqc/` transcript-level QC report (against the reference gene panel)
    - `cellcycleqc/` cell-cycle scoring QC report
    - `modelqc/` model-based QC scoring report
    - `analysis/`
      - `overview/` downstream QC analysis overview report
      - `rna_qc_annotated.h5ad` AnnData object annotated with the combined QC results
      - `category/` downstream per-category QC analysis report
      - `cluster/` downstream clustering QC analysis report
      - `rna_cluster.h5ad` AnnData object with clustering results
    - `staining_log.txt` log of the stainings processed by spoQC
    - `report.html` final self-contained HTML report aggregating every spoQC step above
  - `spoQC_tmp/` intermediate data consumed by later spoQC steps
    - `generalqc_output_hqcr.parquet`, `bubbleqc_output_hqcr.parquet`, `doubletqc_output_hqcr.parquet`, `voidqc_output_hqcr.parquet`, `cellqc_output_hqcr.parquet` per-step HQCR contribution scores
    - `ambient_output_genes.parquet` ambient RNA gene-signal estimate
    - `hqcr_output_mask_raw.parquet` / `hqcr_output_mask_smoothed_raw.parquet` combined raw/smoothed HQCR mask
    - `hqcr_output_mask_smoothed_celltype_refined.parquet` cell-type-refined smoothed HQCR mask
    - `metrices/hqpr/${staining}/` per-staining HQPR metrics
    - `hqpr_${staining}_output_mask_raw/` / `hqpr_${staining}_output_mask_smoothed_raw/` per-staining raw/smoothed HQPR mask
    - `metrices/hqtr/` HQTR metrics
    - `hqtr_output_ac_prob/` HQTR ambient-contamination probabilities
    - `hqtr_output_qv_prob/` HQTR quality-value probabilities
    - `hqtr_output_mask_raw/` / `hqtr_output_mask_smoothed_raw/` raw/smoothed HQTR mask
- `multiqc/`
  - `multiqc_report.html`: a standalone HTML file that can be viewed in your web browser.
  - `multiqc_data/`: directory containing parsed statistics from the different tools used in the pipeline.
  - `multiqc_plots/`: directory containing static images from the report in various formats.

</details>

## Preview mode

<details markdown="1">
<summary>Output files</summary>

- `preview/`
  - `baysor/`
    - `preview/`
      - `preview.html` segmentation preview

</details>

## Additional Functionality

### SpatialData

The pipeline create spatialdata objects (data bundles) on various stages (see metromap in the [README](../README.md))

<details markdown="1">
<summary>Output files</summary>

- `spatialdata/`
  - `write/${meta.id}/spatialdata/` spatialdata bundle of the raw data
  - `meta/${meta.id}/spatialdata_spatialaxe_final/` spatialdata bundle of the final data with metadata
    - `sdata['raw_table'].uns['spatialdata_attrs']` provenance metadata
    - `sdata['raw_table'].uns['experiment_xenium']` experimental metadata
    - `sdata['raw_table'].uns['gene_panel']` gene panel metadata

</details>

### Xenium Ranger Import Segmentation

This step is needed to import segemntations from different methods into the xenium bundle and is called at different stages of the pipeline.

<details markdown="1">
<summary>Output files</summary>

- `xeniumranger/`
  - `import_segementation/`
    - `${meta.id}/` directory containing the output xenium bundle of Xenium

</details>

### MultiQC

<details markdown="1">
<summary>Output files</summary>

- `multiqc/`
  - `multiqc_report.html`: a standalone HTML file that can be viewed in your web browser.
  - `multiqc_data/`: directory containing parsed statistics from the different tools used in the pipeline.
  - `multiqc_plots/`: directory containing static images from the report in various formats.

</details>

[MultiQC](http://multiqc.info) is a visualization tool that generates a single HTML report summarising all samples in your project. Most of the pipeline QC results are visualised in the report and further statistics are available in the report data directory.

The pipeline has special steps which also allow the software versions to be reported in the MultiQC output for future traceability. For more information about how to use MultiQC reports, see <http://multiqc.info>.

### Pipeline information

<details markdown="1">
<summary>Output files</summary>

- `pipeline_info/`
  - Reports generated by Nextflow: `execution_report.html`, `execution_timeline.html`, `execution_trace.txt` and `pipeline_dag.dot`/`pipeline_dag.svg`.
  - Reports generated by the pipeline: `pipeline_report.html`, `pipeline_report.txt` and `software_versions.yml`. The `pipeline_report*` files will only be present if the `--email` / `--email_on_fail` parameter's are used when running the pipeline.
  - Reformatted samplesheet files used as input to the pipeline: `samplesheet.valid.csv`.
  - Parameters used by the pipeline run: `params.json`.

</details>

[Nextflow](https://www.nextflow.io/docs/latest/tracing.html) provides excellent functionality for generating various reports relevant to the running and execution of the pipeline. This will allow you to troubleshoot errors with the running of the pipeline, and also provide you with other information such as launch commands, run times and resource usage.
