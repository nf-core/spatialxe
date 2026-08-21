# nf-core/spatialaxe: Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.0.1dev - [date]

Initial release of nf-core/spatialaxe, created with the [nf-core](https://nf-co.re/) template.

### `Added`

- Added the entire **spoQC tool and subworkflow**: `subworkflows/local/spoqc/main.nf` plus 31 new local modules under `modules/local/spoQC/` (`ambient`, `analysis_category`, `analysis_cluster`, `analysis_overview`, `annotation`, `bubble`, `cell`, `cellcycle`, `combine_masks`, `doublet`, `finalreport`, `general`, `hqcr_celltype`, `hqcr_ident`, `hqpr_bounding_box`, `hqpr_celltype`, `hqpr_clustering`, `hqpr_metrices`, `hqpr_refinement`, `hqtr_ac`, `hqtr_bounding_box`, `hqtr_celltype`, `hqtr_clustering`, `hqtr_metrices`, `hqtr_qv`, `hqtr_refinement`, `marker`, `model`, `transcript`, `void`, `whole_slide`), each with its own `main.nf`, `meta.yml`, and nf-test suite (`tests/main.nf.test` + snapshot).
- Wired spoQC into `workflows/spatialaxe.nf`.
- Added `bin/spatialdata_write.py` support for spoQC's SpatialData output.
- New test configs: `conf/tests/test_spoqc.config` and `conf/tests/test_full_spoqc.config`
- New samplesheet for a full test: `assets/samplesheet_full.csv` with `conf/test_full.config`.
- `conf/base.config`: adding combinatorial label system — separate `process_{tiny,low,mid,high,xl}_cpus`, `_mem`, and `_time` labels, each scaling with `task.attempt`.
- Added `nf-core/unzip` module usage so the pipeline can unpack the larger test dataset.
- samplesheet redefinition: `sample,bundle,image,annotation,stainings`, samplesheet allows for two additional optional columns `annotation,stainings` that are useful for the QC subworkflow.
- `spatialdata_write_meta_merge/main.nf`: Change to subworkflow to account for proper `qc` mode.
- Change to `bin/spatialdata_write.py`: Adding an `all` mode to set all available features to `True`, which is important for QC.
- **Image QC and transcript QC**, ported from the internal nf-xenium-processing repository. Adds the `QC` subworkflow (`subworkflows/local/qc/`) wrapping `IMAGE_QC` and `TRANSCRIPT_QC` (`subworkflows/local/image_qc/`, `subworkflows/local/transcript_qc/`), the `image_qc` and `transcript_qc` local modules, and their analysis scripts, report notebooks and threshold configs in `bin/`. Image QC computes focus, SNR and morphology metrics; transcript QC computes per-transcript, per-cell and per-FoV metrics with a bounded-memory streaming reader. Each renders an HTML report with the nf-core `quarto/notebook` module.
- 17 new QC parameters under the `qc_options` schema group, including `image_qc_gpus` (image QC is GPU-optional; the value is both the accelerator request and the device cap passed to the script), `tile_size`, `neg_control_prefix`, and figure/SNR/streaming toggles.
- `conf/base.config`: new `process_gpu_qc` label for the optionally-GPU image QC step, routed to the GPU queue in the `aws` profile.
- `.github/workflows/python-tests.yml` plus `bin/tests/`: the QC analysis scripts' Python unit tests, which were previously never executed in CI. Run in a runtime environment (`bin/tests/environment.yml`) because the scripts import their heavy dependencies at module scope.

### `Fixed`

- Change to input validation in `spatialaxe.nf`: Moving `morphology_focus/` folder into `bundle_optional_files` because not all Xenium bundles have such a folder.

### `Dependencies`

### `Deprecated`

- `nextflow.config`: bumped the `nf-schema` plugin version.
- `subworkflows/nf-core/utils_nfschema_plugin/main.nf`: added a new `cli_typecast` input (pass `null` to keep default behavior), renamed the `parametersSchema` option key to `parameters_schema` across the help/summary/validate option maps, and fixed how the `--help` text value is resolved.
- `subworkflows/nf-core/utils_nextflow_pipeline/main.nf`: bumped the version.
- `subworkflows/local/utils_nfcore_spatialaxe_pipeline/main.nf`: passes the new `cli_typecast` argument (`null`) through to `UTILS_NFSCHEMA_PLUGIN`.
- `nextflow_schema.json`: cleanup driven by the new schema version.

## 1.0.1 - [06.08.2026]

Hotfix to tackle some bugs

### `Added`

- Template update for nf-core/tools version 4.0.3
- Adding new conf/tests folder
- Adding new test for coordinate mode to check the bugfixes
- Remove default outdir='results' for test profiles (tests)

### `Fixed`

- Only pass `--expansion-distance` to `xeniumranger import-segmentation` for nuclei-based imports. It was applied to every import, but xeniumranger rejects it for transcript-assignment imports (proseg/baysor/segger) and cells-only imports with `ERROR: --expansion-distance requires --nuclei`.
- Preserve the URI scheme (e.g. `s3://`) when building Xenium bundle child paths, so bundle validation works when the work directory is on object storage (S3/GCS/Azure). Previously `Path.toString()` dropped the scheme and the resulting path was resolved on the local filesystem, causing `Xenium bundle does not exist` / `NoSuchFileException` failures on AWS Batch.

### `Dependencies`

### `Deprecated`

## 1.0.0 - [18.06.2026]

Initial release of nf-core/spatialaxe, created with the [nf-core](https://nf-co.re/) template.

### `Added`

### `Fixed`

### `Dependencies`

### `Deprecated`
