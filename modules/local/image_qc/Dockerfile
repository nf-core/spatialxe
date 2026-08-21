# Container for the image QC module (analysis + Quarto report).
# Built from environment.yml in this directory. Tagged and pushed as
# quay.io/dongzehe/image_qc:1.0.0 (to be migrated to the nf-core org for release).
#
#   docker build -t quay.io/<namespace>/image_qc:1.0.0 modules/local/image_qc
#
# Note: behind a corporate proxy / private conda mirror you may need to provide a
# CA bundle and channel config at build time (e.g. via BuildKit --mount=type=secret);
# a standard build resolves the pinned packages directly from conda-forge/bioconda.
FROM mambaorg/micromamba:2.0.5
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean -a -f -y \
    && rm -f /tmp/environment.yml
ENV PATH="/opt/conda/bin:$PATH" \
    QUARTO_DENO=/opt/conda/bin/deno \
    QUARTO_DENO_DOM=/opt/conda/lib/deno_dom.so \
    QUARTO_PANDOC=/opt/conda/bin/pandoc \
    QUARTO_ESBUILD=/opt/conda/bin/esbuild \
    QUARTO_TYPST=/opt/conda/bin/typst \
    QUARTO_DART_SASS=/opt/conda/bin/sass \
    QUARTO_SHARE_PATH=/opt/conda/share/quarto \
    QUARTO_CONDA_PREFIX=/opt/conda \
    ES2_LIBRARY=/opt/conda/lib/libGLESv2.so.2
