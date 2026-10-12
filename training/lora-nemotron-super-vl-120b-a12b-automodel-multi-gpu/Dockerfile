FROM nvcr.io/nvidia/nemo-automodel:26.08.00

# Upstream commit introducing native Nemotron 3.5 Super VL support.
ARG AUTOMODEL_COMMIT=6ae5ca23895d1e06c81cc6103732a20510833fda
LABEL org.opencontainers.image.revision=$AUTOMODEL_COMMIT

RUN cd /opt/Automodel \
    && git fetch --depth=1 https://github.com/NVIDIA-NeMo/Automodel.git "$AUTOMODEL_COMMIT" \
    && git checkout --force --detach FETCH_HEAD

# Install pinned Python and media dependencies from the upstream lockfile.
# PyTorch/torchvision/torchao/Triton are supplied by the NGC image.
# CUDA extras (TE, DeepEP, Mamba, grouped GEMM) are also retained from the image.
# --no-deps avoids re-resolving those inherited packages or replacing their builds.
ENV UV_HTTP_TIMEOUT=300

RUN cd /opt/Automodel \
    && uv export --quiet --frozen --no-default-groups --extra vlm --extra vlm-media \
         --no-emit-project --prune torch --prune torchvision --prune torchao \
         --prune triton --prune megatron-fsdp \
         --output-file /tmp/vlm-requirements.txt \
    && uv pip install --no-config --no-deps --python /opt/venv/bin/python --require-hashes \
         --requirements /tmp/vlm-requirements.txt \
    && uv pip install --no-config --no-deps --no-build-isolation \
         --python /opt/venv/bin/python --editable /opt/Automodel \
    && rm /tmp/vlm-requirements.txt
