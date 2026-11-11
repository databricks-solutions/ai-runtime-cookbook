ARG BASE_IMAGE=databricksruntime/air:dcs-base-aws-runtime
FROM ${BASE_IMAGE}

# Optional package-index overrides for controlled build networks. Build args
# are available to RUN steps but are not persisted in the final image.
ARG UV_INDEX_URL=https://pypi.org/simple
ARG UV_INSECURE_HOST=""

ENV PYTHONUNBUFFERED=1 \
    XRV_CACHE_DIR=/opt/torchxrayvision

# Match the PyTorch build validated for the AIR base image. Keep torch and
# torchvision out of requirements.txt so the resolver cannot replace them.
RUN uv pip install --no-cache \
    torch==2.6.0 torchvision==0.21.0

COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --no-cache -r /tmp/requirements.txt

# Cache the public weights in the image so inference does not depend on a
# runtime download. AIR may run as a non-root user, so make the cache readable.
RUN /opt/venv/bin/python -c \
    "import os; import torchxrayvision as xrv; xrv.models.DenseNet(weights='densenet121-res224-all', cache_dir=os.environ['XRV_CACHE_DIR'])" \
    && chmod -R a+rX "$XRV_CACHE_DIR"
