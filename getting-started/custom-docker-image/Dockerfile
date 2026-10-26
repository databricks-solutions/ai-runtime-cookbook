# The AWS AIR base supplies Python, uv, CUDA, and the platform networking libraries.
FROM databricksruntime/air:dcs-base-aws-runtime

# River is absent from the base image. Install it in the base Python environment.
# An optional BuildKit secret selects a package index without storing its URL
# in the image or build history.
RUN --mount=type=secret,id=uv_index_url,required=false \
    if [ -f /run/secrets/uv_index_url ]; then \
      UV_INDEX_URL="$(cat /run/secrets/uv_index_url)" uv pip install --no-cache river==0.22.0; \
    else \
      uv pip install --no-cache river==0.22.0; \
    fi

# AIR does not honor WORKDIR; workload.yaml runs this absolute path.
COPY train.py /app/train.py
