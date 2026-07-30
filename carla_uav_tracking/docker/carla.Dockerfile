# All-in-one CARLA server + Python client for ACoT-UAV-Track data generation.
#
# The base image already ships the simulator (/home/carla/CarlaUE4.sh) and the
# PythonAPI. On top of it we add a Python 3.8 client toolchain (the pip `carla`
# wheel has no cp36 build, and the base image's system python is 3.6) plus the
# repo's runtime deps, so a single container can BOTH run CarlaUE4 servers
# (one per GPU via scripts/launch_parallel.py) and run the data-generation client.
FROM carlasim/carla:0.9.15

# ---------------------------------------------------------------------------
# System + Python client toolchain (needs root)
# ---------------------------------------------------------------------------
USER root

# deadsnakes gives us python3.8 on the base image's Ubuntu; the extra libs cover
# CARLA's offscreen Vulkan rendering and OpenCV(headless) at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-distutils python3.8-dev \
        libvulkan1 \
        libgl1 libglib2.0-0 libgomp1 \
        libpng16-16 libjpeg-turbo8 \
    && curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8 \
    && ln -sf /usr/bin/python3.8 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Python client deps (carla pin MUST match the server version above)
COPY docker/requirements.txt /tmp/requirements.txt
RUN python3.8 -m pip install --no-cache-dir -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Remap the image's `carla` user to the host uid/gid so bind-mounted repo files
# and generated data are owned by the host user, not root.
# ---------------------------------------------------------------------------
ARG HOST_UID=1003
ARG HOST_GID=1015
RUN (groupmod -o -g ${HOST_GID} carla || groupadd -o -g ${HOST_GID} carla) \
    && usermod -o -u ${HOST_UID} -g ${HOST_GID} carla \
    && mkdir -p /workspace /data \
    && chown -R ${HOST_UID}:${HOST_GID} /home/carla /workspace /data

# CARLA_ROOT lets scripts/launch_parallel.py find CarlaUE4.sh in this image.
ENV CARLA_ROOT=/home/carla \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

WORKDIR /workspace
USER carla
