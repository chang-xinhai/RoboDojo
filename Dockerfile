# syntax=docker/dockerfile:1
# RoboDojo simulator / evaluation image (Ubuntu 22.04 + CUDA 13.0).
#
# Scope: this image contains ONLY the RoboDojo simulation-evaluation side
# (Isaac Sim, IsaacLab, CuRobo, the RoboDojo Python stack) plus the lightweight
# XPolicyLab client_server + policy deploy adapters used by src/eval_client.
# It deliberately does NOT install any policy-specific environment, dependency
# set, or checkpoints (e.g. GR00T_N17). The policy server runs OUTSIDE this
# container and is reached over TCP (see docker/README.md).
#
# Build:
#   See docker/README.md for the required host-derived CUROBO_VERSION build arg.
#   This target has not been built or GPU-validated for the native migration.
#
# Run (Linux host, policy server already listening on the host):
#   docker run --rm -it --gpus all --network host --ipc host \
#     -v "$PWD/Assets:/workspace/RoboDojo/Assets:ro" \
#     -v "$PWD/eval_result:/workspace/RoboDojo/eval_result" \
#     robodojo:cuda13.0 \
#     bash scripts/robodojo.sh client --task stack_bowls \
#       --policy-name GR00T_N17 --policy-host 127.0.0.1 --policy-port 9999 --eval-num 1

ARG CUDA_IMAGE=nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04@sha256:36c66a3ad4608580cf937f7ec3add9323a610956cfe8c9e2f99ef2ea0c896f01
FROM ${CUDA_IMAGE}

# Exact setuptools_scm version exported from the clean, pinned host cuRobo source.
# No fallback: inventing a version would hide the source packaging identity.
ARG CUROBO_VERSION
RUN test -n "${CUROBO_VERSION}" || { echo "Pass --build-arg CUROBO_VERSION from the source checkout (docker/README.md)." >&2; exit 1; }

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/root/miniconda3/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    OMNI_KIT_ACCEPT_EULA=YES \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    TERM=xterm-256color \
    PIP_USER=0 \
    PYTHONNOUSERSITE=1

# ── Optional China mirrors (build args; defaults = official upstreams) ─────────
# The image is unchanged for normal builds. On a China network, enable mirrors:
#   docker build \
#     --build-arg UBUNTU_MIRROR=mirrors.tuna.tsinghua.edu.cn \
#     --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
#     --build-arg MINICONDA_URL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     --build-arg CONDA_CHANNEL_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/anaconda \
#     --build-arg CUROBO_VERSION="$CUROBO_VERSION" -t robodojo:cuda13.0 .
# (docker/smoke_docker.sh sets all of these when ROBODOJO_CN_MIRRORS=1.)
ARG UBUNTU_MIRROR=""
ARG PIP_INDEX_URL="https://pypi.org/simple"
ARG MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
ARG CONDA_CHANNEL_MIRROR=""
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

# ── System dependencies ──────────────────────────────────────────────────────
# setup_system() from scripts/install.sh (cmake, build-essential, ffmpeg) plus
# the headless OpenGL/EGL/Vulkan runtime libraries Isaac Sim needs, and
# netcat for the policy-server connectivity check documented in docker/README.md.
RUN if [ -n "${UBUNTU_MIRROR}" ]; then \
        sed -i "s@http://archive.ubuntu.com/ubuntu@http://${UBUNTU_MIRROR}/ubuntu@g; s@http://security.ubuntu.com/ubuntu@http://${UBUNTU_MIRROR}/ubuntu@g" /etc/apt/sources.list 2>/dev/null || true; \
        rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia-ml.list 2>/dev/null || true; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        git \
        git-lfs \
        wget \
        curl \
        ca-certificates \
        ffmpeg \
        netcat-openbsd \
        libgl1 \
        libglu1-mesa \
        libglib2.0-0 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libsm6 \
        libice6 \
        libxrandr2 \
        libxi6 \
        libxcursor1 \
        libxinerama1 \
        libegl1 \
        libvulkan1 \
        vulkan-tools \
        mesa-utils \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ── Headless EGL / Vulkan ICD (NVIDIA) ───────────────────────────────────────
# Point the GL/Vulkan loaders at the NVIDIA driver that the NVIDIA Container
# Toolkit injects at runtime, so Isaac Sim can render cameras headless.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF
RUN mkdir -p /usr/share/vulkan/icd.d && \
    cat > /usr/share/vulkan/icd.d/nvidia_icd.json <<'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.3.242"
    }
}
EOF

WORKDIR /workspace/RoboDojo

# ── Miniconda + RoboDojo env ─────────────────────────────────────────────────
# setup_conda() from scripts/install.sh: Miniconda under $HOME/miniconda3 and a
# Python 3.12 env named `RoboDojo` (the simulator conda env used by all scripts).
RUN wget -q "${MINICONDA_URL}" -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /root/miniconda3 && \
    rm -f /tmp/miniconda.sh && \
    conda config --set always_yes yes && \
    if [ -n "${CONDA_CHANNEL_MIRROR}" ]; then \
        conda config --add default_channels "${CONDA_CHANNEL_MIRROR}/pkgs/main" && \
        conda config --add default_channels "${CONDA_CHANNEL_MIRROR}/pkgs/r" && \
        conda config --set show_channel_urls yes && \
        conda config --set channel_alias "${CONDA_CHANNEL_MIRROR}/cloud"; \
    fi && \
    (conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true) && \
    (conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true) && \
    conda create -n RoboDojo python=3.12 -y && \
    conda clean -afy

# Run every subsequent build step inside the RoboDojo env.
SHELL ["conda", "run", "--no-capture-output", "-n", "RoboDojo", "/bin/bash", "-c"]

# ── Native simulator dependencies (same installer as the host) ────────────────
# .dockerignore excludes all .git files/directories before copying. Source
# version inference for cuRobo uses the verified host value only during install;
# the final runtime reads its installed distribution metadata.
COPY scripts/install.sh scripts/requirements.txt /workspace/RoboDojo/scripts/
COPY third_party/ /workspace/RoboDojo/third_party/
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO="${CUROBO_VERSION}" \
    bash scripts/install.sh --from isaaclab

# ── RoboDojo source + XPolicyLab client/adapter code ─────────────────────────
# Explicit copies (not `COPY .`) so we do not clobber the compiled artifacts
# already built under third_party/. Heavy policy source, weights, checkpoints,
# assets, and runtime outputs are excluded via .dockerignore.
COPY env/ /workspace/RoboDojo/env/
COPY env_cfg/ /workspace/RoboDojo/env_cfg/
COPY task/ /workspace/RoboDojo/task/
COPY src/ /workspace/RoboDojo/src/
COPY utils/ /workspace/RoboDojo/utils/
COPY scripts/ /workspace/RoboDojo/scripts/
COPY XPolicyLab/ /workspace/RoboDojo/XPolicyLab/
COPY pyproject.toml README.md LICENSE /workspace/RoboDojo/

# ── Headless RTX render deps + single Vulkan ICD (late, cheap layers) ─────────
SHELL ["/bin/bash", "-c"]
# libXt.so.6 is required by Isaac Sim's MaterialX render libs
# (libusd_usdBakeMtlx.so, libMaterialXRender*.so). Its dependencies libsm6/libice6
# are installed above. Container loading/rendering still requires validation.
RUN apt-get update && apt-get install -y --no-install-recommends libxt6 \
    && rm -rf /var/lib/apt/lists/*
# Force a SINGLE Vulkan ICD. This image bakes /usr/share/vulkan/icd.d/nvidia_icd.json
# AND the NVIDIA Container Toolkit injects /etc/vulkan/icd.d/nvidia_icd.json at run
# time; the loader then sees two ICDs for the same GPU ("multiple installable client
# drivers") and RTX device init wedges on the first rendered frame (tiled cameras
# hang right after buffer allocation). Pinning the toolkit-injected, driver-matched
# ICD avoids that duplicate-loader path. This retained setup is unvalidated on
# the new target. --gpus with
# NVIDIA_DRIVER_CAPABILITIES=all guarantees that path exists at run time.
ENV VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json

# ── Entrypoint ───────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV PYTHONPATH=/workspace/RoboDojo

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
