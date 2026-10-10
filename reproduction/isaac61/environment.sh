#!/usr/bin/env bash
# Source from bash after activating RoboDojo's Python 3.12 environment.
ROBODOJO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_HOME="${ROBODOJO_CUDA_HOME:-/usr/local/cuda-12.8}"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX="$CXX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0
export CMAKE_CUDA_ARCHITECTURES=120
export MAX_JOBS=8
export CMAKE_BUILD_PARALLEL_LEVEL=8
export PYTHONNOUSERSITE=1
export TERM=xterm-256color
export OMNI_KIT_ACCEPT_EULA=YES
export ROBODOJO_KIT_ARGS="--/app/tokens/data=$ROBODOJO_ROOT/.cache/isaac61/data --/app/tokens/cache=$ROBODOJO_ROOT/.cache/isaac61/cache --/app/tokens/logs=$ROBODOJO_ROOT/.cache/isaac61/logs --/app/extensions/registryEnabled=false --/exts/isaacsim.sensors.camera/renderer=RayTracedLighting"
