# syntax=docker/dockerfile:1.12

ARG BASE_RUNTIME_IMAGE=dustynv/l4t-ml:r36.4.0
ARG SD_WEBUI_URL=https://github.com/AUTOMATIC1111/stable-diffusion-webui
# v1.10.0
ARG SD_WEBUI_VERSION=c19d04436496ab29ddca4758a792831ae41b31de
# CUDA 架构列表：当前设备是 Jetson Orin (sm_87)，只编译 8.7 可以大幅减少编译时间
# 如果需要兼容其他设备，可以设置为 "7.2;8.7"（Jetson Xavier + Orin）
ARG CUDA_ARCH_LIST=8.7
# 是否跳过 xformers 编译（跳过后使用 SDP attention，编译快但性能略低）
ARG SKIP_XFORMERS=0

FROM ${BASE_RUNTIME_IMAGE} AS final

ARG DEBIAN_FRONTEND=noninteractive
ARG SD_WEBUI_URL
ARG SD_WEBUI_VERSION
ARG CUDA_ARCH_LIST
ARG SKIP_XFORMERS
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
# 使用国内镜像源，加快构建阶段依赖下载
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
# 为编译 xformers 提供 CUDA 路径（r36.4.0 通常是 /usr/local/cuda）
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PATH=/home/user/.local/bin:${PATH}

RUN <<EOF
    set -eu

    apt-get update

    apt-get install -y --no-install-recommends \
        build-essential \
        gosu \
        git \
        google-perftools \
        python3-dev \
        python3-venv \
        python3-pip \
        cmake \
        ninja-build \
        pkg-config \
        libopenblas-dev \
        libomp-dev \
        libtinfo5 \
        gcc-10 \
        g++-10 \
        curl \
        unzip

    apt-get clean
    rm -rf /var/lib/apt/lists/*
EOF

RUN <<'EOF'
    set -eu

    cat <<'PY' >/opt/torch_noop.py
def main():
    print("Skip installing torch/torchvision; base image already provides Jetson builds.")

if __name__ == "__main__":
    main()
PY

    chmod 644 /opt/torch_noop.py
EOF

RUN <<EOF
    set -eu

    if ! getent group user > /dev/null; then
        groupadd -o -g 1000 user
    fi

    if ! id -u user >/dev/null 2>&1; then
        useradd -m -o -u 1000 -g user user
    fi

    mkdir -p /code
    chown -R user:user /code
EOF

RUN <<EOF
    set -eu

    gosu user git clone --depth=1 "${SD_WEBUI_URL}" /code/stable-diffusion-webui
    cd /code/stable-diffusion-webui
    gosu user git fetch origin "${SD_WEBUI_VERSION}"
    gosu user git checkout "${SD_WEBUI_VERSION}"
EOF

# 修改 webui 启动脚本，跳过运行时再次安装 requirements 和 xformers（构建阶段已预装）
RUN <<'EOF'
    set -eu

    python3 - <<'PY'
from pathlib import Path

p = Path("/code/stable-diffusion-webui/modules/launch_utils.py")
text = p.read_text()
# 跳过 requirements 安装
req_needle = '    run_pip(f"install -r \\"{requirements_file}\\"", "requirements")'
req_replacement = '    print("Skip installing requirements (already handled in Docker build).")'
if req_needle in text:
    text = text.replace(req_needle, req_replacement)

# 跳过 xformers 安装
xfm_needle = '    run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")'
xfm_replacement = '    print("Skip installing xformers (already built into Docker image).")'
if xfm_needle in text:
    text = text.replace(xfm_needle, xfm_replacement)

p.write_text(text)
PY
EOF

WORKDIR /code/stable-diffusion-webui

ENV PYTHONPATH=/opt
ENV TORCH_COMMAND="torch_noop"
ENV TORCHVISION_COMMAND="torch_noop"
ENV REINSTALL_TORCH=0
ENV CC=/usr/bin/gcc-10
ENV CXX=/usr/bin/g++-10
ENV TORCH_CUDA_ARCH_LIST="7.2"
ENV FORCE_CUDA=1
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libtcmalloc.so.4

RUN <<EOF
    set -eu

    gosu user python3 -m venv --system-site-packages venv
    gosu user venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel
EOF

RUN <<EOF
    set -eu

    # 安装基础依赖
    gosu user venv/bin/pip install --no-cache-dir \
        numpy \
        packaging \
        pillow \
        pytorch-lightning
EOF

RUN <<EOF
    set -eu

    # 替换 webui 的 PC 版锁死依赖，使其兼容 Jetson (aarch64 + Python 3.10)
    # 保持 numpy 1.26.x（不降级到 1.24.4），因为需要 dtypes 属性且兼容 torch
    if [ -f requirements_versions.txt ]; then
        sed -i '/^onnxruntime-gpu==/d' requirements_versions.txt || true
    fi
    if [ -f requirements.txt ]; then
        sed -i '/^onnxruntime-gpu==/d' requirements.txt || true
    fi

    # 安装调整后的 requirements（包含 gradio/omegaconf/safetensors/kornia/einops/transformers/diskcache/GitPython 等）
    gosu user venv/bin/pip install --no-cache-dir -r requirements_versions.txt || true
    
    # 强制安装 numpy 1.26.4（最后一个支持 dtypes 的 1.x 版本，且兼容 torch）
    # 这确保 venv 使用自己的 numpy，而不是系统的 numpy 2.2.6
    gosu user venv/bin/pip install --no-cache-dir --force-reinstall "numpy==1.26.4" || {
        echo "Warning: numpy 1.26.4 installation failed, trying 1.26.2"
        gosu user venv/bin/pip install --no-cache-dir --force-reinstall "numpy==1.26.2" || {
            echo "Warning: numpy 1.26.2 also failed, keeping existing version"
        }
    }
    
    # 安装 triton（用于 xformers 的额外优化，支持 aarch64）
    echo "安装 triton 以启用 xformers 的额外优化..."
    gosu user venv/bin/pip install --no-cache-dir triton || {
        echo "Warning: triton installation failed, xformers will work but with fewer optimizations"
    }
EOF

RUN <<EOF
    set -eu

    # 根据 SKIP_XFORMERS 参数决定是否编译 xformers
    if [ "${SKIP_XFORMERS}" = "1" ]; then
        echo "跳过 xformers 编译，将使用 SDP attention"
        exit 0
    fi

    # 编译 xformers（确保构建环境能看到 torch）
    # 将 CUDA_ARCH_LIST 传递给 bash -lc（使用环境变量方式）
    export CUDA_ARCH_LIST="${CUDA_ARCH_LIST}"
    gosu user bash -lc '
        set -eu
        
        # 验证 venv 能看到 torch（简化验证，避免引号冲突）
        /code/stable-diffusion-webui/venv/bin/python3 -c "import torch" || {
            echo "ERROR: torch not found in venv, cannot build xformers"
            exit 1
        }
        
        cd /tmp
        rm -rf xformers
        git clone --depth=1 --branch v0.0.23.post1 --recursive https://github.com/facebookresearch/xformers.git
        cd xformers
        git submodule update --init --recursive
        
        # 设置 CUDA 架构列表（使用构建参数，避免编译时间过长）
        # 检查设备架构：nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        # sm_72 (7.2): Jetson Xavier, sm_87 (8.7): Jetson AGX Orin/Orin NX
        # 当前设备是 sm_87，所以默认只编译 8.7，大幅减少编译时间
        export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST"
        export FORCE_CUDA="1"
        
        # 限制并行编译任务数，避免内存耗尽（根据系统内存调整，Jetson 设备建议 1-2）
        export MAX_JOBS=1
        
        # 使用 --no-build-isolation 让构建环境能看到 venv 里的 torch
        CUDA_HOME=/usr/local/cuda \
          /code/stable-diffusion-webui/venv/bin/pip install --no-cache-dir --no-build-isolation -r requirements.txt
        CUDA_HOME=/usr/local/cuda \
          /code/stable-diffusion-webui/venv/bin/pip install --no-cache-dir --no-build-isolation -v .
    '

    rm -rf /tmp/xformers
EOF

RUN <<EOF
    set -eu

    gosu user ./webui.sh \
        --exit \
        --api \
        --skip-torch-cuda-test
EOF

RUN <<EOF
    set -eu

    mkdir -p /data
    chown -R user:user /data

    # 创建 extensions 目录（如果不存在）
    mkdir -p /data/extensions
    chown -R user:user /data/extensions

    mkdir -p /home/user/.cache/huggingface
    chown -R user:user /home/user/.cache

    mkdir -p /code/stable-diffusion-webui/log
    chown -R user:user /code/stable-diffusion-webui/log

    # 创建符号链接指向 /data/extensions
    rm -rf /code/stable-diffusion-webui/extensions
    ln -s /data/extensions /code/stable-diffusion-webui/extensions
    chown -h user:user /code/stable-diffusion-webui/extensions || true
EOF

RUN <<EOF
    set -eu

    cat > /entrypoint.sh <<'SCRIPT'
#!/bin/bash
set -eu
cd /code/stable-diffusion-webui

# 确保 venv 的包优先于系统包（避免系统 numpy 2.2.6 干扰）
export PYTHONPATH=/code/stable-diffusion-webui/venv/lib/python3.10/site-packages:${PYTHONPATH:-}

# xformers CUDA 兼容性设置
# 如果 CUDA kernels 不可用，让 xformers 回退到更兼容的实现
export XFORMERS_DISABLE_FLASH_ATTENTION=0
export XFORMERS_DISABLE_CUTLASS=0
# 设置 CUDA 架构（如果 xformers 需要）
export TORCH_CUDA_ARCH_LIST="8.7"

# 验证 venv 中的 numpy 版本（应该是 1.26.x）
NUMPY_VER=$(/code/stable-diffusion-webui/venv/bin/python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "unknown")
if [[ ! "$NUMPY_VER" =~ ^1\.26\. ]]; then
    echo "Warning: venv numpy version is $NUMPY_VER, expected 1.26.x. Attempting to fix..."
    /code/stable-diffusion-webui/venv/bin/pip install --no-cache-dir --force-reinstall "numpy==1.26.4" || true
fi

# 确保所有必要的目录存在且有正确权限
# stable-diffusion-webui 需要的目录列表
REQUIRED_DIRS="/data/extensions /data/models /data/models/hypernetworks /data/models/Stable-diffusion /data/models/VAE /data/models/Lora /data/models/ControlNet /data/models/ESRGAN /data/models/embeddings /data/outputs /data/config"

if [ "$(id -u)" = "0" ]; then
    # 以 root 运行：先创建目录并设置权限，再切换用户
    # 确保整个 /data 目录及其所有子目录都有正确的权限
    mkdir -p /data
    chown -R user:user /data 2>/dev/null || true
    chmod -R u+rwX /data 2>/dev/null || true
    
    # 创建必要的子目录
    for dir in $REQUIRED_DIRS; do
        mkdir -p "$dir"
        chown -R user:user "$dir" 2>/dev/null || true
        chmod -R u+rwX "$dir" 2>/dev/null || true
    done
    exec gosu user ./webui.sh "$@"
else
    # 以非 root 运行：直接创建目录
    mkdir -p /data 2>/dev/null || true
    for dir in $REQUIRED_DIRS; do
        mkdir -p "$dir" 2>/dev/null || true
    done
    exec ./webui.sh "$@"
fi
SCRIPT

    chmod +x /entrypoint.sh
EOF

ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "--data-dir", "/data", "--xformers", "--listen", "--server-name", "0.0.0.0", "--port", "7860", "--api", "--enable-insecure-extension-access" ]

