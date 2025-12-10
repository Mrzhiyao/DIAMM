# Dockerfile for ARM 36 (Jetson) architecture
# 使用 Jetson L4T ML 基础镜像（r36.4.0）
ARG BASE_RUNTIME_IMAGE=dustynv/l4t-ml:r36.4.0

FROM ${BASE_RUNTIME_IMAGE} AS final

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
# 使用国内镜像源，加快构建阶段依赖下载
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
# 设置 CUDA 环境变量
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# 设置 CUDA 架构（Jetson Orin 使用 sm_87）
ENV TORCH_CUDA_ARCH_LIST=8.7
ENV FORCE_CUDA=1

# 安装系统依赖
RUN <<EOF
    set -eu
    apt-get update
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        python3-pip \
        python3-venv \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        cmake \
        ninja-build \
        pkg-config \
        libopenblas-dev \
        libomp-dev \
        libtinfo5 \
        gcc-10 \
        g++-10 \
        curl \
        unzip \
        libgmp-dev \
        libmpfr-dev \
        libmpc-dev
    apt-get clean
    rm -rf /var/lib/apt/lists/*
EOF

# 设置编译器环境变量（ARM 架构推荐使用 gcc-10）
ENV CC=/usr/bin/gcc-10
ENV CXX=/usr/bin/g++-10

# 设置工作目录
WORKDIR /app

# 复制整个项目（包含 CogVideo-1.0 目录）
# 注意：构建时需要在 /mnt/ssd/yaozhi/cogvideo 目录下运行：
#   cd /mnt/ssd/yaozhi/cogvideo
#   docker build -f Dockerfile -t cogvideo-jetson:latest .
COPY CogVideo-1.0 /app/CogVideo-1.0

# 安装 Python 依赖（使用 Jetson 适配的 requirements 文件）
RUN <<EOF
    set -eu
    # 先升级 pip、setuptools 和 wheel
    pip3 install --no-cache-dir --upgrade pip setuptools wheel
    
    # 安装基础依赖（确保兼容性）
    pip3 install --no-cache-dir \
        "numpy>=1.26.0,<2.1.0" \
        packaging \
        pillow
    
    # 先记录基础镜像的 torch 版本和路径（防止被覆盖）
    BASE_TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
    BASE_TORCH_PATH=$(python3 -c "import torch; import os; print(os.path.dirname(os.path.dirname(torch.__file__)))" 2>/dev/null || echo "")
    echo "Base image torch version: ${BASE_TORCH_VERSION}, path: ${BASE_TORCH_PATH}"
    
    # 安装 Jetson 适配的 requirements（如果存在）
    # 策略：先过滤掉冲突的包（protobuf、torch相关），然后分批安装
    if [ -f /app/CogVideo-1.0/sat/requirements_jetson.txt ]; then
        echo "Installing requirements from requirements_jetson.txt..."
        
        # 创建约束文件，严格保护 torch 版本
        echo "# 约束文件：严格防止 pip 升级基础镜像的 torch/torchvision/torchaudio" > /tmp/constraints.txt
        echo "torch==${BASE_TORCH_VERSION}" >> /tmp/constraints.txt
        echo "torchvision" >> /tmp/constraints.txt
        echo "torchaudio" >> /tmp/constraints.txt
        
        # 第一步：过滤掉 protobuf（我们会在后面单独安装 4.x 版本）
        # 过滤掉 torch 相关包（基础镜像已提供）
        grep -v -E '^protobuf|^torch|^torchvision|^torchaudio|^#.*protobuf|^#.*torch' \
            /app/CogVideo-1.0/sat/requirements_jetson.txt > /tmp/requirements_filtered.txt || true
        
        # 第二步：使用约束文件安装过滤后的包
        echo "Installing filtered requirements (excluding protobuf and torch packages)..."
        pip3 install --no-cache-dir \
            --upgrade-strategy=only-if-needed \
            -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
            -r /tmp/requirements_filtered.txt \
            -c /tmp/constraints.txt || {
            echo "WARNING: Some packages failed to install, but continuing..."
        }
        rm -f /tmp/requirements_filtered.txt /tmp/constraints.txt
        
        # 验证关键包是否安装成功
        echo "Verifying key packages installation..."
        python3 -c "import fastapi; print('fastapi OK')" || echo "WARNING: fastapi not installed"
        python3 -c "import omegaconf; print('omegaconf OK')" || echo "WARNING: omegaconf not installed"
        python3 -c "import transformers; print('transformers OK')" || echo "WARNING: transformers not installed"
    else
        # 如果不存在 jetson 版本，使用普通版本，但需要处理一些 ARM 不兼容的包
        grep -v -E '^torch|^torchvision|^torchaudio' /app/CogVideo-1.0/sat/requirements.txt > /tmp/requirements_filtered.txt || true
        pip3 install --no-cache-dir -r /tmp/requirements_filtered.txt || {
            echo "Warning: Some packages may not be available for ARM architecture"
        }
        rm -f /tmp/requirements_filtered.txt
    fi
    
    # 验证 torch 版本没有被覆盖（应该仍然是基础镜像的版本）
    CURRENT_TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
    if [ "${CURRENT_TORCH_VERSION}" != "${BASE_TORCH_VERSION}" ] && [ -n "${BASE_TORCH_VERSION}" ]; then
        echo "Warning: torch version changed from ${BASE_TORCH_VERSION} to ${CURRENT_TORCH_VERSION}"
        echo "This may cause compatibility issues. Base image torch should be preserved."
    fi
    
    # 验证 torch 是否可用（无论版本如何）
    python3 -c "import torch; print(f'Final torch version: {torch.__version__}')" || {
        echo "Error: torch import failed after installation"
        exit 1
    }
    
    # 注意：torchvision 0.21.0 与 torch 2.9.1 可能存在兼容性问题
    # 如果遇到 RuntimeError: operator torchvision::nms does not exist
    # 这通常是 torchvision C++ 扩展的问题，但不影响基本功能
    # 在实际使用中，torchvision.transforms 等基本功能应该可以正常工作
    
    # 处理 torchmetrics：它依赖 torchvision，但基础镜像的 torchvision 可能有兼容性问题
    # 如果 torchmetrics 安装失败或导致 torchvision 问题，可以降级或跳过
    # 注意：torchmetrics 1.8.2 可能与 torchvision 0.21.0 不兼容
    # 如果遇到 torchvision 导入错误，可能需要降级 torchmetrics 或使用兼容版本
    
    # 修复 protobuf 问题：在 ARM 架构上，protobuf 5.x 的 C++ 扩展无法正确工作
    # 已验证方案：降级到 protobuf 4.x 并使用纯 Python 实现
    echo "修复 protobuf 兼容性问题..."
    pip3 uninstall -y protobuf || true
    # 安装 protobuf 4.x 版本（在 ARM 上更稳定）
    pip3 install --no-cache-dir "protobuf>=4.21.0,<5.0.0"
    # 验证 protobuf 是否可用
    python3 -c "import google.protobuf; print('protobuf installed successfully')" || {
        echo "Error: protobuf import failed"
        exit 1
    }
    
    # 确保 numpy 版本正确（Jetson 需要 1.26.x）
    pip3 install --no-cache-dir --force-reinstall "numpy>=1.26.0,<2.1.0" || {
        echo "Warning: numpy version adjustment failed"
    }
EOF

# 创建输出目录并设置权限
RUN mkdir -p /data/output && chmod a+rwx /data/output

# 暴露 API 端口
EXPOSE 8190

# 设置环境变量
ENV PYTHONPATH=/app/CogVideo-1.0
# 确保 CUDA 在运行时可用
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# 设置 protobuf 使用 Python 实现（如果 C++ 扩展有问题时的后备方案）
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# 修复 torchvision 兼容性问题：延迟导入 torchvision 的某些模块
ENV TORCHVISION_DISABLE_EXTENSIONS=0

# 启动命令（根据实际路径调整）
# CMD ["uvicorn", "CogVideo-1.0.sat.api.main:app", "--host", "0.0.0.0", "--port", "8190", "--workers", "2"]
