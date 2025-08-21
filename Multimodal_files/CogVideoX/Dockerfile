# Dockerfile
# 使用官方 CUDA 基础镜像（根据实际 CUDA 版本调整）
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
# 配置网络参数

# 安装系统依赖并配置 Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libbz2-dev \
    liblzma-dev\
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 验证 CUDA 工具链
RUN nvcc --version  # 应该显示 CUDA 11.8
RUN which nvcc      # 应该输出 /usr/local/cuda/bin/nvcc

# 设置工作目录
WORKDIR /app

# 复制整个项目（包含 CogVideo-1.0 目录）
COPY CogVideo-1.0 /app/CogVideo-1.0

#COPY /app/CogVideo-1.0/sat/requirements.txt /tmp/requirements.txt

# 设置环境变量
ENV PATH="/usr/local/cuda/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# 编译安装 Python 3.11.11
RUN wget https://www.python.org/ftp/python/3.11.11/Python-3.11.11.tgz && \
    tar -xzf Python-3.11.11.tgz && \
    cd Python-3.11.11 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf Python-3.11.11*

# 设置 Python 3.11 为默认版本
RUN update-alternatives --install /usr/local/bin/python3 python3 /usr/local/bin/python3.11 1 && \
    update-alternatives --install /usr/local/bin/python python /usr/local/bin/python3.11 1 && \
    update-alternatives --install /usr/local/bin/pip3 pip3 /usr/local/bin/pip3.11 1



# 安装 Python 依赖（假设 requirements.txt 在项目根目录）
#RUN pip3 install --no-cache-dir -r /app/CogVideo-1.0/sat/requirements.txt \
#    && pip3 install --no-cache-dir 

# 安装 Python 依赖（修复后的命令）
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip3 install --upgrade pip==24.0 && \  
    pip3 install --no-cache-dir -r /app/CogVideo-1.0/sat/requirements.txt 

# 验证安装
RUN . venv/bin/activate && \
    python3 -c "import sys; print(sys.path)" && \
    pip3 check


# 创建输出目录并设置权限
RUN mkdir -p /data/output && chmod a+rwx /data/output

# 暴露 API 端口
EXPOSE 8190

# 设置环境变量
ENV PYTHONPATH=/app/CogVideo-1.0
#
## 启动命令（根据实际路径调整）
#CMD ["uvicorn", "CogVideo-10.sat.api.main:app", "--host", "0.0.0.0", "--port", "8190", "--workers", "2"]

