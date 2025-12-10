# 基础镜像
FROM dustynv/vllm:0.9.2-r36.4-cu128-24.04

# 设置构建时可选代理参数
ARG HTTP_PROXY
ARG HTTPS_PROXY

# 设置代理环境变量（国内网络必须使用代理）
ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

# 切换到 bash
SHELL ["/bin/bash", "-c"]

# 安装 LMDeploy (源码安装)
RUN pip install git+https://github.com/InternLM/lmdeploy.git \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu128 \
    --extra-index-url https://mirrors.ustc.edu.cn/pypi/simple/

# 安装视觉语言模型所需的依赖
RUN pip install qwen_vl_utils \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu128 \
    --extra-index-url https://mirrors.ustc.edu.cn/pypi/simple/

# 安装图像处理相关依赖（如果还没有）
RUN pip install Pillow opencv-python-headless \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu128 \
    --extra-index-url https://mirrors.ustc.edu.cn/pypi/simple/

CMD ["tail", "-f", "/dev/null"]
