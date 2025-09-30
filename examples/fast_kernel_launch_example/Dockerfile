FROM ubuntu:24.04 AS base

SHELL [ "/bin/bash", "-c" ]

# 安装系统依赖和Python 3.10
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        apt-transport-https \
        ca-certificates \
        bash \
        curl \
        vim \
        wget \
        git \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libncurses5-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libffi-dev \
        libnss3-dev \
        libgdbm-dev \
        liblzma-dev \
        libev-dev \
        gcc-10 \
        g++-10 \
        make \
        cmake \
        unzip \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3.10-distutils \
        python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/tmp/* \
    && rm -rf /tmp/*


# 检查并删除可能存在的其他高版本 GCC
RUN for version in 11 12 13 14 15; do \
        if [ -d "/usr/lib/gcc/x86_64-linux-gnu/$version" ] && [ "$version" != "10" ]; then \
            echo "Removing GCC $version" && \
            rm -rf "/usr/lib/gcc/x86_64-linux-gnu/$version"; \
        fi; \
    done

# 验证 GCC 版本
RUN echo "Current GCC versions:" && \
    ls -la /usr/lib/gcc/x86_64-linux-gnu/ && \
    echo "Default GCC version:" && \
    gcc --version | head -n1

# 先配置pip镜像源
RUN mkdir -p /root/.pip \
    && echo '[global]' > /root/.pip/pip.conf \
    && echo 'index-url = https://pypi.tuna.tsinghua.edu.cn/simple' >> /root/.pip/pip.conf \
    && echo 'trusted-host = pypi.tuna.tsinghua.edu.cn' >> /root/.pip/pip.conf \
    && echo 'timeout = 300' >> /root/.pip/pip.conf

# 设置Python 3.10为默认，不升级系统包
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# 设置gcc-10和g++-10为默认
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100

# 下载Ascend-cann-toolkit
RUN wget --quiet --header="Referer: https://www.hiascend.com/" \
    https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/Ascend-cann-toolkit_8.3.RC1_linux-x86_64.run \
    -O ~/Ascend-cann-toolkit.run

# 下载opp_legacy
RUN wget --quiet --header="Referer: https://www.hiascend.com/" \
    https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/cann-910b-opp_legacy-8.3.RC1-linux-x86_64.run \
    -O ~/cann-910b-opp_legacy.run

# 安装CANN Toolkit
RUN chmod +x ~/Ascend-cann-toolkit.run \
    && ~/Ascend-cann-toolkit.run --quiet --full \
    && rm -f ~/Ascend-cann-toolkit.run

# 安装CANN 910B OPP Legacy
RUN chmod +x ~/cann-910b-opp_legacy.run \
    && ~/cann-910b-opp_legacy.run --quiet --full \
    && rm -f ~/cann-910b-opp_legacy.run

# 安装PyTorch和torch_npu
RUN pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install torch_npu==2.6.0

# 设置 Ascend 环境变量
RUN if [ -f "/usr/local/Ascend/set_env.sh" ]; then \
        source /usr/local/Ascend/set_env.sh && \
        echo "Sourced /usr/local/Ascend/set_env.sh"; \
    else \
        echo "Warning: /usr/local/Ascend/set_env.sh not found"; \
    fi

# 设置 LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/Ascend/latest/lib64:/usr/local/Ascend/latest/x86_64-linux/devlib:${LD_LIBRARY_PATH}

ENTRYPOINT ["/bin/bash"]
