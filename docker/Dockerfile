FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# 基础工具 + zsh
RUN apt update && apt install -y \
    wget curl git zsh build-essential \
    htop vim tmux sudo locales \
    libgl1 libglib2.0-0 unzip \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -ya

# 安装 oh-my-zsh + 插件
RUN sh -c "$(wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)" "" --unattended && \
    git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting && \
    sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc && \
    chsh -s /bin/zsh

# 添加 conda 初始化到 zshrc
RUN echo '__conda_setup="$('\''/opt/conda/bin/conda'\'' '\''shell.bash'\'' '\''hook'\'' 2> /dev/null)"' >> ~/.zshrc && \
    echo 'if [ $? -eq 0 ]; then' >> ~/.zshrc && \
    echo '    eval "$__conda_setup"' >> ~/.zshrc && \
    echo 'else' >> ~/.zshrc && \
    echo '    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then' >> ~/.zshrc && \
    echo '        . "/opt/conda/etc/profile.d/conda.sh"' >> ~/.zshrc && \
    echo '    else' >> ~/.zshrc && \
    echo '        export PATH="/opt/conda/bin:$PATH"' >> ~/.zshrc && \
    echo '    fi' >> ~/.zshrc && \
    echo 'fi' >> ~/.zshrc && \
    echo 'unset __conda_setup' >> ~/.zshrc

# 设置工作目录
WORKDIR /workspace

SHELL ["/bin/bash", "-c"]
CMD ["zsh"]
