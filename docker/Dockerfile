ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"
ARG MMCV="1.5.0"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# add new sudo user
ARG GID=1000
ARG UID=1000
ENV USERNAME mmseg
ENV HOME /home/$USERNAME
RUN groupadd -f -g ${GID} ${USERNAME}
RUN useradd -m $USERNAME -u ${UID} -g ${GID} && \
    echo "$USERNAME:$USERNAME" | chpasswd && \
    usermod --shell /bin/bash $USERNAME && \
    usermod -aG sudo $USERNAME && \
    mkdir /etc/sudoers.d && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx sudo curl wget bash-completion \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

USER $USERNAME
WORKDIR /home/$USERNAME
SHELL ["/bin/bash", "-c"]

# Install MMCV
ARG PYTORCH
ARG CUDA
ARG MMCV
RUN pip install --no-cache-dir mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html

# Install MMSegmentation
ARG MMSEG_VERISON=master
RUN git clone https://github.com/open-mmlab/mmsegmentation.git /home/$USERNAME/mmsegmentation -b ${MMSEG_VERISON}
WORKDIR /home/$USERNAME/mmsegmentation
ENV FORCE_CUDA="1"
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -e .

# Install package for tools
RUN pip install seaborn scipy

# Set PATH
RUN echo "export PATH=$PATH:$HOME/.local/bin" >> ~/.bashrc
