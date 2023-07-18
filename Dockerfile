# Created by: George Corrêa de Araújo (george.gcac@gmail.com)

FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

# used to make generated files belong to actual user
ARG GROUPID=901
ARG GROUPNAME=deeplearning
ARG USERID=901
ARG USERNAME=dl

# Environment variables
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="pip --no-cache-dir install --upgrade" && \
    DOWNLOAD_FILE="curl -LO" && \

# ==================================================================
# Create a system group with name deeplearning and id 901 to avoid
#    conflict with existing uids on the host system
# Create a system user with id 901 that belongs to group deeplearning
# When building the image, these values will be replaced by actual
#    user info
# ------------------------------------------------------------------

    groupadd -r $GROUPNAME -g $GROUPID && \
    useradd -u $USERID -m -g $GROUPNAME $USERNAME && \

# ==================================================================
# install libraries via apt-get
# ------------------------------------------------------------------

    rm -rf /var/lib/apt/lists/* && \
    # temporary solution for bug
    # see https://forums.developer.nvidia.com/t/gpg-error-http-developer-download-nvidia-com-compute-cuda-repos-ubuntu1804-x86-64/212904/3
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-get update && \

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        bc \
        curl \
        git \
        libffi-dev \
        rsync \
        wget && \

# ==================================================================
# install python libraries via pip
# ------------------------------------------------------------------

    $PIP_INSTALL \
        pip \
        setuptools \
        wheel && \
    $PIP_INSTALL \
        comet-ml \
        datasets \
        ipdb \
        ipython \
        kornia \
        matplotlib \
        numpy \
        pillow \
        piq \
        prettytable \
        pytorch-lightning \
        "pytorch-lightning[extra]" \
        tensorboard \
        torch_optimizer \
        tqdm && \

# ==================================================================
# install python libraries via git
# ------------------------------------------------------------------

    $PIP_INSTALL git+https://github.com/jonbarron/robust_loss_pytorch && \

# ==================================================================
# send telegram message
# ------------------------------------------------------------------

    $PIP_INSTALL \
        # https://github.com/rahiel/telegram-send/issues/115#issuecomment-1368728425
        python-telegram-bot==13.5 \
        telegram-send && \

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    /opt/conda/bin/conda clean -ya && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

USER $USERNAME

# Expose TensorBoard ports
EXPOSE 6006 7000
