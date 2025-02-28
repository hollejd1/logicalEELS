FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0 as base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git pip \
    libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Do NOT modify UID from 1000
# vscode updates this UID to the user building the container
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt update \
    && apt install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Cleanup
# RUN pip3 cache purge
RUN apt autoremove -y
RUN apt clean

USER $USERNAME
ENV SHELL=/bin/bash

# Allow container to stop. See: https://github.com/microsoft/vscode-remote-release/issues/3512#issuecomment-1267053890
# NOTE: May affect benchmarking
# NOTE: Long interval is necessary so container is not closed during vscode initialization
CMD while sleep 120 && ps aux | egrep 'vscode.*[b]ootstrap-fork.*extensionHost.*' > /dev/null; do :; done