#!/bin/bash

# NOTE: Because of the issue in related to stopping the container, see: 
# https://github.com/microsoft/vscode-remote-release/issues/3512#issuecomment-1267053890
# The following installation commands have to be executed within the timeframe
# set in the CMD command in the dockerfile.

# Uncomment the sections you need and add your own commands
# These are dependencies of tensorflow
# Note that cuda-pytorch image used already installs many of these tools

# # Update system
# apt-get update
# apt-get upgrade -y

# # Install Linux tools and Python 3
# apt-get install software-properties-common git wget curl \
#     python3-dev python3-pip python3-wheel python3-setuptools -y

# Install Python packages
python3 -m pip install --upgrade pip
pip3 install --user -r .devcontainer/requirements.txt

# Install recommended packages
# Some of these support x11 forwarding on linux
# apt-get install zlib1g g++ freeglut3-dev \
#     libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev -y

# Clean up - for smaller image size
# pip3 cache purge # not needed since cache is already disabled
# sudo apt-get autoremove -y
# sudo apt-get clean
