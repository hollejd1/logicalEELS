# Important Notes for Jetson Profiling and Testing

Use vscode devcontainers to develop and test the code on the Jetson. 
Use the `.devcontainer/requirements.txt` and `./devcontainer/install-dev-tools.sh`
to install the correct python libraries.

To open the project in the devcontainer, use vscode to remote connect into the jetson board.  This will prompt you for your jetson board's password. Once the project is open, use the following command:

```bash
# MacOS
cmd+shift+p -> Remote-Containers: Rebuild and Reopen in Devcontainer

# Other
ctrl+shift+p -> Remote-Containers: Rebuild and Reopen in Devcontainer
```

This will take a few minutes to download the docker image and prepare it as a development environment.

**Important**: Once in the container, install the requirements with the following command:

```bash
./.devcontainer/install-dev-tools.sh
```

The stack should provide cuda tools, and python libraries to run tensorflow and tensorrt models.

# Regenerate TRT Engine

TRT engine has to match specific versions of TensorRT and CUDA. If the versions are not the same, the engine will not work. To regenerate the engine, use the following command:

```bash
/usr/src/tensorrt/bin/trtexec  --onnx=EELS.onnx --saveEngine=trt.engine --fp16 --verbose
```

# Jetson Versions

```bash
# Display what is installed from external repos
apt-cache show nvidia-jetpack
```

```bash
# Display library versions
git clone https://github.com/jetsonhacks/jetsonUtilities.git
cd jetsonUtilities
python jetsonInfo.py
```

Current configs (inside the docker image):

```
JHU configs:
NVIDIA NVIDIA Jetson AGX Orin Developer Kit
L4T 36.4.0 [ JetPack UNKNOWN ]
   Ubuntu 22.04.5 LTS
   Kernel Version: 5.15.148-tegra
CUDA 12.6.68
   CUDA Architecture: 8.7
OpenCV version: 4.8.0
   OpenCV Cuda: NO
CUDNN: ii libcudnn9
TensorRT: 10.3.0.30
Vision Works: NOT_INSTALLED
VPI: 3.2.4
Vulcan: 1.3.204
```

```
PNNL-N Configs:
NVIDIA UNKNOWN
 L4T 36.3.0 [ JetPack UNKNOWN ]
   Ubuntu 22.04.3 LTS
   Kernel Version: 5.15.136-tegra
 CUDA 12.6.68
   CUDA Architecture: NONE
 OpenCV version: 4.8.0
   OpenCV Cuda: NO
 CUDNN: ii libcudnn9
 TensorRT: 10.3.0.30
 Vision Works: NOT_INSTALLED
 VPI: 3.2.4
 Vulcan: 
 ```

# Profile a Python File

Run:

```bash
nsys profile -t cuda python3 time_model.py
```

This will generate a file called `report1.qdrep` which can be opened with the NVIDIA Nsight Systems GUI.

Nsigth Systems GUI can be installed from: https://developer.nvidia.com/nsight-systems/get-started
