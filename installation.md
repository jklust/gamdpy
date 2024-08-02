# Installation

## Linux

To install `rumdpy` you need a computer 
with a Nvidia GPU, and the following software installed:

1. the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), and
2. the [numba](https://numba.pydata.org/numba-doc/latest/user/installing.html) python package with CUDA GPU support (install `cudatoolkit`).

Ensure that this is working before proceeding with the installation of `rumdpy`.

### Using pip

Install the package from the GitLab repository:

```sh
pip install git+https://gitlab.com/tbs.cph/rumdpy-dev
```

### From source (for developers)

If you want to inspect or modify the source code, the package can installed by cloning the source code 
from GitLab to a local directory (change `[some_directory]` to the desired path):

```sh
cd [some_directory]
git clone https://gitlab.com/tbs.cph/rumdpy-dev.git/  # Clone latest developers version
cd rumdpy-dev
python3 -m venv venv  # Create virtual enviroment
. venv/bin/activate   # ... and activate
pip install -e .      # Install rumdpy 
```

Update to latest version by executing

```sh
git pull
```

in the `rumdpy-dev` directory.

## Windows (using WSL)

The following show how to install `rumdpy` 
on windows using Windows Subsystem For Linux (WSL).

### Install WSL
Open PowerShell or Windows Command Prompt in administrator mode by right-clicking and selecting "Run as administrator", enter the command

```sh
wsl --install
```

press enter and then restart your machine. 
The default installation is Ubuntu, for others check: <https://learn.microsoft.com/en-us/windows/wsl/install>

### Install python and pip on WSL

- open Windows Command Prompt
- in the tab bar click on "v" and select ubuntu
```sh 
sudo apt-get update
sudo apt-get install python3.10
sudo apt-get install pip
```

### Install miniconda 

See <https://docs.anaconda.com/miniconda/>

```sh
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

### Install cuda

```sh
miniconda3/condabin/conda install cudatoolkit
sudo apt install nvidia-cuda-toolkit
```

- modify `.bashrc` adding: `export LD_LIBRARY_PATH="/usr/lib/wsl/lib/"` from <https://github.com/numba/numba/issues/7104>


### Install rumdpy

```sh
pip install git+https://gitlab.com/tbs.cph/rumdpy-dev.git
```

## Windows (using Anaconda)

### Install Anaconda

Install Anaconda: <https://docs.anaconda.com/anaconda/install/windows/>

### Install rumdpy 

Finally, we install `rumdpy` (and `pip`) using Powershell Prompt in Anaconda

- open Anaconda Powershell as admin (from search)

```sh
conda update -n base -c defaults conda
conda install anaconda::pip
conda install anaconda::git
conda config --set channel_priority flexible
conda install cudatoolkit
pip install git+https://gitlab.com/tbs.cph/rumdpy-dev.git
```
