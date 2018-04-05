#!/bin/bash
set -e


# Install conda
# --------------------------------------------------
# v4.4 is the current conda version, but 4.3 is what currently ships with
# miniconda, and v4.4 is easier to install. We update before installing.

echo "==> Installing Conda"

CONDA_URL="https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
CONDA_PREFIX="/opt/conda"

# Download conda
echo "--> Downloading conda"
curl $CONDA_URL > miniconda.sh
chmod +x miniconda.sh

# Perform a forced (-f) batch (-b) install to $CONDA_PREFIX (-p)
echo "--> Installing conda"
mkdir -p /opt/conda
./miniconda.sh -f -b -p $CONDA_PREFIX
source "$CONDA_PREFIX/bin/activate"

# Update to at least 4.4
echo "--> Updating conda"
conda update --all --quiet --yes

# Enable conda for all users, requires conda 4.4
echo "--> Adding conda to the global profile"
ln -sf "$CONDA_PREFIX/etc/profile.d/conda.sh" /etc/profile.d/conda.sh
echo "conda activate" >> /etc/profile.d/conda.sh

# Make Spark use conda's python.
echo "--> Configuring Spark to use conda"
echo "export PYTHONHASHSEED=0" >> /etc/profile.d/conda.sh
echo "spark.executorEnv.PYTHONHASHSEED=0" >> /etc/spark/conf/spark-defaults.conf
echo "export PYSPARK_PYTHON=$CONDA_PREFIX/bin/python" | tee -a \
    /etc/profile.d/conda.sh \
    /etc/environment \
    /usr/lib/spark/conf/spark-env.sh


# Install packages and updates
# --------------------------------------------------

echo "==> Updating the operating system"
export DEBIAN_FRONTEND=noninteractive
apt-get -y update
apt-get -y upgrade

echo "==> Installing conda packages"
conda config --set always_yes true
conda config --set changeps1 false
conda install -q numpy scipy opencv keras


# Install GPU drivers
# --------------------------------------------------
# Beta feature of GCP: https://cloud.google.com/dataproc/docs/concepts/compute/gpus
# GPU Pricing: https://cloud.google.com/compute/pricing#gpus

echo "==> Configuring the GPU"

# Detect NVIDIA GPU
apt-get install -y pciutils
if (lspci | grep -q NVIDIA)
then
	echo "--> Detected NVIDIA GPU"

	# Unload the open source nouveau drivers.
	echo "--> Unloading the nouveau drivers"
	modprobe -r nouveau

	# Add non-free Debian 8 Jessie backports packages.
	# See https://www.debian.org/distrib/packages#note
	echo "--> Enabling non-free Debian 8 Jessie backports"
	sed 's/main/contrib/p;s/contrib/non-free/' \
	  /etc/apt/sources.list.d/backports.list \
	  > /etc/apt/sources.list.d/backports-non-free.list
	apt-get update

	# Install proprietary NVIDIA Drivers and CUDA.
	# See https://wiki.debian.org/NvidiaGraphicsDrivers
	# Without --no-install-recommends this takes a very long time.
	echo "--> Installing NVIDIA drivers and CUDA"
	export DEBIAN_FRONTEND=noninteractive
	apt-get install -y linux-headers-$(uname -r)
	apt-get install -y -t jessie-backports --no-install-recommends \
	  nvidia-cuda-toolkit nvidia-kernel-common nvidia-driver nvidia-smi

	# Create a system wide NVBLAS config.
	# See http://docs.nvidia.com/cuda/nvblas/
	echo "--> Configuring NVBLAS"
	export NVBLAS_CONFIG_FILE=/etc/nvidia/nvblas.conf
	echo "NVBLAS_CONFIG_FILE=${NVBLAS_CONFIG_FILE}" >> /etc/environment

	echo "# Insert here the CPU BLAS fallback library of your choice."                >> ${NVBLAS_CONFIG_FILE}
	echo "# The standard libblas.so.3 defaults to OpenBLAS, which does not have the"  >> ${NVBLAS_CONFIG_FILE}
	echo "# requisite CBLAS API."                                                     >> ${NVBLAS_CONFIG_FILE}
	echo "NVBLAS_CPU_BLAS_LIB /usr/lib/libblas/libblas.so"                            >> ${NVBLAS_CONFIG_FILE}
	echo "# Use all GPUs"                                                             >> ${NVBLAS_CONFIG_FILE}
	echo "NVBLAS_GPU_LIST ALL"                                                        >> ${NVBLAS_CONFIG_FILE}

	# Rebooting during an initialization action is not recommended, so just
	# dynamically load kernel modules. If you want to run an X server, it is
	# recommended that you schedule a reboot to occur after the initialization
	# action finishes.
	echo "--> Loading the NVIDIA drivers"
	modprobe nvidia-current
	modprobe nvidia-drm
	modprobe nvidia-uvm
	modprobe drm

	# Restart any NodeManagers so they pick up the NVBLAS config.
	echo "--> Restarting Hadoop NodeManagers"
	if systemctl status hadoop-yarn-nodemanager
	then
	  systemctl restart hadoop-yarn-nodemanager
	fi
else
	echo "--> No NVIDIA GPU detected"
fi
