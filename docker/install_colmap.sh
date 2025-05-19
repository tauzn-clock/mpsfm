apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev

apt-get update && apt-get install -y \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc


cd /
git clone https://github.com/cvg/pyceres.git
cd pyceres
git checkout tags/v2.1
python3 -m pip install /pyceres

cd /
git clone https://github.com/Zador-Pataki/colmap.git
cd colmap
mkdir build
cd build
compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.')
cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${compute_cap} -flto=auto
ninja
ninja install

pip3 install -e /colmap