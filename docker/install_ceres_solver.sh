# CMake
apt-get update && apt-get install -y \
    cmake \
    libgoogle-glog-dev libgflags-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libsuitesparse-dev

cd /
git clone --recursive https://github.com/ceres-solver/ceres-solver
cd ceres-solver
git checkout tags/2.1.0
mkdir build
cd build
cmake ..
make -j3
make test
# Optionally install Ceres, it can also be exported using CMake which
# allows Ceres to be used without requiring installation, see the documentation
# for the EXPORT_BUILD_DIR option for more information.
make install