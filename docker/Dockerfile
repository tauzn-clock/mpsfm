FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    bash \
    lsb-release \
    git \
    curl \
    python3 \
    python3-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
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

RUN apt-get update && apt-get install -y \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc

# Install Ceres.
RUN apt-get install -y --no-install-recommends --no-install-suggests wget && \
    wget "http://ceres-solver.org/ceres-solver-2.1.0.tar.gz" && \
    tar zxf ceres-solver-2.1.0.tar.gz && \
    mkdir ceres-build && \
    cd ceres-build && \
    cmake ../ceres-solver-2.1.0 -GNinja && \
    ninja install

COPY install_colmap.sh /tmp/install_colmap.sh
RUN chmod +x /tmp/install_colmap.sh && \
    /tmp/install_colmap.sh && \
    rm /tmp/install_colmap.sh

RUN git clone https://github.com/tauzn-clock/mpsfm.git 

WORKDIR /mpsfm
RUN pip3 install --upgrade pip
#RUN pip3 install -r requirements.txt

CMD ["bash"]