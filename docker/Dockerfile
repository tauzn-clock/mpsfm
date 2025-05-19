FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

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

RUN echo test

COPY install_ceres_solver.sh /tmp/install_ceres_solver.sh
RUN chmod +x /tmp/install_ceres_solver.sh && \
    /tmp/install_ceres_solver.sh && \
    rm /tmp/install_ceres_solver.sh

COPY install_colmap.sh /tmp/install_colmap.sh
RUN chmod +x /tmp/install_colmap.sh && \
    /tmp/install_colmap.sh && \
    rm /tmp/install_colmap.sh

RUN git config --global url."https://github.com/".insteadOf git@github.com:
RUN git config --global url."https://".insteadOf git://
RUN git clone --recursive https://github.com/tauzn-clock/mpsfm.git 

WORKDIR /mpsfm
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install -e .

CMD ["bash"]