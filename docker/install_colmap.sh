apt remove -y nvidia-cuda-toolkit

cd /
git clone https://github.com/Zador-Pataki/colmap.git
cd colmap
mkdir build
cd build
compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.')
cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${compute_cap}
ninja
ninja install