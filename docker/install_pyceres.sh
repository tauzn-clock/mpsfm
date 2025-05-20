cd /
git clone https://github.com/cvg/pyceres.git
cd pyceres
git checkout tags/v2.4
mkdir build
cd build
cmake ..
python3 install -e /pyceres