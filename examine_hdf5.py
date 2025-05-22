import h5py
import numpy as np

def hdf5_to_dict(h5file):
    def recurse(h5obj):
        out = {}
        for key, item in h5obj.items():
            if isinstance(item, h5py.Group):
                out[key] = recurse(item)
            elif isinstance(item, h5py.Dataset):
                out[key] = item[()]  # Convert to NumPy or scalar
        return out

    with h5py.File(h5file, "r") as f:
        return recurse(f)

data = hdf5_to_dict("/mpsfm/custom_dataset/cache_dir/depths-mast3r_matcher.h5")

print(data["400.png"]["450.png"]["400.png"])

print(data["200.png"]["400.png"]["400.png"])