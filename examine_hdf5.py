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

data = hdf5_to_dict("/mpsfm/custom_dataset/cache_dir/feats-superpoint-nms4-n-6000-rmax1600.h5")

print(data.keys())
print(data["100.png"].keys())

for k in data["100.png"].keys():
    print(data["100.png"][k])

print(data["100.png"]["scores"].max(), data["100.png"]["scores"].min())
    
from PIL import Image
import matplotlib.pyplot as plt

DATA_DIR = "/mpsfm/custom_dataset"
IMG_1 = "100.png"
IMG_2 = "120.png"

img1 = Image.open(f"{DATA_DIR}/images/{IMG_1}")
img2 = Image.open(f"{DATA_DIR}/images/{IMG_2}")

img1 = np.array(img1)
img2 = np.array(img2)

fig, ax = plt.subplots()
ax.imshow(img1)
ax.plot(data[IMG_1]["keypoints"][:,0], data[IMG_1]["keypoints"][:,1], "bx")

plt.axis("off")
plt.savefig("img1.png", bbox_inches='tight', pad_inches=0)

fig, ax = plt.subplots()
ax.imshow(img2)
ax.plot(data[IMG_2]["keypoints"][:,0], data[IMG_2]["keypoints"][:,1], "bx")

plt.axis("off")
plt.savefig("img2.png", bbox_inches='tight', pad_inches=0)

key_pts_desc_1 = data[IMG_1]["descriptors"]
key_pts_desc_2 = data[IMG_2]["descriptors"]

key_pts_coord_1 = data[IMG_1]["keypoints"]
key_pts_coord_2 = data[IMG_2]["keypoints"]

# Match keypoints using L2 distance

store = []

for i in range(key_pts_desc_1.shape[1]):    
    best_match = -1
    best_match_dist = -float("inf")
    for j in range(key_pts_desc_2.shape[1]):
        dist = np.dot(key_pts_desc_1[:, i], key_pts_desc_2[:, j])
        if dist > best_match_dist:
            best_match_dist = dist
            best_match = j
    
    if best_match_dist > 0.9:
        store.append([i,best_match])

print(len(store))

# Visualize matches
fig, ax = plt.subplots()

img_combined = np.zeros((img1.shape[0], img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
img_combined[:, :img1.shape[1]] = img1
img_combined[:, img1.shape[1]:] = img2 

ax.imshow(img_combined)

for i, j in store:
    x1, y1 = key_pts_coord_1[i]
    x2, y2 = key_pts_coord_2[j]
    ax.plot([x1, x2 + img1.shape[1]], [y1, y2], "r-", linewidth=0.5)

plt.axis("off")
plt.savefig("matches.png", bbox_inches='tight', pad_inches=0)