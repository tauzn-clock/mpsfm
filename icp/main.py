import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import yaml

from depth_to_3d import depth_to_3d
from interpolate import interpolate

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

DATA_DIR = "/mpsfm/custom_dataset"
IMG_1 = "180.png"
IMG_2 = "200.png"

#Get intrinsic
with open(f"{DATA_DIR}/intrinsics.yaml") as f:
    intrinsics = yaml.safe_load(f)
    intrinsics = intrinsics[1]['params']
    print(intrinsics)

data = hdf5_to_dict(f"{DATA_DIR}/cache_dir/feats-superpoint-nms4-n-6000-rmax1600.h5")

img1 = Image.open(f"{DATA_DIR}/images/{IMG_1}")
img2 = Image.open(f"{DATA_DIR}/images/{IMG_2}")
depth1 = Image.open(f"{DATA_DIR}/depth/{IMG_1}")
depth2 = Image.open(f"{DATA_DIR}/depth/{IMG_2}")

img1 = np.array(img1)
img2 = np.array(img2)
depth1 = np.array(depth1) / 1000.0
depth2 = np.array(depth2) / 1000.0

print(depth1.max(), depth2.max())

key_pts_desc_1 = data[IMG_1]["descriptors"]
key_pts_desc_2 = data[IMG_2]["descriptors"]

key_pts_coord_1 = data[IMG_1]["keypoints"]
key_pts_coord_2 = data[IMG_2]["keypoints"]

key_confidence_1 = data[IMG_1]["scores"]
key_confidence_2 = data[IMG_2]["scores"]

print("Before:", len(key_confidence_1), len(key_confidence_2))

conf_bound = 0.01

print(key_pts_coord_1.shape)

key_pts_desc_1 = key_pts_desc_1[:,key_confidence_1 > conf_bound] 
key_pts_coord_1 = key_pts_coord_1[key_confidence_1 > conf_bound,:]
key_confidence_1 = key_confidence_1[key_confidence_1 > conf_bound] 

key_pts_desc_2 = key_pts_desc_2[:,key_confidence_2 > conf_bound] 
key_pts_coord_2 = key_pts_coord_2[key_confidence_2 > conf_bound,:]
key_confidence_2 = key_confidence_2[key_confidence_2 > conf_bound]

print("After:", len(key_confidence_1), len(key_confidence_2))

matches = []

for i in range(key_pts_desc_1.shape[1]):    
    best_match = -1
    best_match_dist = -float("inf")
    for j in range(key_pts_desc_2.shape[1]):
        dist = np.dot(key_pts_desc_1[:, i], key_pts_desc_2[:, j])
        if dist > best_match_dist:
            best_match_dist = dist
            best_match = j
    
    if best_match_dist > 0.9:
        matches.append([i,best_match])
        
print("Matches", len(matches))

# Visualize matches
fig, ax = plt.subplots()

img_combined = np.zeros((img1.shape[0], img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
img_combined[:, :img1.shape[1]] = img1
img_combined[:, img1.shape[1]:] = img2 

ax.imshow(img_combined)

for i, j in matches:
    x1, y1 = key_pts_coord_1[i]
    x2, y2 = key_pts_coord_2[j]
    ax.plot([x1, x2 + img1.shape[1]], [y1, y2], "r-", linewidth=0.5)

plt.axis("off")
plt.savefig("matches.png", bbox_inches='tight', pad_inches=0)

pcd_1 = depth_to_3d(depth1, intrinsics)
pcd_2 = depth_to_3d(depth2, intrinsics)

H, W = depth1.shape

pcd_1 = pcd_1.reshape((H, W, 3))
pcd_2 = pcd_2.reshape((H, W, 3))
    
# RANSAC to find relative pose

best_tf = None
pts_cnt = 0
pts_bound = 0.1

all_features_1 = []
all_features_2 = []

for i, j in matches:
    x1, y1 = key_pts_coord_1[i]
    x2, y2 = key_pts_coord_2[j]
    
    all_features_1.append(interpolate(pcd_1, x1, y1))
    all_features_2.append(interpolate(pcd_2, x2, y2))

all_features_1 = np.array(all_features_1)
all_features_2 = np.array(all_features_2)

#print(all_features_1.shape, all_features_2.shape)
#R = np.array([[1, 0, 0], [0, 0.707, 0.707], [0, -0.707, 0.707]])
#all_features_2 = np.dot(all_features_1, R.T)  + np.array([1,2,3])

for _ in range(5000):
    sample = np.random.choice(len(matches), 3, replace=False)
    
    pts1 = []
    pts2 = []
    for i in range(len(sample)):
        pts1.append(all_features_1[sample[i],:])
        pts2.append(all_features_2[sample[i],:])
    
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
        
    # Tf from pts1 to pts2
    def rigid_transform_3D(P, Q):
        assert P.shape == Q.shape

        # Step 1: Compute centroids
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)
        
        # Step 2: Center the points
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q

        # Step 3: Compute covariance matrix
        H = P_centered.T @ Q_centered

        # Step 4: SVD
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Step 5: Fix reflection
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        # Step 6: Compute translation
        t = centroid_Q - R @ centroid_P
        
        return R, t
    
    R, t = rigid_transform_3D(pts1, pts2)
        
    all_features_1_transformed = np.dot(all_features_1, R.T) + t
    
    diff = np.linalg.norm(all_features_2 - all_features_1_transformed, axis=1)
    
    # Count number of diff is less than bound
    
    within_bound = diff < pts_bound
        
    if (within_bound.sum() > pts_cnt):
        pts_cnt = within_bound.sum()
        best_tf = (R,t)
    
    #exit()

print(pts_cnt)
print(best_tf)