import numpy as np
import pathlib
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

from mpsfm.test.simple import SimpleTest
from mpsfm.utils.tools import load_cfg
from mpsfm.vars import gvars

from mpsfm.utils.io import read_image
from mpsfm.utils.viz import plot_images, plot_keypoints

VISUALISE = False

# select pipeline: mpsfm/sfm/configs
cname = "sp-mast3r-dense"
conf = load_cfg(gvars.SFM_CONFIG_DIR / f"{cname}.yaml", return_name=False)

data_dir = gvars.ROOT / "local/example" # pathlib.Path("/mpsfm/custom_dataset")
imanmes = sorted(list((data_dir / "images").iterdir()))

if VISUALISE:
    fig, axs = plot_images([read_image(imname) for imname in imanmes], dpi=30)
    fig.savefig(data_dir / "visualise/all_images.png", dpi=300)

# setup the experiment
experiment = SimpleTest(conf, data_dir=data_dir)
# reconstruct
in_imanmes = [el.name for el in imanmes]
out_rec = experiment(
    imnames=in_imanmes,
    data_dir=data_dir,
)

if VISUALISE:
    # Prior depths
    p95 = np.percentile([image.depth.data for image in out_rec.images.values()], 95)
    fig, ax = plot_images([image.depth.data_prior for image in out_rec.images.values()], dpi=50, cmaps="viridis", vmax=p95)
    fig.savefig(data_dir / "visualise/depth_prior.png", dpi=300)
    # Refined depths
    fig, ax = plot_images([image.depth.data for image in out_rec.images.values()], dpi=50, cmaps="viridis", vmax=p95)
    fig.savefig(data_dir / "visualise/depth_refined.png", dpi=300)

if VISUALISE and False:
    fig = out_rec.vis_depth_maps(data_dir / "images", name="refined", dmap_rescale=0.8)
    fig = out_rec.vis_depth_maps(fig=fig, images_dir=data_dir / "images", name="prior", prior=True, dmap_rescale=0.8)
    fig = out_rec.vis_cameras(fig)

    fig.show()
    
if VISUALISE and False:
    # visualize sparse anchors (blue) and dense points (red)
    sparse_dense_mode = "sparse_mask"
    fig = out_rec.vis_colmap_points(mode=sparse_dense_mode, name=sparse_dense_mode)
    uncertainty_mode = "uncert"
    fig = out_rec.vis_colmap_points(fig=fig, mode=uncertainty_mode, name=uncertainty_mode)
    fig = out_rec.vis_cameras(fig, color="green", name="cameras")
    fig.show()



print("Propagated uncertainties per pixel...")
intstds = [image.calculate_int_covs_for_entire_image(False, False) for image in tqdm(out_rec.images.values())]

if VISUALISE:

    priorstds = [image.depth.uncertainty for image in out_rec.images.values()]

    p90 = np.percentile(np.hstack(intstds), 90)

    # Prior uncertainties
    fig, ax = plot_images([std for std in priorstds], dpi=50, vmax=p90, cmaps="viridis")
    fig.savefig(data_dir / "visualise/depth_uncertainty_prior.png", dpi=300)

    # Propagated uncertainties
    fig, ax = plot_images([std for std in intstds], dpi=50, vmax=p90, cmaps="viridis")
    fig.savefig(data_dir / "visualise/depth_uncertainty.png", dpi=300)

if VISUALISE and False:
    # Find 2D points used to optimize the depth
    koords_with_3d = [image.keypoint_coords_with_3d() * image.camera.sx for image in out_rec.images.values()]
    plot_keypoints(koords_with_3d, colors="r", ps=2)

if VISUALISE:
    p80 = np.percentile(np.hstack(intstds), 80)
    intcov_masks = {imid: intstds < p80 for imid, intstds in zip(out_rec.images.keys(), intstds)}
    fig = out_rec.vis_depth_maps(data_dir / "images", name="refined", input_masks=intcov_masks)
    fig = out_rec.vis_cameras(fig)
    fig.show()

