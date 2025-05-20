import numpy as np
import pathlib
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

from mpsfm.test.simple import SimpleTest
from mpsfm.utils.tools import load_cfg
from mpsfm.vars import gvars

# select pipeline: mpsfm/sfm/configs
cname = "sp-mast3r-dense"
conf = load_cfg(gvars.SFM_CONFIG_DIR / f"{cname}.yaml", return_name=False)

data_dir = pathlib.Path("/mpsfm/custom_dataset")
imanmes = sorted(list((data_dir / "images").iterdir()))

# setup the experiment
experiment = SimpleTest(conf, data_dir=data_dir)
# reconstruct
in_imanmes = [el.name for el in imanmes]
out_rec = experiment(
    imnames=in_imanmes,
    data_dir=data_dir,
)