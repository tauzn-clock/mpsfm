import os
import yaml
from PIL import Image

DATA_PATH = "/scratchdata/indoor_lift"

current_dir = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(f"{current_dir}/images"):
    os.system(f"rm -rf {current_dir}/images")

os.system(f"mkdir {current_dir}/images")

for i in range(0,1000, 100):
    img = Image.open(f"{DATA_PATH}/rgb/{i}.png")
    img.save(f"{current_dir}/images/{i}.png")

fx = 306.9346923828125
fy = 306.8908386230469
cx = 318.58868408203125
cy = 198.37969970703125

intrinsics = {
    1: {
        "params": [fx, fy, cx, cy],
        "images": "all",
    }
}
with open(f"{current_dir}/intrinsics.yaml", "w") as f:
    yaml.dump(intrinsics, f)