import os
from PIL import Image

# === 修改为你的真实路径 ===
SRC_ROOT = r"D:/Users/111/Downloads/datasets/training"

SRC_IMG = os.path.join(SRC_ROOT, "images")
SRC_GT  = os.path.join(SRC_ROOT, "1st_manual")

DST_IMG = "data/imgs"
DST_GT  = "data/masks"

os.makedirs(DST_IMG, exist_ok=True)
os.makedirs(DST_GT, exist_ok=True)

for img_name in os.listdir(SRC_IMG):
    if not img_name.endswith(".tif"):
        continue

    # 1. 复制图像
    Image.open(os.path.join(SRC_IMG, img_name)) \
         .save(os.path.join(DST_IMG, img_name))

    # 2. 对应的 GT 名字（DRIVE 固定规则）
    # 21_training.tif -> 21_manual1.gif
    gt_name = img_name.replace("_training.tif", "_manual1.gif")

    gt = Image.open(os.path.join(SRC_GT, gt_name)).convert("L")
    gt.save(os.path.join(DST_GT, img_name.replace(".tif", ".png")))

print("DRIVE training data prepared for UNet.")
