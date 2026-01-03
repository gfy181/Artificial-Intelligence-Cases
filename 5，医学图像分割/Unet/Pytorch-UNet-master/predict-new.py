import torch
import numpy as np
from PIL import Image
from pathlib import Path

from unet import UNet
from utils.data_loading import BasicDataset


def predict_img(
    net,
    img_path,
    device,
    scale=0.5,
    threshold=0.5
):
    net.eval()

    # 1. 读取原始图像
    img = Image.open(img_path)

    # 2. 使用和训练时完全一致的预处理
    img = BasicDataset.preprocess(
        mask_values=[0, 255],   # binary segmentation
        pil_img=img,
        scale=scale,
        is_mask=False
    )

    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        probs = torch.sigmoid(output)
        mask = (probs > threshold).float()

    # 去掉 batch 和 channel 维
    mask = mask.squeeze().cpu().numpy()

    return mask


def mask_to_image(mask: np.ndarray):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == '__main__':
    # ======== 配置区 ========
    model_path = Path('checkpoints/checkpoint_epoch5.pth')
    image_path = Path(r'D:\Users\111\Downloads\datasets\test\images\01_test.tif')
    output_path = Path('predict_01_test.png')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ======== 加载模型 ========
    net = UNet(n_channels=3, n_classes=1)
    state_dict = torch.load(model_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 255])
    net.load_state_dict(state_dict)
    net.to(device)

    # ======== 推理 ========
    mask = predict_img(net, image_path, device)

    result = mask_to_image(mask)
    result.save(output_path)

    print(f'Saved prediction to {output_path}')
