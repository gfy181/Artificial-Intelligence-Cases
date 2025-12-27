# YOLOv12 在 COCO 2017 数据集上的训练流程说明

本 README 详细记录了在 **AutoDL 服务器** 上，基于 **YOLOv12** 框架对 **COCO 2017 数据集**进行训练的完整实验流程，适用于实验复现、课程设计或科研记录。

---

## 一、实验环境准备

1. 前往 **AutoDL** 平台租用合适配置的 GPU 服务器  
2. 环境选择：**官方预配置的 YOLOv12 环境**（包含：
   - YOLOv12 源码
   - 已配置完成的 Conda 环境）


---

## 二、下载 COCO 2017 数据集

### 1. 下载 YOLO 格式的 COCO 标签文件

在 `yolov12/datasets` 目录下执行：

```bash
curl -L -o coco2017labels.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip
```

解压标签文件：

```bash
unzip coco2017labels.zip
```

解压后会生成：

```text
datasets/coco/
├── coco/
│   ├── labels/
│   │   ├── train2017/
│   │   └── val2017/
│   └── annotations/
```

---

### 2. 下载 COCO 2017 图像数据

进入图像目录：

```bash
mkdir -p yolov12/datasets/coco/images
cd yolov12/datasets/coco/images
```

下载图像压缩包：

```bash
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
```

解压图像：

```bash
unzip train2017.zip
unzip val2017.zip
```

最终图像目录结构如下：

```text
datasets/coco/images/
├── train2017/
└── val2017/
```

---

## 三、创建 COCO 数据集配置文件（coco.yaml）

文件路径：

```text
yolov12/datasets/coco/coco.yaml
```

文件内容如下：

```yaml
# COCO 2017 for YOLO (local images version)

path: /root/autodl-tmp/yolov12/datasets/coco

train: images/train2017
val: images/val2017

nc: 80

names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
```

---

## 四、创建训练脚本

在 `yolov12` 根目录下创建 `train.py`：

```python
import warnings
from ultralytics import YOLO

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    model = YOLO(model='ultralytics/cfg/models/v12/yolov12.yaml')

    model.train(
        data='/root/autodl-tmp/yolov12/datasets/coco/coco.yaml',
        imgsz=640,
        epochs=80,
        batch=16,
        workers=4,
        device='0',
        optimizer='SGD',
        project='runs/train',
        name='yolo12n',
    )
```

---

## 五、启动训练

在 `yolov12` 目录下执行：

```bash
python train.py
```

---

## 六、训练结果与权重文件

训练完成后，结果将保存在：

```text
runs/train/yolo12n/
```

主要文件包括：

- `weights/best.pt`：验证集性能最优模型
- `weights/last.pt`：最后一个 epoch 的模型
- `results.png`：训练过程曲线（loss / mAP）

---

## 七、说明

- 本实验流程基于 **Ultralytics YOLOv12**
- COCO 标签采用官方 YOLO 格式转换版本
- 适用于实验复现、课程设计、科研训练记录

