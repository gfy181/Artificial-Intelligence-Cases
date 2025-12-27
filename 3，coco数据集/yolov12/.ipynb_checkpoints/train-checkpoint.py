import warnings
import os  # 导入 os 模块用于执行系统命令
from ultralytics import YOLO

warnings.filterwarnings('ignore')  # 忽略所有警告信息，避免训练过程中出现的警告影响输出

if __name__ == '__main__':
    # 加载模型，使用 YAML 配置文件
    model = YOLO(model=r'ultralytics/cfg/models/v12/yolov12.yaml')
    # 训练模型
    model.train(
        data="/root/autodl-tmp/yolov12/datasets/coco/coco.yaml",# 数据集配置文件路径
        imgsz=640,  # 输入图像大小
        epochs=80,  # 训练轮数
        batch=16,  # 批次大小
        workers=4,  # 数据加载线程数（多线程加速数据加载）
        device='0',  # 使用 GPU
        optimizer='SGD',  # 随机梯度下降优化器
        project='runs/train',  # 训练结果保存路径
        name='yolo12n',  # 实验名称
    )
