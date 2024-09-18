**Scale-Focused DETR**: Enhancing Small Object Detection with Attention Mechanism
===

By [Ru Qiu], [Hongzhi Fu].

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)
[![GitHub license](https://img.shields.io/github/license/xiuqhou/Salience-DETR.svg?color=blue)](https://github.com/xiuqhou/Salience-DETR/blob/master/LICENSE)
![GitHub stars](https://img.shields.io/github/forks/qiqidonebyte/Salience-DETR)
![GitHub forks](https://img.shields.io/github/forks/qiqidonebyte/Salience-DETR)

## ✨研究亮点:

1. 我们深入分析了针对小目标识别的两阶段DETR类方法中存在的[尺度偏差和查询冗余](id_1)问题。
2. 我们提出了小目标特征多尺度窗口权重注意力方法能较好的捕捉[细粒度的物体轮廓](#id_2)，并优化了小目标的显著性监督下降低计算复杂度的分层过滤算法。
3. Scale-Focused DETR在全球海域救援挑战数据集SeaDronesSee任务上分别提升了 **+4.0%**  AP，在SeaDronesSee上只使用了大约 *
   *50\%** FLOPs 实现了相当的精度。

## 模型库

基于SeaDronesSee数据集进行了一系列的测试。

### 训练12轮

| 模型                 | 主干网      |  AP  | AP<sub>50 | AP<sub>75 | AP<sub>S | AP<sub>M | AP<sub>L |   时间    |
|--------------------|----------|:----:|:---------:|:---------:|:--------:|:--------:|:--------:|:-------:|
| Scale-Focused DETR | ResNet50 | 50.0 |   67.7    |   54.2    |   33.3   |   54.4   |   64.4   | 5:12:03 |

### 训练24轮

| 模型                 | 主干网      |  AP  | AP<sub>50 | AP<sub>75 | AP<sub>S | AP<sub>M | AP<sub>L |    时间    |
|--------------------|----------|:----:|:---------:|:---------:|:--------:|:--------:|:--------:|:--------:|
| Scale-Focused DETR | ResNet50 | 51.2 |   68.9    |   55.7    |   33.9   |   55.5   |   65.6   | 22:10:13 |

## 📁准备数据集

请按照如下格式下载 [COCO 2017](https://cocodataset.org/) 数据集或准备您自己的数据集，并将他们放在 `data/` 目录下。

```shell
coco/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```

## 📚︎安装环境


1. 创建并激活conda环境：
    
    ```shell
    conda create -n sfdetr python=3.8
    conda activate sfdetr
    ```
2. 根据官方步骤 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) 安装pytorch
   
   ```shell
   conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
   ```

3. 安装其他依赖：

    ```shell
    conda install --file requirements.txt -c conda-forge
    ```
## 📚︎训练模型

```shell
nohup python main.py > output20240712.log 2>&1 &
```

## 📚︎全球排名

```url
https://macvi.org/leaderboard/airborne/seadronessee/object-detection-v2
```