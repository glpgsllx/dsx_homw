# BERT情感分类模型

基于BERT的汽车评论情感分类模型，可以分析评论文本中不同方面（如价格、配置、油耗等）的情感极性。

## 项目结构

```
.
├── README.md              # 项目说明文档
├── requirements.txt       # 依赖库列表
├── dataset.py             # 数据处理模块
├── model.py               # 模型定义模块
├── train.py               # 训练脚本
├── predict.py             # 预测脚本
└── data/                  # 数据目录
    ├── train.txt          # 训练数据
    └── test.txt           # 测试数据
```

## 数据格式

数据采用以下格式：

```
文本内容\t领域1#情感值 领域2#情感值 ...
```

例如：

```
这辆车价格便宜，但是油耗高，配置还可以。	价格#1 油耗#-1 配置#0
```

情感值说明：
- `-1`: 负面情感
- `0`: 中性情感
- `1`: 正面情感

## 环境配置

1. 安装依赖：

```bash
pip install -r requirements.txt
```

## 模型训练

### 基本BERT情感分类模型

```bash
python train.py --train_file data/train.txt --test_file data/test.txt --model_dir models --epochs 5
```

### 领域感知BERT情感分类模型

```bash
python train.py --train_file data/train.txt --test_file data/test.txt --model_dir models --epochs 5 --use_aspect
```

### 使用Weights & Biases (wandb) 跟踪实验

本项目集成了Weights & Biases用于可视化训练过程和实验跟踪。运行以下命令首先登录wandb：

```bash
wandb login
```

然后在训练时添加wandb相关参数：

```bash
python train.py --train_file data/train.txt --test_file data/test.txt --model_dir models --epochs 5 --wandb_project "bert-sentiment" --wandb_entity "你的用户名"
```

通过wandb界面，您可以查看：
- 训练和验证loss曲线
- 准确率、精确率、召回率和F1分数曲线
- 混淆矩阵
- 超参数比较

### 参数说明

- `--train_file`: 训练数据文件路径
- `--test_file`: 测试数据文件路径
- `--model_dir`: 模型保存目录
- `--bert_model`: BERT预训练模型名称，默认为bert-base-chinese
- `--max_length`: 最大序列长度，默认为128
- `--batch_size`: 批次大小，默认为16
- `--learning_rate`: 学习率，默认为2e-5
- `--epochs`: 训练轮数，默认为5
- `--use_aspect`: 是否使用领域感知模型
- `--aspect_embedding_dim`: 领域嵌入维度，默认为50
- `--wandb_project`: wandb项目名称，默认为"bert-sentiment-classification"
- `--wandb_entity`: wandb实体名称（用户名或组织名），默认为None

## 模型预测

### 交互式预测

```bash
python predict.py --model_path models/bert_sentiment_model.pth
```

### 预测单个文本

```bash
python predict.py --model_path models/bert_sentiment_model.pth --input "这辆车价格便宜，但是油耗高，配置还可以。" --aspects 价格 油耗 配置
```

### 批量预测

```bash
python predict.py --model_path models/bert_sentiment_model.pth --input data/test.txt --is_file --output predictions.json
```

### 参数说明

- `--model_path`: 模型文件路径
- `--input`: 输入文本或文件
- `--output`: 预测结果输出文件，默认为predictions.json
- `--aspects`: 评论领域列表
- `--is_file`: 输入是否为文件
- `--max_length`: 最大序列长度，默认为128

## 模型说明

本项目实现了两种基于BERT的情感分类模型：

1. **基本BERT情感分类模型**：直接使用BERT提取文本特征进行情感分类。
2. **领域感知BERT情感分类模型**：结合领域信息，更准确地预测不同领域的情感极性。

这两种模型都可以有效分析汽车评论中的情感极性，帮助理解用户对不同方面的评价。 