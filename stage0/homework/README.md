# BERT汽车评论分类

基于BERT的汽车评论分析系统，可同时进行**主题识别**（多标签分类）和**情感分析**（三分类），用于分析汽车相关评论文本。

## 项目介绍

本项目针对汽车评论数据，实现了两个核心功能：
1. **主题识别**：识别评论中涉及的汽车相关主题（动力、价格、油耗等）
2. **情感分析**：分析评论或特定主题的情感极性（正面、中性、负面）

支持三种工作模式：
- 单独的主题分类任务
- 单独的情感分类任务
- 同时进行主题识别和情感分析的多任务模型

## 目录结构

```
.
├── README.md              # 项目说明文档
├── requirements.txt       # 依赖库列表
├── model.py               # 模型定义
├── dataset.py             # 数据处理
├── train.py               # 训练脚本
├── predict.py             # 预测脚本
├── model/                 # BERT预训练模型目录
└── data/                  # 数据目录
    ├── train.txt          # 训练数据
    └── test.txt           # 测试数据
```

## 数据格式

数据采用以下格式：

```
文本内容\t主题1#情感值 主题2#情感值 ...
```

例如：

```
一直 92 ， 偶尔 出去 了 不 了解 当地 油品 加 95 ( 97 ) 。 5万 公里 从没 遇到 问题 ， 省油 ， 动力 也 充足 ， 加 95 也 没感觉 有啥 不同 。	油耗#1 动力#1
```

情感值说明：
- `-1`: 负面情感
- `0`: 中性情感
- `1`: 正面情感

主题分类：包括10个主题类别
- 动力、价格、内饰、配置、安全性、外观、操控、油耗、空间、舒适性

## 模型架构

本项目实现了三种模型：

1. **BertTopicClassifier**：基于BERT的主题分类模型
   - 多标签分类（一段评论可能涉及多个主题）
   - 使用BCEWithLogitsLoss损失函数

2. **BertSentimentClassifier**：基于BERT的情感分类模型
   - 单标签多分类（-1, 0, 1三种情感极性）
   - 使用CrossEntropyLoss损失函数

3. **BertMultiTaskClassifier**：多任务模型，同时进行主题识别和情感分析
   - 共享BERT编码层
   - 将主题分类和情感分类作为两个独立任务
   - 使用加权损失函数进行联合训练

## 环境配置

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 准备BERT预训练模型（放在`./model`目录下）

## 使用方法

### 训练模型

#### 主题分类模型

```bash
python train.py --mode topic --train_file data/train.txt --test_file data/test.txt --model_dir saved_models --epochs 5
```

#### 情感分类模型

```bash
python train.py --mode sentiment --train_file data/train.txt --test_file data/test.txt --model_dir saved_models --epochs 5
```

#### 多任务模型（同时进行主题分类和情感分析）

```bash
python train.py --mode multitask --train_file data/train.txt --test_file data/test.txt --model_dir saved_models --epochs 5 --topic_weight 1.0 --sentiment_weight 1.0
```

### 使用模型进行预测

#### 单个文本预测

```bash
# 主题分类
python predict.py --model_type topic --model_path saved_models/bert_topic_model.pth --input "这辆车价格便宜，但是油耗高，配置还可以。"

# 情感分类
python predict.py --model_type sentiment --model_path saved_models/bert_sentiment_model.pth --input "这辆车价格便宜，但是油耗高，配置还可以。"

# 多任务预测（主题+情感）
python predict.py --model_type multitask --model_path saved_models/bert_multitask_model.pth --input "这辆车价格便宜，但是油耗高，配置还可以。"
```

#### 交互式预测

```bash
python predict.py --model_type multitask --model_path saved_models/bert_multitask_model.pth
```

#### 批量文件预测

```bash
python predict.py --model_type multitask --model_path saved_models/bert_multitask_model.pth --input data/test.txt --is_file --output predictions.json
```

## 参数说明

### 训练参数

- `--train_file`: 训练数据文件路径
- `--test_file`: 测试数据文件路径
- `--model_dir`: 模型保存目录
- `--bert_model`: BERT预训练模型路径
- `--max_length`: 最大序列长度，默认为128
- `--batch_size`: 批次大小，默认为32
- `--learning_rate`: 学习率，默认为2e-5
- `--epochs`: 训练轮数，默认为5
- `--topic_threshold`: 主题预测阈值，默认为0.5
- `--mode`: 训练模式，可选["topic", "sentiment", "multitask"]
- `--topic_weight`: 多任务学习中主题分类的损失权重
- `--sentiment_weight`: 多任务学习中情感分类的损失权重

### 预测参数

- `--model_path`: 模型文件路径
- `--model_type`: 模型类型，可选["topic", "sentiment", "multitask"]
- `--input`: 输入文本或文件
- `--output`: 预测结果输出文件，默认为predictions.json
- `--is_file`: 输入是否为文件
- `--max_length`: 最大序列长度，默认为128
- `--threshold`: 主题预测阈值，默认为0.5

## 性能指标

主题分类使用以下指标：
- Hamming Loss: 衡量错误预测的主题标签比例
- Precision, Recall, F1（micro和macro平均）

情感分类使用以下指标：
- Accuracy: 准确率
- Precision, Recall, F1（macro平均） 