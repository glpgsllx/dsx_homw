import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss, classification_report

from dataset import CarReviewDataset, create_dataloaders, get_predefined_topics
from model import BertTopicClassifier, BertSentimentClassifier, BertMultiTaskClassifier

def train_topic_classifier(model, dataloader, optimizer, scheduler, criterion, device):
    """训练主题分类器一个epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="训练主题分类器")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        topic_labels = batch['topic_labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        loss = criterion(outputs, topic_labels)
        
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def train_sentiment_classifier(model, dataloader, optimizer, scheduler, criterion, device):
    """训练情感分类器一个epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="训练情感分类器")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sentiment_labels = batch['sentiment_label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        loss = criterion(outputs, sentiment_labels)
        
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def train_multitask_classifier(model, dataloader, optimizer, scheduler, topic_criterion, sentiment_criterion, device, topic_weight=1.0, sentiment_weight=1.0):
    """训练多任务分类器一个epoch"""
    model.train()
    total_loss = 0
    topic_total_loss = 0
    sentiment_total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="训练多任务分类器")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        topic_labels = batch['topic_labels'].to(device)
        sentiment_labels = batch['sentiment_label'].to(device)
        
        optimizer.zero_grad()
        
        topic_outputs, sentiment_outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        topic_loss = topic_criterion(topic_outputs, topic_labels)
        sentiment_loss = sentiment_criterion(sentiment_outputs, sentiment_labels)
        
        # 总损失为主题损失和情感损失的加权和
        loss = topic_weight * topic_loss + sentiment_weight * sentiment_loss
        
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        topic_total_loss += topic_loss.item()
        sentiment_total_loss += sentiment_loss.item()
        
        progress_bar.set_postfix({
            'loss': loss.item(),
            'topic_loss': topic_loss.item(),
            'sentiment_loss': sentiment_loss.item()
        })
    
    return total_loss / len(dataloader), topic_total_loss / len(dataloader), sentiment_total_loss / len(dataloader)

def evaluate_topic_classifier(model, dataloader, criterion, device, threshold=0.5, topic_list=None):
    """评估主题分类器"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []  # 存储原始预测概率
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估主题分类器"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            topic_labels = batch['topic_labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = criterion(outputs, topic_labels)
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(topic_labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 打印预测概率分布
    print("\n预测概率分布统计:")
    for i, topic in enumerate(topic_list):
        topic_probs = all_probs[:, i]
        print(f"\n{topic}:")
        print(f"最小值: {topic_probs.min():.4f}")
        print(f"最大值: {topic_probs.max():.4f}")
        print(f"平均值: {topic_probs.mean():.4f}")
        print(f"标准差: {topic_probs.std():.4f}")
        print(f"预测为1的比例: {(topic_probs > threshold).mean():.4f}")
        print(f"实际为1的比例: {all_labels[:, i].mean():.4f}")
    
    # 只计算macro平均的指标
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # 计算每个类别的F1分数
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # 如果提供了主题列表，打印每个主题的F1分数
    if topic_list is not None:
        print("\n每个主题的F1分数:")
        for i, topic in enumerate(topic_list):
            if i < len(f1_per_class):
                print(f"{topic}: {f1_per_class[i]:.4f}")
                # 打印详细的混淆矩阵
                true_pos = ((all_labels[:, i] == 1) & (all_preds[:, i] == 1)).sum()
                false_pos = ((all_labels[:, i] == 0) & (all_preds[:, i] == 1)).sum()
                false_neg = ((all_labels[:, i] == 1) & (all_preds[:, i] == 0)).sum()
                true_neg = ((all_labels[:, i] == 0) & (all_preds[:, i] == 0)).sum()
                print(f"TP: {true_pos}, FP: {false_pos}, FN: {false_neg}, TN: {true_neg}")
    
    return {
        'loss': total_loss / len(dataloader),
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class
    }

def evaluate_sentiment_classifier(model, dataloader, criterion, device):
    """评估情感分类器"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估情感分类器"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = criterion(outputs, sentiment_labels)
            
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(sentiment_labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # 打印分类报告
    print("\n情感分类报告:")
    print(classification_report(all_labels, all_preds, target_names=['负面', '中性', '正面'], digits=4))
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_multitask_classifier(model, dataloader, topic_criterion, sentiment_criterion, device, threshold=0.5, topic_list=None):
    """评估多任务分类器"""
    model.eval()
    topic_total_loss = 0
    sentiment_total_loss = 0
    
    # 主题分类
    topic_all_preds = []
    topic_all_labels = []
    topic_all_probs = []
    
    # 情感分类
    sentiment_all_preds = []
    sentiment_all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估多任务分类器"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            topic_labels = batch['topic_labels'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)
            
            topic_outputs, sentiment_outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            
            topic_loss = topic_criterion(topic_outputs, topic_labels)
            sentiment_loss = sentiment_criterion(sentiment_outputs, sentiment_labels)
            
            topic_total_loss += topic_loss.item()
            sentiment_total_loss += sentiment_loss.item()
            
            # 主题预测
            topic_probs = torch.sigmoid(topic_outputs)
            topic_all_probs.extend(topic_probs.cpu().numpy())
            topic_all_labels.extend(topic_labels.cpu().numpy())
            
            # 情感预测
            _, sentiment_preds = torch.max(sentiment_outputs, dim=1)
            sentiment_all_preds.extend(sentiment_preds.cpu().numpy())
            sentiment_all_labels.extend(sentiment_labels.cpu().numpy())
    
    # 转为numpy数组
    topic_all_probs = np.array(topic_all_probs)
    topic_all_labels = np.array(topic_all_labels)
    sentiment_all_preds = np.array(sentiment_all_preds)
    sentiment_all_labels = np.array(sentiment_all_labels)
    
    # 为每个主题计算最优阈值
    topic_thresholds = []
    print("\n主题预测概率分布统计:")
    for i, topic in enumerate(topic_list):
        topic_probs = topic_all_probs[:, i]
        # 使用预测概率的平均值作为阈值
        optimal_threshold = topic_probs.mean()
        topic_thresholds.append(optimal_threshold)
        
        print(f"\n{topic}:")
        print(f"最小值: {topic_probs.min():.4f}")
        print(f"最大值: {topic_probs.max():.4f}")
        print(f"平均值: {topic_probs.mean():.4f}")
        print(f"标准差: {topic_probs.std():.4f}")
        print(f"选择的阈值: {optimal_threshold:.4f}")
        print(f"预测为1的比例: {(topic_probs > optimal_threshold).mean():.4f}")
        print(f"实际为1的比例: {topic_all_labels[:, i].mean():.4f}")
    
    # 使用动态阈值进行预测
    topic_all_preds = np.zeros_like(topic_all_probs)
    for i in range(len(topic_list)):
        topic_all_preds[:, i] = (topic_all_probs[:, i] > topic_thresholds[i]).astype(float)
    
    # 计算主题分类指标 - 只使用macro平均
    topic_precision_macro = precision_score(topic_all_labels, topic_all_preds, average='macro', zero_division=0)
    topic_recall_macro = recall_score(topic_all_labels, topic_all_preds, average='macro', zero_division=0)
    topic_f1_macro = f1_score(topic_all_labels, topic_all_preds, average='macro', zero_division=0)
    
    # 计算每个主题的F1分数
    topic_f1_per_class = f1_score(topic_all_labels, topic_all_preds, average=None, zero_division=0)
    
    # 如果提供了主题列表，打印每个主题的F1分数和混淆矩阵
    if topic_list is not None:
        print("\n每个主题的F1分数和统计:")
        for i, topic in enumerate(topic_list):
            if i < len(topic_f1_per_class):
                print(f"\n{topic}:")
                print(f"F1分数: {topic_f1_per_class[i]:.4f}")
                print(f"使用的阈值: {topic_thresholds[i]:.4f}")
                # 打印详细的混淆矩阵
                true_pos = ((topic_all_labels[:, i] == 1) & (topic_all_preds[:, i] == 1)).sum()
                false_pos = ((topic_all_labels[:, i] == 0) & (topic_all_preds[:, i] == 1)).sum()
                false_neg = ((topic_all_labels[:, i] == 1) & (topic_all_preds[:, i] == 0)).sum()
                true_neg = ((topic_all_labels[:, i] == 0) & (topic_all_preds[:, i] == 0)).sum()
                print(f"TP: {true_pos}, FP: {false_pos}, FN: {false_neg}, TN: {true_neg}")
                print(f"正样本数量: {(topic_all_labels[:, i] == 1).sum()}")
                print(f"负样本数量: {(topic_all_labels[:, i] == 0).sum()}")
    
    # 计算情感分类指标
    sentiment_accuracy = accuracy_score(sentiment_all_labels, sentiment_all_preds)
    sentiment_precision = precision_score(sentiment_all_labels, sentiment_all_preds, average='macro', zero_division=0)
    sentiment_recall = recall_score(sentiment_all_labels, sentiment_all_preds, average='macro', zero_division=0)
    sentiment_f1 = f1_score(sentiment_all_labels, sentiment_all_preds, average='macro', zero_division=0)
    
    # 打印情感分类报告
    print("\n情感分类报告:")
    print(classification_report(sentiment_all_labels, sentiment_all_preds, target_names=['负面', '中性', '正面'], digits=4))
    
    return {
        'topic_loss': topic_total_loss / len(dataloader),
        'sentiment_loss': sentiment_total_loss / len(dataloader),
        'topic_precision_macro': topic_precision_macro,
        'topic_recall_macro': topic_recall_macro,
        'topic_f1_macro': topic_f1_macro,
        'sentiment_accuracy': sentiment_accuracy,
        'sentiment_precision': sentiment_precision,
        'sentiment_recall': sentiment_recall,
        'sentiment_f1': sentiment_f1
    }

def main():
    parser = argparse.ArgumentParser(description="BERT汽车评论分类训练脚本")
    parser.add_argument("--train_file", default="data/train.txt", help="训练数据文件路径")
    parser.add_argument("--test_file", default="data/test.txt", help="测试数据文件路径")
    parser.add_argument("--model_dir", default="./saved_models", help="模型保存目录")
    parser.add_argument("--bert_model", default="./model", help="BERT预训练模型名称")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--topic_threshold", type=float, default=0.5, help="主题预测阈值")
    parser.add_argument("--wandb_project", default="car-review-classification", help="wandb项目名称")
    parser.add_argument("--wandb_entity", default=None, help="wandb实体名称")
    parser.add_argument("--mode", choices=["topic", "sentiment", "multitask"], default="multitask", help="训练模式")
    parser.add_argument("--topic_weight", type=float, default=1.0, help="多任务学习中主题分类的损失权重")
    parser.add_argument("--sentiment_weight", type=float, default=1.0, help="多任务学习中情感分类的损失权重")
    args = parser.parse_args()
    
    # 初始化wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "topic_threshold": args.topic_threshold,
            "mode": args.mode,
            "topic_weight": args.topic_weight,
            "sentiment_weight": args.sentiment_weight
        }
    )
    
    # 创建模型保存目录
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 检查是否有GPU可用
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("使用MPS设备")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用GPU设备")
    else:
        device = torch.device("cpu")
        print("使用CPU设备")
    
    # 加载BERT分词器
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    
    # 创建数据加载器
    train_loader, test_loader, topic_mapping = create_dataloaders(
        args.train_file, args.test_file, tokenizer,
        batch_size=args.batch_size, max_length=args.max_length
    )
    
    # 获取预定义主题列表
    topic_list = get_predefined_topics()
    print(f"使用的主题列表: {topic_list}")
    
    # 获取训练集和验证集
    train_size = int(0.9 * len(train_loader.dataset))
    val_size = len(train_loader.dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_loader.dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    if args.mode == "topic":
        # 创建主题分类模型
        model = BertTopicClassifier(
            bert_model_name=args.bert_model,
            num_topics=len(topic_list)
        )
        model.to(device)
        
        # 定义损失函数（多标签分类使用BCEWithLogitsLoss）
        criterion = nn.BCEWithLogitsLoss()
        
        # 定义优化器
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        
        # 计算总的训练步数
        total_steps = len(train_loader) * args.epochs
        
        # 创建学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # 训练主题分类模型
        best_val_f1 = 0
        best_model_state = None
        
        for epoch in range(args.epochs):
            print(f"\n====== Epoch {epoch+1}/{args.epochs} ======")
            
            # 训练
            train_loss = train_topic_classifier(model, train_loader, optimizer, scheduler, criterion, device)
            print(f"训练损失: {train_loss:.4f}")
            
            # 验证
            val_metrics = evaluate_topic_classifier(
                model, val_loader, criterion, device, 
                threshold=args.topic_threshold, topic_list=topic_list
            )
            
            print(f"验证损失: {val_metrics['loss']:.4f}")
            print(f"验证精确率(macro): {val_metrics['precision_macro']:.4f}")
            print(f"验证召回率(macro): {val_metrics['recall_macro']:.4f}")
            print(f"验证F1分数(macro): {val_metrics['f1_macro']:.4f}")

            
            # 记录到wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items() if k != 'f1_per_class'}
            })
            
            # 保存最佳模型
            if val_metrics['f1_macro'] > best_val_f1:
                best_val_f1 = val_metrics['f1_macro']
                best_model_state = model.state_dict().copy()
                print(f"保存新的最佳模型，F1分数(macro): {val_metrics['f1_macro']:.4f}")
        
        # 加载最佳模型状态
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 在测试集上评估
        print("\n====== 测试集评估 ======")
        test_metrics = evaluate_topic_classifier(
            model, test_loader, criterion, device, 
            threshold=args.topic_threshold, topic_list=topic_list
        )
        
        print(f"测试损失: {test_metrics['loss']:.4f}")
        print(f"测试精确率(macro): {test_metrics['precision_macro']:.4f}")
        print(f"测试召回率(macro): {test_metrics['recall_macro']:.4f}")
        print(f"测试F1分数(macro): {test_metrics['f1_macro']:.4f}")

        
        # 记录到wandb
        wandb.log({
            **{f"test_{k}": v for k, v in test_metrics.items() if k != 'f1_per_class'}
        })
        
        # 保存模型
        model_path = os.path.join(args.model_dir, "bert_topic_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'topic_list': topic_list,
            'topic_mapping': topic_mapping,
            'args': vars(args)
        }, model_path)
        
        print(f"模型已保存到 {model_path}")
    
    elif args.mode == "sentiment":
        # 创建情感分类模型
        model = BertSentimentClassifier(
            bert_model_name=args.bert_model,
            num_classes=3  # -1, 0, 1 -> 0, 1, 2
        )
        model.to(device)
        
        # 定义损失函数（单标签多分类使用CrossEntropyLoss）
        criterion = nn.CrossEntropyLoss()
        
        # 定义优化器
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        
        # 计算总的训练步数
        total_steps = len(train_loader) * args.epochs
        
        # 创建学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # 训练情感分类模型
        best_val_f1 = 0
        best_model_state = None
        
        for epoch in range(args.epochs):
            print(f"\n====== Epoch {epoch+1}/{args.epochs} ======")
            
            # 训练
            train_loss = train_sentiment_classifier(model, train_loader, optimizer, scheduler, criterion, device)
            print(f"训练损失: {train_loss:.4f}")
            
            # 验证
            val_metrics = evaluate_sentiment_classifier(model, val_loader, criterion, device)
            
            print(f"验证损失: {val_metrics['loss']:.4f}")
            print(f"验证准确率: {val_metrics['accuracy']:.4f}")
            print(f"验证精确率: {val_metrics['precision']:.4f}")
            print(f"验证召回率: {val_metrics['recall']:.4f}")
            print(f"验证F1分数: {val_metrics['f1']:.4f}")
            
            # 记录到wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })
            
            # 保存最佳模型
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_model_state = model.state_dict().copy()
                print(f"保存新的最佳模型，F1分数: {val_metrics['f1']:.4f}")
        
        # 加载最佳模型状态
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 在测试集上评估
        print("\n====== 测试集评估 ======")
        test_metrics = evaluate_sentiment_classifier(model, test_loader, criterion, device)
        
        print(f"测试损失: {test_metrics['loss']:.4f}")
        print(f"测试准确率: {test_metrics['accuracy']:.4f}")
        print(f"测试精确率: {test_metrics['precision']:.4f}")
        print(f"测试召回率: {test_metrics['recall']:.4f}")
        print(f"测试F1分数: {test_metrics['f1']:.4f}")
        
        # 记录到wandb
        wandb.log({
            **{f"test_{k}": v for k, v in test_metrics.items()}
        })
        
        # 保存模型
        model_path = os.path.join(args.model_dir, "bert_sentiment_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': vars(args)
        }, model_path)
        
        print(f"模型已保存到 {model_path}")
    
    else:  # 多任务学习
        # 创建多任务分类模型
        model = BertMultiTaskClassifier(
            bert_model_name=args.bert_model,
            num_topics=len(topic_list),
            num_sentiments=3  # -1, 0, 1 -> 0, 1, 2
        )
        model.to(device)
        
        # 定义损失函数
        topic_criterion = nn.BCEWithLogitsLoss()
        sentiment_criterion = nn.CrossEntropyLoss()
        
        # 定义优化器
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        
        # 计算总的训练步数
        total_steps = len(train_loader) * args.epochs
        
        # 创建学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # 训练多任务分类模型
        best_val_f1 = 0
        best_model_state = None
        
        for epoch in range(args.epochs):
            print(f"\n====== Epoch {epoch+1}/{args.epochs} ======")
            
            # 训练
            train_loss, train_topic_loss, train_sentiment_loss = train_multitask_classifier(
                model, train_loader, optimizer, scheduler, 
                topic_criterion, sentiment_criterion, device,
                topic_weight=args.topic_weight, 
                sentiment_weight=args.sentiment_weight
            )
            print(f"训练总损失: {train_loss:.4f}")
            print(f"训练主题损失: {train_topic_loss:.4f}")
            print(f"训练情感损失: {train_sentiment_loss:.4f}")
            
            # 验证
            val_metrics = evaluate_multitask_classifier(
                model, val_loader, topic_criterion, sentiment_criterion, 
                device, threshold=args.topic_threshold, topic_list=topic_list
            )
            
            print(f"验证主题损失: {val_metrics['topic_loss']:.4f}")
            print(f"验证情感损失: {val_metrics['sentiment_loss']:.4f}")
            print(f"验证主题F1分数(macro): {val_metrics['topic_f1_macro']:.4f}")
            print(f"验证情感准确率: {val_metrics['sentiment_accuracy']:.4f}")
            print(f"验证情感F1分数: {val_metrics['sentiment_f1']:.4f}")
            
            # 记录到wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_topic_loss": train_topic_loss,
                "train_sentiment_loss": train_sentiment_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })
            
            # 使用主题F1和情感F1的平均值作为模型选择指标
            val_combined_f1 = (val_metrics['topic_f1_macro'] + val_metrics['sentiment_f1']) / 2
            
            # 保存最佳模型
            if val_combined_f1 > best_val_f1:
                best_val_f1 = val_combined_f1
                best_model_state = model.state_dict().copy()
                print(f"保存新的最佳模型，组合F1分数: {val_combined_f1:.4f}")
        
        # 加载最佳模型状态
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 在测试集上评估
        print("\n====== 测试集评估 ======")
        test_metrics = evaluate_multitask_classifier(
            model, test_loader, topic_criterion, sentiment_criterion,
            device, threshold=args.topic_threshold, topic_list=topic_list
        )
        
        print(f"测试主题损失: {test_metrics['topic_loss']:.4f}")
        print(f"测试情感损失: {test_metrics['sentiment_loss']:.4f}")
        print(f"测试主题F1分数(macro): {test_metrics['topic_f1_macro']:.4f}")
        print(f"测试情感准确率: {test_metrics['sentiment_accuracy']:.4f}")
        print(f"测试情感精确率: {test_metrics['sentiment_precision']:.4f}")
        print(f"测试情感召回率: {test_metrics['sentiment_recall']:.4f}")
        print(f"测试情感F1分数: {test_metrics['sentiment_f1']:.4f}")
        
        # 记录到wandb
        wandb.log({
            **{f"test_{k}": v for k, v in test_metrics.items()}
        })
        
        # 保存模型
        model_path = os.path.join(args.model_dir, "bert_multitask_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'topic_list': topic_list,
            'topic_mapping': topic_mapping,
            'args': vars(args)
        }, model_path)
        
        print(f"模型已保存到 {model_path}")
    
    # 完成wandb运行
    wandb.finish()

if __name__ == "__main__":
    main() 