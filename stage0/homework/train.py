import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score

from dataset import CarReviewDataset, create_dataloaders, get_aspect_mapping
from model import BertSentimentClassifier, BertAspectSentimentClassifier

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    # 使用tqdm显示进度条
    progress_bar = tqdm(dataloader, desc="训练")
    
    for batch in progress_bar:
        # 将数据移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['sentiment'].to(device)
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        
        # 剪裁梯度，避免梯度爆炸
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        
        # 更新学习率
        scheduler.step()
        
        # 累加损失
        total_loss += loss.item()
        
        # 更新进度条
        progress_bar.set_postfix({'loss': loss.item()})
    
    # 返回平均损失
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="验证"):
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['sentiment'].to(device)
            
            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 累加损失
            total_loss += loss.item()
            
            # 预测类别（取最大概率的类别）
            _, preds = torch.max(outputs, dim=1)
            
            # 收集预测和标签，用于计算指标
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    # 返回损失和指标
    return total_loss / len(dataloader), accuracy, precision, recall, f1, all_preds, all_labels

def train_aspect_aware_model(train_dataloader, val_dataloader, model, criterion, optimizer, scheduler, device, num_epochs, aspect_mapping):
    """训练具有领域感知的模型"""
    model.train()
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f"\n====== Epoch {epoch+1}/{num_epochs} ======")
        
        # 训练
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc="训练")
        
        for batch in progress_bar:
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['sentiment'].to(device)
            aspects = batch['aspect']
            
            # 将领域转换为ID
            aspect_ids = torch.tensor([aspect_mapping[aspect] for aspect in aspects], dtype=torch.long).to(device)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, aspect_ids=aspect_ids)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 剪裁梯度
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            
            # 更新学习率
            scheduler.step()
            
            # 累加损失
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
        
        # 计算平均训练损失
        avg_train_loss = total_loss / len(train_dataloader)
        
        # 验证
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="验证"):
                # 将数据移动到设备
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['sentiment'].to(device)
                aspects = batch['aspect']
                
                # 将领域转换为ID
                aspect_ids = torch.tensor([aspect_mapping[aspect] for aspect in aspects], dtype=torch.long).to(device)
                
                # 前向传播
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, aspect_ids=aspect_ids)
                
                # 计算损失
                loss = criterion(outputs, targets)
                
                # 累加损失
                val_loss += loss.item()
                
                # 预测类别
                _, preds = torch.max(outputs, dim=1)
                
                # 收集预测和标签
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
        
        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_dataloader)
        
        # 计算评估指标
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds, average='macro')
        val_recall = recall_score(all_labels, all_preds, average='macro')
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # 记录到wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1
        })
        
        # 打印结果
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"训练损失: {avg_train_loss:.4f}")
        print(f"验证损失: {avg_val_loss:.4f}")
        print(f"验证准确率: {val_accuracy:.4f}")
        print(f"验证精确率: {val_precision:.4f}")
        print(f"验证召回率: {val_recall:.4f}")
        print(f"验证F1分数: {val_f1:.4f}")
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            print(f"保存新的最佳模型，F1分数: {val_f1:.4f}")
    
    # 加载最佳模型状态
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def main():
    parser = argparse.ArgumentParser(description="BERT情感分类训练脚本")
    parser.add_argument("--train_file", default="data/train.txt", help="训练数据文件路径")
    parser.add_argument("--test_file", default="data/test.txt", help="测试数据文件路径")
    parser.add_argument("--model_dir", default="./model_new", help="模型保存目录")
    parser.add_argument("--bert_model", default="./model", help="BERT预训练模型名称")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--use_aspect", action="store_true", help="是否使用领域感知模型")
    parser.add_argument("--aspect_embedding_dim", type=int, default=50, help="领域嵌入维度")
    parser.add_argument("--wandb_project", default="bert-sentiment-classification", help="wandb项目名称")
    parser.add_argument("--wandb_entity", default=None, help="wandb实体名称")
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
            "use_aspect": args.use_aspect,
            "aspect_embedding_dim": args.aspect_embedding_dim if args.use_aspect else None,
            "model_type": "BertAspectSentimentClassifier" if args.use_aspect else "BertSentimentClassifier"
        }
    )
    
    # 创建模型保存目录
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 检查是否有GPU可用
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        # torch.mps.empty_cache()
        print("使用MPS设备(M4)")
    else:
        device = torch.device("cpu")
        print("使用CPU设备")
    
    # 加载BERT分词器
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    
    # 创建数据加载器
    train_loader, test_loader = create_dataloaders(
        args.train_file, args.test_file, tokenizer,
        batch_size=args.batch_size, max_length=args.max_length
    )
    
    # 获取验证集（从训练集中分割）
    train_size = int(0.9 * len(train_loader.dataset))
    val_size = len(train_loader.dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_loader.dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    if args.use_aspect:
        # 获取领域映射
        aspect_mapping = get_aspect_mapping(args.train_file)
        print(f"识别到的领域: {aspect_mapping}")
        
        # 创建领域感知模型
        model = BertAspectSentimentClassifier(
            bert_model_name=args.bert_model,
            num_classes=3,  # -1, 0, 1 -> 0, 1, 2
            num_aspects=len(aspect_mapping),
            aspect_embedding_dim=args.aspect_embedding_dim
        )
    else:
        # 创建基本情感分类模型
        model = BertSentimentClassifier(
            bert_model_name=args.bert_model,
            num_classes=3  # -1, 0, 1 -> 0, 1, 2
        )
    
    # 将模型移动到设备
    model.to(device)
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 计算总的训练步数
    total_steps = len(train_loader) * args.epochs
    
    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # 默认不进行预热
        num_training_steps=total_steps
    )
    
    # 训练模型
    if args.use_aspect:
        model = train_aspect_aware_model(
            train_loader, val_loader, model, criterion, optimizer, scheduler,
            device, args.epochs, aspect_mapping
        )
    else:
        best_val_f1 = 0
        best_model_state = None
        
        for epoch in range(args.epochs):
            print(f"\n====== Epoch {epoch+1}/{args.epochs} ======")
            
            # 训练
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
            print(f"训练损失: {train_loss:.4f}")
            
            # 验证
            test_loss, test_accuracy, test_precision, test_recall, test_f1, test_preds, test_labels = evaluate(model, val_loader, criterion, device)
            print(f"验证损失: {test_loss:.4f}")
            print(f"验证准确率: {test_accuracy:.4f}")
            print(f"验证精确率: {test_precision:.4f}")
            print(f"验证召回率: {test_recall:.4f}")
            print(f"验证F1分数: {test_f1:.4f}")
            
            # 记录到wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": test_loss,
                "val_accuracy": test_accuracy,
                "val_precision": test_precision,
                "val_recall": test_recall,
                "val_f1": test_f1
            })
            
            # 保存最佳模型
            if test_f1 > best_val_f1:
                best_val_f1 = test_f1
                best_model_state = model.state_dict().copy()
                print(f"保存新的最佳模型，F1分数: {test_f1:.4f}")
        
        # 加载最佳模型状态
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
    
    # 在测试集上评估
    print("\n====== 测试集评估 ======")
    if args.use_aspect:
        # 领域感知模型的测试评估
        model.eval()
        test_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="测试"):
                # 将数据移动到设备
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['sentiment'].to(device)
                aspects = batch['aspect']
                
                # 将领域转换为ID
                aspect_ids = torch.tensor([aspect_mapping[aspect] for aspect in aspects], dtype=torch.long).to(device)
                
                # 前向传播
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, aspect_ids=aspect_ids)
                
                # 计算损失
                loss = criterion(outputs, targets)
                
                # 累加损失
                test_loss += loss.item()
                
                # 预测类别
                _, preds = torch.max(outputs, dim=1)
                
                # 收集预测和标签
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
        
        # 计算平均测试损失
        avg_test_loss = test_loss / len(test_loader)
        
        # 计算评估指标
        test_accuracy = accuracy_score(all_labels, all_preds)
        test_precision = precision_score(all_labels, all_preds, average='macro')
        test_recall = recall_score(all_labels, all_preds, average='macro')
        test_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # 记录到wandb
        wandb.log({
            "test_loss": avg_test_loss,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        })
        
        print(f"测试损失: {avg_test_loss:.4f}")
        print(f"测试准确率: {test_accuracy:.4f}")
        print(f"测试精确率: {test_precision:.4f}")
        print(f"测试召回率: {test_recall:.4f}")
        print(f"测试F1分数: {test_f1:.4f}")
        
        # 打印详细的分类报告
        print("\n分类报告:")
        print(classification_report(all_labels, all_preds, digits=4))
    else:
        # 基本模型的测试评估
        test_loss, test_accuracy, test_precision, test_recall, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
        
        # 记录到wandb
        wandb.log({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        })
        
        print(f"测试损失: {test_loss:.4f}")
        print(f"测试准确率: {test_accuracy:.4f}")
        print(f"测试精确率: {test_precision:.4f}")
        print(f"测试召回率: {test_recall:.4f}")
        print(f"测试F1分数: {test_f1:.4f}")
        
        # 打印详细的分类报告
        print("\n分类报告:")
        print(classification_report(test_labels, test_preds, digits=4))
    
    # 创建混淆矩阵
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        preds=all_preds,
        y_true=all_labels,
        class_names=["负面", "中性", "正面"]
    )})
    
    # 保存模型
    model_path = os.path.join(args.model_dir, "bert_sentiment_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)
    }, model_path)
    
    # 完成wandb运行
    wandb.finish()
    
    print(f"模型已保存到 {model_path}")
    
if __name__ == "__main__":
    main() 