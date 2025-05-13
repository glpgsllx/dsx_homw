import argparse
import torch
import os
import json
from transformers import BertTokenizer
from model import BertSentimentClassifier, BertAspectSentimentClassifier
from dataset import get_aspect_mapping

def load_model(model_path):
    """加载训练好的模型"""
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    
    # 加载模型参数
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    args = checkpoint['args']
    
    # 创建模型
    if args.get('use_aspect', False):
        # 加载领域映射
        aspect_mapping = get_aspect_mapping(args['train_file'])
        model = BertAspectSentimentClassifier(
            bert_model_name=args['bert_model'],
            num_classes=3,  # -1, 0, 1 -> 0, 1, 2
            num_aspects=len(aspect_mapping),
            aspect_embedding_dim=args.get('aspect_embedding_dim', 50)
        )
        return model, checkpoint, aspect_mapping
    else:
        model = BertSentimentClassifier(
            bert_model_name=args['bert_model'],
            num_classes=3  # -1, 0, 1 -> 0, 1, 2
        )
        return model, checkpoint, None

def predict_sentiment(text, aspect, model, tokenizer, aspect_mapping=None, max_length=128, device='cpu'):
    """预测文本的情感极性"""
    # 预处理文本
    encoded_input = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # 将输入数据移动到设备
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        if aspect_mapping is not None:
            # 领域感知模型
            if aspect in aspect_mapping:
                aspect_id = torch.tensor([aspect_mapping[aspect]], dtype=torch.long).to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, aspect_ids=aspect_id)
            else:
                # 找不到对应的领域，使用一个默认值
                print(f"警告: 找不到领域 '{aspect}'，使用默认领域")
                aspect_id = torch.tensor([0], dtype=torch.long).to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, aspect_ids=aspect_id)
        else:
            # 基本模型
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # 获取预测结果
    _, predicted = torch.max(outputs, dim=1)
    sentiment_idx = predicted.item()
    
    # 将预测结果转换为情感极性 (-1, 0, 1)
    sentiment = sentiment_idx - 1
    
    return sentiment

def batch_predict_from_file(input_file, output_file, model, tokenizer, aspect_mapping=None, max_length=128, device='cpu'):
    """从文件批量预测情感"""
    results = []
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 处理每一行
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 解析输入
        parts = line.split('\t')
        if len(parts) < 2:
            continue
            
        text = parts[0]
        
        # 处理标签（如果存在）
        labels = parts[1].split() if len(parts) > 1 else []
        
        line_results = {'text': text, 'predictions': []}
        
        # 预测每个领域的情感
        if labels:
            for label in labels:
                if '#' in label:
                    aspect, _ = label.split('#')
                    sentiment = predict_sentiment(
                        text, aspect, model, tokenizer, aspect_mapping, max_length, device
                    )
                    line_results['predictions'].append({
                        'aspect': aspect,
                        'sentiment': sentiment
                    })
        
        results.append(line_results)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"批量预测结果已保存到 {output_file}")

def predict_single_text(text, aspects, model, tokenizer, aspect_mapping=None, max_length=128, device='cpu'):
    """预测单个文本的情感极性"""
    results = {'text': text, 'predictions': []}
    
    # 预测每个领域的情感
    for aspect in aspects:
        sentiment = predict_sentiment(
            text, aspect, model, tokenizer, aspect_mapping, max_length, device
        )
        results['predictions'].append({
            'aspect': aspect,
            'sentiment': sentiment
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="BERT情感分类预测脚本")
    parser.add_argument("--model_path", default="models/bert_sentiment_model.pth", help="模型文件路径")
    parser.add_argument("--input", help="输入文本或文件")
    parser.add_argument("--output", default="predictions.json", help="预测结果输出文件")
    parser.add_argument("--aspects", nargs='+', help="评论领域列表")
    parser.add_argument("--is_file", action="store_true", help="输入是否为文件")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    args = parser.parse_args()
    
    # 检查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    model, checkpoint, aspect_mapping = load_model(args.model_path)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载分词器
    bert_model_name = checkpoint['args']['bert_model']
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    if args.is_file:
        # 批量预测
        if not args.input:
            raise ValueError("需要提供输入文件路径")
        
        print(f"从文件 {args.input} 批量预测...")
        batch_predict_from_file(
            args.input, args.output, model, tokenizer,
            aspect_mapping, args.max_length, device
        )
    else:
        # 单个文本预测
        if not args.input:
            # 交互式预测
            print("进入交互式预测模式:")
            
            while True:
                text = input("请输入文本 (输入'quit'退出): ")
                if text.lower() == 'quit':
                    break
                
                # 获取领域列表
                if args.aspects:
                    aspects = args.aspects
                elif aspect_mapping:
                    aspects = list(aspect_mapping.keys())
                else:
                    aspects = ["价格", "配置", "动力", "操控", "油耗", "舒适性", "内饰", "安全性"]
                
                # 预测
                results = predict_single_text(
                    text, aspects, model, tokenizer,
                    aspect_mapping, args.max_length, device
                )
                
                # 打印结果
                print("\n预测结果:")
                for pred in results['predictions']:
                    sentiment_text = "负面" if pred['sentiment'] == -1 else "中性" if pred['sentiment'] == 0 else "正面"
                    print(f"领域: {pred['aspect']}, 情感: {sentiment_text} ({pred['sentiment']})")
                print()
        else:
            # 单个文本，非交互式
            text = args.input
            
            # 获取领域列表
            if args.aspects:
                aspects = args.aspects
            elif aspect_mapping:
                aspects = list(aspect_mapping.keys())
            else:
                aspects = ["价格", "配置", "动力", "操控", "油耗", "舒适性", "内饰", "安全性"]
            
            # 预测
            results = predict_single_text(
                text, aspects, model, tokenizer,
                aspect_mapping, args.max_length, device
            )
            
            # 打印结果
            print("\n预测结果:")
            for pred in results['predictions']:
                sentiment_text = "负面" if pred['sentiment'] == -1 else "中性" if pred['sentiment'] == 0 else "正面"
                print(f"领域: {pred['aspect']}, 情感: {sentiment_text} ({pred['sentiment']})")
            
            # 保存结果
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到 {args.output}")

if __name__ == "__main__":
    main() 