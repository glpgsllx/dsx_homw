import argparse
import torch
import os
import json
from transformers import BertTokenizer
from model import BertTopicClassifier, BertSentimentClassifier, BertMultiTaskClassifier

def load_model(model_path, model_type="multitask"):
    """加载训练好的模型"""
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    
    # 加载模型参数
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    args = checkpoint['args']
    
    # 根据模型类型创建对应的模型
    if model_type == "topic":
        topic_list = checkpoint.get('topic_list', [])
        model = BertTopicClassifier(
            bert_model_name=args['bert_model'],
            num_topics=len(topic_list)
        )
        return model, checkpoint, topic_list
    elif model_type == "sentiment":
        model = BertSentimentClassifier(
            bert_model_name=args['bert_model'],
            num_classes=3  # -1, 0, 1 -> 0, 1, 2
        )
        return model, checkpoint, None
    else:  # 多任务模型
        topic_list = checkpoint.get('topic_list', [])
        model = BertMultiTaskClassifier(
            bert_model_name=args['bert_model'],
            num_topics=len(topic_list),
            num_sentiments=3  # -1, 0, 1 -> 0, 1, 2
        )
        return model, checkpoint, topic_list

def predict_topics(text, model, tokenizer, topic_list, max_length=128, threshold=0.5, device='cpu'):
    """预测文本的主题"""
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # 使用sigmoid激活函数获取概率
        probs = torch.sigmoid(outputs)
        # 根据阈值获取预测
        predictions = (probs > threshold).float().cpu().numpy()[0]
    
    # 获取预测主题
    predicted_topics = []
    topic_probs = {}
    
    for i, pred in enumerate(predictions):
        if pred == 1:
            if i < len(topic_list):
                topic = topic_list[i]
                predicted_topics.append(topic)
                # 记录概率值
                topic_probs[topic] = float(probs[0][i].cpu().numpy())
    
    return predicted_topics, topic_probs

def predict_sentiment(text, model, tokenizer, max_length=128, device='cpu'):
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 获取预测结果
        _, predicted = torch.max(outputs, dim=1)
        sentiment_idx = predicted.item()
        
        # 将预测结果转换为情感极性 (-1, 0, 1)
        sentiment = sentiment_idx - 1
    
    return sentiment

def predict_multitask(text, model, tokenizer, topic_list, max_length=128, threshold=0.5, device='cpu'):
    """预测文本的主题和情感"""
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
        topic_outputs, sentiment_outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # 主题预测
        topic_probs = torch.sigmoid(topic_outputs)
        topic_predictions = (topic_probs > threshold).float().cpu().numpy()[0]
        
        # 情感预测
        _, sentiment_predicted = torch.max(sentiment_outputs, dim=1)
        sentiment_idx = sentiment_predicted.item()
        sentiment = sentiment_idx - 1  # 转回 -1, 0, 1
    
    # 获取预测主题
    predicted_topics = []
    topic_probs_dict = {}
    
    for i, pred in enumerate(topic_predictions):
        if pred == 1:
            if i < len(topic_list):
                topic = topic_list[i]
                predicted_topics.append(topic)
                # 记录概率值
                topic_probs_dict[topic] = float(topic_probs[0][i].cpu().numpy())
    
    return predicted_topics, topic_probs_dict, sentiment

def process_text(text, model_type, model, tokenizer, topic_list=None, max_length=128, threshold=0.5, device='cpu'):
    """根据模型类型处理文本"""
    if model_type == "topic":
        predicted_topics, topic_probs = predict_topics(
            text, model, tokenizer, topic_list, max_length, threshold, device
        )
        return {
            'text': text,
            'predicted_topics': predicted_topics,
            'topic_probabilities': topic_probs
        }
    elif model_type == "sentiment":
        sentiment = predict_sentiment(text, model, tokenizer, max_length, device)
        return {
            'text': text,
            'sentiment': sentiment,
            'sentiment_text': '负面' if sentiment == -1 else '中性' if sentiment == 0 else '正面'
        }
    else:  # 多任务模型
        predicted_topics, topic_probs, sentiment = predict_multitask(
            text, model, tokenizer, topic_list, max_length, threshold, device
        )
        return {
            'text': text,
            'predicted_topics': predicted_topics,
            'topic_probabilities': topic_probs,
            'sentiment': sentiment,
            'sentiment_text': '负面' if sentiment == -1 else '中性' if sentiment == 0 else '正面'
        }

def batch_predict_from_file(input_file, output_file, model_type, model, tokenizer, topic_list=None, max_length=128, threshold=0.5, device='cpu'):
    """从文件批量预测"""
    results = []
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 处理每一行
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 解析输入（只使用文本部分）
        parts = line.split('\t')
        text = parts[0]
        
        # 预测
        result = process_text(
            text, model_type, model, tokenizer, 
            topic_list, max_length, threshold, device
        )
        results.append(result)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"批量预测结果已保存到 {output_file}")

def main():
    parser = argparse.ArgumentParser(description="BERT汽车评论分类预测脚本")
    parser.add_argument("--model_path", default="saved_models/bert_multitask_model.pth", help="模型文件路径")
    parser.add_argument("--model_type", choices=["topic", "sentiment", "multitask"], default="multitask", help="模型类型")
    parser.add_argument("--input", help="输入文本或文件")
    parser.add_argument("--output", default="predictions.json", help="预测结果输出文件")
    parser.add_argument("--is_file", action="store_true", help="输入是否为文件")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--threshold", type=float, default=0.5, help="主题预测阈值")
    args = parser.parse_args()
    
    # 检查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    model, checkpoint, topic_list = load_model(args.model_path, args.model_type)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载分词器
    bert_model_name = checkpoint['args']['bert_model']
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    if args.model_type in ["topic", "multitask"] and topic_list:
        print(f"可识别的主题: {topic_list}")
    
    if args.is_file:
        # 批量预测
        if not args.input:
            raise ValueError("需要提供输入文件路径")
        
        print(f"从文件 {args.input} 批量预测...")
        batch_predict_from_file(
            args.input, args.output, args.model_type, model, tokenizer,
            topic_list, args.max_length, args.threshold, device
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
                
                # 预测
                result = process_text(
                    text, args.model_type, model, tokenizer, 
                    topic_list, args.max_length, args.threshold, device
                )
                
                # 打印结果
                print("\n预测结果:")
                if args.model_type in ["topic", "multitask"]:
                    if result['predicted_topics']:
                        print("检测到的主题:")
                        for topic in result['predicted_topics']:
                            print(f"- {topic} (概率: {result['topic_probabilities'][topic]:.4f})")
                    else:
                        print("未检测到明确的主题")
                
                if args.model_type in ["sentiment", "multitask"]:
                    print(f"情感极性: {result['sentiment_text']} ({result['sentiment']})")
                print()
        else:
            # 单个文本，非交互式
            text = args.input
            
            # 预测
            result = process_text(
                text, args.model_type, model, tokenizer, 
                topic_list, args.max_length, args.threshold, device
            )
            
            # 打印结果
            print("\n预测结果:")
            if args.model_type in ["topic", "multitask"]:
                if result['predicted_topics']:
                    print("检测到的主题:")
                    for topic in result['predicted_topics']:
                        print(f"- {topic} (概率: {result['topic_probabilities'][topic]:.4f})")
                else:
                    print("未检测到明确的主题")
            
            if args.model_type in ["sentiment", "multitask"]:
                print(f"情感极性: {result['sentiment_text']} ({result['sentiment']})")
            
            # 保存结果
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到 {args.output}")

if __name__ == "__main__":
    main() 