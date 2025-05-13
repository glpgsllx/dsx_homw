import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import numpy as np

class CarReviewDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=128, topic_mapping=None):
        """
        汽车评论数据集
        
        Args:
            data_file: 数据文件路径
            tokenizer: BERT分词器
            max_length: 最大序列长度
            topic_mapping: 主题到ID的映射字典
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data, self.all_topics, self.topic_mapping = self._load_data(data_file, topic_mapping)
    
    def _load_data(self, data_file, provided_topic_mapping=None):
        """加载并预处理数据"""
        data_list = []
        all_topics = set()
        topic_counts = {}  # 添加主题计数字典
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 分离文本和标签
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                    
                text = parts[0]
                
                # 处理标签
                labels = parts[1].split()
                topics = []
                sentiments = []
                
                for label in labels:
                    if '#' in label:
                        topic, sentiment = label.split('#')
                        topics.append(topic)
                        all_topics.add(topic)
                        # 统计主题出现次数
                        topic_counts[topic] = topic_counts.get(topic, 0) + 1
                        
                        # 将情感值转换为数字: -1, 0, 1
                        sentiment_value = int(sentiment)
                        # 映射到0, 1, 2用于分类
                        sentiment_idx = sentiment_value + 1
                        sentiments.append(sentiment_idx)
                
                data_list.append({
                    'text': text,
                    'topics': topics,
                    'sentiments': sentiments
                })
        
        # 确定主题映射
        if provided_topic_mapping:
            topic_mapping = provided_topic_mapping
        else:
            all_topics = sorted(list(all_topics))
            topic_mapping = {topic: idx for idx, topic in enumerate(all_topics)}
        
        # 打印主题分布统计
        print("\n数据集主题分布统计:")
        total_samples = len(data_list)
        for topic in topic_mapping.keys():
            count = topic_counts.get(topic, 0)
            percentage = (count / total_samples) * 100
            print(f"{topic}: {count} 样本 ({percentage:.2f}%)")
        
        return data_list, sorted(list(all_topics)), topic_mapping
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        topics = item['topics']
        sentiments = item['sentiments']
        
        # BERT分词
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 创建主题的多热向量
        topic_labels = torch.zeros(len(self.topic_mapping), dtype=torch.float)
        for topic in topics:
            if topic in self.topic_mapping:
                topic_idx = self.topic_mapping[topic]
                topic_labels[topic_idx] = 1.0
        
        # 创建情感标签 - 如果有多个主题，取情感的平均值
        # 如果sum>0为正向，sum<0为负向，sum=0为中性
        if len(sentiments) > 0:
            raw_sentiment_values = [s - 1 for s in sentiments]  # 转回-1, 0, 1
            avg_sentiment = np.mean(raw_sentiment_values)
            if avg_sentiment > 0:
                sentiment_label = 2  # 正向
            elif avg_sentiment < 0:
                sentiment_label = 0  # 负向
            else:
                sentiment_label = 1  # 中性
        else:
            sentiment_label = 1  # 默认中性
        
        # 初始化包含所有主题的情感映射，默认为中性(1)
        topic_sentiment_map = {topic: 1 for topic in self.topic_mapping.keys()}
        
        # 更新实际出现的主题的情感
        for i, topic in enumerate(topics):
            if i < len(sentiments) and topic in self.topic_mapping:
                topic_sentiment_map[topic] = sentiments[i]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'topic_labels': topic_labels,
            'sentiment_label': torch.tensor(sentiment_label, dtype=torch.long),
            'topic_sentiment_map': topic_sentiment_map,
            'text': text,
            'topics': topics
        }
    
    def get_topics_list(self):
        """返回所有主题的列表"""
        return self.all_topics
    
    def get_topic_mapping(self):
        """返回主题到ID的映射"""
        return self.topic_mapping

def get_predefined_topics():
    """获取预定义的10个主题"""
    return ["动力", "价格", "内饰", "配置", "安全性", "外观", "操控", "油耗", "空间", "舒适性"]

def get_topic_mapping(predefined=True):
    """获取主题映射字典"""
    if predefined:
        topics = get_predefined_topics()
        return {topic: idx for idx, topic in enumerate(topics)}
    return {}

def create_dataloaders(train_file, test_file, tokenizer, batch_size=16, max_length=128):
    """创建训练和测试数据加载器"""
    topic_mapping = get_topic_mapping(predefined=True)
    
    train_dataset = CarReviewDataset(train_file, tokenizer, max_length, topic_mapping)
    test_dataset = CarReviewDataset(test_file, tokenizer, max_length, topic_mapping)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader, topic_mapping 