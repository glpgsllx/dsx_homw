import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer

class CarReviewDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=128):
        """
        汽车评论数据集
        
        Args:
            data_file: 数据文件路径
            tokenizer: BERT分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_file)
    
    def _load_data(self, data_file):
        """加载并预处理数据"""
        data_list = []
        
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
                for label in labels:
                    if '#' in label:
                        aspect, sentiment = label.split('#')
                        # 将情感值转换为数字: -1, 0, 1
                        sentiment = int(sentiment)
                        # 将-1,0,1映射为0,1,2便于分类
                        sentiment_idx = sentiment + 1
                        
                        data_list.append({
                            'text': text,
                            'aspect': aspect,
                            'sentiment': sentiment_idx
                        })
        
        return data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # BERT分词
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sentiment': torch.tensor(item['sentiment'], dtype=torch.long),
            'aspect': item['aspect']
        }

def get_aspect_mapping(train_file):
    """获取评论领域的映射字典"""
    aspects = set()
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                continue
                
            labels = parts[1].split()
            for label in labels:
                if '#' in label:
                    aspect, _ = label.split('#')
                    aspects.add(aspect)
    
    return {aspect: idx for idx, aspect in enumerate(sorted(aspects))}

def create_dataloaders(train_file, test_file, tokenizer, batch_size=16, max_length=128):
    """创建训练和测试数据加载器"""
    train_dataset = CarReviewDataset(train_file, tokenizer, max_length)
    test_dataset = CarReviewDataset(test_file, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader

if __name__ == "__main__":
    # 测试代码
    from transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained('./model')
    dataset = CarReviewDataset('data/train.txt', tokenizer)
    print(f"数据集大小: {len(dataset)}")
    print(f"样本示例: {dataset[0]}")
    
    # 测试数据加载器
    train_loader, test_loader = create_dataloaders(
        'data/train.txt', 'data/test.txt', tokenizer, batch_size=16
    )
    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}") 