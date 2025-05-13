import torch
import torch.nn as nn
from transformers import BertModel

class BertTopicClassifier(nn.Module):
    """BERT主题分类模型，用于识别评论的主题（多标签分类）"""
    def __init__(self, bert_model_name='./model', num_topics=10, dropout_rate=0.1):
        """
        基于BERT的主题分类模型
        
        Args:
            bert_model_name: BERT预训练模型名称
            num_topics: 主题数量
            dropout_rate: Dropout比率
        """
        super(BertTopicClassifier, self).__init__()
        
        # 加载BERT预训练模型
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # 获取BERT隐藏层大小
        self.hidden_size = self.bert.config.hidden_size
        
        # 分类器 - 多标签分类不使用softmax，而是对每个类别独立使用sigmoid
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size // 2, num_topics)
        )
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        
        Args:
            input_ids: 输入序列的token IDs
            attention_mask: 注意力掩码
            
        Returns:
            logits: 各个主题的logits值
        """
        # 获取BERT的输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用[CLS]标记的最后隐藏状态作为句子表示
        pooled_output = outputs.pooler_output
        
        # 通过分类器
        logits = self.classifier(pooled_output)
        
        return logits

class BertSentimentClassifier(nn.Module):
    """基于BERT的情感分类模型"""
    def __init__(self, bert_model_name='./model', num_classes=3, dropout_rate=0.1):
        """
        基于BERT的情感分类模型
        
        Args:
            bert_model_name: BERT预训练模型名称
            num_classes: 分类数量（-1,0,1对应的0,1,2三类）
            dropout_rate: Dropout比率
        """
        super(BertSentimentClassifier, self).__init__()
        
        # 加载BERT预训练模型
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # 获取BERT隐藏层大小
        self.hidden_size = self.bert.config.hidden_size
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        
        Args:
            input_ids: 输入序列的token IDs
            attention_mask: 注意力掩码
            
        Returns:
            logits: 分类logits
        """
        # 获取BERT的输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用[CLS]标记的最后隐藏状态作为句子表示
        pooled_output = outputs.pooler_output
        
        # 通过分类器
        logits = self.classifier(pooled_output)
        
        return logits

class BertMultiTaskClassifier(nn.Module):
    """BERT多任务分类模型，同时进行主题识别和情感分类"""
    def __init__(self, bert_model_name='./model', num_topics=10, num_sentiments=3, dropout_rate=0.1):
        """
        基于BERT的多任务分类模型
        
        Args:
            bert_model_name: BERT预训练模型名称
            num_topics: 主题数量
            num_sentiments: 情感类别数量
            dropout_rate: Dropout比率
        """
        super(BertMultiTaskClassifier, self).__init__()
        
        # 加载BERT预训练模型
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # 获取BERT隐藏层大小
        self.hidden_size = self.bert.config.hidden_size
        
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )
        
        # 主题分类器 - 多标签分类
        self.topic_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size // 2, num_topics)
        )
        
        # 情感分类器 - 单标签多分类
        self.sentiment_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size // 2, num_sentiments)
        )
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        
        Args:
            input_ids: 输入序列的token IDs
            attention_mask: 注意力掩码
            
        Returns:
            topic_logits: 主题分类的logits
            sentiment_logits: 情感分类的logits
        """
        # 获取BERT的输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用[CLS]标记的最后隐藏状态作为句子表示
        pooled_output = outputs.pooler_output
        
        # 通过共享特征层
        shared_features = self.shared(pooled_output)
        
        # 主题分类
        topic_logits = self.topic_classifier(shared_features)
        
        # 情感分类
        sentiment_logits = self.sentiment_classifier(shared_features)
        
        return topic_logits, sentiment_logits 