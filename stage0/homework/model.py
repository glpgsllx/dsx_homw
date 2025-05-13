import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BertSentimentClassifier(nn.Module):
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
        
class BertAspectSentimentClassifier(nn.Module):
    """具有领域特征的BERT情感分类模型"""
    def __init__(self, bert_model_name='./model', num_classes=3, 
                 num_aspects=10, aspect_embedding_dim=50, dropout_rate=0.1):
        super(BertAspectSentimentClassifier, self).__init__()
        
        # 加载BERT预训练模型
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # 获取BERT隐藏层大小
        self.hidden_size = self.bert.config.hidden_size
        
        # 评论领域的嵌入层
        self.aspect_embeddings = nn.Embedding(num_aspects, aspect_embedding_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size + aspect_embedding_dim, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, aspect_ids):
        """
        前向传播
        
        Args:
            input_ids: 输入序列的token IDs
            attention_mask: 注意力掩码
            aspect_ids: 评论领域ID
            
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
        
        # 获取领域嵌入
        aspect_embeddings = self.aspect_embeddings(aspect_ids)
        
        # 拼接BERT表示和领域嵌入
        combined_features = torch.cat([pooled_output, aspect_embeddings], dim=1)
        
        # 通过分类器
        logits = self.classifier(combined_features)
        
        return logits 