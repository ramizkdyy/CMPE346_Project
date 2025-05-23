import torch
import torch.nn as nn
from transformers import (
    BertModel, 
    DistilBertModel, 
    RobertaModel
)

class BERTClassifier(nn.Module):
    """
    BERT based text classification model
    """
    def __init__(self, model_name, num_classes, dropout=0.1):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(
            model_name,
            torchscript=True,  
            return_dict=False
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs[1]
        
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        
        return logits

class DistilBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.1):
        super().__init__()
        
        self.distilbert = DistilBertModel.from_pretrained(
            model_name,
            torchscript=True,
            return_dict=False
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_state = outputs[0][:, 0]
        
        hidden_state = self.dropout(hidden_state)
        
        logits = self.classifier(hidden_state)
        
        return logits

class RoBERTaClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.1):
        super().__init__()
        
        self.roberta = RobertaModel.from_pretrained(
            model_name,
            torchscript=True,
            return_dict=False
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_state = outputs[0][:, 0]
        
        hidden_state = self.dropout(hidden_state)
        
        logits = self.classifier(hidden_state)
        
        return logits

class FastTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout, pad_idx):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.relu = nn.ReLU()
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            
            embedded = embedded * mask
            
            pooled = embedded.sum(dim=1)
            
            count = mask.sum(dim=1)
            
            pooled = pooled / (count + 1e-8)
        else:
            pooled = embedded.mean(dim=1)
        
        
        hidden = self.fc1(pooled)
        
        hidden = self.relu(hidden)
        
        # Dropout
        hidden = self.dropout(hidden)
        
        output = self.fc2(hidden)
        
        return output