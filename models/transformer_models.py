import torch
import torch.nn as nn
from transformers import (
    BertModel, 
    DistilBertModel, 
    RobertaModel
)

class BERTClassifier(nn.Module):
    """
    BERT tabanlı metin sınıflandırma modeli
    """
    def __init__(self, model_name, num_classes, dropout=0.1):
        super().__init__()
        
        # BERT modelini yükle
        self.bert = BertModel.from_pretrained(model_name)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Sınıflandırma katmanı
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # BERT çıktılarını al
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # [CLS] token'ının çıktısını al
        pooled_output = outputs.pooler_output
        
        # Dropout uygula
        pooled_output = self.dropout(pooled_output)
        
        # Sınıflandırma
        logits = self.classifier(pooled_output)
        
        return logits

class DistilBERTClassifier(nn.Module):
    """
    DistilBERT tabanlı metin sınıflandırma modeli
    """
    def __init__(self, model_name, num_classes, dropout=0.1):
        super().__init__()
        
        # DistilBERT modelini yükle
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Sınıflandırma katmanı
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # DistilBERT çıktılarını al
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Son hidden state'in [CLS] token'ını al
        hidden_state = outputs.last_hidden_state[:, 0]
        
        # Dropout uygula
        hidden_state = self.dropout(hidden_state)
        
        # Sınıflandırma
        logits = self.classifier(hidden_state)
        
        return logits

class RoBERTaClassifier(nn.Module):
    """
    RoBERTa tabanlı metin sınıflandırma modeli
    """
    def __init__(self, model_name, num_classes, dropout=0.1):
        super().__init__()
        
        # RoBERTa modelini yükle
        self.roberta = RobertaModel.from_pretrained(model_name)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Sınıflandırma katmanı
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # RoBERTa çıktılarını al
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Pooler çıktısını al
        # Not: RoBERTa'da pooler_output farklı olabilir, son hidden state'in ilk token'ını kullanabiliriz
        hidden_state = outputs.last_hidden_state[:, 0]
        
        # Dropout uygula
        hidden_state = self.dropout(hidden_state)
        
        # Sınıflandırma
        logits = self.classifier(hidden_state)
        
        return logits

class FastTextClassifier(nn.Module):
    """
    FastText benzeri basit bir derin öğrenme modeli
    
    Not: Bu gerçek FastText değil, FastText'in basit bir benzeri
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout, pad_idx):
        super().__init__()
        
        # Embedding katmanı
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Gizli katman
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        
        # Çıkış katmanı
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Aktivasyon
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch size, seq len]
        
        # Embeddings
        embedded = self.embedding(input_ids)
        # embedded: [batch size, seq len, embedding dim]
        
        # Ortalama pooling (attention mask kullanarak)
        if attention_mask is not None:
            # attention_mask'i genişlet
            mask = attention_mask.unsqueeze(-1).float()
            
            # Maskelenmiş ortalama
            embedded = embedded * mask
            
            # Toplam
            pooled = embedded.sum(dim=1)
            
            # Geçerli token sayısı
            count = mask.sum(dim=1)
            
            # Ortalama (sıfıra bölünmeyi önle)
            pooled = pooled / (count + 1e-9)
        else:
            # Basit ortalama
            pooled = embedded.mean(dim=1)
        
        # pooled: [batch size, embedding dim]
        
        # İlk doğrusal katman
        hidden = self.fc1(pooled)
        
        # Aktivasyon
        hidden = self.relu(hidden)
        
        # Dropout
        hidden = self.dropout(hidden)
        
        # Son doğrusal katman
        output = self.fc2(hidden)
        
        return output