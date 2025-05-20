import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    LSTM tabanlı metin sınıflandırma modeli
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()
        
        # Embedding katmanı
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # LSTM katmanı
        self.lstm = nn.LSTM(embedding_dim, 
                          hidden_dim, 
                          num_layers=n_layers, 
                          bidirectional=bidirectional, 
                          dropout=dropout if n_layers > 1 else 0,
                          batch_first=True)
        
        # Doğrusal çıkış katmanı
        # Bidirectional ise hidden_dim'i 2 ile çarpıyoruz
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch size, seq len]
        
        # embeddings üret
        embedded = self.embedding(input_ids)
        # embedded: [batch size, seq len, embedding dim]
        
        # Eğer attention mask varsa, sadece geçerli token'ların çıktılarını al
        if attention_mask is not None:
            # LSTM çıktılarını al
            outputs, (hidden, cell) = self.lstm(embedded)
        else:
            # Attention mask yoksa normal LSTM çıktılarını al
            outputs, (hidden, cell) = self.lstm(embedded)
        
        # Son hidden state'i al
        # Eğer bidirectional ise son hidden state'in her iki yönünü birleştir
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # hidden: [batch size, hidden dim]
        
        # Dropout uygula
        hidden = self.dropout(hidden)
        
        # Doğrusal katmanı geçir
        output = self.fc(hidden)
        # output: [batch size, output dim]
        
        return output