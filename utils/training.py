import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = criterion(outputs, labels)
            
            epoch_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    
    num_classes = len(np.unique(all_labels))
    
    f1 = f1_score(all_labels, all_preds, 
                  average='weighted' if num_classes > 2 else 'binary')
    
    return epoch_loss / len(dataloader), accuracy, f1

def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, device, 
                num_epochs=2, patience=1, model_save_path=None):
    train_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []
    
    best_val_f1 = 0
    
    counter = 0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        
        val_loss, val_acc, val_f1 = evaluate(model, val_dataloader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        
        print(f"\tEğitim Kaybı: {train_loss:.4f} | Doğrulama: Kayıp={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
            counter = 0
        else:
            counter += 1
            
        if counter >= patience:
            print(f"Erken durdurma: {patience} epoch boyunca iyileşme olmadı.")
            break
    
    total_time = time.time() - start_time
    print(f"Toplam eğitim süresi: {total_time:.2f} saniye")
    
    return train_losses, val_losses, val_accs, val_f1s

def test_model(model, test_dataloader, criterion, device):
    test_loss, test_acc, test_f1 = evaluate(model, test_dataloader, criterion, device)
    
    print(f"Test Sonucu: Kayıp={test_loss:.4f}, Doğruluk={test_acc:.4f}, F1={test_f1:.4f}")
    
    return test_loss, test_acc, test_f1