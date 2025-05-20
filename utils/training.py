import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Modeli bir epoch boyunca eğitir
    
    Args:
        model: Eğitilecek model
        dataloader: Eğitim veri yükleyici
        optimizer: Optimizer
        criterion: Kayıp fonksiyonu
        device: Cihaz (cpu veya cuda)
        
    Returns:
        float: Epoch'un ortalama kaybı
    """
    model.train()
    epoch_loss = 0
    
    # tqdm göstergesi kullanma
    for batch in dataloader:
        # Veriyi cihaza taşı
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Gradyanları sıfırla
        optimizer.zero_grad()
        
        # Modeli çalıştır
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Kaybı hesapla
        loss = criterion(outputs, labels)
        
        # Geri yayılım
        loss.backward()
        
        # Parametreleri güncelle
        optimizer.step()
        
        # Kaybı topla
        epoch_loss += loss.item()
    
    # Ortalama kaybı döndür
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """
    Modeli değerlendirir
    
    Args:
        model: Değerlendirilecek model
        dataloader: Değerlendirme veri yükleyici
        criterion: Kayıp fonksiyonu
        device: Cihaz (cpu veya cuda)
        
    Returns:
        tuple: (ortalama kayıp, doğruluk, F1 skoru)
    """
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        # tqdm göstergesi kullanma
        for batch in dataloader:
            # Veriyi cihaza taşı
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Modeli çalıştır
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Kaybı hesapla
            loss = criterion(outputs, labels)
            
            # Kaybı topla
            epoch_loss += loss.item()
            
            # Tahminleri hesapla
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Tahminleri ve etiketleri topla
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Metrikleri hesapla
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Sınıf sayısını belirle
    num_classes = len(np.unique(all_labels))
    
    # F1 skorunu hesapla
    # Sınıf sayısı 2'den fazla ise 'weighted' average kullan, aksi takdirde 'binary'
    f1 = f1_score(all_labels, all_preds, 
                  average='weighted' if num_classes > 2 else 'binary')
    
    return epoch_loss / len(dataloader), accuracy, f1

def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, device, 
                num_epochs=2, patience=1, model_save_path=None):
    """
    Modeli eğitir ve değerlendirir
    
    Args:
        model: Eğitilecek model
        train_dataloader: Eğitim veri yükleyici
        val_dataloader: Doğrulama veri yükleyici
        optimizer: Optimizer
        criterion: Kayıp fonksiyonu
        device: Cihaz (cpu veya cuda)
        num_epochs (int): Epoch sayısı
        patience (int): Erken durdurma için sabır
        model_save_path (str): Model kayıt yolu
        
    Returns:
        tuple: (eğitim kayıpları, doğrulama kayıpları, doğrulama doğrulukları, doğrulama F1 skorları)
    """
    # Metrikleri kaydetmek için listeler
    train_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []
    
    # En iyi doğrulama F1 skoru
    best_val_f1 = 0
    
    # Erken durdurma sayacı
    counter = 0
    
    # Başlangıç zamanı
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Modeli eğit
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        
        # Modeli değerlendir
        val_loss, val_acc, val_f1 = evaluate(model, val_dataloader, criterion, device)
        
        # Metrikleri kaydet
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        
        # Sonuçları yazdır
        print(f"\tEğitim Kaybı: {train_loss:.4f} | Doğrulama: Kayıp={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
        
        # En iyi modeli kaydet
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
            counter = 0
        else:
            counter += 1
            
        # Erken durdurma
        if counter >= patience:
            print(f"Erken durdurma: {patience} epoch boyunca iyileşme olmadı.")
            break
    
    # Toplam eğitim süresini hesapla
    total_time = time.time() - start_time
    print(f"Toplam eğitim süresi: {total_time:.2f} saniye")
    
    return train_losses, val_losses, val_accs, val_f1s

def test_model(model, test_dataloader, criterion, device):
    """
    Modeli test eder
    
    Args:
        model: Test edilecek model
        test_dataloader: Test veri yükleyici
        criterion: Kayıp fonksiyonu
        device: Cihaz (cpu veya cuda)
        
    Returns:
        tuple: (test kaybı, test doğruluğu, test F1 skoru)
    """
    # Modeli değerlendir
    test_loss, test_acc, test_f1 = evaluate(model, test_dataloader, criterion, device)
    
    # Sonuçları yazdır
    print(f"Test Sonucu: Kayıp={test_loss:.4f}, Doğruluk={test_acc:.4f}, F1={test_f1:.4f}")
    
    return test_loss, test_acc, test_f1