import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer
from sklearn.metrics import classification_report

# Yerel modüller
from utils.data_loader import get_dataloaders
from utils.training import train_model, test_model
from models.lstm_model import LSTMClassifier
from models.transformer_models import (
    BERTClassifier, 
    DistilBERTClassifier, 
    RoBERTaClassifier, 
    FastTextClassifier
)

# Cihazı belirle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan cihaz: {device}")

# Konfigürasyon - EXTRA HIZLI
CONFIG = {
    "batch_size": 32,      # Artırılmış batch boyutu
    "max_length": 64,      # Daha kısa token uzunluğu
    "num_epochs": 1,       # SADECE 1 EPOCH
    "learning_rate": 5e-5, # Biraz daha yüksek öğrenme oranı
    "patience": 1,         # Daha az bekleme süresi
    
    # Örnek sayıları
    "max_train_samples": 100,    # Çok az eğitim örnekleri (süper hızlı test için)
    "max_test_samples": 50,     # Çok az test örnekleri
    
    # LSTM konfigürasyonu - hafifleştirilmiş
    "lstm_config": {
        "embedding_dim": 100,  # Daha küçük embedding boyutu
        "hidden_dim": 128,     # Daha küçük gizli katman
        "n_layers": 1,         # Tek katman
        "bidirectional": True,
        "dropout": 0.2,
    },
    
    # FastText konfigürasyonu - hafifleştirilmiş
    "fasttext_config": {
        "embedding_dim": 100,  # Daha küçük embedding boyutu
        "hidden_dim": 128,     # Daha küçük gizli katman
        "dropout": 0.2,
    },
    
    # Ödevdeki gibi TÜM modelleri çalıştır
    "models_to_run": ["LSTM", "BERT", "DistilBERT", "RoBERTa", "FastText"],
    
    # TÜM kişiler için TÜM veri setleri
    "people_datasets": {
        "data_ramiz": ["imdb", "amazon", "twitter"],
        "data_yusuf": ["bbc", "ag_news", "20newsgroups"],
        "data_ipek": ["jigsaw", "wikipedia", "hate_speech"]
    }
}

# Sonuçları saklamak için tablo
results = []

def run_lstm_experiment(person, dataset_name, train_dataloader, val_dataloader, test_dataloader, num_classes):
    """
    LSTM modeli ile deney yapar
    """
    print(f"\n{'='*50}")
    print(f"LSTM - {person} - {dataset_name}")
    print(f"{'='*50}")
    
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Modeli oluştur
    model = LSTMClassifier(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=CONFIG["lstm_config"]["embedding_dim"],
        hidden_dim=CONFIG["lstm_config"]["hidden_dim"],
        output_dim=num_classes,
        n_layers=CONFIG["lstm_config"]["n_layers"],
        bidirectional=CONFIG["lstm_config"]["bidirectional"],
        dropout=CONFIG["lstm_config"]["dropout"],
        pad_idx=tokenizer.pad_token_id
    )
    
    # Modeli cihaza taşı
    model = model.to(device)
    
    # Optimizasyon ve kayıp fonksiyonu
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    # Modeli eğit
    train_losses, val_losses, val_accs, val_f1s = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=CONFIG["num_epochs"],
        patience=CONFIG["patience"]
    )
    
    # Modeli test et
    test_loss, test_acc, test_f1 = test_model(
        model=model,
        test_dataloader=test_dataloader,
        criterion=criterion,
        device=device
    )
    
    # Sonuçları kaydet
    results.append({
        "Person": person,
        "Dataset": dataset_name,
        "Model": "LSTM",
        "Accuracy": test_acc,
        "F1": test_f1
    })
    
    return test_acc, test_f1

def run_bert_experiment(person, dataset_name, train_dataloader, val_dataloader, test_dataloader, num_classes):
    """
    BERT modeli ile deney yapar
    """
    print(f"\n{'='*50}")
    print(f"BERT - {person} - {dataset_name}")
    print(f"{'='*50}")
    
    # Modeli oluştur
    model = BERTClassifier(
        model_name='bert-base-uncased',
        num_classes=num_classes
    )
    
    # Modeli cihaza taşı
    model = model.to(device)
    
    # Optimizasyon ve kayıp fonksiyonu
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    # Modeli eğit
    train_losses, val_losses, val_accs, val_f1s = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=CONFIG["num_epochs"],
        patience=CONFIG["patience"]
    )
    
    # Modeli test et
    test_loss, test_acc, test_f1 = test_model(
        model=model,
        test_dataloader=test_dataloader,
        criterion=criterion,
        device=device
    )
    
    # Sonuçları kaydet
    results.append({
        "Person": person,
        "Dataset": dataset_name,
        "Model": "BERT",
        "Accuracy": test_acc,
        "F1": test_f1
    })
    
    return test_acc, test_f1

def run_distilbert_experiment(person, dataset_name, train_dataloader, val_dataloader, test_dataloader, num_classes):
    """
    DistilBERT modeli ile deney yapar
    """
    print(f"\n{'='*50}")
    print(f"DistilBERT - {person} - {dataset_name}")
    print(f"{'='*50}")
    
    # Modeli oluştur
    model = DistilBERTClassifier(
        model_name='distilbert-base-uncased',
        num_classes=num_classes
    )
    
    # Modeli cihaza taşı
    model = model.to(device)
    
    # Optimizasyon ve kayıp fonksiyonu
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    # Modeli eğit
    train_losses, val_losses, val_accs, val_f1s = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=CONFIG["num_epochs"],
        patience=CONFIG["patience"]
    )
    
    # Modeli test et
    test_loss, test_acc, test_f1 = test_model(
        model=model,
        test_dataloader=test_dataloader,
        criterion=criterion,
        device=device
    )
    
    # Sonuçları kaydet
    results.append({
        "Person": person,
        "Dataset": dataset_name,
        "Model": "DistilBERT",
        "Accuracy": test_acc,
        "F1": test_f1
    })
    
    return test_acc, test_f1

def run_roberta_experiment(person, dataset_name, train_dataloader, val_dataloader, test_dataloader, num_classes):
    """
    RoBERTa modeli ile deney yapar
    """
    print(f"\n{'='*50}")
    print(f"RoBERTa - {person} - {dataset_name}")
    print(f"{'='*50}")
    
    # Modeli oluştur
    model = RoBERTaClassifier(
        model_name='roberta-base',
        num_classes=num_classes
    )
    
    # Modeli cihaza taşı
    model = model.to(device)
    
    # Optimizasyon ve kayıp fonksiyonu
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    # Modeli eğit
    train_losses, val_losses, val_accs, val_f1s = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=CONFIG["num_epochs"],
        patience=CONFIG["patience"]
    )
    
    # Modeli test et
    test_loss, test_acc, test_f1 = test_model(
        model=model,
        test_dataloader=test_dataloader,
        criterion=criterion,
        device=device
    )
    
    # Sonuçları kaydet
    results.append({
        "Person": person,
        "Dataset": dataset_name,
        "Model": "RoBERTa",
        "Accuracy": test_acc,
        "F1": test_f1
    })
    
    return test_acc, test_f1

def run_fasttext_experiment(person, dataset_name, train_dataloader, val_dataloader, test_dataloader, num_classes):
    """
    FastText modeli ile deney yapar
    """
    print(f"\n{'='*50}")
    print(f"FastText - {person} - {dataset_name}")
    print(f"{'='*50}")
    
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Modeli oluştur
    model = FastTextClassifier(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=CONFIG["fasttext_config"]["embedding_dim"],
        hidden_dim=CONFIG["fasttext_config"]["hidden_dim"],
        output_dim=num_classes,
        dropout=CONFIG["fasttext_config"]["dropout"],
        pad_idx=tokenizer.pad_token_id
    )
    
    # Modeli cihaza taşı
    model = model.to(device)
    
    # Optimizasyon ve kayıp fonksiyonu
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    # Modeli eğit
    train_losses, val_losses, val_accs, val_f1s = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=CONFIG["num_epochs"],
        patience=CONFIG["patience"]
    )
    
    # Modeli test et
    test_loss, test_acc, test_f1 = test_model(
        model=model,
        test_dataloader=test_dataloader,
        criterion=criterion,
        device=device
    )
    
    # Sonuçları kaydet
    results.append({
        "Person": person,
        "Dataset": dataset_name,
        "Model": "FastText",
        "Accuracy": test_acc,
        "F1": test_f1
    })
    
    return test_acc, test_f1

def run_experiments():
    """
    Tüm deneyleri çalıştırır
    """
    # Kişiler ve veri setleri - CONFIG'den alınıyor
    people = CONFIG["people_datasets"]
    
    # Tokenizer'lar
    tokenizers = {
        "LSTM": BertTokenizer.from_pretrained('bert-base-uncased'),
        "BERT": BertTokenizer.from_pretrained('bert-base-uncased'),
        "DistilBERT": DistilBertTokenizer.from_pretrained('distilbert-base-uncased'),
        "RoBERTa": RobertaTokenizer.from_pretrained('roberta-base'),
        "FastText": BertTokenizer.from_pretrained('bert-base-uncased')
    }
    
    # Deneyler
    experiment_funcs = {
        "LSTM": run_lstm_experiment,
        "BERT": run_bert_experiment,
        "DistilBERT": run_distilbert_experiment,
        "RoBERTa": run_roberta_experiment,
        "FastText": run_fasttext_experiment
    }
    
    # Sonuçlar için dizini oluştur
    os.makedirs("results", exist_ok=True)
    
    # data_loader.py dosyasına max_samples bilgisini iletmek için çevresel değişken ayarla
    os.environ["MAX_TRAIN_SAMPLES"] = str(CONFIG["max_train_samples"])
    os.environ["MAX_TEST_SAMPLES"] = str(CONFIG["max_test_samples"])
    
    # Her bir kişi için
    for person, datasets in people.items():
        # Her bir veri seti için
        for dataset in datasets:
            # Her bir model için
            for model_name in CONFIG["models_to_run"]:
                try:
                    experiment_func = experiment_funcs[model_name]
                    
                    print(f"\n{'-'*50}")
                    print(f"Başlatılıyor: {model_name} - {person} - {dataset}")
                    print(f"{'-'*50}")
                    
                    # Veri yükleyiciyi al
                    tokenizer = tokenizers[model_name]
                    train_dataloader, val_dataloader, test_dataloader, num_classes = get_dataloaders(
                        person=person,
                        dataset_name=dataset,
                        tokenizer=tokenizer,
                        batch_size=CONFIG["batch_size"],
                        max_length=CONFIG["max_length"]
                    )
                    
                    # Deneyi çalıştır
                    test_acc, test_f1 = experiment_func(
                        person=person,
                        dataset_name=dataset,
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        test_dataloader=test_dataloader,
                        num_classes=num_classes
                    )
                    
                    print(f"{model_name} - {person} - {dataset} tamamlandı.")
                    print(f"Test Doğruluğu: {test_acc:.4f}, Test F1: {test_f1:.4f}")
                    
                    # Her model+veri seti kombinasyonu için sonuçları ara yedekle
                    results_df = pd.DataFrame(results)
                    results_df.to_csv("results/all_results_partial.csv", index=False)
                    
                except Exception as e:
                    print(f"Hata: {model_name} - {person} - {dataset}")
                    print(e)
    
    # Sonuçları DataFrame'e dönüştür
    results_df = pd.DataFrame(results)
    
    # Sonuçları kaydet
    results_df.to_csv("results/all_results.csv", index=False)
    
    # Sonuçları göster
    print("\nTüm Sonuçlar:")
    print(results_df)
    
    # Her bir kişi için ayrı bir sonuç tablosu oluştur ve kaydet
    for person in people.keys():
        person_results = results_df[results_df["Person"] == person]
        person_results.to_csv(f"results/{person}_results.csv", index=False)
        
        # Kişiye özgü pivot tabloları oluştur
        pivot_acc = person_results.pivot_table(
            index="Dataset", 
            columns="Model", 
            values="Accuracy"
        )
        
        pivot_f1 = person_results.pivot_table(
            index="Dataset", 
            columns="Model", 
            values="F1"
        )
        
        # Kişiye özgü pivot tabloları kaydet
        pivot_acc.to_csv(f"results/{person}_accuracy_pivot.csv")
        pivot_f1.to_csv(f"results/{person}_f1_pivot.csv")
        
        # Kişiye özgü sonuçları göster
        print(f"\n{person} Sonuçları:")
        print(person_results)
        
        print(f"\n{person} Doğruluk Pivot Tablosu:")
        print(pivot_acc)
        
        print(f"\n{person} F1 Pivot Tablosu:")
        print(pivot_f1)
    
    # Genel pivot tablolar - tüm kişiler
    pivot_acc = results_df.pivot_table(
        index=["Person", "Dataset"], 
        columns="Model", 
        values="Accuracy"
    )
    
    pivot_f1 = results_df.pivot_table(
        index=["Person", "Dataset"], 
        columns="Model", 
        values="F1"
    )
    
    # Pivot tabloları kaydet
    pivot_acc.to_csv("results/accuracy_pivot.csv")
    pivot_f1.to_csv("results/f1_pivot.csv")
    
    # Pivot tabloları göster
    print("\nDoğruluk Pivot Tablosu:")
    print(pivot_acc)
    
    print("\nF1 Pivot Tablosu:")
    print(pivot_f1)

if __name__ == "__main__":
    run_experiments()