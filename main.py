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

# Konfigürasyon
CONFIG = {
    "batch_size": 16,
    "max_length": 128,
    "num_epochs": 3,
    "learning_rate": 2e-5,
    "patience": 2,
    "lstm_config": {
        "embedding_dim": 300,
        "hidden_dim": 256,
        "n_layers": 2,
        "bidirectional": True,
        "dropout": 0.25,
    },
    "fasttext_config": {
        "embedding_dim": 300,
        "hidden_dim": 256,
        "dropout": 0.25,
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
    # Kişiler ve veri setleri
    people = {
        "data_ramiz": ["imdb", "amazon", "twitter"],
        "data_yusuf": ["bbc", "ag_news", "20newsgroups"],
        "data_ipek": ["jigsaw", "wikipedia", "hate_speech"]
    }
    
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
    
    # Her bir kişi için
    for person, datasets in people.items():
        # Her bir veri seti için
        for dataset in datasets:
            # Her bir model için
            for model_name, experiment_func in experiment_funcs.items():
                try:
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
    
    # Pivot tablo oluştur - Kişi ve veri seti bazında doğruluk
    pivot_acc = results_df.pivot_table(
        index=["Person", "Dataset"], 
        columns="Model", 
        values="Accuracy"
    )
    
    # Pivot tablo oluştur - Kişi ve veri seti bazında F1
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