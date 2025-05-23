import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer
from sklearn.metrics import classification_report

from utils.data_loader import get_dataloaders
from utils.training import train_model, test_model
from models.lstm_model import LSTMClassifier
from models.transformer_models import (
    BERTClassifier, 
    DistilBERTClassifier, 
    RoBERTaClassifier, 
    FastTextClassifier
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan cihaz: {device}")

CONFIG = {
    "batch_size": 32,      
    "max_length": 64,      
    "num_epochs": 1,       
    "learning_rate": 5e-5, 
    "patience": 1,         
    
    
    "max_train_samples": 100,
    "max_test_samples": 50,     
    

    "lstm_config": {
        "embedding_dim": 100,  
        "hidden_dim": 128,     
        "n_layers": 1,         
        "bidirectional": True,
        "dropout": 0.2,
    },
    
    "fasttext_config": {
        "embedding_dim": 100, 
        "hidden_dim": 128,     
        "dropout": 0.2,
    },
    
    "models_to_run": ["LSTM", "BERT", "DistilBERT", "RoBERTa", "FastText"],
    
    "people_datasets": {
        "data_ramiz": ["imdb", "amazon", "twitter"],
        "data_yusuf": ["bbc", "ag_news", "20newsgroups"],
        "data_ipek": ["jigsaw", "wikipedia", "hate_speech"]
    }
}

results = []

def run_lstm_experiment(person, dataset_name, train_dataloader, val_dataloader, test_dataloader, num_classes):
    print(f"\n{'='*50}")
    print(f"LSTM - {person} - {dataset_name}")
    print(f"{'='*50}")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
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
    
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
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
    
    test_loss, test_acc, test_f1 = test_model(
        model=model,
        test_dataloader=test_dataloader,
        criterion=criterion,
        device=device
    )
    
    results.append({
        "Person": person,
        "Dataset": dataset_name,
        "Model": "LSTM",
        "Accuracy": test_acc,
        "F1": test_f1
    })
    
    return test_acc, test_f1

def run_bert_experiment(person, dataset_name, train_dataloader, val_dataloader, test_dataloader, num_classes):
    print(f"\n{'='*50}")
    print(f"BERT - {person} - {dataset_name}")
    print(f"{'='*50}")
    
    model = BERTClassifier(
        model_name='bert-base-uncased',
        num_classes=num_classes
    )
    
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
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
    
    test_loss, test_acc, test_f1 = test_model(
        model=model,
        test_dataloader=test_dataloader,
        criterion=criterion,
        device=device
    )
    
    results.append({
        "Person": person,
        "Dataset": dataset_name,
        "Model": "BERT",
        "Accuracy": test_acc,
        "F1": test_f1
    })
    
    return test_acc, test_f1

def run_distilbert_experiment(person, dataset_name, train_dataloader, val_dataloader, test_dataloader, num_classes):
    print(f"\n{'='*50}")
    print(f"DistilBERT - {person} - {dataset_name}")
    print(f"{'='*50}")
    
    model = DistilBERTClassifier(
        model_name='distilbert-base-uncased',
        num_classes=num_classes
    )
    
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
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
    
    test_loss, test_acc, test_f1 = test_model(
        model=model,
        test_dataloader=test_dataloader,
        criterion=criterion,
        device=device
    )
    
    results.append({
        "Person": person,
        "Dataset": dataset_name,
        "Model": "DistilBERT",
        "Accuracy": test_acc,
        "F1": test_f1
    })
    
    return test_acc, test_f1

def run_roberta_experiment(person, dataset_name, train_dataloader, val_dataloader, test_dataloader, num_classes):
    print(f"\n{'='*50}")
    print(f"RoBERTa - {person} - {dataset_name}")
    print(f"{'='*50}")
    
    model = RoBERTaClassifier(
        model_name='roberta-base',
        num_classes=num_classes
    )
    
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
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
    
    test_loss, test_acc, test_f1 = test_model(
        model=model,
        test_dataloader=test_dataloader,
        criterion=criterion,
        device=device
    )
    
    results.append({
        "Person": person,
        "Dataset": dataset_name,
        "Model": "RoBERTa",
        "Accuracy": test_acc,
        "F1": test_f1
    })
    
    return test_acc, test_f1

def run_fasttext_experiment(person, dataset_name, train_dataloader, val_dataloader, test_dataloader, num_classes):
    print(f"\n{'='*50}")
    print(f"FastText - {person} - {dataset_name}")
    print(f"{'='*50}")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    model = FastTextClassifier(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=CONFIG["fasttext_config"]["embedding_dim"],
        hidden_dim=CONFIG["fasttext_config"]["hidden_dim"],
        output_dim=num_classes,
        dropout=CONFIG["fasttext_config"]["dropout"],
        pad_idx=tokenizer.pad_token_id
    )
    
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
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
    
    # test the model
    test_loss, test_acc, test_f1 = test_model(
        model=model,
        test_dataloader=test_dataloader,
        criterion=criterion,
        device=device
    )
    
    # Saving the results
    results.append({
        "Person": person,
        "Dataset": dataset_name,
        "Model": "FastText",
        "Accuracy": test_acc,
        "F1": test_f1
    })
    
    return test_acc, test_f1

def run_experiments():
    people = CONFIG["people_datasets"]
    
    # Tokenizers
    tokenizers = {
        "LSTM": BertTokenizer.from_pretrained('bert-base-uncased'),
        "BERT": BertTokenizer.from_pretrained('bert-base-uncased'),
        "DistilBERT": DistilBertTokenizer.from_pretrained('distilbert-base-uncased'),
        "RoBERTa": RobertaTokenizer.from_pretrained('roberta-base'),
        "FastText": BertTokenizer.from_pretrained('bert-base-uncased')
    }
    
    # Experiments
    experiment_funcs = {
        "LSTM": run_lstm_experiment,
        "BERT": run_bert_experiment,
        "DistilBERT": run_distilbert_experiment,
        "RoBERTa": run_roberta_experiment,
        "FastText": run_fasttext_experiment
    }
    
    os.makedirs("results", exist_ok=True)
    
    os.environ["MAX_TRAIN_SAMPLES"] = str(CONFIG["max_train_samples"])
    os.environ["MAX_TEST_SAMPLES"] = str(CONFIG["max_test_samples"])
    
    for person, datasets in people.items():
        for dataset in datasets:
            for model_name in CONFIG["models_to_run"]:
                try:
                    experiment_func = experiment_funcs[model_name]
                    
                    print(f"\n{'-'*50}")
                    print(f"Başlatılıyor: {model_name} - {person} - {dataset}")
                    print(f"{'-'*50}")
                    
                    tokenizer = tokenizers[model_name]
                    train_dataloader, val_dataloader, test_dataloader, num_classes = get_dataloaders(
                        person=person,
                        dataset_name=dataset,
                        tokenizer=tokenizer,
                        batch_size=CONFIG["batch_size"],
                        max_length=CONFIG["max_length"]
                    )
                    
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
                    
                    results_df = pd.DataFrame(results)
                    results_df.to_csv("results/all_results_partial.csv", index=False)
                    
                except Exception as e:
                    print(f"Hata: {model_name} - {person} - {dataset}")
                    print(e)
    
    results_df = pd.DataFrame(results)
    
    results_df.to_csv("results/all_results.csv", index=False)
    
    print("\nTüm Sonuçlar:")
    print(results_df)
    
    for person in people.keys():
        person_results = results_df[results_df["Person"] == person]
        person_results.to_csv(f"results/{person}_results.csv", index=False)
        
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
        
        pivot_acc.to_csv(f"results/{person}_accuracy_pivot.csv")
        pivot_f1.to_csv(f"results/{person}_f1_pivot.csv")
        
        print(f"\n{person} Sonuçları:")
        print(person_results)
        
        print(f"\n{person} Doğruluk Pivot Tablosu:")
        print(pivot_acc)
        
        print(f"\n{person} F1 Pivot Tablosu:")
        print(pivot_f1)
    
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
    
    pivot_acc.to_csv("results/accuracy_pivot.csv")
    pivot_f1.to_csv("results/f1_pivot.csv")
    
    print("\nDoğruluk Pivot Tablosu:")
    print(pivot_acc)
    
    print("\nF1 Pivot Tablosu:")
    print(pivot_f1)

if __name__ == "__main__":
    run_experiments()