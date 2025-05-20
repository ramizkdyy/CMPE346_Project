import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader

class TextClassificationDataset(Dataset):
    """
    Metin sınıflandırması için PyTorch dataset sınıfı
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        return item

def load_sentiment_dataset(dataset_name, tokenizer, batch_size=16, max_length=512):
    """
    Duygu analizi veri setlerini yükler
    """
    if dataset_name == "imdb":
        # IMDB veri seti
        print(f"Yükleniyor: IMDB veri seti")
        dataset = load_dataset("imdb")
        train_texts = dataset["train"]["text"]
        train_labels = dataset["train"]["label"]
        test_texts = dataset["test"]["text"]
        test_labels = dataset["test"]["label"]
        num_classes = 2
        
    elif dataset_name == "amazon":
        # Amazon veri seti
        print(f"Yükleniyor: Amazon veri seti")
        dataset = load_dataset("amazon_reviews_multi", "en")
        train_texts = dataset["train"]["review_body"]
        train_labels = [score-1 for score in dataset["train"]["stars"]]  # 1-5 -> 0-4
        test_texts = dataset["test"]["review_body"]
        test_labels = [score-1 for score in dataset["test"]["stars"]]
        num_classes = 5
        
    elif dataset_name == "twitter":
        # Twitter duygu analizi
        print(f"Yükleniyor: Twitter duygu analizi veri seti")
        dataset = load_dataset("sentiment140")
        train_texts = dataset["train"]["text"]
        train_labels = dataset["train"]["sentiment"]
        test_texts = dataset["test"]["text"]
        test_labels = dataset["test"]["sentiment"]
        num_classes = 3
    
    else:
        raise ValueError(f"Bilinmeyen veri seti: {dataset_name}")
    
    # Veri setlerinin boyutlarını kontrol et ve daha küçük bir alt küme al
    max_samples = 10000
    if len(train_texts) > max_samples:
        print(f"Eğitim seti {len(train_texts)} örnekten {max_samples} örneğe küçültülüyor")
        indices = np.random.choice(len(train_texts), max_samples, replace=False)
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        
    if len(test_texts) > max_samples//5:
        print(f"Test seti {len(test_texts)} örnekten {max_samples//5} örneğe küçültülüyor")
        indices = np.random.choice(len(test_texts), max_samples//5, replace=False)
        test_texts = [test_texts[i] for i in indices]
        test_labels = [test_labels[i] for i in indices]
    
    # Doğrulama seti için eğitim setini ayır
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42
    )
    
    print(f"Eğitim seti boyutu: {len(train_texts)}")
    print(f"Doğrulama seti boyutu: {len(val_texts)}")
    print(f"Test seti boyutu: {len(test_texts)}")
    
    # Dataset nesnelerini oluştur
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length)
    
    # DataLoader nesnelerini oluştur
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_dataloader, val_dataloader, test_dataloader, num_classes

def load_news_dataset(dataset_name, tokenizer, batch_size=16, max_length=512):
    """
    Haber kategori sınıflandırma veri setlerini yükler
    """
    if dataset_name == "bbc":
        # BBC News veri seti
        print(f"Yükleniyor: BBC News veri seti")
        dataset = load_dataset("SetFit/bbc-news")
        texts = dataset["train"]["text"]
        labels = dataset["train"]["label_text"]
        
        # Etiketleri sayısal indekslere dönüştür
        unique_labels = list(set(labels))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = [label_to_id[label] for label in labels]
        
        # Eğitim/test verisini ayır
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, numeric_labels, test_size=0.2, random_state=42
        )
        num_classes = len(unique_labels)
        
    elif dataset_name == "ag_news":
        # AG News veri seti
        print(f"Yükleniyor: AG News veri seti")
        dataset = load_dataset("ag_news")
        train_texts = dataset["train"]["text"]
        train_labels = dataset["train"]["label"]
        test_texts = dataset["test"]["text"]
        test_labels = dataset["test"]["label"]
        num_classes = 4
        
    elif dataset_name == "20newsgroups":
        # 20 Newsgroups veri seti
        print(f"Yükleniyor: 20 Newsgroups veri seti")
        dataset = load_dataset("newsgroup")
        texts = dataset["train"]["text"]
        labels = dataset["train"]["label"]
        
        # Eğitim/test verisini ayır
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        num_classes = 20
    
    else:
        raise ValueError(f"Bilinmeyen veri seti: {dataset_name}")
    
    # Veri setlerinin boyutlarını kontrol et ve daha küçük bir alt küme al
    max_samples = 10000
    if len(train_texts) > max_samples:
        print(f"Eğitim seti {len(train_texts)} örnekten {max_samples} örneğe küçültülüyor")
        indices = np.random.choice(len(train_texts), max_samples, replace=False)
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        
    if len(test_texts) > max_samples//5:
        print(f"Test seti {len(test_texts)} örnekten {max_samples//5} örneğe küçültülüyor")
        indices = np.random.choice(len(test_texts), max_samples//5, replace=False)
        test_texts = [test_texts[i] for i in indices]
        test_labels = [test_labels[i] for i in indices]
    
    # Doğrulama seti için eğitim setini ayır
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42
    )
    
    print(f"Eğitim seti boyutu: {len(train_texts)}")
    print(f"Doğrulama seti boyutu: {len(val_texts)}")
    print(f"Test seti boyutu: {len(test_texts)}")
    
    # Dataset nesnelerini oluştur
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length)
    
    # DataLoader nesnelerini oluştur
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_dataloader, val_dataloader, test_dataloader, num_classes

def load_toxic_dataset(dataset_name, tokenizer, batch_size=16, max_length=512):
    """
    Toksik yorum tespiti veri setlerini yükler
    """
    if dataset_name == "jigsaw":
        # Jigsaw Toxic Comment veri seti
        print(f"Yükleniyor: Jigsaw Toxic Comment veri seti")
        dataset = load_dataset("jigsaw_toxicity_pred")
        texts = dataset["train"]["comment_text"]
        labels = dataset["train"]["toxic"]
        
        # Eğitim/test verisini ayır
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        num_classes = 2
        
    elif dataset_name == "wikipedia":
        # Wikipedia Toksik Yorum veri seti
        print(f"Yükleniyor: Wikipedia Toksik Yorum veri seti")
        dataset = load_dataset("wikipedia_toxicity_subtypes")
        texts = dataset["train"]["comment"]
        labels = dataset["train"]["toxicity"]
        
        # İkili sınıflandırma için eşik değeri
        threshold = 0.5
        binary_labels = [1 if label >= threshold else 0 for label in labels]
        
        # Eğitim/test verisini ayır
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, binary_labels, test_size=0.2, random_state=42
        )
        num_classes = 2
        
    elif dataset_name == "hate_speech":
        # Twitter Hate Speech veri seti
        print(f"Yükleniyor: Twitter Hate Speech veri seti")
        dataset = load_dataset("hate_speech18")
        texts = dataset["train"]["text"]
        labels = dataset["train"]["label"]
        
        # Eğitim/test verisini ayır
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        num_classes = 2
    
    else:
        raise ValueError(f"Bilinmeyen veri seti: {dataset_name}")
    
    # Veri setlerinin boyutlarını kontrol et ve daha küçük bir alt küme al
    max_samples = 10000
    if len(train_texts) > max_samples:
        print(f"Eğitim seti {len(train_texts)} örnekten {max_samples} örneğe küçültülüyor")
        indices = np.random.choice(len(train_texts), max_samples, replace=False)
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        
    if len(test_texts) > max_samples//5:
        print(f"Test seti {len(test_texts)} örnekten {max_samples//5} örneğe küçültülüyor")
        indices = np.random.choice(len(test_texts), max_samples//5, replace=False)
        test_texts = [test_texts[i] for i in indices]
        test_labels = [test_labels[i] for i in indices]
    
    # Doğrulama seti için eğitim setini ayır
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42
    )
    
    print(f"Eğitim seti boyutu: {len(train_texts)}")
    print(f"Doğrulama seti boyutu: {len(val_texts)}")
    print(f"Test seti boyutu: {len(test_texts)}")
    
    # Dataset nesnelerini oluştur
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length)
    
    # DataLoader nesnelerini oluştur
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_dataloader, val_dataloader, test_dataloader, num_classes

def get_dataloaders(person, dataset_name, tokenizer, batch_size=16, max_length=512):
    """
    Verilen kişi ve veri seti için dataloader'ları döndürür
    
    Args:
        person (str): Kişi (data_ramiz, data_yusuf, data_ipek)
        dataset_name (str): Veri seti adı
        tokenizer: Modelin tokenizer'ı
        batch_size (int): Batch büyüklüğü
        max_length (int): Maksimum token uzunluğu
        
    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader, num_classes)
    """
    if person == "data_ramiz":
        return load_sentiment_dataset(dataset_name, tokenizer, batch_size, max_length)
    elif person == "data_yusuf":
        return load_news_dataset(dataset_name, tokenizer, batch_size, max_length)
    elif person == "data_ipek":
        return load_toxic_dataset(dataset_name, tokenizer, batch_size, max_length)
    else:
        raise ValueError(f"Bilinmeyen kişi: {person}")