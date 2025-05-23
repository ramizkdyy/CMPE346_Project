import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
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

def load_sentiment_dataset(dataset_name, tokenizer, batch_size=32, max_length=64):
    max_train_samples = int(os.environ.get("MAX_TRAIN_SAMPLES", 500))
    max_test_samples = int(os.environ.get("MAX_TEST_SAMPLES", 100))
    
    if dataset_name == "imdb":
        print(f"Yükleniyor: IMDB veri seti")
        dataset = load_dataset("imdb")
        train_texts = dataset["train"]["text"]
        train_labels = dataset["train"]["label"]
        test_texts = dataset["test"]["text"]
        test_labels = dataset["test"]["label"]
        num_classes = 2
        
    elif dataset_name == "amazon":
        print(f"Yükleniyor: Amazon veri seti (dummy data)")
        train_texts = [f"This is a dummy amazon review {i}" for i in range(1000)]
        train_labels = np.random.randint(0, 5, size=1000).tolist()  # 0-4 arası
        test_texts = [f"This is a dummy amazon test review {i}" for i in range(200)]
        test_labels = np.random.randint(0, 5, size=200).tolist()
        num_classes = 5
        
    elif dataset_name == "twitter":
        print(f"Yükleniyor: Twitter duygu analizi veri seti (dummy data)")
        train_texts = [f"This is a dummy twitter post {i}" for i in range(1000)]
        train_labels = np.random.randint(0, 3, size=1000).tolist()  # 0-2 arası
        test_texts = [f"This is a dummy twitter test post {i}" for i in range(200)]
        test_labels = np.random.randint(0, 3, size=200).tolist()
        num_classes = 3
    
    else:
        raise ValueError(f"Bilinmeyen veri seti: {dataset_name}")
    
    if len(train_texts) > max_train_samples:
        print(f"Eğitim seti {len(train_texts)} örnekten {max_train_samples} örneğe küçültülüyor")
        indices = np.random.choice(len(train_texts), max_train_samples, replace=False)
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        
    if len(test_texts) > max_test_samples:
        print(f"Test seti {len(test_texts)} örnekten {max_test_samples} örneğe küçültülüyor")
        indices = np.random.choice(len(test_texts), max_test_samples, replace=False)
        test_texts = [test_texts[i] for i in indices]
        test_labels = [test_labels[i] for i in indices]
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42
    )
    
    print(f"Eğitim seti boyutu: {len(train_texts)}")
    print(f"Doğrulama seti boyutu: {len(val_texts)}")
    print(f"Test seti boyutu: {len(test_texts)}")
    
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_dataloader, val_dataloader, test_dataloader, num_classes

def load_news_dataset(dataset_name, tokenizer, batch_size=32, max_length=64):
    max_train_samples = int(os.environ.get("MAX_TRAIN_SAMPLES", 500))
    max_test_samples = int(os.environ.get("MAX_TEST_SAMPLES", 100))
    
    if dataset_name == "bbc":
        print(f"Yükleniyor: BBC News veri seti")
        dataset = load_dataset("SetFit/bbc-news")
        texts = dataset["train"]["text"]
        labels = dataset["train"]["label_text"]
        
        unique_labels = list(set(labels))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = [label_to_id[label] for label in labels]
        
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, numeric_labels, test_size=0.2, random_state=42
        )
        num_classes = len(unique_labels)
        
    elif dataset_name == "ag_news":
        print(f"Yükleniyor: AG News veri seti")
        dataset = load_dataset("ag_news")
        train_texts = dataset["train"]["text"]
        train_labels = dataset["train"]["label"]
        test_texts = dataset["test"]["text"]
        test_labels = dataset["test"]["label"]
        num_classes = 4
        
    elif dataset_name == "20newsgroups":
        print(f"Yükleniyor: 20 Newsgroups veri seti (dummy data)")
        train_texts = [f"This is a dummy newsgroup post {i}" for i in range(1000)]
        train_labels = np.random.randint(0, 20, size=1000).tolist()  # 0-19 arası
        test_texts = [f"This is a dummy newsgroup test post {i}" for i in range(200)]
        test_labels = np.random.randint(0, 20, size=200).tolist()
        num_classes = 20
    
    else:
        raise ValueError(f"Bilinmeyen veri seti: {dataset_name}")
    
    if len(train_texts) > max_train_samples:
        print(f"Eğitim seti {len(train_texts)} örnekten {max_train_samples} örneğe küçültülüyor")
        indices = np.random.choice(len(train_texts), max_train_samples, replace=False)
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        
    if len(test_texts) > max_test_samples:
        print(f"Test seti {len(test_texts)} örnekten {max_test_samples} örneğe küçültülüyor")
        indices = np.random.choice(len(test_texts), max_test_samples, replace=False)
        test_texts = [test_texts[i] for i in indices]
        test_labels = [test_labels[i] for i in indices]
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42
    )
    
    print(f"Eğitim seti boyutu: {len(train_texts)}")
    print(f"Doğrulama seti boyutu: {len(val_texts)}")
    print(f"Test seti boyutu: {len(test_texts)}")
    
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_dataloader, val_dataloader, test_dataloader, num_classes

def load_toxic_dataset(dataset_name, tokenizer, batch_size=32, max_length=64):
    max_train_samples = int(os.environ.get("MAX_TRAIN_SAMPLES", 500))
    max_test_samples = int(os.environ.get("MAX_TEST_SAMPLES", 100))
    
    
    if dataset_name == "jigsaw":
        print(f"Yükleniyor: Jigsaw Toxic Comment veri seti (dummy data)")
        train_texts = [f"This is a dummy jigsaw toxic comment {i}" for i in range(1000)]
        train_labels = np.random.randint(0, 2, size=1000).tolist()  
        test_texts = [f"This is a dummy jigsaw toxic test comment {i}" for i in range(200)]
        test_labels = np.random.randint(0, 2, size=200).tolist()
        num_classes = 2
        
    elif dataset_name == "wikipedia":
        print(f"Yükleniyor: Wikipedia Toksik Yorum veri seti (dummy data)")
        train_texts = [f"This is a dummy wikipedia toxic comment {i}" for i in range(1000)]
        train_labels = np.random.randint(0, 2, size=1000).tolist()
        test_texts = [f"This is a dummy wikipedia toxic test comment {i}" for i in range(200)]
        test_labels = np.random.randint(0, 2, size=200).tolist()
        num_classes = 2
        
    elif dataset_name == "hate_speech":
        print(f"Yükleniyor: Twitter Hate Speech veri seti (dummy data)")
        train_texts = [f"This is a dummy hate speech comment {i}" for i in range(1000)]
        train_labels = np.random.randint(0, 2, size=1000).tolist()  # 0-1 arası
        test_texts = [f"This is a dummy hate speech test comment {i}" for i in range(200)]
        test_labels = np.random.randint(0, 2, size=200).tolist()
        num_classes = 2
    
    else:
        raise ValueError(f"Bilinmeyen veri seti: {dataset_name}")
    
    if len(train_texts) > max_train_samples:
        print(f"Eğitim seti {len(train_texts)} örnekten {max_train_samples} örneğe küçültülüyor")
        indices = np.random.choice(len(train_texts), max_train_samples, replace=False)
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        
    if len(test_texts) > max_test_samples:
        print(f"Test seti {len(test_texts)} örnekten {max_test_samples} örneğe küçültülüyor")
        indices = np.random.choice(len(test_texts), max_test_samples, replace=False)
        test_texts = [test_texts[i] for i in indices]
        test_labels = [test_labels[i] for i in indices]
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42
    )
    
    print(f"Eğitim seti boyutu: {len(train_texts)}")
    print(f"Doğrulama seti boyutu: {len(val_texts)}")
    print(f"Test seti boyutu: {len(test_texts)}")
    
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_dataloader, val_dataloader, test_dataloader, num_classes

def get_dataloaders(person, dataset_name, tokenizer, batch_size=32, max_length=64):
    if person == "data_ramiz":
        return load_sentiment_dataset(dataset_name, tokenizer, batch_size, max_length)
    elif person == "data_yusuf":
        return load_news_dataset(dataset_name, tokenizer, batch_size, max_length)
    elif person == "data_ipek":
        return load_toxic_dataset(dataset_name, tokenizer, batch_size, max_length)
    else:
        raise ValueError(f"Bilinmeyen kişi: {person}")