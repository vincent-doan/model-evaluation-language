import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import BertTokenizerFast

from .base import BaseDataset


class AGNewsDataset(BaseDataset):
    def __init__(self, data_dir, download, tokenizer_name, max_length=128):
        super().__init__(data_dir, download)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.train_split = None
        self.test_split = None

    def load(self):
        dataset = load_dataset("ag_news")
        self.train_split = dataset["train"]
        self.test_split = dataset["test"]
        texts = self.train_split["text"]
        labels = self.train_split["label"]
        self.samples = np.array(texts)
        self.labels = np.array(labels)

    def tokenize_samples(self, texts):
        tokenized = self.tokenizer(texts.tolist() if isinstance(texts, np.ndarray) else texts,
                                   padding="max_length",
                                   truncation=True,
                                   max_length=self.max_length,
                                   return_tensors="pt")
        return tokenized

    def preprocess(self, train_set_size, test_set_size, unlabelled_set_size=-1, seed=2411):
        np.random.seed(seed)
        
        # Stratified split for train and unlabelled sets
        stratified_split = StratifiedShuffleSplit(n_splits=1, train_size=train_set_size, random_state=seed)
        train_indices, unlabelled_indices = next(stratified_split.split(self.samples, self.labels))
        
        # Obtain the train and unlabelled samples
        X_train_texts = self.samples[train_indices]
        y_train = self.labels[train_indices]
        X_unlabelled_texts = self.samples[unlabelled_indices]
        y_unlabelled = self.labels[unlabelled_indices]
        if unlabelled_set_size > 0:
            X_unlabelled_texts = X_unlabelled_texts[:unlabelled_set_size]
            y_unlabelled = y_unlabelled[:unlabelled_set_size]
        
        # Tokenize the texts
        X_train_enc = self.tokenize_samples(X_train_texts)
        X_unlabelled_enc = self.tokenize_samples(X_unlabelled_texts)
        
        # Use the official test split from the loaded dataset
        X_test_enc = self.tokenize_samples(self.test_split["text"])
        y_test = self.test_split["label"]
        
        return (X_train_enc, torch.tensor(y_train, dtype=torch.long)), \
            (X_test_enc, torch.tensor(y_test, dtype=torch.long)), \
            (X_unlabelled_enc, torch.tensor(y_unlabelled, dtype=torch.long))
