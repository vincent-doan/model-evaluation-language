import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import BertTokenizerFast

from .base import BaseDataset


class DBpediaDataset(BaseDataset):
    def __init__(self, data_dir, download, tokenizer_name, max_length=256):
        """
        Initialize the DBpedia dataset.
        Args:
            data_dir (str): Directory where data is or will be stored.
            download (bool): Whether to download the dataset.
            tokenizer_name (str): The name of the tokenizer to use.
            max_length (int): Maximum tokenized sequence length.
        """
        super().__init__(data_dir, download)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.train_split = None
        self.test_split = None

    def load(self):
        """
        Loads the DBpedia dataset using Hugging Face's datasets library.
        DBpedia_14 has two splits: "train" and "test".
        We use the "content" field as the document and "label" as the class.
        """
        self.train_split = load_dataset("dbpedia_14", split="train").shuffle(seed=2411).select(range(50000))
        self.test_split = load_dataset("dbpedia_14", split="test").shuffle(seed=2411).select(range(20000))
        texts = self.train_split["content"]
        labels = self.train_split["label"]
        self.samples = np.array(texts)
        self.labels = np.array(labels)

    def tokenize_samples(self, texts):
        """
        Tokenize a list (or numpy array) of texts.
        The tokenizer pads and truncates sequences to max_length.
        Returns:
            A dictionary containing input_ids, attention_mask, etc.
        """
        tokenized = self.tokenizer(
            texts.tolist() if isinstance(texts, np.ndarray) else texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return tokenized

    def preprocess(self, train_set_size, test_set_size=0, unlabelled_set_size=-1, seed=2411):
        """
        Splits the training data (from the official DBpedia train split) using a stratified split into:
            - A labeled training set (with train_set_size samples)
            - An unlabelled set (the remainder; optionally limited by unlabelled_set_size)
        Also tokenizes the official test split.
        Returns a 3-tuple:
            - (X_train_enc, y_train): the labeled training data.
            - (X_test_enc, y_test): the test data.
            - (X_unlabelled_enc, y_unlabelled): the unlabelled training data.
        """
        np.random.seed(seed)
        strat_split = StratifiedShuffleSplit(n_splits=1, train_size=train_set_size, random_state=seed)
        train_indices, unlabelled_indices = next(strat_split.split(self.samples, self.labels))
        
        X_train_texts = self.samples[train_indices]
        y_train = self.labels[train_indices]
        X_unlabelled_texts = self.samples[unlabelled_indices]
        y_unlabelled = self.labels[unlabelled_indices]
        
        if unlabelled_set_size > 0:
            X_unlabelled_texts = X_unlabelled_texts[:unlabelled_set_size]
            y_unlabelled = y_unlabelled[:unlabelled_set_size]
        
        # Tokenize the training, unlabelled, and test texts
        X_train_enc = self.tokenize_samples(X_train_texts)
        X_unlabelled_enc = self.tokenize_samples(X_unlabelled_texts)
        X_test_enc = self.tokenize_samples(self.test_split["content"])
        y_test = self.test_split["label"]
        
        return (X_train_enc, torch.tensor(y_train, dtype=torch.long)), \
               (X_test_enc, torch.tensor(y_test, dtype=torch.long)), \
               (X_unlabelled_enc, torch.tensor(y_unlabelled, dtype=torch.long))
