import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import BertTokenizerFast

from .base import BaseDataset


class IMDBDataset(BaseDataset):
    def __init__(self, data_dir, download, tokenizer_name, max_length=256):
        """
        Initialize the IMDB dataset.
        Args:
            data_dir (str): Directory where data is or will be stored.
            download (bool): Whether to download the dataset.
            tokenizer_name (str): The name of the tokenizer to use.
            max_length (int): Maximum length of tokenized sequences.
        """
        super().__init__(data_dir, download)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.train_split = None
        self.test_split = None

    def load(self):
        """
        Load the IMDb dataset using Hugging Face's datasets library.
        IMDb contains two splits: train and test.
        The train split is used for training and can be further split into a labeled training set
        and an unlabelled set if desired.
        """
        dataset = load_dataset("imdb")
        self.train_split = dataset["train"]
        self.test_split = dataset["test"]
        texts = self.train_split["text"]
        labels = self.train_split["label"]
        self.samples = np.array(texts)
        self.labels = np.array(labels)

    def tokenize_samples(self, texts):
        """
        Tokenize a list or array of texts into a dictionary containing input_ids,
        attention_mask, etc., using the configured tokenizer.
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
        Perform a stratified split on the training split to obtain a labeled
        training set and an unlabelled set.
        The test set is taken from the official IMDb test split.
        Args:
            train_set_size (int): Number of samples for the labeled training set.
            test_set_size (int): (Not used here, as the official test split is loaded.)
            unlabelled_set_size (int): Limit for the unlabelled training set. If -1, use all.
            seed (int): Random seed for reproducibility.
        Returns:
            A tuple of three elements:
              - (X_train_enc, y_train): Labeled training data.
              - (X_test_enc, y_test): Test data.
              - (X_unlabelled_enc, y_unlabelled): Unlabelled training data.
        """
        np.random.seed(seed)
        # Use stratified split so that both classes are represented proportionally
        stratified_split = StratifiedShuffleSplit(n_splits=1, train_size=train_set_size, random_state=seed)
        train_indices, unlabelled_indices = next(stratified_split.split(self.samples, self.labels))
        
        X_train_texts = self.samples[train_indices]
        y_train = self.labels[train_indices]
        X_unlabelled_texts = self.samples[unlabelled_indices]
        y_unlabelled = self.labels[unlabelled_indices]
        if unlabelled_set_size > 0:
            X_unlabelled_texts = X_unlabelled_texts[:unlabelled_set_size]
            y_unlabelled = y_unlabelled[:unlabelled_set_size]
        
        # Tokenize training and unlabelled texts
        X_train_enc = self.tokenize_samples(X_train_texts)
        X_unlabelled_enc = self.tokenize_samples(X_unlabelled_texts)
        
        # Tokenize the official test split
        X_test_enc = self.tokenize_samples(self.test_split["text"])
        y_test = self.test_split["label"]
        
        return (X_train_enc, torch.tensor(y_train, dtype=torch.long)), \
               (X_test_enc, torch.tensor(y_test, dtype=torch.long)), \
               (X_unlabelled_enc, torch.tensor(y_unlabelled, dtype=torch.long))
