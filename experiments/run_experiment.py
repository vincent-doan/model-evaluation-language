import argparse
import os
import json
import yaml
import gc
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------------------------
# Import dataset classes
# ------------------------------------------------------------------------
from src.datasets.ag_news import AGNewsDataset
from src.datasets.imdb import IMDBDataset
from src.datasets.dbpedia import DBpediaDataset

# ------------------------------------------------------------------------
# Import model classes
# ------------------------------------------------------------------------
from src.models.bert_base_uncased import BertBaseUncased
from src.models.bert_large_uncased import BertLargeUncased

# ------------------------------------------------------------------------
# Trainer & metrics
# ------------------------------------------------------------------------
from src.training.trainer import train_model
from src.evaluation.metrics import compute_custom_metrics

# ------------------------------------------------------------------------
# Postprocessing
# ------------------------------------------------------------------------
from experiments.postprocess import postprocess_results

# ------------------------------------------------------------------------
# Registry / mapping from string to classes
# ------------------------------------------------------------------------
DATASET_MAP = {
    "ag_news": AGNewsDataset,
    "imdb": IMDBDataset,
    "dbpedia": DBpediaDataset
}

MODEL_MAP = {
    "bert-base-uncased": BertBaseUncased,
    "bert-large-uncased": BertLargeUncased
}

# ------------------------------------------------------------------------
# Default parameters
# ------------------------------------------------------------------------
PARAMS = {
    "ag_news": {
        "seed": 2411,
        "num_classes": 4
    },
    "imdb": {
        "seed": 2411,
        "num_classes": 2
    },
    "dbpedia": {
        "seed": 2411,
        "num_classes": 14
    }
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_version", type=int, default=0,
                        help="Version identifier for saving results.")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Which dataset to use (overrides config.yml if provided).")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Which model to use (overrides config.yml if provided).")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="Which GPU to use (default: 0).")
    parser.add_argument("--train_set_size", type=int,
                        help="Size of the training set.")
    parser.add_argument("--test_set_size", type=int, default=0,
                        help="Size of the test set.")
    parser.add_argument("--unlabelled_set_size", type=int, default=-1,
                        help="Size of the unlabelled set.")
    
    args = parser.parse_args()
    
    # ===================================================================
    # 0.1. Read command line arguments
    # ===================================================================
    experiment_version = 'v' + str(args.experiment_version)
    gpu_id = args.gpu_id
    dataset_name = args.dataset_name
    model_name = args.model_name
    
    # ===================================================================
    # 0.2. Check validity of dataset and model names
    # ===================================================================
    if dataset_name not in DATASET_MAP:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_MAP.keys())}")
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_MAP.keys())}")
    
    # ===================================================================
    # 1. Load configuration from experiments/config.yml
    # ===================================================================
    with open("experiments/config.yml", "r") as f:
        cfg = yaml.safe_load(f)
    model_configs = cfg["model"][model_name]
    keys = list(model_configs.keys())
    values = list(model_configs.values())
    config_list = [dict(zip(keys, combo)) for combo in product(*values)]

    # ===================================================================
    # 2. Create dataset instance & load data
    # ===================================================================
    ds_class = DATASET_MAP[dataset_name]
    if dataset_name in ["ag_news", "imdb", "dbpedia"]:
        dataset_obj = ds_class(
            data_dir="./data",
            download=True,
            tokenizer_name=model_name
        )
        dataset_obj.load()
        (X_train, y_train), (X_test, y_test), (X_unlabelled, y_unlabelled) = dataset_obj.preprocess(
            train_set_size=args.train_set_size,
            test_set_size=args.test_set_size,
            unlabelled_set_size=args.unlabelled_set_size,
            seed=PARAMS[dataset_name]["seed"]
        )
    else:
        #  -- Add other datasets here if needed --
        pass
    
    # ===================================================================
    # 3. Function to prepare DataLoaders
    # ===================================================================
    def create_loader(X, y, batch_size, shuffle):
        dataset = TensorDataset(X["input_ids"], X["attention_mask"], y)
        def collate_fn(batch):
            input_ids = torch.stack([item[0] for item in batch])
            attention_mask = torch.stack([item[1] for item in batch])
            labels = torch.stack([item[2] for item in batch])
            return ({"input_ids": input_ids, "attention_mask": attention_mask}, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    # ==================================================================
    # 4. Loop over all configurations
    # ==================================================================
    all_result_entries = []
    for config in config_list:
        # Unpack configuration
        num_epochs = config["num_epochs"]
        learning_rate = config["learning_rate"]
        batch_size = config["batch_size"]
        dropout_rate = config.get("dropout_rate", 0.1)

        # Create DataLoaders for training and test splits
        train_loader = create_loader(X_train, y_train, batch_size, shuffle=True)
        test_loader = create_loader(X_test, y_test, batch_size, shuffle=False)

        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        # Initialize model
        if model_name in ["bert-base-uncased", "bert-large-uncased"]:
            model = MODEL_MAP[model_name](num_labels=PARAMS[dataset_name]["num_classes"], dropout_rate=dropout_rate)
        else:
            # -- Add other models here if needed --
            pass
        model.to(device)

        # Train model
        CACHED_FOLDER_PATH = f"results/{experiment_version}/{dataset_name}/{model_name}"
        os.makedirs(CACHED_FOLDER_PATH, exist_ok=True)
        CHECKPOINT_PATH = os.path.join(CACHED_FOLDER_PATH, "best_checkpoint.pth")
        trained_model, train_accuracy, test_accuracy = train_model(model, train_loader, test_loader, device, num_epochs, learning_rate, CHECKPOINT_PATH)

        # Group unlabelled data (using tokenized representations)
        TRAIN_LOGITS_PATH = os.path.join(CACHED_FOLDER_PATH, "cached_train_logits.npy")
        TRAIN_LABELS_PATH = os.path.join(CACHED_FOLDER_PATH, "cached_train_labels.npy")
        UNLABELLED_LOGITS_PATH = os.path.join(CACHED_FOLDER_PATH, "cached_unlabelled_logits.npy")
        UNLABELLED_LABELS_PATH = os.path.join(CACHED_FOLDER_PATH, "cached_unlabelled_labels.npy")
        grouping, train_logits, unlabelled_logits = dataset_obj.group_unlabelled(
            X_train, 
            X_unlabelled, 
            trained_model, 
            device
        )

        # Save logits and labels
        np.save(TRAIN_LOGITS_PATH, train_logits)
        np.save(TRAIN_LABELS_PATH, y_train.cpu().numpy())
        np.save(UNLABELLED_LOGITS_PATH, unlabelled_logits)
        np.save(UNLABELLED_LABELS_PATH, y_unlabelled.cpu().numpy())
        with open(f"results/{experiment_version}/{dataset_name}/{model_name}/grouping.json", "w") as f:
            json.dump(grouping, f)

        # Compute custom metrics. (For our text task, custom metrics are computed similarly.)
        train_loss, distance, measure, _distance, _measure = compute_custom_metrics(
            cached_folder_path=f"results/{experiment_version}/{dataset_name}/{model_name}",
            grouping=grouping,
            device=device
        )

        # Log results
        result_entry = {
            "dataset": dataset_name,
            "model_name": model_name,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "batch_size": batch_size,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_accuracy, 4),
            "test_accuracy": round(test_accuracy, 4),
            "distance": round(distance, 4),
            "measure": round(measure, 4),
            "_distance": round(_distance, 4),
            "_measure": round(_measure, 4)
        }
        all_result_entries.append(result_entry)
        print(f"Completed configuration: {config}")
        print(f"Result: {result_entry}")
        
        # Remove the cached .npy files
        os.remove(TRAIN_LOGITS_PATH)
        os.remove(TRAIN_LABELS_PATH)
        os.remove(UNLABELLED_LOGITS_PATH)
        os.remove(UNLABELLED_LABELS_PATH)

        # Clean up after each iteration
        del model, trained_model
        del train_logits, unlabelled_logits
        del train_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()

    # Save all results
    results_csv_path = f"results/{experiment_version}/{dataset_name}/{model_name}/results.csv"
    results_df = pd.DataFrame(all_result_entries)
    results_df.to_csv(results_csv_path, index=False)
    print(f"Saved results to {results_csv_path}")
    print("Postprocessing experiment results...")
    postprocess_results("accuracy", top_k=15, results_path=results_csv_path)
    print("Postprocessed output saved!")


if __name__ == "__main__":
    main()
