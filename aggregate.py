#!/usr/bin/env python
import argparse
import os
import json
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


def parse_comparison_file(file_path):
    """
    Parse the comparison-accuracy.json file and return
    a dictionary mapping each row label to its value.
    
    The JSON structure typically has keys like:
      "tau_train_accuracy", "tau_measure",
      "top_k_accuracy_train_accuracy_list", "top_k_accuracy_measure_list", etc.
    We want:
      tau_train -> from "tau_train_accuracy"
      tau_measure -> from "tau_measure"
      acc@1_train -> from top_k_accuracy_train_accuracy_list[0]
      acc@1_measure -> from top_k_accuracy_measure_list[0]
      acc@3_train -> from top_k_accuracy_train_accuracy_list[2]
      ...
      acc@15_measure -> from top_k_accuracy_measure_list[14]
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    
    # The final row labels we want
    row_labels = [
        "tau_train",
        "tau_measure",
        "tau@3_train",
        "tau@3_measure",
        "tau@5_train",
        "tau@5_measure",
        "tau@10_train",
        "tau@10_measure",
        "tau@15_train",
        "tau@15_measure",
        "acc@1_train",
        "acc@1_measure",
        "acc@3_train",
        "acc@3_measure",
        "acc@5_train",
        "acc@5_measure",
        "acc@10_train",
        "acc@10_measure",
        "acc@15_train",
        "acc@15_measure",
    ]
    results_map = {rl: None for rl in row_labels}
    
    # 1.1. tau_train is "tau_train_accuracy" in JSON
    if "tau_train_accuracy" in data:
        results_map["tau_train"] = data["tau_train_accuracy"]["all"]
    else:
        # fallback if you used "tau_train" or something else
        results_map["tau_train"] = None

    # 2.1. tau_measure is "tau_measure" in JSON
    if "tau_measure" in data:
        results_map["tau_measure"] = data["tau_measure"]["all"]
    else:
        results_map["tau_measure"] = None

    # 1.2. tau@k_train is "tau_train_accuracy_list" in JSON
    if "tau_train_accuracy" in data:
        tau_train_list = data["tau_train_accuracy"]
        for k in [3, 5, 10, 15]:
            key = f"tau@{k}_train"
            results_map[key] = tau_train_list[f"top_{k}"]

    # 2.2. tau@k_measure is "tau_measure_list" in JSON
    if "tau_measure" in data:
        tau_measure_list = data["tau_measure"]
        for k in [3, 5, 10, 15]:
            key = f"tau@{k}_measure"
            results_map[key] = tau_measure_list[f"top_{k}"]

    # 3. top_k_accuracy_train_accuracy_list
    train_acc_list = data.get("top_k_accuracy_train_accuracy_list", [])
    # 4. top_k_accuracy_measure_list
    measure_list = data.get("top_k_accuracy_measure_list", [])
    
    # We want the k values: 1,3,5,10,15
    # We'll map them to zero-based indices: k-1
    ks = [1, 3, 5, 10, 15]
    for k in ks:
        idx = k - 1
        # e.g. "acc@3_train" => from train_acc_list[2]
        train_key = f"acc@{k}_train"
        measure_key = f"acc@{k}_measure"
        
        # If the list isn't long enough, default to None
        if idx < len(train_acc_list):
            results_map[train_key] = train_acc_list[idx]
        else:
            results_map[train_key] = None
        
        if idx < len(measure_list):
            results_map[measure_key] = measure_list[idx]
        else:
            results_map[measure_key] = None
    
    # sort results_map exactly like the order in row_labels
    results_map = {k: results_map[k] for k in row_labels}
    return results_map


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate comparison-accuracy.json files into a single CSV."
    )
    parser.add_argument("--experiment_version", type=str, default="0",
                        help="Which version folder to look in, e.g. '0' => results/v0/")
    args = parser.parse_args()
    version_str = f"v{args.experiment_version}"
    
    # The directory we’ll search: results/v0, results/v1, etc.
    base_dir = os.path.join("results", version_str)
    if not os.path.isdir(base_dir):
        print(f"Directory {base_dir} does not exist!")
        return
    
    # We want to find all JSON files of the form:
    #   results/vX/<dataset>/<model>/comparison-accuracy.json
    # We'll build a table with row labels in the index
    # and columns for each <model>_<dataset>.
    
    # We'll store data in a nested dictionary:
    # table[row_label][column_label] = value
    row_labels = [
        "tau_train",
        "tau_measure",
        "tau@3_train",
        "tau@3_measure",
        "tau@5_train",
        "tau@5_measure",
        "tau@10_train",
        "tau@10_measure",
        "tau@15_train",
        "tau@15_measure",
        "acc@1_train",
        "acc@1_measure",
        "acc@3_train",
        "acc@3_measure",
        "acc@5_train",
        "acc@5_measure",
        "acc@10_train",
        "acc@10_measure",
        "acc@15_train",
        "acc@15_measure",
        "avg_train_acc",
        "avg_test_acc",
        "avg_train_acc_top_15",
        "avg_test_acc_top_15",
    ]
    table = {rl: {} for rl in row_labels}  # row_label -> { column_label -> value }
    
    # Walk through the base_dir structure
    # e.g. results/v0/cifar10/cnn_0/comparison-accuracy.json
    # We'll interpret "cifar10" as dataset, "cnn_0" as model
    for dataset_name in sorted(os.listdir(base_dir)):
        dataset_path = os.path.join(base_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        
        # Inside dataset_name, we have model directories
        for model_name in sorted(os.listdir(dataset_path)):
            model_path = os.path.join(dataset_path, model_name)
            if not os.path.isdir(model_path):
                continue
            
            json_file = os.path.join(model_path, "comparison-accuracy.json")
            if os.path.isfile(json_file):
                column_label = f"{model_name}-{dataset_name}"
                parsed_data = parse_comparison_file(json_file)
                # Fill in table
                for rl in row_labels:
                    if rl in ["avg_train_acc", "avg_test_acc", "avg_train_acc_top_15", "avg_test_acc_top_15"]:
                        continue
                    table[rl][column_label] = round(parsed_data[rl], 4)
            else:
                print(f"File not found: {json_file}")

            result_file = os.path.join(model_path, "results.csv")
            if os.path.isfile(result_file):
                column_label = f"{model_name}-{dataset_name}"
                results = pd.read_csv(result_file)
                # Overall average train/test accuracy
                avg_train_acc = results['train_accuracy'].mean()
                avg_test_acc = results['test_accuracy'].mean()
                table['avg_train_acc'][column_label] = round(avg_train_acc, 4)
                table['avg_test_acc'][column_label] = round(avg_test_acc, 4)
                # Top 15 average train/test accuracy
                avg_train_acc_top_15 = results.sort_values(by='test_accuracy', ascending=False)['train_accuracy'][:15].mean()
                avg_test_acc_top_15 = results.sort_values(by='test_accuracy', ascending=False)['test_accuracy'][:15].mean()
                table['avg_train_acc_top_15'][column_label] = round(avg_train_acc_top_15, 4)
                table['avg_test_acc_top_15'][column_label] = round(avg_test_acc_top_15, 4)
            else:
                print(f"File not found: {result_file}")
    
    # Convert table to a DataFrame
    df = pd.DataFrame(table).T  # we stored row_label -> column_label, so transpose
    # By default, row labels become columns and column labels become rows,
    # so we invert that with .T. Another approach is to store data differently.
    
    # We actually want each row_label as the DF's index,
    # and each column_label as a DF column. So let's do:
    #    df = pd.DataFrame.from_dict(table, orient="index")
    # That way row labels are the dictionary keys. Then each column_label is a DF column.
    df = pd.DataFrame.from_dict(table, orient="index")
    
    # Reorder rows to the desired order (just in case):
    df = df.reindex(row_labels)
    
    # Save to CSV
    out_csv = os.path.join(base_dir, "aggregated.csv")
    df.to_csv(out_csv, index=True)
    print(f"Aggregated CSV saved to: {out_csv}")

    # ===================================
    # ========== VISUALIZATION ==========
    # ===================================
    NUM_DATASETS = 2
    NUM_MODELS = 3

    experiment_versions = [version_str]
    for v_idx, v in enumerate(experiment_versions):
        fig, axes = plt.subplots(NUM_DATASETS, NUM_MODELS, figsize=(15, 10))
        dataset_names = [f for f in sorted(os.listdir(f"results/{v}")) if os.path.isdir(f"results/{v}/{f}")]
        for d_idx, d in enumerate(dataset_names):
            model_names = [f for f in sorted(os.listdir(f"results/{v}/{d}")) if os.path.isdir(f"results/{v}/{d}/{f}")]
            for m_idx, m in enumerate(model_names):
                file_path = f"results/{v}/{d}/{m}/results.csv"
                df = pd.read_csv(file_path)
                sns.scatterplot(x=df["measure"], y=df["_measure"], ax=axes[d_idx, m_idx])
                sns.regplot(x=df["measure"], y=df["_measure"], scatter=False, ax=axes[d_idx, m_idx])
                axes[d_idx, m_idx].set_xlabel("Measure")
                axes[d_idx, m_idx].set_ylabel("_Measure")
                axes[d_idx, m_idx].set_title(f"{d} - {m}")
                axes[d_idx, m_idx].grid(True)
                corr, _ = pearsonr(df["measure"], df["_measure"])
                axes[d_idx, m_idx].text(0.5, 0.9, f"Correlation: {corr:.4f}", horizontalalignment="center", verticalalignment="center", transform=axes[d_idx, m_idx].transAxes)
        plt.suptitle(f"Scatter Plot of Measure vs _Measure for {experiment_versions[v_idx]}")
        plt.tight_layout()
        plt.savefig(f"results/{v}/scatterplot.png")
        plt.close()
        print("Scatter plot saved to: ", f"results/{v}/scatterplot.png")


if __name__ == "__main__":
    main()
