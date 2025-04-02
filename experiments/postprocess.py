import math
import json

import pandas as pd


def postprocess_results(gt_metric_name, top_k, results_path):
    """
    Loads the results CSV from the results_path,
    sorts and ranks the configurations based on multiple metrics,
    computes Kendall tau correlations and top-k accuracies, and saves
    the output to a JSON file.
    """
    try:
        results = pd.read_csv(results_path)
    except FileNotFoundError:
        print(f"Results file {results_path} not found.")
        return

    # Decide sort order: lower rank is better for loss/measure, higher for accuracy.
    def get_sort_order(metric_name):
        if "accuracy" in metric_name:
            return False
        elif "loss" in metric_name or "measure" in metric_name:
            return True

    # Sort based on loss-related metrics
    loss_metrics = ["train_loss", "measure"]
    results = results.sort_values(
        by=loss_metrics,
        ascending=[get_sort_order(metric) for metric in loss_metrics],
        kind="mergesort"
    ).reset_index(drop=True)

    # Sort based on accuracy-related metrics
    accuracy_metrics = ["train_accuracy", "test_accuracy"]
    results = results.sort_values(
        by=accuracy_metrics,
        ascending=[get_sort_order(metric) for metric in accuracy_metrics],
        kind="mergesort"
    ).reset_index(drop=True)

    # Create ranking columns based on various metrics.
    metrics = ["train_loss", "train_accuracy", "test_accuracy", "measure"]
    for metric in metrics:
        col_name = f"order_based_on_{metric}"
        results[col_name] = results[metric].rank(method='min', ascending=get_sort_order(metric)) - 1

    # Sort based on the ground truth metric.
    results = results.sort_values(
        by=f"test_{gt_metric_name}",
        ascending=get_sort_order(f"test_{gt_metric_name}"),
        kind="mergesort"
    ).reset_index(drop=True)

    # Compute Kendall tau between test_{gt_metric_name} and candidate metrics.
    candidate_metrics = ["train_loss", "train_accuracy", "measure"]
    ks = [3, 5, 10, 15]
    taus = [[] for _ in range(len(candidate_metrics))]
    for i, metric in enumerate(candidate_metrics):
        test_rank = results[f"order_based_on_test_{gt_metric_name}"]
        tau_all = test_rank.corr(results[f"order_based_on_{metric}"], method="kendall")
        taus[i].append(tau_all)
        for k in ks:
            tau_k = test_rank[:k].corr(results[f"order_based_on_{metric}"][:k], method="kendall")
            taus[i].append(tau_k)

    # Compute top-k accuracies.
    top_k_accuracies_list = [[] for _ in range(len(candidate_metrics))]
    for k in range(0, top_k):
        for metric in [f"test_{gt_metric_name}", "train_loss", "train_accuracy", "measure"]:
            results[f"top_{k+1}_accuracy_{metric}"] = results[f"order_based_on_{metric}"] <= k

        for i, metric in enumerate(candidate_metrics):
            mask_test = results[f"top_{k+1}_accuracy_test_{gt_metric_name}"] if f"top_{k+1}_accuracy_test_{gt_metric_name}" in results.columns else results[f"top_{k+1}_accuracy_test_accuracy"]
            mask_candidate = results[f"top_{k+1}_accuracy_{metric}"]
            proportion = len(results[mask_test & mask_candidate]) / len(results[mask_test])
            top_k_accuracies_list[i].append(min(round(proportion, 4), 1))

    output = {
        "gt_metric_name": gt_metric_name
    }
    for i, metric in enumerate(candidate_metrics):
        output[f"tau_{metric}"] = {
            "all": round(taus[i][0], 4),
            "top_3": round(taus[i][1], 4),
            "top_5": round(taus[i][2], 4),
            "top_10": round(taus[i][3], 4),
            "top_15": round(taus[i][4], 4)
        }
        output[f"top_k_accuracy_{metric}_list"] = top_k_accuracies_list[i]

    output_path = "/".join(results_path.split("/")[:-1]) + f"/comparison-{gt_metric_name}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4, default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else o)
