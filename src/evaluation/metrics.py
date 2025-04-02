import numpy as np
import torch


def compute_custom_metrics(
        cached_folder_path,
        best_train_accuracy,
        best_test_accuracy,
        grouping,
        device
    ):
    """
    Computes metrics using cached logits to avoid redundant passes.
    """
    # Load cached logits
    cached_train_logits = torch.tensor(np.load(f"{cached_folder_path}/cached_train_logits.npy"))
    cached_unlabelled_logits = torch.tensor(np.load(f"{cached_folder_path}/cached_unlabelled_logits.npy"))

    # Load cached labels
    y_train = np.load(f"{cached_folder_path}/cached_train_labels.npy")
    y_train = torch.tensor(y_train).to(device)
    y_unlabelled = np.load(f"{cached_folder_path}/cached_unlabelled_labels.npy")
    y_unlabelled = torch.tensor(y_unlabelled).to(device)

    # Compute train loss and accuracy using cached logits
    train_probs = torch.softmax(cached_train_logits, dim=-1)
    train_accuracy = best_train_accuracy
    train_loss = 1 - train_accuracy

    # Compute test loss and accuracy (since it's smaller, we still pass through the trained_model)
    test_accuracy = best_test_accuracy
    test_loss = 1 - test_accuracy

    # Compute clustering-based metric
    total_distance = 0.0
    _total_distance = 0.0
    unlabelled_set_size = cached_unlabelled_logits.shape[0]

    for centroid_id_str, member_ids in grouping.items():
        centroid_id = int(centroid_id_str)
        centroid_probs = train_probs[centroid_id].numpy()

        member_probs = torch.softmax(cached_unlabelled_logits[member_ids], dim=-1).numpy()
        in_cluster_distance = np.mean(np.abs(member_probs - centroid_probs), axis=0).sum()
        total_distance += in_cluster_distance * (len(member_ids) / unlabelled_set_size)

        # Actual measure: 0-1 loss on unlabelled members and centroid
        centroid_loss = int(y_train[centroid_id].item() != centroid_probs.argmax())
        member_loss = np.mean((y_unlabelled[member_ids].cpu().numpy() != member_probs.argmax(axis=1)).astype(int))
        _in_cluster_distance = np.abs(member_loss - centroid_loss).sum()
        _total_distance += _in_cluster_distance * (len(member_ids) / unlabelled_set_size)

    # Aggregate custom metrics
    measure = total_distance + train_loss
    _measure = _total_distance + train_loss

    return train_loss, train_accuracy, test_loss, test_accuracy, total_distance, measure, _total_distance, _measure
