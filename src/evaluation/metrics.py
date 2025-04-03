import numpy as np
import torch


def compute_custom_metrics(
        cached_folder_path,
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
    train_probs = torch.softmax(cached_train_logits, dim=-1).to(device)
    y_train_one_hot = torch.nn.functional.one_hot(y_train, num_classes=train_probs.shape[1])
    train_loss = torch.mean(torch.abs(train_probs - y_train_one_hot.float())).item()

    # Compute clustering-based metric
    total_distance = 0.0
    _total_distance = 0.0
    unlabelled_set_size = cached_unlabelled_logits.shape[0]

    for centroid_id_str, member_ids in grouping.items():
        centroid_id = int(centroid_id_str)
        centroid_probs = train_probs[centroid_id].cpu().numpy()

        member_probs = torch.softmax(cached_unlabelled_logits[member_ids], dim=-1).numpy()
        in_cluster_distance = np.mean(np.abs(member_probs - centroid_probs), axis=1).sum() / len(member_ids)
        total_distance += in_cluster_distance * (len(member_ids) / unlabelled_set_size)

        # Actual measure: MAE loss on unlabelled members and centroid
        centroid_loss = np.mean(np.abs(y_train_one_hot[centroid_id].cpu().numpy() - centroid_probs))
        y_unlabelled_one_hot = torch.nn.functional.one_hot(y_unlabelled[member_ids], num_classes=member_probs.shape[1]).cpu().numpy()
        member_loss = np.mean(np.abs(member_probs - y_unlabelled_one_hot), axis=1)
        _in_cluster_distance = np.sum(np.abs(member_loss - centroid_loss)) / len(member_ids)
        _total_distance += _in_cluster_distance * (len(member_ids) / unlabelled_set_size)

    # Aggregate custom metrics
    measure = total_distance + train_loss
    _measure = _total_distance + train_loss

    return train_loss, total_distance, measure, _total_distance, _measure
