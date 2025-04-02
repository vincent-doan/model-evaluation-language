import faiss
import numpy as np
import torch
from tqdm import tqdm


class BaseDataset:
    def __init__(self, data_dir="./data", download=True):
        self.data_dir = data_dir
        self.download = download
        self.samples = None
        self.labels = None

    def load(self):
        raise NotImplementedError("Subclasses must implement load()")

    def augment(self, samples, augmentation_multiplier, **kwargs):
        return samples

    def preprocess(self, train_set_size, test_set_size, unlabelled_set_size=-1, seed=2411):
        raise NotImplementedError("Subclasses must implement preprocess()")
    
    def group_unlabelled(self, X_train_enc, X_unlabelled_enc, trained_model, device):
        
        def get_logits(model, inputs, device, batch_size=256):
            model.eval()
            all_logits = []
            n = inputs["input_ids"].size(0)
            for i in tqdm(range(0, n, batch_size), desc="Computing logits", total=n // batch_size):
                batch = {k: v[i: i+batch_size].to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = model.get_logits(**batch)
                all_logits.append(logits.cpu())
            return torch.cat(all_logits, dim=0).numpy()
    
        # Compute logits
        train_logits = get_logits(trained_model, X_train_enc, device)
        unlabelled_logits = get_logits(trained_model, X_unlabelled_enc, device)

        # Normalize for nearest-neighbor search
        def normalize_vectors(vectors):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return vectors / norms
    
        train_norm = normalize_vectors(train_logits)
        unlabelled_norm = normalize_vectors(unlabelled_logits)
    
        # Use FAISS for efficient nearest-neighbor search
        d = train_norm.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(train_norm)
        _, I = index.search(unlabelled_norm, 1)
        I = I.reshape(-1)
    
        # Store groupings
        from collections import defaultdict
        grouping = defaultdict(list)
        for idx, centroid_idx in enumerate(I):
            grouping[int(centroid_idx)].append(int(idx))
    
        return grouping, train_logits, unlabelled_logits
