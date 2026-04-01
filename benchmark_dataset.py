import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from src import AdaFaceVerifier
import itertools
import json

def main():
    parser = argparse.ArgumentParser(description="Benchmarking AdaFace on a Dataset")
    parser.add_argument("--data_dir", default="assets/dataset", help="Path to dataset (each subfolder = 1 person)")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Dataset directory not found: {args.data_dir}")
        return

    # Initialize verifier
    verifier = AdaFaceVerifier(args.config)
    
    identities = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    print(f"Found {len(identities)} identities.")

    # 1. Extract all embeddings
    all_embeddings = {} 
    
    for identity in tqdm(identities, desc="Extracting embeddings"):
        identity_path = os.path.join(args.data_dir, identity)
        images = [f for f in os.listdir(identity_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        identity_embs = []
        for img_name in images:
            img_path = os.path.join(identity_path, img_name)
            try:
                emb = verifier.get_embedding(img_path)
                identity_embs.append(emb)
            except Exception as e:
                # Silently skip errors in large datasets
                pass
                
        if identity_embs:
            all_embeddings[identity] = identity_embs

    # 2. Calculate Positives (Same Person)
    positives = []
    for identity, embs in all_embeddings.items():
        if len(embs) > 1:
            for i, j in itertools.combinations(range(len(embs)), 2):
                sim = torch.dot(embs[i], embs[j]).item()
                positives.append(sim)

    # 3. Calculate Negatives (Different People)
    negatives = []
    id_list = list(all_embeddings.keys())
    num_neg_samples = 10000 
    
    for i, j in itertools.combinations(range(len(id_list)), 2):
        id1, id2 = id_list[i], id_list[j]
        sim = torch.dot(all_embeddings[id1][0], all_embeddings[id2][0]).item()
        negatives.append(sim)
        if len(negatives) >= num_neg_samples:
            break

    # 4. Results
    if not positives or not negatives:
        print("Not enough data.")
        return

    pos_mean, pos_std = np.mean(positives), np.std(positives)
    neg_mean, neg_std = np.mean(negatives), np.std(negatives)
    
    suggested_threshold = (pos_mean + neg_mean) / 2

    print("\n" + "="*50)
    print(f"{'Metric':<25} | {'Value':<10}")
    print("-" * 50)
    print(f"{'Positives (Same Person)':<25} | {pos_mean:.4f} (std: {pos_std:.4f})")
    print(f"{'Negatives (Diff Person)':<25} | {neg_mean:.4f} (std: {neg_std:.4f})")
    print(f"{'Recommended Threshold':<25} | {suggested_threshold:.4f}")
    print("-" * 50)
    
    # 5. Save Metrics for DVC
    metrics = {
        "pos_mean": pos_mean,
        "pos_std": pos_std,
        "neg_mean": neg_mean,
        "neg_std": neg_std,
        "suggested_threshold": suggested_threshold,
        "num_positives": len(positives),
        "num_negatives": len(negatives)
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved to metrics.json")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
