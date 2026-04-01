import sys
import os
import torch
from src import AdaFaceVerifier

def main():
    photos = ["foto1.jpg", "foto2.jpg", "foto3.jpg", "foto4.jpg", "foto5.jpg"]
    
    # Check if files exist
    existing_photos = [f for f in photos if os.path.exists(f)]
    if not existing_photos:
        print("No photos found in current directory.")
        return
        
    print(f"Loading verifier for {len(existing_photos)} photos: {existing_photos}...")
    try:
        verifier = AdaFaceVerifier("configs/config.yaml")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Extract all embeddings first
    embeddings = {}
    for photo in existing_photos:
        print(f"Extracting embedding for {photo}...")
        embeddings[photo] = verifier.get_embedding(photo)

    # Compare all pairs (Upper triangle + diagonal)
    print("\nSimilarity Matrix:")
    print("-" * 65)
    header = "       | " + " | ".join([f"{p:<8}" for p in existing_photos])
    print(header)
    print("-" * 65)

    for p1 in existing_photos:
        row = f"{p1:<6} | "
        for p2 in existing_photos:
            # We already have normalized embeddings, so dot product is enough
            sim = torch.dot(embeddings[p1], embeddings[p2]).item()
            row += f"{sim:8.4f} | "
        print(row)
    print("-" * 65)

    # Summary of matches (above threshold)
    print("\nMatches (threshold > 0.45):")
    for i, p1 in enumerate(existing_photos):
        for j, p2 in enumerate(existing_photos):
            if i < j: # Avoid self and duplicate pairs
                sim = torch.dot(embeddings[p1], embeddings[p2]).item()
                if sim > verifier.threshold:
                    print(f"✅ {p1} vs {p2}: {sim:.4f} (MATCH)")
                else:
                    print(f"❌ {p1} vs {p2}: {sim:.4f} (DIFF)")

if __name__ == "__main__":
    main()
