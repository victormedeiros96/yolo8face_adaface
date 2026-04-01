import argparse
import sys
import os
from src import AdaFaceVerifier

def main():
    parser = argparse.ArgumentParser(description="CVLFace (AdaFace) User Verification Tool")
    parser.add_argument("--img1", help="Path to first image (query/image1)")
    parser.add_argument("--img2", help="Path to second image (verification/image2)")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--mode", choices=["verify", "identify"], default="verify", help="Mode of operation")
    parser.add_argument("--gallery", nargs="+", help="Gallery images for identification (supports glob like *.jpg)")
    parser.add_argument("--token", help="Hugging Face Token (optional, if private)")

    args = parser.parse_args()

    # Set HF Token if provided via CLI
    if args.token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.token

    # Initialize verifier
    try:
        verifier = AdaFaceVerifier(args.config)
    except Exception as e:
        print(f"Error initializing verifier: {e}")
        sys.exit(1)

    if args.mode == "verify":
        if not args.img1 or not args.img2:
            print("Usage: python main.py --mode verify --img1 <path1> --img2 <path2>")
            return
            
        print(f"Verifying {args.img1} vs {args.img2}...")
        is_same, similarity = verifier.verify(args.img1, args.img2)
        
        print("\n" + "="*40)
        print(f"Similarity Score: {similarity:.4f}")
        print(f"Threshold:        {verifier.threshold}")
        print("-" * 40)
        print(f"Result:           {'SAME PERSON ✅' if is_same else 'DIFFERENT PERSON ❌'}")
        print("="*40 + "\n")

    elif args.mode == "identify":
        if not args.img1 or not args.gallery:
            print("Usage: python main.py --mode identify --img1 <path> --gallery assets/gallery/*.jpg")
            return
            
        results = verifier.identify(args.img1, args.gallery)
        print("\n" + "="*50)
        print(f"Query Image: {args.img1}")
        print("-" * 50)
        print(f"{'Path':<40} | {'Sim':<6}")
        print("-" * 50)
        for path, sim in results[:10]: # Top 10
            match_str = " (MATCH)" if sim > verifier.threshold else ""
            print(f"{path:<40} | {sim:.4f}{match_str}")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()
