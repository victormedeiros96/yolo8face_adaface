import io
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from src import AdaFaceVerifier
from src.utils import get_similarity

app = FastAPI(title="AdaFace High Performance API")

# Global Verifier Instance (Loaded once at startup)
verifier = AdaFaceVerifier("configs/config.yaml")

def load_image_from_bytes(file_bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_bgr

@app.post("/verify")
async def verify(img1: UploadFile = File(...), img2: UploadFile = File(...)):
    """Compare two faces and return similarity."""
    try:
        b1 = await img1.read()
        b2 = await img2.read()
        
        i1 = load_image_from_bytes(b1)
        i2 = load_image_from_bytes(b2)
        
        match, score = verifier.verify(i1, i2)
        
        # Converte para escala de 0 a 100 para facilitar o entendimento
        score_percent = round(max(0.0, float(score)) * 100, 2)
        
        return {
            "match": match,
            "score_cosine": float(score),
            "score_percent": f"{score_percent}%",
            "threshold": verifier.threshold,
            "threshold_percent": f"{verifier.threshold * 100}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract")
async def extract_embeddings(files: List[UploadFile] = File(...)):
    """
    HIGH SPEED BATCH EXTRACTION.
    Send multiple files to process them in parallel on the GPU.
    """
    try:
        imgs = []
        for file in files:
            content = await file.read()
            img = load_image_from_bytes(content)
            if img is not None:
                imgs.append(img)
        
        if not imgs:
            return {"embeddings": []}

        # Run batch inference
        embs_tensors = verifier.get_embeddings_batch(imgs)
        
        # Convert tensors to list for JSON response
        results = [emb.cpu().numpy().tolist() for emb in embs_tensors]
        
        return {
            "count": len(results),
            "embeddings": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ready", "device": str(verifier.device)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
