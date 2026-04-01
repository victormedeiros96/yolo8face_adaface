import time
import torch
import os
import numpy as np
import cv2
from src.verification import AdaFaceVerifier
from src.utils import detector, preprocess_image
from PIL import Image

def profile_production_speed(img_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nPROFILING PRODUCTION SPEED (YOLOv8 + AdaFace) on {device}")
    
    img_bgr = cv2.imread(img_path)
    
    # 1. WARMUP (Aquece a GPU e carrega todos os modelos na VRAM)
    print("Iniciando Warmup (Aquece a GPU)...")
    _ = detector(img_bgr, verbose=False)
    _ = preprocess_image(img_path)
    print("Warmup concluido! Disparando bateria de testes...\n")
    print("-" * 55)
    
    # 2. RUN 10 TIMES for statistics
    times = []
    num_runs = 10
    
    for i in range(num_runs):
        start = time.perf_counter()
        
        # O pipeline completo que importa para você está aqui:
        _ = preprocess_image(img_path)
        
        t_total = (time.perf_counter() - start) * 1000
        times.append(t_total)
        print(f"Run {i+1:<2}:   {t_total:.2f} ms")

    print("-" * 55)
    avg_time = np.mean(times)
    print(f"AVERAGE PIPELINE TIME:      {avg_time:.2f} ms")
    print(f"FPS ESTIMADO (PROD):        {int(1000/avg_time)} FPS")
    
if __name__ == "__main__":
    test_img = "foto1.jpg"
    if os.path.exists(test_img):
        profile_production_speed(test_img)
