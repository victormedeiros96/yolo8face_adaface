import torch
import numpy as np
import os
import cv2
from PIL import Image
from torchvision.transforms import Normalize, ToPILImage
from ultralytics import YOLO

# 1. Initialize YOLOv8-Face
detector = YOLO('model_weights/yolov8n-face.pt')
detector.to('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_image(img_path_or_bgr, target_size=(112, 112)):
    """
    Single image preprocessing fallback.
    """
    if isinstance(img_path_or_bgr, str):
        img_bgr = cv2.imread(img_path_or_bgr)
    else:
        img_bgr = img_path_or_bgr
        
    if img_bgr is None:
        return None

    results = detector(img_bgr, verbose=False)[0]
    if len(results.boxes) > 0:
        box = results.boxes.xyxy[0].cpu().numpy().astype(int)
        face_img = img_bgr[max(0, box[1]):min(img_bgr.shape[0], box[3]), 
                           max(0, box[0]):min(img_bgr.shape[1], box[2])]
        face_img = cv2.resize(face_img, target_size)
    else:
        face_img = cv2.resize(img_bgr, target_size)

    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
    norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return norm(face_tensor).unsqueeze(0).to(detector.device)

def preprocess_batch(imgs_list, target_size=(112, 112)):
    """
    HIGH PERFORMANCE BATCH PREPROCESSING.
    Expects list of BGR images or paths.
    """
    # 1. Load all images if they are paths
    processed_imgs = []
    for img in imgs_list:
        if isinstance(img, str):
            img_bgr = cv2.imread(img)
        else:
            img_bgr = img
        if img_bgr is not None:
            processed_imgs.append(img_bgr)

    if not processed_imgs:
        return None

    # 2. YOLO Batch Inference
    results_list = detector(processed_imgs, verbose=False)
    
    batch_tensors = []
    norm = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    for i, results in enumerate(results_list):
        orig_img = processed_imgs[i]
        if len(results.boxes) > 0:
            box = results.boxes.xyxy[0].cpu().numpy().astype(int)
            face_img = orig_img[max(0, box[1]):min(orig_img.shape[0], box[3]), 
                                max(0, box[0]):min(orig_img.shape[1], box[2])]
            face_img = cv2.resize(face_img, target_size)
        else:
            face_img = cv2.resize(orig_img, target_size)
            
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
        batch_tensors.append(norm(tensor))

    # Stack to (B, 3, 112, 112)
    return torch.stack(batch_tensors).to(detector.device)

def get_similarity(feat1, feat2):
    return torch.dot(feat1.view(-1), feat2.view(-1)).item()
