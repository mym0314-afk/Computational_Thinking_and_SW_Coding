import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def load_image(path):
  if not path.lower().endswith(('.png', '.jpg', '.jpeg')):
    raise ValueError("PNG/JPG 이미지 파일만 지원합니다.")
  
  img = cv2.imread(path)
  if img is None:
    raise FileNotFoundError(f"이미지를 불러올 수 없습니다.")
  
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

def hr_to_lr(img, scale=4):
  h, w, _ = img.shape
  return cv2.resize(img, (w//scale, h//scale), interpolation=cv2.INTER_CUBIC)

def bicubic_sr(lr, scale=4):
  h, w, _ = lr.shape
  return cv2.resize(lr, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

def edsr_sr(model, lr_img, device):
  lr = torch.from_numpy(lr_img).float()/255.
  lr = lr.permute(2, 0, 1).unsqueeze(0).to(device)
  
  with torch.no_grad():
    sr = model(lr)
    
  sr = sr.squeeze().permute(1, 2, 0).cpu().numpy()
  sr = np.clip(sr*255, 0, 255).astype(np.uint8)
  return sr

def evaluate(hr, sr):
  p = peak_signal_noise_ratio(hr, sr)
  s = structural_similarity(hr, sr, shannel_axis=2)
  return p, s