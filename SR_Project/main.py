import os
import torch
import cv2
import matplotlib.pyplot as plt

from model import EDSR
from utils import *

IMAGE_PATH = "input/test.png"
SCALE = 4
PRETRAINED_PATH = "pretrained/edsr_x4.pt"

os.makedirs("results", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

hr = load_image(IMAGE_PATH)
lr = hr_to_lr(hr, SCALE)

model = EDSR(scale=SCALE).to(device)
state = torch.load(PRETRAINED_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

sr_bicubic = bicubic_sr(lr, SCALE)
sr_edsr = edsr_sr(model, lr, device)

p_bi, s_bi = evaluate(hr, sr_bicubic)
p_ed, s_ed = evaluate(hr, sr_edsr)

print(f"Bicubic PSNR: {p_bi:.2f}, SSIM: {s_bi:.4f}")
print(f"EDSR    PSNR: (p_ed:.2f), SSIM: (s_ed:.4f)")

cv2.imwrite("results/SR_EDSR.png", cv2.cvtColor(sr_edsr, cv2.COLOR_RGB2BGR))

plt.figure(figsize=(18, 6))
titles = ["HR", "LR", "Bicubic", "EDSR"]
images = [hr, lr, sr_bicubic, sr_edsr]

for i in range(4):
  plt.subplot(1, 4, i+1)
  plt.imshow(images[i])
  plt.title(titles[i])
  plt.axis("off")
  
plt.show()