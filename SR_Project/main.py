import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from model import SRCNN

IMAGE_PATH = "input/test.png"
SCALE = 2
EPOCHS = 100
LR = 1e-5

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

hr = cv2.imread(IMAGE_PATH)
hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

# YCbCr 변환
hr_ycbcr = cv2.cvtColor(hr, cv2.COLOR_RGB2YCrCb)
hr_y = hr_ycbcr[:, :, 0]

h, w = hr_y.shape
lr = cv2.resize(hr_y, (w // SCALE, h // SCALE), interpolation=cv2.INTER_CUBIC)
bicubic_y = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)

def to_tensor(img):
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float() / 255.

hr_t = to_tensor(hr_y).to(device)
lr_t = to_tensor(bicubic_y).to(device)

model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 학습, 에포크 100
model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    sr = model(lr_t)
    loss = criterion(sr, hr_t)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    sr_y = model(lr_t)

sr_y = sr_y.squeeze().cpu().numpy()
sr_y = np.clip(sr_y * 255, 0, 255).astype(np.uint8)

# 복원
sr_ycbcr = hr_ycbcr.copy()
sr_ycbcr[:, :, 0] = sr_y
sr_rgb = cv2.cvtColor(sr_ycbcr, cv2.COLOR_YCrCb2RGB)

bicubic_ycbcr = hr_ycbcr.copy()
bicubic_ycbcr[:, :, 0] = bicubic_y.astype(np.uint8)
bicubic_rgb = cv2.cvtColor(bicubic_ycbcr, cv2.COLOR_YCrCb2RGB)

# PSNR
psnr_bi = peak_signal_noise_ratio(hr_y, bicubic_y)
psnr_sr = peak_signal_noise_ratio(hr_y, sr_y)

print(f"Bicubic PSNR: {psnr_bi:.2f}")
print(f"SRCNN   PSNR: {psnr_sr:.2f}")

# 시각화
plt.figure(figsize=(10, 8))

titles = ["Original", "Bicubic", "SRCNN", "HR"]
images = [hr, bicubic_rgb, sr_rgb, hr]

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()