# 📷 Super-Resolution using SRCNN (Y Channel Based)

본 프로젝트는 Super-Resolution(SR) 문제를 다루며,  
SRCNN(Super-Resolution Convolutional Neural Network)을 직접 구현하여  
YCbCr 색공간의 Y 채널(밝기 정보)만을 이용해 해상도 복원을 수행한다.

본 구현은 SRCNN 원 논문 방식을 따르며,  
Bicubic interpolation과의 성능 비교를 통해 SR의 효과를 분석한다.

---

## 🔍 Super-Resolution이란?

Super-Resolution(SR)은 저해상도(LR) 이미지로부터  
고해상도(HR) 이미지를 복원하는 컴퓨터 비전 문제이다.

본 프로젝트에서는 다음 문제를 학습한다.

Bicubic으로 업샘플된 흐릿한 이미지 → 원본 고해상도 이미지

---

## 🎨 YCbCr 색공간과 Y 채널 사용 이유

이미지는 RGB가 아닌 YCbCr 색공간으로 변환하여 처리한다.

- Y : 밝기 (Luminance)
- Cb, Cr : 색상 성분

이미지의 윤곽선, 에지, 디테일 정보는 대부분 Y 채널에 존재하며,  
사람의 시각 또한 색보다 밝기 변화에 더 민감하다.

SRCNN, FSRCNN, VDSR, EDSR 등 대부분의 SR 논문은  
Y 채널 기준 PSNR을 사용하므로, 본 프로젝트 역시 동일한 방식을 따른다.

학습과 평가는 Y 채널만 사용하며,  
Cb/Cr 채널은 그대로 유지하여 최종 RGB 이미지를 복원한다.

---

## 🧠 모델 구조 (SRCNN)

SRCNN은 다음과 같은 3-layer CNN 구조를 가진다.

Input (Bicubic Y)  
→ Conv(9×9) + ReLU  
→ Conv(5×5) + ReLU  
→ Conv(5×5)  
→ Output (SR Y)

- 입력/출력: 1채널 (Y 채널)
- Loss Function: MSE (Mean Squared Error)

---

## 📁 프로젝트 구조
``` bash
SR_Project/
├─ input/
│  └─ test.png
├─ model.py
├─ main.py
├─ README.md
└─ requirements.txt
```

---

## ⚙️ 실행 환경

- Python 3.9 이상
- CPU 환경 (GPU 없어도 실행 가능)
- Windows + VS Code

---

## 📦 라이브러리 설치

```bash
pip install -r requirements.txt
```
---

## 실행 방법
```bash
python main.py
```
