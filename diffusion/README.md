
# DDPM for MNIST Generation

이 프로젝트는 PyTorch를 사용하여 기본적인 DDPM(Denoising Diffusion Probabilistic Model)을 구현한 것입니다. MNIST 손글씨 숫자 데이터셋을 학습하여 새로운 숫자 이미지를 생성하는 것을 목표로 합니다.

## 프로젝트 구조

```
diffusion/
├── README.md           # 본 파일, 프로젝트 설명
├── config.py           # 모델 및 학습 관련 하이퍼파라미터 설정
├── main.py             # 전체 학습 프로세스를 실행하는 메인 스크립트
├── sampling.py         # 학습된 모델로 새로운 이미지를 생성하는 스크립트
├── dataloader.py       # MNIST 데이터셋을 불러오는 데이터로더
├── diffusion_utils.py  # Forward/Reverse Process 등 확산 모델의 핵심 로직
├── model.py            # 노이즈 예측을 위한 U-Net 모델 아키텍처
└── train.py            # 모델 학습을 위한 훈련 루프
```

- **`config.py`**: 학습 에폭, 배치 크기, 이미지 크기, Diffusion 타임스텝 등 모든 주요 하이퍼파라미터를 관리합니다.
- **`dataloader.py`**: `torchvision`을 사용하여 MNIST 데이터셋을 불러오고, 학습에 맞게 전처리합니다.
- **`diffusion_utils.py`**: DDPM의 핵심 수학적 로직을 포함합니다. 노이즈를 추가하는 Forward Process와, 노이즈를 점진적으로 제거하여 이미지를 생성하는 Reverse Process(Sampling) 함수들이 정의되어 있습니다.
- **`model.py`**: 노이즈 낀 이미지와 타임스텝을 입력받아 추가된 노이즈를 예측하는 U-Net 신경망을 정의합니다.
- **`train.py`**: U-Net 모델이 노이즈를 정확히 예측하도록 학습시키는 훈련 루프를 구현합니다. 학습된 모델 가중치는 `diffusion/checkpoints/` 폴더에 저장됩니다.
- **`main.py`**: 모든 구성 요소를 결합하여 전체 학습 과정을 시작하는 메인 진입점입니다.
- **`sampling.py`**: 학습이 완료된 모델을 불러와 새로운 숫자 이미지를 생성하고, `diffusion/outputs/` 폴더에 이미지 파일로 저장합니다.

## 핵심 개념

DDPM은 두 가지 과정으로 동작합니다.
1.  **Forward Process (확산 과정)**: 원본 이미지에 점진적으로 가우시안 노이즈를 추가하여, 이미지가 완전한 노이즈가 될 때까지 여러 단계를 거칩니다.
2.  **Reverse Process (복원 과정)**: U-Net 모델을 학습시켜, 완전한 노이즈에서 시작하여 점진적으로 노이즈를 제거함으로써 원본과 유사한 새로운 이미지를 생성합니다.

## 사용법

### 1. 필요 라이브러리 설치

코드를 실행하기 위해 `torch`, `torchvision`, `tqdm`이 필요합니다.

```bash
pip install torch torchvision tqdm
```

### 2. 모델 학습

먼저 모델을 학습시켜야 합니다. 프로젝트의 루트 디렉토리에서 다음 명령어를 실행하세요. 학습은 `config.py`에 설정된 `DEVICE`에 따라 CPU 또는 GPU에서 실행됩니다.

```bash
python -m diffusion.main
```

학습이 진행되면서 각 에폭의 평균 손실이 출력되고, 에폭이 끝날 때마다 모델 체크포인트가 `diffusion/checkpoints/` 폴더에 `.pth` 파일로 저장됩니다.

### 3. 새로운 이미지 생성 (샘플링)

학습이 완료되면, 저장된 모델을 사용하여 새로운 이미지를 생성할 수 있습니다. 다음 명령어를 실행하세요.

```bash
python -m diffusion.sampling
```

이 스크립트는 `checkpoints` 폴더에서 마지막 에폭의 모델을 불러와 16개의 새로운 숫자 이미지를 생성하고, `diffusion/outputs/generated_images.png` 파일로 저장합니다.
