
# Vanilla Transformer 구현체

이 프로젝트는 PyTorch를 사용하여 기본적인 Vanilla Transformer 모델을 구현한 것입니다. 주어진 숫자 시퀀스를 뒤집는 간단한 Sequence-to-Sequence 작업을 예시로 사용합니다.

## 프로젝트 구조

```
Transformer/
├── README.md           # 본 파일, 프로젝트 설명
├── config.py           # 모델 및 학습 관련 하이퍼파라미터 설정
├── main.py             # 전체 학습 프로세스를 실행하는 메인 스크립트
├── dataloader/
│   └── dataloader.py   # 학습에 사용할 데이터셋 및 데이터로더 생성
├── model/
│   └── model.py        # Transformer 모델 아키텍처 정의
└── train/
    └── train.py        # 모델 학습 및 평가를 위한 훈련 루프
```

- **`config.py`**: 모델의 크기, 학습률, 에폭 수 등 모든 주요 하이퍼파라미터를 관리합니다. CPU 환경을 위해 모델 크기가 축소되었습니다.
- **`model/model.py`**: `PositionalEncoding`과 PyTorch의 내장 `nn.Transformer` 모듈을 사용하여 Transformer 모델을 정의합니다.
- **`dataloader/dataloader.py`**: 임의의 숫자 시퀀스와 그 시퀀스를 뒤집은 타겟 시퀀스를 생성하는 PyTorch `Dataset`을 구현합니다. `<PAD>`, `<SOS>`, `<EOS>` 특수 토큰을 사용하여 시퀀스를 처리합니다.
- **`train/train.py`**: 실제 모델 학습이 이루어지는 곳으로, 학습 및 검증 단계를 위한 함수를 포함합니다.
- **`main.py`**: 모든 구성 요소를 가져와 모델, 데이터로더, 옵티마이저 등을 초기화하고 전체 학습 과정을 시작하는 진입점입니다.

## 사용법

### 1. 필요 라이브러리 설치

코드를 실행하기 위해 PyTorch가 필요합니다.

```bash
pip install torch
```

### 2. 학습 시작

프로젝트의 루트 디렉토리에서 다음 명령어를 실행하여 학습을 시작할 수 있습니다.

```bash
python -m Transformer.main
```

스크립트가 실행되면, 설정된 에폭 수만큼 학습이 진행되며 각 에폭마다 훈련 손실(Train Loss)과 검증 손실(Val. Loss)이 출력됩니다.
