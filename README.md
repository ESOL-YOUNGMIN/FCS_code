# Machine-Learning-Based Prediction of Key Operational Parameters in a CH4/H2/Air Swirl Combustor from a Flame Chemiluminescence Spectrum
---

## Overview / 개요

This repository provides the complete source code for reproducing the results presented in the paper. The code implements a two-stage deep learning framework that predicts three combustion operational parameters from a single flame chemiluminescence spectrum (FCS) input (1600 wavelength points).

본 저장소는 논문의 결과를 재현하기 위한 전체 소스 코드를 제공합니다. 본 코드는 단일 화염 화학발광 스펙트럼(FCS) 입력(1600개 파장 포인트)으로부터 3가지 연소 운전 변수를 예측하는 2단계 딥러닝 프레임워크를 구현합니다.

---

## Terminology / 용어 정의

| Symbol | Description | 설명 | Unit | 
|--------|-------------|------|------|
| FCS | Flame Chemiluminescence Spectrum | 
| V̇ (Vdot) | Total combustion flow rate | 총 연소 유량 | L/min |
| φ (phi) | Global equivalence ratio | 총괄 당량비 | - |
| X_H2 (XH2) | H2 blend ratio | H2 혼합비 | mol% | 

---

## Repository Structure / 저장소 구조

```
.
├── data_preprocessing.py            # Data loading, checkerboard splitting, normalization
│                                    # 데이터 로드, 체커보드 분할, 정규화
│
├── model_CAE/                       # Stage 1: Convolutional Autoencoder (feature extractor)
│   │                                # 1단계: 합성곱 오토인코더 (특징 추출기)
│   ├── BOHB_CAE.py                  #   Hyperparameter optimization via BOHB
│   └── Optimized_CAE.py             #   Training with optimized hyperparameters
│
├── model_proposed/                  # Stage 2: Proposed model (frozen CAE encoder + regressors)
│   │                                # 2단계: 제안 모델 (동결된 CAE 인코더 + 회귀 모델)
│   ├── BOHB_regressor_Vdot.py       #   BOHB optimization for V̇ regressor
│   ├── BOHB_regressor_phi.py        #   BOHB optimization for φ regressor
│   ├── BOHB_regressor_XH2.py        #   BOHB optimization for X_H2 regressor
│   ├── Optimized_regressor_Vdot.py  #   Optimized V̇ regressor training
│   ├── Optimized_regressor_phi.py   #   Optimized φ regressor training
│   ├── Optimized_regressor_XH2.py   #   Optimized X_H2 regressor training
│   └── MACs_calculate.py            #   Computational complexity (MACs) calculation
│
├── model_benchmark1/                # Benchmark model 1: Three single-output CNNs
│   │                                # 벤치마크 모델 1: 단일 출력 CNN 3개
│   ├── BOHB_benchmark1_*.py         #   BOHB optimization
│   └── Optimized_benchmark1_*.py    #   Training with optimized hyperparameters
│
├── model_derivative/                # Derivative model: CAE + multi-output regressor
│   │                                # 파생 모델: CAE + 다중 출력 회귀 모델
│   ├── BOHB_derivative.py           #   BOHB optimization
│   └── Optimized_derivative.py      #   Training with optimized hyperparameters
│
├── model_benchmark2/                # Benchmark model 2: Multi-output CNN
│   │                                # 벤치마크 모델 2: 다중 출력 CNN
│   ├── BOHB_benchmark2.py           #   BOHB optimization
│   └── Optimized_benchmark2.py      #   Training with optimized hyperparameters
│
├── Grad_RAM/                        # Gradient-weighted Regression Activation Mapping
│   │                               
│   ├── Grad_RAM_Vdot.py             #   Grad-RAM for V̇
│   ├── Grad_RAM_phi.py              #   Grad-RAM for φ
│   ├── Grad_RAM_XH2.py              #   Grad-RAM for X_H2
│   ├── find_outlier.py              #   Outlier detection / 이상치 탐지
│   └── lowdataplot.py               #   Low-data regime visualization
│
├── dataplot/                        # Data visualization / 데이터 시각화
│   ├── dataplot.py
│   └── dataplot_OHCH.py
│
├── checkpoint/                      # Saved model weights / 저장된 모델 가중치
│   ├── CAE/                         #   Pre-trained CAE encoder
│   ├── combine_Vdot/                #   Proposed V̇ regressor
│   ├── combine_phi/                 #   Proposed φ regressor
│   ├── combine_XH2/                 #   Proposed X_H2 regressor
│   └── combine_Total/               #   Multi-output model
│
└── _legacy/                         # Deprecated code / 사용되지 않는 코드
```

---

## Model Architecture / 모델 구조

The proposed model adopts a two-stage training approach.

제안 모델은 2단계 학습 방식을 사용합니다.

### Stage 1 — Convolutional Autoencoder (CAE) / 1단계 — 합성곱 오토인코더

The CAE is trained for spectrum reconstruction. After training, the encoder is frozen and reused as a shared feature extractor in Stage 2.

CAE는 스펙트럼 복원을 위해 학습됩니다. 학습 후 인코더는 동결되어 2단계에서 공유 특징 추출기로 재사용됩니다.

```
Input (1600, 1)
  → Conv1D(31, kernel=4, stride=4, ReLU)   → (400, 31)
  → Conv1D(42, kernel=5, stride=5, ReLU)   → (80, 42)
  → Conv1D(27, kernel=8, stride=8, ReLU)   → (10, 27)
  → Conv1D(22, kernel=10, stride=1, ReLU)  → (1, 22)  [latent features]
  → Conv1DTranspose (symmetric decoder)
  → Output (1600, 1)
```

### Stage 2 — Parameter-Specific Regressors / 2단계 — 변수별 회귀 모델

Each regressor shares the frozen CAE encoder and has independently optimized Dense layers.

각 회귀 모델은 동결된 CAE 인코더를 공유하며, 독립적으로 최적화된 Dense 레이어를 갖습니다.

| Regressor / 회귀 모델 | Dense neurons | Dense layers | Total params | Trainable params |
|----------------------|---------------|-------------|-------------|-----------------|
| V̇ (Vdot) | 163 | 4 | 105,877 | 84,109 |
| φ (phi) | 241 | 3 | 144,197 | 122,429 |
| X_H2 (XH2) | 288 | 1 | 28,681 | 6,913 |

**Unique parameters (shared encoder counted once) / 고유 파라미터 수 (공유 인코더 1회 계산): 235,219**

All hyperparameters were optimized using BOHB (Bayesian Optimization HyperBand).

모든 하이퍼파라미터는 BOHB를 사용하여 최적화되었습니다.

---

## Data Description / 데이터 설명

- **441 experimental conditions**: 7 flow rates (80–140 L/min) × 7 equivalence ratios (0.7–1.0) × 9 H2 blend ratios (0–30 mol%)
- **500 spectra per condition**, each with 1600 wavelength points
- **Data split**: Checkerboard pattern based on condition index parity (even → train, odd → test/validation)
- **Normalization**: Spectra divided by global max; labels scaled to [0, 1] via min-max normalization

---

- **441개 실험 조건**: 유량 7단계 (80–140 L/min) × 당량비 7단계 (0.7–1.0) × H2 몰분율 9단계 (0–30 mol%)
- **조건당 500개 스펙트럼**, 각 1600개 파장 포인트
- **데이터 분할**: 조건 인덱스 홀짝에 기반한 체커보드 패턴 (짝수 → 학습, 홀수 → 테스트/검증)
- **정규화**: 스펙트럼은 전체 최대값으로 나눔; 레이블은 min-max 정규화로 [0, 1] 범위로 스케일링

---

## How to Run / 실행 방법

All scripts assume sequential execution within the same Python session unless otherwise noted. Run `data_preprocessing.py` first to load and prepare the data.

별도 표기가 없는 한 모든 스크립트는 동일한 Python 세션 내에서 순차적으로 실행합니다. 먼저 `data_preprocessing.py`를 실행하여 데이터를 로드하고 준비합니다.

### Step 1: Data Preprocessing / 1단계: 데이터 전처리
```bash
python data_preprocessing.py
```

### Step 2: Train CAE / 2단계: CAE 학습
```bash
# (Optional) Hyperparameter optimization / (선택) 하이퍼파라미터 최적화
python model_CAE/BOHB_CAE.py

# Train with optimized hyperparameters / 최적화된 하이퍼파라미터로 학습
python model_CAE/Optimized_CAE.py
```

### Step 3: Train Regressors / 3단계: 회귀 모델 학습
```bash
# (Optional) Hyperparameter optimization / (선택) 하이퍼파라미터 최적화
python model_proposed/BOHB_regressor_Vdot.py

# Train with optimized hyperparameters / 최적화된 하이퍼파라미터로 학습
python model_proposed/Optimized_regressor_Vdot.py
python model_proposed/Optimized_regressor_phi.py
python model_proposed/Optimized_regressor_XH2.py
```

### Step 4: Evaluation / 4단계: 평가
```bash
python Grad_RAM/Grad_RAM_Vdot.py     # Grad-RAM analysis / Grad-RAM 분석
```

---

## Requirements / 요구 사항

| Package | Version |
|---------|---------|
| Python | 3.9.18 |
| TensorFlow | 2.9.0 |
| Keras | 2.9.0 |
| NumPy | 1.23.2 |
| SciPy | 1.11.3 |
| Pandas | 2.1.1 |
| scikit-learn | 1.3.1 |
| hpbandster | 0.7.4 |
| ConfigSpace | 0.7.1 |
| OpenCV | 4.8.1 |
| Matplotlib | 3.8.0 |
| Seaborn | 0.13.0 |
| psutil | 5.9.0 |

---

## Operating Environment / 운영 환경

| Item / 항목 | Specification / 사양 |
|-------------|---------------------|
| Hardware | MacBook Air (M2, 2022) |
| Processor / 프로세서 | Apple M2 (8-core CPU) |
| RAM | 8 GB |
| OS / 운영체제 | macOS (arm64) |
| Language / 언어 | Python 3.9.18 |
| Framework / 프레임워크 | TensorFlow 2.9.0 / Keras 2.9.0 |
| RAM usage during inference / 추론 시 RAM 사용량 | ~283 MB (Peak RSS) |
| Model weights on disk / 모델 가중치 디스크 크기 | ~2.86 MB (3 models combined) |
