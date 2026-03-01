# Sentiment Model Comparison (IMDb)

## 1. Project Goal | 프로젝트 목표

This project fine-tunes a pre-trained Transformer model on the IMDb dataset for binary sentiment classification.

이 프로젝트는 IMDb 영화 리뷰 데이터를 활용하여 사전학습된 Transformer 모델을 fine-tuning 하고,  
하이퍼파라미터 변화에 따른 성능 변화를 분석하는 것을 목표로 합니다.

### 핵심 목표

- Baseline 모델 구축
- Epoch 변화에 따른 성능 분석
- 데이터 양 변화에 따른 영향 분석
- 실험 기반 비교 및 해석

---

## 2. Dataset | 데이터셋

- **Dataset**: IMDb movie reviews  
- **Source**: Hugging Face `datasets`  
- **Task**: Binary classification (positive / negative)  
- **Property**: Balanced dataset  

IMDb 데이터는 긍정/부정이 균형 잡힌 이진 분류 데이터입니다.

---

## 3. Model | 사용 모델

### distilbert-base-uncased

**선택 이유:**

- Lightweight Transformer (BERT 대비 가벼움)
- CPU 환경에서도 빠른 학습 가능
- 성능 대비 효율 우수

---

## 4. Experiments | 실험 설계

| Model       | Train Samples | Epochs | Accuracy | F1     |
|------------|--------------|--------|----------|--------|
| distilbert | 2000         | 1      | 0.851    | 0.848  |
| distilbert | 2000         | 3      | 0.877    | 0.877  |
| distilbert | 5000         | 1      | 0.881    | 0.879  |

---

## 5. Analysis | 결과 분석

### Epoch 증가 실험

- Epoch 1 → 3 증가 시 성능 상승 (0.851 → 0.877)
- 동일 데이터 반복 학습이 일반화 성능 향상에 기여
- 단, Epoch를 과도하게 증가시키면 **Overfitting** 발생 가능

### 데이터 크기 실험

- 학습 데이터 증가 시 성능 향상 여부 분석
- 데이터 양과 모델 성능의 상관관계 확인 예정

---

## 6. Training Details | 학습 설정

- **Optimizer**: AdamW  
- **Learning Rate**: 2e-5  
- **Batch Size**: 16  
- **Max Length**: 256  
- **Evaluation Metrics**: Accuracy, F1 Score  

---


## 7. How to Run | 실행 방법

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Baseline Model

```bash
python src/train.py
```

### 3. Change Experiment Settings (Environment Variables)

#### Change Epoch

```bash
$env:EPOCHS="3"
python src/train.py
```

#### Change Training Data Size

```bash
$env:MAX_TRAIN_SAMPLES="5000"
python src/train.py
```

---

## 8. Future Work | 향후 계획

### Planned Improvements

1. **Inference Script 추가**
   - `predict.py` 구현
   - 단일 문장 입력 → 감정 예측 출력

2. **REST API 구축**
   - FastAPI 기반 서버 구현
   - `/predict` 엔드포인트 설계

3. **Frontend 연동**
   - Simple HTML + JavaScript
   - 사용자 입력 → API 호출 → 결과 표시

4. **모델 비교 실험**
   - BERT vs DistilBERT 성능 비교
   - Accuracy / F1 / Training Time 비교 분석

5. **Training Efficiency 분석**
   - Training Time vs Performance 관계 분석
   - 효율성 중심 모델 선택 기준 도출

---

## 9. Key Takeaways | 프로젝트 핵심 포인트

### Technical Insights

- 사전학습 모델을 활용한 **Fine-tuning 실험 설계 수행**
- 통제변수 기반의 **Controlled Variable Experiment 진행**
- Epoch와 데이터 크기 변화에 따른 **성능 영향 분석**
- 실험 결과를 `metrics.csv`로 기록하여 **재현성 확보**

### Experimental Findings

- Epoch 증가 시 성능 향상 경향 확인
- 데이터 규모 확대 시 추가 성능 개선 가능성 존재
- 과적합 방지를 위한 적절한 학습 전략 필요성 인지
