# Sentiment Analysis Web Service (IMDb)

A sentiment analysis project that fine-tunes a Transformer model on the IMDb dataset and deploys it as a simple web service using **FastAPI** and a lightweight **web interface**.

IMDb 영화 리뷰 데이터를 활용하여 Transformer 기반 감정 분석 모델을 fine-tuning하고,  
FastAPI와 웹 인터페이스를 통해 실제 서비스 형태로 배포하는 프로젝트입니다.

---

# 1. Project Goal | 프로젝트 목표

This project aims to build a sentiment classification model using a pre-trained Transformer and analyze how training configurations affect model performance.

이 프로젝트의 목표는 다음과 같습니다.

### Goals

- Transformer 기반 감정 분석 모델 구축
- 실험을 통해 하이퍼파라미터 영향 분석
- 모델 추론 API 구축
- 간단한 웹 인터페이스를 통한 서비스 데모 구현

---

# 2. Dataset | 데이터셋

**Dataset:** IMDb Movie Reviews  
**Source:** Hugging Face `datasets`

Task:


Binary Sentiment Classification
Positive / Negative


특징:

- 약 **50,000개의 영화 리뷰 데이터**
- 긍정 / 부정이 **균형 잡힌 데이터셋**

---

# 3. Model | 사용 모델

### distilbert-base-uncased

선택 이유:

- BERT보다 **가벼운 Transformer 모델**
- CPU 환경에서도 빠른 학습 가능
- NLP classification task에서 높은 효율

---

# 4. Experiments | 실험 설계

| Model | Train Samples | Epochs | Accuracy | F1 |
|------|---------------|-------|--------|------|
| distilbert | 2000 | 1 | 0.851 | 0.848 |
| distilbert | 2000 | 3 | 0.877 | 0.877 |
| distilbert | 5000 | 1 | 0.881 | 0.879 |

---

# 5. Analysis | 결과 분석

### Epoch 증가 실험

Epoch 1 → 3 증가 시 성능 상승


Accuracy
0.851 → 0.877


동일 데이터 반복 학습이 모델 성능 향상에 기여.

단, Epoch를 과도하게 증가시키면 **Overfitting 위험** 존재.

---

### 데이터 크기 실험

학습 데이터 증가 시 성능 향상 확인.


Train Samples
2000 → 5000


더 많은 데이터가 모델 일반화 성능에 긍정적 영향을 줄 가능성 확인.

---

# 6. Training Details | 학습 설정

### Training Details

| Setting | Value |
|------|------|
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Max Length | 256 |
| Metrics | Accuracy, F1 |

---

# 7. Web MVP Demo | 웹 데모

학습된 모델을 실제 서비스처럼 사용하기 위해  
FastAPI 기반 추론 API와 간단한 웹 인터페이스를 구현했습니다.

### Features

- 텍스트 입력 기반 감정 분석
- Positive / Negative 뱃지 표시
- Confidence Score Progress Bar
- FastAPI 기반 REST API
- HTML / CSS / JavaScript 웹 인터페이스

웹 인터페이스에서 문장을 입력하면 감정 분석 결과가 표시됩니다.

# Demo
Example

## Input
I love this movie

## Output
POSITIVE
Confidence: 79.5%

---

# 8. System Architecture

```text
User (Browser)
      │
      ▼
Frontend (HTML / CSS / JS)
      │
      ▼
Fetch API
      │
      ▼
FastAPI Backend
      │
      ▼
Transformer Model (DistilBERT)
      │
      ▼
Prediction Result

이 구조는 실제 AI 서비스의 **end-to-end pipeline**을 단순화한 형태입니다.

---

# 9. Project Structure

```text
sentiment-model-comparison
│
├── backend/
│   └── app.py
│
├── frontend/
│   └── index.html
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── results/
│   └── metrics.csv
│
├── README.md
└── requirements.txt
---

# 10. How to Run | 실행 방법

### Install dependencies


pip install -r requirements.txt


---

### Train Model


python src/train.py


---

### Run Backend Server


cd backend
uvicorn app:app --reload --port 8000


---

### Run Frontend


cd frontend
python -m http.server 5500


---

### Open Web Page


http://127.0.0.1:5500


---

# 11. Key Takeaways | 프로젝트 핵심 포인트

### Technical

- Transformer 기반 NLP 모델 fine-tuning
- 실험 설계를 통한 모델 성능 비교
- FastAPI 기반 추론 서버 구현
- 웹 인터페이스를 통한 모델 서비스 데모

### Engineering

- AI 모델 → API → Web Interface 연결
- 실제 서비스와 유사한 **AI deployment pipeline 경험**

---

# 12. Future Improvements

- BERT vs DistilBERT 성능 비교
- Training time vs performance 분석
- React 기반 프론트엔드 업그레이드
- Cloud 배포 (AWS / Docker)
