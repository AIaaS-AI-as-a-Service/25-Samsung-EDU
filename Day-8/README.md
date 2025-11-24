# Samsung Education Day-8: Machine Learning Classification

## 📚 과정 개요

이 과정은 4가지 주요 분류 알고리즘을 **실무 데이터**와 **교육용 데이터**를 혼합하여 학습합니다.

**소요 시간**: 3-4시간  
**난이도**: 중급  
**사전 지식**: Python 기초, NumPy, Pandas 기초

---

## 📁 디렉토리 구조

```
new_day-8/
├── data/                       # 데이터셋
│   ├── secom.data             # SECOM 센서 데이터
│   ├── secom_labels.data      # SECOM 라벨
│   └── secom.names            # 데이터셋 설명
├── notebooks/                  # Jupyter 노트북
│   ├── 1-Logistic_Regression_SECOM.ipynb
│   ├── 2-NaiveBayes_Text.ipynb
│   ├── 3-KNN_Classic.ipynb
│   └── 4-SVM_SECOM.ipynb
├── requirements.txt           # Python 패키지 의존성
├── environment.yml            # Conda 환경 파일
└── README.md                  # 이 파일
```

---

## 🎯 학습 목표

### 1️⃣ Logistic Regression (60분) - **SECOM 데이터**
- 실제 반도체 제조 공정 데이터 분석
- 고차원 데이터 전처리 (결측치, Feature Selection)
- 클래스 불균형 처리 (`class_weight='balanced'`)
- Feature Importance 분석

### 2️⃣ Naive Bayes (45분) - **20 Newsgroups 텍스트**
- 텍스트 데이터 Vectorization (CountVectorizer, TF-IDF)
- Multinomial vs Bernoulli Naive Bayes
- 독립성 가정의 실용성
- 텍스트 분류 응용 (스팸 필터, 감정 분석)

### 3️⃣ K-Nearest Neighbors (45분) - **Iris + Wine**
- KNN 작동 원리 (거리 기반 분류)
- k 값 선택의 중요성
- Feature Scaling 필수성
- 차원의 저주 (Curse of Dimensionality)

### 4️⃣ Support Vector Machine (60분) - **SECOM 데이터**
- Linear vs RBF Kernel 비교
- 하이퍼파라미터 튜닝 (C, gamma)
- 고차원 데이터에서의 SVM 강점
- Logistic Regression과 성능 비교

---

## 🚀 환경 설정

### 방법 1: Conda 환경 사용 (권장)

```bash
# 환경 생성
conda env create -f environment.yml

# 환경 활성화
conda activate day8-ml

# Jupyter Notebook 실행
jupyter notebook
```

### 방법 2: pip 사용

```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt

# Jupyter Notebook 실행
jupyter notebook
```

---

## 📊 데이터셋 정보

### SECOM (SEmiCOnductor Manufacturing)
- **출처**: UCI Machine Learning Repository
- **샘플**: 1,567개
- **특성**: 590개 (센서 측정값)
- **문제**: 불량품 예측 (Pass/Fail)
- **클래스 불균형**: Pass 93.4% vs Fail 6.6%
- **특징**: 결측치 많음, 실제 제조 공정 데이터
- **사용**: Logistic Regression, SVM

### 20 Newsgroups
- **출처**: scikit-learn built-in dataset
- **샘플**: ~2,800개 (4개 카테고리 선택)
- **특성**: 텍스트 (가변 길이)
- **문제**: 뉴스 기사 카테고리 분류
- **사용**: Naive Bayes

### Iris & Wine
- **출처**: scikit-learn built-in datasets
- **특성**: 저차원 (4-13개)
- **문제**: 다중 클래스 분류
- **사용**: KNN

---

## 🔧 주요 기술 스택

- **Python**: 3.9+
- **NumPy**: 배열 연산
- **Pandas**: 데이터 처리
- **Scikit-learn**: 머신러닝 알고리즘
- **Matplotlib & Seaborn**: 시각화

---

## 📝 실습 순서

1. **환경 설정 확인**
   ```bash
   conda activate day8-ml
   jupyter notebook
   ```

2. **노트북 실행 순서**
   - `1-Logistic_Regression_SECOM.ipynb` → SECOM 데이터 이해
   - `2-NaiveBayes_Text.ipynb` → 텍스트 분류 기초
   - `3-KNN_Classic.ipynb` → 거리 기반 분류
   - `4-SVM_SECOM.ipynb` → SECOM 재방문 (비교)

3. **각 노트북의 셀을 순서대로 실행**

---

## 💡 학습 포인트

### 데이터 전처리
- **결측치 처리**: Imputation (평균값, 중앙값)
- **Feature Selection**: 결측치 비율로 필터링
- **Scaling**: StandardScaler (KNN, SVM 필수)
- **Vectorization**: CountVectorizer, TF-IDF (텍스트)

### 클래스 불균형
- **문제**: 소수 클래스 무시
- **해결책**: `class_weight='balanced'`, SMOTE
- **평가**: Accuracy 대신 F1-score, Precision, Recall

### 하이퍼파라미터 튜닝
- **GridSearchCV**: 최적 파라미터 탐색
- **Cross-Validation**: 과적합 방지
- **Early Stopping**: 불필요한 계산 방지

---

## 🎓 실무 적용 사례

### Logistic Regression + SECOM
- 반도체 불량 예측 시스템
- 중요 센서 식별로 모니터링 비용 절감

### Naive Bayes + 텍스트
- 이메일 스팸 필터
- 고객 리뷰 감정 분석
- 문서 자동 분류

### KNN
- 추천 시스템 (유사 사용자 찾기)
- 이상 탐지 (Local Outlier Factor)

### SVM
- 이미지 분류 (Feature Extraction 후)
- 의료 진단 (고차원 특성)

---

## ⚠️ 주의사항

1. **메모리**: SECOM 데이터 로딩 시 ~50MB RAM 필요
2. **실행 시간**: GridSearchCV 사용 시 5-10분 소요 가능
3. **Jupyter Kernel**: 노트북 실행 전 `day8-ml` 커널 선택 확인
4. **데이터 경로**: 노트북은 `../data/` 경로 가정

---

## 📚 참고 자료

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [UCI SECOM Dataset](https://archive.ics.uci.edu/ml/datasets/SECOM)
- [20 Newsgroups Dataset](http://qwone.com/~jason/20Newsgroups/)

---

## 🤝 문의 및 피드백

실습 중 문제가 발생하면:
1. 먼저 에러 메시지 확인
2. 환경 설정 재확인 (`conda list`)
3. 데이터 파일 존재 확인 (`ls data/`)

---

**Happy Learning! 🚀**
