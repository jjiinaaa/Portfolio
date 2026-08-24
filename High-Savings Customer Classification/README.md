# 고저축(High-Balance) 고객 예측 — Decision Tree 기반 분류

**기계학습 Term Project 2**

---

## 1. 프로젝트 배경 및 문제 정의

### 1-1. 기존 문제 정의의 한계와 신규 타겟 재정의

Term Project 1에서는 데이터셋에 내재된 `churn`(이탈 여부) 라벨을 그대로 종속 변수로 삼아 이진 분류를 수행했다. 그러나 이는 이미 정의된 문제를 수동적으로 적용하는 데 그친다는 한계가 있었다. 이에 본 프로젝트는 `churn` 라벨을 폐기하고, 동일한 데이터에서 새로운 예측 대상을 직접 정의했다.

새롭게 정의한 문제는 **고객의 프로필만으로 해당 고객이 "고저축 고객"일 가능성을 예측**하는 것이다. 은행 입장에서 계좌 잔고는 핵심 자금 조달원(은행의 부채)이므로, 어떤 고객이 자산을 많이 예치하는 성향을 갖는지 사전에 식별하는 것은 마케팅·리스크 관리 측면에서 중요한 문제다.

**라벨 정의 (Option B):**
- 전체 고객 `balance`의 **중앙값(median)**을 임계값으로 설정
- `balance >= median` → `y = 1` (고저축 고객)
- `balance < median` → `y = 0` (일반 고객)

중앙값을 기준으로 삼은 이유는 (1) "평균적 고객보다 많이 예치하는가"라는 직관적 의미를 가지면서, (2) 두 클래스의 표본 수를 절반씩 균형 있게 유지해 분류 성능을 편향 없이 평가할 수 있기 때문이다.

### 1-2. 라벨 누수(Label Leakage) 방지

라벨 생성의 근거가 된 `balance`는 **입력 변수(X)에서 반드시 제외**한다. 만약 `balance`를 입력에 포함하면 정답이 입력 안에 그대로 들어 있게 되어, 모델은 예측이 아니라 라벨의 정의를 그대로 복원하는 trivial problem이 된다. 따라서 아래 9개 특징만으로 고저축 여부를 예측하도록 설계했다.

| 구분 | 사용 변수 |
|---|---|
| 입력(X) | credit_score, country, gender, age, tenure, products_number, credit_card, active_member, estimated_salary |
| 라벨(y) | balance ≥ median → 1, else 0 |
| 제외 | customer_id(식별자), churn(구 라벨), balance(라벨 재료) |

---

## 2. 알고리즘 선정: Decision Tree

본 문제는 범주형(country, gender)과 수치형(credit_score, age, tenure 등) 변수가 섞인 정형 데이터의 이진 분류이며, 단순 정확도를 넘어 **"어떤 특성의 고객이 고저축 고객인가"를 사람이 이해할 수 있는 규칙으로 제시**하는 것이 목표다.

데이터 특성상 다음 두 가지가 예상된다.
1. **변수 간 상호작용**: 저축 성향은 거주 국가의 금융 환경, 연령·거래 기간 등에 따라 조건부로 달라질 수 있다.
2. **비단조적 관계**: 예컨대 보유 상품 수가 많다고 잔고가 비례해서 증가한다고 보기 어렵다.

**Decision Tree**는 특징 공간을 임계값 기준으로 재귀 분할하므로 구간별·비단조적 관계를 별도 변환 없이 표현하고, 계층 구조 자체가 변수 간 상호작용을 자동으로 반영하며, 스케일 정규화가 불필요하고, 각 분기 경로가 그대로 If-Then 규칙이 되어 해석 목적에 부합한다.

**다른 알고리즘을 배제한 이유:**
- **Logistic Regression**: 선형 결합 기반이라 결정 경계가 직선이며, 비단조 관계·상호작용을 반영하려면 구간화·상호작용항을 사람이 직접 설계해야 한다.
- **SVM (RBF kernel)**: 비선형 경계는 그릴 수 있지만 고차원 커널 공간에 형성되어 원래 특징 기준의 규칙으로 환원되지 않는다 → 분류는 가능하나 근거 제시 불가.
- **MLP**: 상호작용·비단조 관계 학습은 가능하나 블랙박스 모델이라 분류 근거를 규칙으로 제시할 수 없어 배제했다.

---

## 3. 실험 설정

- 언어/환경: Python (pandas, numpy, scikit-learn, matplotlib)
- 데이터: 10,000개 샘플, **8:2 stratified split** (두 클래스 비율을 학습/검증에서 동일하게 유지), `random_state=42` 고정
- 범주형 변수: `country`, `gender` → one-hot encoding
- 비교 모델: Decision Tree (제약 없음 / `max_depth=4, min_samples_split=10`), Logistic Regression, Random Forest, SVM(RBF)

---

## 4. 실행 방법

```bash
pip install -r requirements.txt
python main.py
```

데이터 파일(`Bank Customer Churn Prediction.csv`, [Kaggle 링크](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset))을 `data/` 폴더에 위치시킨 후 실행한다.

실행 결과(모델 비교표, feature importance, confusion matrix, decision tree 시각화)는 `results/` 폴더에 저장된다.

---

## 5. 결과

> `python main.py` 실행 후 아래 이미지들이 `results/`에 생성됩니다.

- `results/model_comparison.csv` — 알고리즘별 Train/Test Accuracy, Precision, Recall, F1
- `results/feature_importance.png` — Decision Tree(Model B) 기준 변수 중요도
- `results/confusion_matrix.png` — Model B 혼동행렬
- `results/decision_tree.png` — Model B 트리 구조 시각화

---

## 6. 프로젝트 구조

```
.
├── main.py
├── requirements.txt
├── README.md
├── data/
│   └── Bank Customer Churn Prediction.csv   (직접 다운로드 필요)
└── results/                                  (실행 시 자동 생성)
```