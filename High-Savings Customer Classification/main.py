# =============================================================
# 기계학습 Term Project 2 - "고저축(많은 저축) 고객 예측"
#
# Target(Option B): balance >= 중앙값 → 1 (고저축 고객), 아니면 0
# 핵심: 라벨의 재료가 된 balance는 입력(X)에서 제외 → 라벨 누수(leakage) 방지
# =============================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

# 결과 이미지 저장 폴더
RESULTS_DIR = "High-Savings Customer Classification/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 0. 데이터 불러오기
FILE_PATH = "High-Savings Customer Classification/data/Bank_Customer_Churn_Prediction.csv"
df = pd.read_csv(FILE_PATH)

# 1. 새로운 라벨(y) 정의 : "고저축 고객" = balance가 중앙값 이상
median_balance = df["balance"].median()
df["y"] = (df["balance"] >= median_balance).astype(int)
print(f"balance 중앙값(임계값) : {median_balance:,.1f}")
print(f"고저축 고객(y=1) 비율  : {df['y'].mean():.3f}  (균형 라벨 → 베이스라인 0.50)\n")

# 2. 입력 X 구성
#    - 식별자(customer_id) 제거
#    - 기존 라벨(churn) 제거
#    - 라벨 재료인 balance 제거 => 누수 방지 핵심
#    - 범주형(country, gender) 원-핫 인코딩
df = df.drop(["customer_id", "churn", "balance"], axis=1)
df = pd.get_dummies(df, columns=["country", "gender"], drop_first=True)

X = df.drop("y", axis=1)
y = df["y"]
print("입력 feature:", list(X.columns), "\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. 알고리즘 비교 (Decision Tree가 본 문제에 적합한 이유를 검증하기 위한 표)
def evaluate(name, model, scale=False):
    clf = make_pipeline(StandardScaler(), model) if scale else model
    clf.fit(X_train, y_train)
    tr = accuracy_score(y_train, clf.predict(X_train))
    pred = clf.predict(X_test)
    return {
        "Model": name,
        "Train Acc": round(tr, 4),
        "Test Acc": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 3),
        "Recall": round(recall_score(y_test, pred), 3),
        "F1": round(f1_score(y_test, pred), 3),
    }

dt_A = DecisionTreeClassifier(random_state=42)                                    # Model A: 제약 없음
dt_B = DecisionTreeClassifier(max_depth=4, min_samples_split=10, random_state=42)  # Model B: 규제

rows = [
    evaluate("DecisionTree (Model A, no limit)", dt_A),
    evaluate("DecisionTree (Model B, depth=4)", dt_B),
    evaluate("LogisticRegression", LogisticRegression(max_iter=1000), scale=True),
    evaluate("RandomForest (200)", RandomForestClassifier(n_estimators=200, random_state=42)),
    evaluate("SVM (RBF)", SVC(kernel="rbf"), scale=True),
]
result_table = pd.DataFrame(rows)
baseline = max(y_test.mean(), 1 - y_test.mean())
print(f"[Baseline] 다수 클래스만 찍을 때 정확도 = {baseline:.4f}")
print("\n[알고리즘 성능 비교]")
print(result_table.to_string(index=False))
result_table.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

# 4. 최종 모델(Model B) 상세 분석
dt_A.fit(X_train, y_train)
dt_B.fit(X_train, y_train)
pred_B = dt_B.predict(X_test)

print("\n[Model B 혼동행렬]  행=실제, 열=예측  (0=Low, 1=High saver)")
print(confusion_matrix(y_test, pred_B))

importance = pd.Series(dt_B.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n[Model B feature importance]")
print(importance.round(3).to_string())

# 5. 시각화 (결과 이미지는 results/ 폴더에 저장 -> README에서 확인 가능)
# 5-1 feature importance 막대
plt.figure(figsize=(8, 4))
importance[importance > 0].sort_values().plot(kind="barh", color="#4C72B0")
plt.title("Model B - Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
plt.close()

# 5-2 혼동행렬 히트맵
cm = confusion_matrix(y_test, pred_B)
plt.figure(figsize=(4.5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Model B - Confusion Matrix")
plt.xticks([0, 1], ["Pred Low", "Pred High"]); plt.yticks([0, 1], ["True Low", "True High"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=13)
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# 5-3 의사결정나무
plt.figure(figsize=(26, 5))
plot_tree(dt_B, feature_names=list(X.columns),
          class_names=["Low saver (0)", "High saver (1)"],
          filled=True, rounded=True, fontsize=9)
plt.title("Decision Tree (Model B, max_depth=4) - Target: High-Balance(>=median) Customer",
          fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "decision_tree.png"), dpi=150)
plt.close()

print(f"\n모든 결과 이미지 및 표가 '{RESULTS_DIR}/' 폴더에 저장되었습니다.")