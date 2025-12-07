import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score
)

data = pd.read_csv("heart_disease_cleveland.csv")

numeric_cols = data.select_dtypes(include=["number"]).columns.drop("target")
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

categorical_cols = data.select_dtypes(include=['object', 'bool']).columns
if len(categorical_cols) > 0:
    data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

scaler = StandardScaler()
scaler.fit(data[numeric_cols])
data[numeric_cols] = pd.DataFrame(scaler.transform(data[numeric_cols]), columns=numeric_cols)

if len(categorical_cols) > 0:
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

data.to_csv("preprocessed_heart_disease_cleveland.csv", index=False)

X = data.drop(columns=["target"], axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)

dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
dt_classifier.fit(X_train, y_train)
y_pred_test = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_test)
print(f"Accuracy: {accuracy}")

y_pred_proba = dt_classifier.predict_proba(X_test)[:, 1]

FPR, TPR, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="bwr")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()

plt.figure(figsize=(5, 5))
plt.plot(FPR, TPR, label=f'AUC = {auc}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
