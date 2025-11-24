import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("Heart_disease_cleveland_new 2.csv")

X_reg = df.drop(columns=['age', 'target'])
y_reg = df['age']

y_clf = (df['target'] > 0).astype(int)
X_clf = df.drop(columns=['target'])

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_reg_train, y_reg_train)
y_reg_pred = lr.predict(X_reg_test)

mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

print(f"Linear Regression Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

ridge = Ridge(alpha=1.0)
ridge.fit(X_reg_train, y_reg_train)
y_reg_pred_ridge = ridge.predict(X_reg_test)
r2_ridge = r2_score(y_reg_test, y_reg_pred_ridge)

print(f"\nRidge Regression R²: {r2_ridge:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_reg_test, y_reg_pred, alpha=0.5)
plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--')
plt.xlabel("Реальный возраст")
plt.ylabel("Предсказанный возраст")
plt.title("Регрессия: Реальный vs Предсказанный возраст")
plt.grid(True)
plt.tight_layout()
plt.savefig("regression_age_plot.png")
plt.show()

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_clf_train, y_clf_train)
y_clf_pred = logreg.predict(X_clf_test)
y_clf_proba = logreg.predict_proba(X_clf_test)[:, 1]

acc = accuracy_score(y_clf_test, y_clf_pred)
print(f"\nClassification Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_clf_test, y_clf_pred, target_names=['No Disease', 'Disease']))

cm = confusion_matrix(y_clf_test, y_clf_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Disease', 'Disease'], 
            yticklabels=['No Disease', 'Disease'])
plt.title("Матрица ошибок")
plt.ylabel("Реальный класс")
plt.xlabel("Предсказанный класс")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

logreg_l1 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
logreg_l1.fit(X_clf_train, y_clf_train)
y_clf_pred_l1 = logreg_l1.predict(X_clf_test)
acc_l1 = accuracy_score(y_clf_test, y_clf_pred_l1)
print(f"\nL1 Logistic Regression Accuracy: {acc_l1:.4f}")

results = pd.DataFrame({
    'Age_Real': y_reg_test.values, 
    'Age_Pred': y_reg_pred, 
    'Disease_Real': y_clf_test.values, 
    'Disease_Pred': y_clf_pred, 
    'Disease_Proba': y_clf_proba
})
results.to_csv("lab2_predictions.csv", index=False)
