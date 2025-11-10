import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("processed_spaceship.csv")

X_reg = df.drop(columns=['Age','Transported'])
y_reg = df['Age']
X_clf = df.drop(columns=['Transported'])
y_clf = df['Transported']

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_reg_train, y_reg_train)
y_reg_pred = lr.predict(X_reg_test)

mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

ridge = Ridge(alpha=1.0)
ridge.fit(X_reg_train, y_reg_train)
y_reg_pred_ridge = ridge.predict(X_reg_test)
r2_ridge = r2_score(y_reg_test, y_reg_pred_ridge)

plt.figure(figsize=(8,6))
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
y_clf_proba = logreg.predict_proba(X_clf_test)[:,1]

acc = accuracy_score(y_clf_test, y_clf_pred)
print(classification_report(y_clf_test, y_clf_pred, target_names=['Не транспортирован','Транспортирован']))

cm = confusion_matrix(y_clf_test, y_clf_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Не трансп.','Трансп.'], yticklabels=['Не трансп.','Трансп.'])
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

results = pd.DataFrame({
    'Age_Real': y_reg_test.values,
    'Age_Pred': y_reg_pred,
    'Transported_Real': y_clf_test.values,
    'Transported_Pred': y_clf_pred,
    'Transported_Proba': y_clf_proba
})
results.to_csv("lab2_predictions.csv", index=False)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("processed_spaceship.csv")

X_reg = df.drop(columns=['Age','Transported'])
y_reg = df['Age']
X_clf = df.drop(columns=['Transported'])
y_clf = df['Transported']

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_reg_train, y_reg_train)
y_reg_pred = lr.predict(X_reg_test)

mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

ridge = Ridge(alpha=1.0)
ridge.fit(X_reg_train, y_reg_train)
y_reg_pred_ridge = ridge.predict(X_reg_test)
r2_ridge = r2_score(y_reg_test, y_reg_pred_ridge)

plt.figure(figsize=(8,6))
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
y_clf_proba = logreg.predict_proba(X_clf_test)[:,1]

acc = accuracy_score(y_clf_test, y_clf_pred)
print(classification_report(y_clf_test, y_clf_pred, target_names=['Не транспортирован','Транспортирован']))

cm = confusion_matrix(y_clf_test, y_clf_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Не трансп.','Трансп.'], yticklabels=['Не трансп.','Трансп.'])
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

results = pd.DataFrame({
    'Age_Real': y_reg_test.values,
    'Age_Pred': y_reg_pred,
    'Transported_Real': y_clf_test.values,
    'Transported_Pred': y_clf_pred,
    'Transported_Proba': y_clf_proba
})
results.to_csv("lab2_predictions.csv", index=False)
