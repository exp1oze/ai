import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (
        mean_squared_error, 
        mean_absolute_error, 
        r2_score, 
        confusion_matrix, 
        classification_report) 

data = pd.read_csv("data.csv")

data = data.drop(columns=["id_number"])

numeric_cols = data.select_dtypes(include=["number"]).columns.drop("charges")
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

categorical_cols = data.select_dtypes(include=['object', 'bool']).columns
if len(categorical_cols) > 0:
    data[categorical_cols] = data[categorical_cols].fillna(
        data[categorical_cols].mode().iloc[0]
    )

scaler = StandardScaler()
scaler.fit(data[numeric_cols])
data[numeric_cols] = pd.DataFrame(
    scaler.transform(data[numeric_cols]), columns=numeric_cols
)

if len(categorical_cols) > 0:
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

data.to_csv("preprocessed_data.csv", index=False)

#Task 1
X = data.drop(columns=["charges"], axis=1)
y = data["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_test, X_val, y_test, y_val = train_test_split(
    X_test, y_test, test_size=0.4, random_state=42
)

polinom = PolynomialFeatures()
polinom.fit(X_train)
X_train = polinom.transform(X_train)
X_test = polinom.transform(X_test)

#Task 2
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_test = linear_model.predict(X_test)

#Task 3
MSE = mean_squared_error(y_test, y_pred_test)
RMSE = np.sqrt(MSE)
MAE = mean_absolute_error(y_test, y_pred_test)
R2 = r2_score(y_test, y_pred_test)
print(f"MSE: {MSE}\nRMSE: {RMSE}\nMAE: {MAE}\nR2: {R2}\n")

data["charges_binary"] = (data["charges"] > data["charges"].median()).astype(int)

X = data.drop(columns=["charges", "charges_binary"])
y = data["charges_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_test, X_val, y_test, y_val = train_test_split(
    X_test, y_test, test_size=0.4, random_state=42
)

polinom = PolynomialFeatures()
polinom.fit(X_train)
X_train = polinom.transform(X_train)
X_test = polinom.transform(X_test)

#Task 4 
logistic_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)
logistic_model.fit(X_train, y_train)
y_pred_test = logistic_model.predict(X_test)

#Task 5
report = classification_report(y_test, y_pred_test)
print(report)

cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="bwr")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()
