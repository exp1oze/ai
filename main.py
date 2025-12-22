import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


print("Генерация данных...")
np.random.seed(42)
X = np.random.randint(0, 2, size=(100, 12))  # 100 примеров, 12 бинарных признаков
Y = np.random.randint(0, 2, size=(100, 2))   # 100 примеров, 2 класса (one-hot, случайные метки)

#np.savetxt('dataIn.txt', X, fmt='%d')
#np.savetxt('dataOut.txt', Y, fmt='%d')

#X = np.loadtxt('dataIn.txt', dtype=float)    # (100, 12)
#Y = np.loadtxt('dataOut.txt', dtype=int)     # (100, 2)

print(f"Загружено {X.shape[0]} примеров с {X.shape[1]} признаками\n")

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = keras.Sequential([
    keras.layers.Dense(10, activation='sigmoid', input_shape=(12,)),  # скрытый слой
    keras.layers.Dense(2, activation='sigmoid')                      # выходной слой
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

print("\nОбучение модели...")
history = model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, Y_test),
    verbose=1
)

# -----------------------------
# 5. Оценка точности
# -----------------------------
Y_pred_prob = model.predict(X_test, verbose=0)
Y_pred = np.argmax(Y_pred_prob, axis=1)
Y_true = np.argmax(Y_test, axis=1)

acc = accuracy_score(Y_true, Y_pred)
print(f"\nТочность на тестовых данных: {acc:.4f} ({acc*100:.2f}%)")

# -----------------------------
# 6. Графики обучения
# -----------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val loss', linewidth=2)
plt.title('Функция потерь')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val accuracy', linewidth=2)
plt.title('Точность')
plt.xlabel('Эпоха')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(alpha=0.3)

plt.suptitle('Обучение MLP с logsig (один скрытый слой)')
plt.tight_layout()
plt.show()

# -----------------------------
# 7. Матрица ошибок
# -----------------------------
cm = confusion_matrix(Y_true, Y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Правящая партия', 'Оппозиция'],
            yticklabels=['Правящая партия', 'Оппозиция'])
plt.title(f'Матрица ошибок\nТочность: {acc*100:.2f}%')
plt.xlabel('Предсказано')
plt.ylabel('Истинное')
plt.tight_layout()
plt.show()

# -----------------------------
# 8. Визуализация предсказаний (PCA до 2D)
# -----------------------------
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

Y_pred_pca = np.argmax(model.predict(X_test, verbose=0), axis=1)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1],
                      c=Y_pred_pca, cmap='bwr', alpha=0.8, edgecolors='k', s=80)
plt.colorbar(scatter, ticks=[0, 1], label='Предсказанный класс (0 — Правящая, 1 — Оппозиция)')
plt.title('Визуализация предсказаний модели на тестовых данных (проекция PCA)')
plt.xlabel('Главная компонента 1')
plt.ylabel('Главная компонента 2')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()