import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Генерация случайных данных
np.random.seed(42)
X = np.random.randint(0, 2, size=(100, 12))  # 100 на 12
Y = np.random.randint(0, 2, size=(100, 2))  # 100 на 2

# Сохранение данных в файлы
np.savetxt('dataIn.txt', X, fmt='%d')
np.savetxt('dataOut.txt', Y, fmt='%d')

#Загрузка данных
X = np.loadtxt('dataIn.txt', dtype=int)
Y = np.loadtxt('dataOut.txt', dtype=int)


#Разделение на тренировочную и тестовую выборки
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=np.argmax(Y, axis=1)
)

#нормализуем для нашей модели
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Создание модели sigmoid это название, то же самое, что и logsig
model = keras.Sequential([
    keras.layers.Dense(10, activation='sigmoid', input_shape=(12,)),    #один скрытый слой
    keras.layers.Dense(2, activation='sigmoid')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Обучение

history = model.fit(X_train, Y_train,
                    epochs=100,       #берём 100 эпох
                    batch_size=16,
                    validation_data=(X_test, Y_test),
                    verbose=1)

#Предсказание и точность
Y_pred_prob = model.predict(X_test, verbose=0)
Y_pred = np.argmax(Y_pred_prob, axis=1)
Y_true = np.argmax(Y_test, axis=1)

acc = accuracy_score(Y_true, Y_pred)
print(f"\nТочность на тестовых данных: {acc:.4f}")

#Графики обучения
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Test loss', linewidth=2)
plt.title('Функция потерь')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train acc', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Test acc', linewidth=2)
plt.title('Точность')
plt.xlabel('Эпоха')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(alpha=0.3)

plt.suptitle('Результаты обучения MLP')
plt.tight_layout()
plt.show()

cm = confusion_matrix(Y_true, Y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=2, linecolor='black',
            xticklabels=['Правящая партия', 'Оппозиция'],
            yticklabels=['Правящая партия', 'Оппозиция'])

plt.title(f'Матрица ошибок', fontsize=16, pad=20)
plt.xlabel('Предсказано моделью', fontsize=12)
plt.ylabel('Истинный класс', fontsize=12)
plt.tight_layout()
plt.show()