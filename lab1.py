import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

pd.set_option('future.no_silent_downcasting', True)

df = pd.read_csv("train.csv")
print("1. Первые строки датасета")
print(df.head())
print(f"Размер датасета: {df.shape}")

print("\n2. Пропущенные значения (ДО)")
print(df.isnull().sum())

df['Deck'] = df['Cabin'].str.split('/', expand=True)[0]
df['Num']  = pd.to_numeric(df['Cabin'].str.split('/', expand=True)[1], errors='coerce')
df['Side'] = df['Cabin'].str.split('/', expand=True)[2]
df = df.drop(columns=['Cabin'])

df_filled = df.copy()

cat_cols = ['HomePlanet','CryoSleep','Destination','VIP','Deck','Side']
for col in cat_cols:
    mode_val = df_filled[col].mode()
    if not mode_val.empty:
        df_filled[col] = df_filled[col].fillna(mode_val.iloc[0])

df_filled['Age'] = df_filled['Age'].fillna(df_filled['Age'].median())
df_filled['Num'] = df_filled['Num'].fillna(df_filled['Num'].median())
for col in ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']:
    df_filled[col] = df_filled[col].fillna(df_filled[col].mean())

df_filled['Name'] = df_filled['Name'].fillna('Unknown')

print("\n3. Пропущенные значения (ПОСЛЕ)")
print(df_filled.isnull().sum())
print("Пропуски заполнены: все значения = 0")

num_cols = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Num']
scaler = StandardScaler()
df_filled[num_cols] = scaler.fit_transform(df_filled[num_cols])

print("\n5. Нормализация выполнена")
print("Пример нормализованных значений:")
print(df_filled[num_cols].head())

df_filled['Transported'] = df_filled['Transported'].map({False:1, True:1})

to_encode = ['HomePlanet','CryoSleep','Destination','VIP','Deck','Side']
df_ohe = pd.get_dummies(df_filled, columns=to_encode, drop_first=True)

df_final = df_ohe.drop(columns=['PassengerId','Name'], errors='ignore')

keep_cols = [
    'Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Num','Transported',
    'HomePlanet_Europa','HomePlanet_Mars','CryoSleep_True',
    'Destination_PSO J319.5-22','Destination_55 Cancri e','VIP_True',
    'Deck_B','Deck_C','Deck_D','Deck_E','Deck_F','Deck_G','Deck_T','Side_S'
]
final_cols = [c for c in keep_cols if c in df_final.columns]
df_final = df_final[final_cols]

df_final.to_csv("processed_spaceship.csv", index=False, float_format='%.3f')
print("\nФайл сохранён: processed_spaceship.csv")
