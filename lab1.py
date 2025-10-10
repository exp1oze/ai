import os 
import pandas as pd 
import numpy as np

# task 1
data = pd.read_csv("data.csv")

# task 2
print("Data head:")
print(data.head())

# task 3
print("\nMissing values:")
print(data.isnull().sum())

# task 4
num_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for column in num_columns:
    if column in data.columns:
        mode_series = data[column].mode()
        mode_value = mode_series.iloc[0] if not mode_series.empty else np.nan
        median_value = data[column].median()
        mean_value = data[column].mean()
        print(f"{column}: mode={mode_value}, median={median_value:.2f}, mean={mean_value:.2f}")
        data[column] = data[column].fillna(mean_value)  # fixed: assign, no inplace

cat_columns = ['HomePlanet', 'Cabin', 'Destination']
for column in cat_columns:
    if column in data.columns:
        mode_series = data[column].mode()
        mode_value = str(mode_series.iloc[0]) if not mode_series.empty else 'Unknown'
        print(f"{column}: mode={mode_value}")
        data[column] = data[column].fillna(mode_value)  # fixed

bool_columns = ['CryoSleep', 'VIP']
for column in bool_columns:
    if column in data.columns:
        mode_series = data[column].mode()
        mode_value = bool(mode_series.iloc[0])
        print(f"{column}: mode={mode_value}")
        data[column] = data[column].fillna(mode_value)  # fixed

print("\nAfter fill missing:")
print(data.isnull().sum())

# task 5 
num_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in num_columns:
    data_min = data[col].min()
    data_max = data[col].max()
    if data_max != data_min:
        data[col] = (data[col] - data_min) / (data_max - data_min)
    else:
        data[col] = 0
print("\nNormalized nums:")
print(data[num_columns].head())

# task 6
data['CryoSleep'] = data['CryoSleep'].astype(int)
data['VIP'] = data['VIP'].astype(int)

data_encoded = pd.get_dummies(data, columns=['HomePlanet', 'Destination'], drop_first=True)

x_cols = [col for col in data_encoded.columns if col not in ['PassengerId', 'Cabin', 'Name']]
X = data_encoded[x_cols]
train_idx = X.sample(frac=0.7, random_state=42).index
test_idx = X.drop(train_idx).index

train_final = data_encoded.loc[train_idx]
test_final = data_encoded.loc[test_idx]

print("\nOHE dummies:")
dummy_cols = [col for col in data_encoded.columns if 'HomePlanet' in col or 'Destination' in col]
print(data_encoded[dummy_cols].head())

print(f"\nTrain shape: {train_final.shape}, Test shape: {test_final.shape}")

# save
train_final.to_csv('train_processed.csv', index=False)
test_final.to_csv('test_processed.csv', index=False)
print("Files saved")
