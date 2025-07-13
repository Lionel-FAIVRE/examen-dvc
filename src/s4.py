import pandas as pd
from sklearn.linear_model import Ridge
import pickle

with open('models/best_param.pkl', 'rb') as f:
    best_params = pickle.load(f)

X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv').values.ravel()  # .ravel() pour éviter une shape (n,1)

model = Ridge(**best_params)
model.fit(X_train_scaled, y_train)

with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

