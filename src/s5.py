import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import json
import pickle


with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load( f)

X_test_scaled = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv').values.ravel()    
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse}")
print(f"R2 : {r2}")

# Creer un dico avec les scores
scores = {
    "mse": mse,
    "r2": r2
}

# Sauvegarder au format JSON
with open('metrics/scores.json', 'w') as f:
    json.dump(scores, f, indent=4)

