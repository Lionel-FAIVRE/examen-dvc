import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import pickle


# 🔹 Charger les data 
X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv').values.ravel()  # .ravel() pour éviter une shape (n,1)

# 🔹 Définir le modèle et la grille de params
model = Ridge()
param_grid = {
    #'alpha': [0.01, 0.1, 1, 10, 100]
    'alpha': [5, 8, 10, 12, 14, 15, 15.5, 16, 16.5, 16.75, 17, 20, 25, 50, 75],  # Valeurs plus larges pour une meilleure exploration
}

# 🔹 GridSearch
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# 🔹 Afficher les meilleurs params
print("Meilleurs paramètres trouvés :", grid_search.best_params_)

with open('models/best_param.pkl', 'wb') as f:
    pickle.dump(grid_search.best_params_, f)
