import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv ( "data/raw_data/raw.csv")

print(df.head(5))
print(df.info())
y = df['silica_concentrate']
X = df.drop(columns=['date','silica_concentrate'])      # On drop la date et la cible, la date n'est pas un paramètre a traiter comme les autres si on devait le faire

# Fait le decoupage XY et Train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)


# sauve les dataframes dans le dossier data/processed
X_train.to_csv('data/processed_data/X_train.csv', index=False)
X_test.to_csv('data/processed_data/X_test.csv', index=False)
y_train.to_csv('data/processed_data/y_train.csv', index=False)
y_test.to_csv('data/processed_data/y_test.csv', index=False)

