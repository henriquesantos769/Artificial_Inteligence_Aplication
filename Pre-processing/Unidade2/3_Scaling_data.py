# Import StandardScaler
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier


wine = load_wine_dataset()

# Inicializer o scale
scaler = StandardScaler()

# exclua do dataset a coluna Quality
X = wine.drop(["Quality"],axis=1)

#normalize o dataset com scaler
X_norm = pd.DataFrame(scaler.fit_transform(X))

#obtenha as labels da coluna Quality
y = wine["Quality"].values

#print a valriância de X
X_norm = scaler.fit_transform(X)
print('variancia',X.var())

#print a variânca do dataset X_norm
print('variancia do dataset normalizado',X_norm.var())

# Divida o dataset em treino e teste com amostragem estratificada
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, stratify = y, random_state=42)

#inicialize o algoritmo KNN
knn = KNeighborsClassifier()
if len(X_train) == len(y_train):
    knn.fit(X_train, y_train)
else:
    print("Erro: o número de amostras em X_train é diferente do número de amostras em y_train")

# Aplique a função fit do KNN
#knn.fit(X_train,y)

# Verifique o acerto do classificador
print('score', knn.score(X_train, y_train))