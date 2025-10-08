from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle
from estatisticas import Estatistica

wine = load_wine()

X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)

# Salva o modelo
filename = "mlp_model_novo_wine.pkl"
with open(filename, "wb") as file:
    pickle.dump(clf, file)

# Carrega o modelo da memória
with open(filename, "rb") as file:
    loaded_mlp = pickle.load(file)

predictions = loaded_mlp.predict(X_test)

estatistica = Estatistica()

estatistica.matriz_confusao(y_test, predictions, wine.target_names, "MLP wine")

print(f"precision score \n {estatistica.precision(y_test, predictions)}")
print(f"accuracy score \n {estatistica.accuracy(y_test, predictions)}")
print(f"recall score \n {estatistica.recall(y_test, predictions)}")
