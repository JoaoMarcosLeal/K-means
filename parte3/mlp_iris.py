from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle
from estatisticas import Estatistica
import time

iris = load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

curr = time.time()
clf = MLPClassifier(random_state=1, max_iter=900).fit(X_train, y_train)
total_time_training = time.time() - curr

# Salva o modelo
filename = "mlp_model_novo_iris.pkl"
with open(filename, "wb") as file:
    pickle.dump(clf, file)

# Carrega o modelo da memória
with open(filename, "rb") as file:
    loaded_mlp = pickle.load(file)

curr = time.time()
predictions = loaded_mlp.predict(X_test)
total_time_clf = time.time() - curr

estatistica = Estatistica()

estatistica.matriz_confusao(y_test, predictions, iris.target_names, "MLP iris")

print(f"precision score \n {estatistica.precision(y_test, predictions)}")
print(f"accuracy score \n {estatistica.accuracy(y_test, predictions)}")
print(f"recall score \n {estatistica.recall(y_test, predictions)}")
print(f"tempo de treinamento \n {total_time_training:.4f}")
print(f"tempo de classificação \n {total_time_clf:.4f}")
