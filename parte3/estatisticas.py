from sklearn.metrics import (
    ConfusionMatrixDisplay,
    recall_score,
    precision_score,
    accuracy_score,
)
import matplotlib.pyplot as plt

class Estatistica:
    def matriz_confusao(self, y_test, y_pred, class_names, title):
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, cmap=plt.cm.Blues, display_labels=class_names
        )
        disp.ax_.set_title(title)

        plt.show()

    def recall(self, y_test, y_pred):
        return recall_score(y_test, y_pred, average="micro")

    def precision(self, y_test, y_pred):
        return precision_score(y_test, y_pred, average="micro")

    def accuracy(self, y_test, y_pred):
        return accuracy_score(y_test, y_pred, normalize=True)
