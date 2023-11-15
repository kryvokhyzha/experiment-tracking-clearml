import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import plot_tree


def evaluate_model(model, X_test, y_test):
    # Load the data in XGBoost format
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Make predictions for test data
    y_pred = model.predict(dtest)
    predictions = [round(value) for value in y_pred]

    # Evaluate predictions
    accuracy = accuracy_score(dtest.get_label(), predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # Plots
    plot_tree(model)
    plt.title("Decision Tree")
    plt.show()

    xgb.plot_importance(model)
    plt.show()

    return accuracy
