import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from xgboost import plot_tree


def evaluate_model(logger, model, X_test, y_test):
    # Load the data in XGBoost format
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Make predictions for test data
    y_pred = model.predict(dtest)
    predictions = [round(value) for value in y_pred]

    # Evaluate predictions
    accuracy = accuracy_score(dtest.get_label(), predictions)
    logger.report_scalar("Accuracy", "acc", value=accuracy, iteration=0)
    logger.report_table(
        title="Metrics",
        series="metrics",
        iteration=0,
        table_plot=pd.DataFrame({"accuracy": [accuracy]}),
    )

    # Plots
    plot_tree(model)
    plt.title("Decision Tree")
    plt.show()

    xgb.plot_importance(model)
    plt.show()

    return accuracy
