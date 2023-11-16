import joblib
import xgboost as xgb
from clearml import Task


def train_model(X_train, y_train, output_model_path):
    # Load the data into XGBoost format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    # Set the parameters
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 4,  # the maximum depth of each tree
        "eta": 0.3,  # the training step for each iteration
        "gamma": 0,
        "max_delta_step": 1,
        "subsample": 1,
        "sampling_method": "uniform",
        "seed": 42,
    }
    Task.current_task().connect(params)

    # Train the XGBoost Model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=25,
        evals=[(dtrain, "train")],
        verbose_eval=0,
    )

    # Save the model
    joblib.dump(model, output_model_path)

    return model
