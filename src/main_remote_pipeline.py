from clearml import TaskTypes
from clearml.automation.controller import PipelineDecorator


@PipelineDecorator.component(
    return_values=["X_train", "y_train", "X_test", "y_test"],
    task_type=TaskTypes.data_processing,
)
def prepare_data(dataset_id=None, dataset_alias=None, dataset_name=None, dataset_project=None):
    # Imports first
    import pandas as pd
    from clearml import Dataset
    from sklearn.model_selection import train_test_split

    # Read the data
    data_path = Dataset.get(
        dataset_id=dataset_id,
        alias=dataset_alias,
        dataset_name=dataset_name,
        dataset_project=dataset_project,
    ).get_local_copy()
    data = pd.read_csv(f"{data_path}/sample.csv", sep=";").fillna(0).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=["target"]), data["target"], test_size=0.33, random_state=42
    )

    return X_train, y_train, X_test, y_test


@PipelineDecorator.component(
    return_values=["model"],
    task_type=TaskTypes.training,
    auto_connect_frameworks={"joblib": False},
)
def train_model(X_train, y_train, output_model_path):
    # Imports first
    import joblib
    import xgboost as xgb
    from clearml import OutputModel, Task

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

    output_model = OutputModel(task=Task.current_task(), tags=["examples", "external"], framework="xgboost")
    output_model.update_weights(weights_filename=output_model_path)

    return model


@PipelineDecorator.component(return_values=["accuracy"], cache=True, task_type=TaskTypes.qc)
def evaluate_model(model, X_test, y_test):
    # Imports first
    import matplotlib.pyplot as plt
    import pandas as pd
    import xgboost as xgb
    from clearml import Task
    from sklearn.metrics import accuracy_score
    from xgboost import plot_tree

    # Load the data in XGBoost format
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Make predictions for test data
    y_pred = model.predict(dtest)
    predictions = [round(value) for value in y_pred]

    # Evaluate predictions
    accuracy = accuracy_score(dtest.get_label(), predictions)
    logger = Task.current_task().get_logger()
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


@PipelineDecorator.pipeline(name="KS-scoring-pipeline", project="KS-scoring", version="0.0.5")
def run_pipeline(output_model_path):
    # Imports first
    from clearml import Task

    # Get the data in XGBoost format
    X_train, y_train, X_test, y_test = prepare_data(
        # dataset_id='62ae7c4f5d234427bc61e77d9230ae19',
        dataset_alias="KS-scoring-example",
        dataset_name="KS-scoring-example",
        dataset_project="KS-scoring",
    )

    # Train an XGBoost model on the data
    model = train_model(X_train=X_train, y_train=y_train, output_model_path=output_model_path)

    # Evaluate the model
    target_metric = evaluate_model(model=model, X_test=X_test, y_test=y_test)
    Task.current_task().get_logger().report_single_value(name="Accuracy", value=target_metric)

    return target_metric


if __name__ == "__main__":
    PipelineDecorator.run_locally()
    run_pipeline(output_model_path="models/xgboost_best_model.joblib")
