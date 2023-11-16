import importlib.metadata as metadata

from clearml import Model, OutputModel, Task

from src.evaluate import evaluate_model
from src.prepare import prepare_data
from src.train import train_model


def main():
    for i in metadata.distributions():
        Task.add_requirements(i.name, i.version)

    task = Task.init(
        project_name="KS-scoring",
        task_name="XGBoost simple example",
        output_uri=True,
        auto_connect_frameworks={
            "joblib": False,
        },
    )

    X_train, y_train, X_test, y_test = prepare_data(
        # dataset_id='62ae7c4f5d234427bc61e77d9230ae19',
        dataset_alias="KS-scoring-example",
        dataset_name="KS-scoring-example",
        dataset_project="KS-scoring",
    )

    output_model_path = "models/xgboost_best_model.joblib"
    model = train_model(X_train=X_train, y_train=y_train, output_model_path=output_model_path)
    target_metric = evaluate_model(logger=task.logger, model=model, X_test=X_test, y_test=y_test)

    output_model = OutputModel(task=task, tags=["examples", "external"], framework="xgboost")
    output_model.update_weights(weights_filename=output_model_path)
    output_model.set_metadata(key="target_metric", value=str(target_metric), v_type="float")
    output_model.set_metadata(key="metric_direction", value="1", v_type="int")

    best_publish_model = Model.query_models(project_name="KS-scoring", max_results=1, only_published=True)
    if best_publish_model:
        best_model = Task.get_task(task_id=best_publish_model[0].task).get_models()["output"][0]

        best_target_metric = float(best_model.get_metadata("target_metric"))
        best_metric_direction = int(best_model.get_metadata("metric_direction"))

        if (best_target_metric < target_metric and best_metric_direction == 1) or (
            best_target_metric > target_metric and best_metric_direction == -1
        ):
            best_model.archive()
            output_model.publish()
    else:
        output_model.publish()


if __name__ == "__main__":
    main()
