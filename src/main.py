from clearml import Task
from src.prepare import prepare_data
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    task = Task.init(project_name='KS-scoring', task_name='XGBoost simple example', output_uri=True)

    X_train, y_train, X_test, y_test = prepare_data(
        dataset_id='62ae7c4f5d234427bc61e77d9230ae19', dataset_alias='KS-scoring-example'
    )
    model = train_model(X_train=X_train, y_train=y_train)
    evaluate_model(model=model, X_test=X_test, y_test=y_test)


if __name__ == '__main__':
    main()
