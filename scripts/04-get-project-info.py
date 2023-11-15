import clearml
import joblib


if __name__ == '__main__':
    task_id = '7e677b57946e48009744786b62e1fb5a'
    task = clearml.Task.get_task(task_id)

    print(task.get_task(task_id).data.script.requirements)

    path = task.models['output'][0].get_local_copy()
    model = joblib.load(path)

    print(model.feature_names)