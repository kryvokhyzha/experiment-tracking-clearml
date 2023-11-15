import pandas as pd
from sklearn.model_selection import train_test_split
from clearml import Dataset


def prepare_data(dataset_id, dataset_alias):
    # Read the data
    data_path = Dataset.get(dataset_id=dataset_id, alias=dataset_alias).get_local_copy()
    data = pd.read_csv(f"{data_path}/sample.csv", sep=';').fillna(0).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=['target']), data['target'], test_size=0.33, random_state=42
    )

    return X_train, y_train, X_test, y_test
