from clearml import Dataset
from config import Config


if __name__ == '__main__':
    # opt = Config()

    # dataset_folder = Dataset.get(dataset_name='KS-scoring-example', dataset_project='KS-scoring').get_local_copy()
    dataset_folder = Dataset.get(
        dataset_id='62ae7c4f5d234427bc61e77d9230ae19', alias='KS-scoring-example'
    ).get_local_copy()
    print(dataset_folder)
