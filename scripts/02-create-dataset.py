from clearml import Dataset
from config import Config


if __name__ == '__main__':
    opt = Config()

    dataset = Dataset.create(
        dataset_name='KS-scoring-example', dataset_project='KS-scoring', dataset_tags=['examples', 'external'],
    )
    dataset.add_files(opt.path_to_samples)
    dataset.upload()
    dataset.finalize()
    dataset.publish()
