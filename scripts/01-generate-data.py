from utils import generate_sample
from config import Config


if __name__ == '__main__':
    opt = Config()
    df = generate_sample('multiclass_classification')
    df.to_csv(opt.path_to_samples / 'sample.csv', index=False, sep=';')
