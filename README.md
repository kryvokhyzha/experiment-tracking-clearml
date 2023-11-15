# Experiment Tracking ClearML

This repository contains the example of ClearML usage.

## Setup python environment

1. Clone the repository using `git clone` command.
2. Open the terminal and go to the project directory using `cd` command.
3. Create virtual environment using `python -m venv venv` or
   `conda create -n venv python=3.10` command. We have used `Python 3.10` during
   development.
4. Activate virtual environment using `source venv/bin/activate` or
   `conda activate venv` command.
5. Install poetry using instructions from
   [here](https://python-poetry.org/docs/#installation). Use
   `with the official installer` section.
6. Set the following option to disable new virtualenv creation:
   ```bash
   poetry config virtualenvs.create false
   ```
7. Install dependencies using `poetry install --no-root -E all` command. The
   `--no-root` flag is needed to avoid installing the package itself.
8. Setup `pre-commit` hooks using `pre-commit install` command. More information
   about `pre-commit` you can find [here](https://pre-commit.com/).
9. Run the test to check the correctness of the project work using following
   command:
   ```bash
   python -m unittest -b
   ```
10. After successful passing of the tests, you can work with the project!
11. If you want to add new dependencies, use `poetry add <package_name>`
    command. More information about `poetry` you can find
    [here](https://python-poetry.org/docs/basic-usage/).
12. If you want to add new tests, use `unittest` library. More information about
    `unittest` you can find
    [here](https://docs.python.org/3/library/unittest.html). All tests should be
    placed in the `tests` directory.
13. All commits should be checked by `pre-commit` hooks. If you want to skip
    this check, use `git commit --no-verify` command. But it is not recommended
    to do this.
14. Also, you can run `pre-commit` hooks manually using
    `pre-commit run --all-files` command.
15. More useful commands you can find in `Makefile`.

## Setup ClearML server

1. See
   [installation guide](https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server_linux_mac/)
   for your platform. If you encounter the `elasticserach` error, try to change
   the volume for this service to:

```
- /opt/clearml/elasticsearch/logs:/usr/share/elasticsearch/logs`
```

2. Run the docker-compose to start the server
3. Initialize ClearML client (firstly, you need to install the python
   dependencies):

```bash
clearml-init
```

4. Run the following command to start the worker:

```bash
clearml-agent daemon --queue default --foreground
```

## Examples

### How to start?

1. Generate the dataset using the following command:

```bash
python scripts/01-generate-data.py
```

2. Create and upload dataset to the ClearML:

```bash
python scripts/02-create-dataset.py
```

3. Train & Evaluate the model using the following command:

```bash
python src/main.py
```

4. Navigate to the ClearML web interface and see the results.
