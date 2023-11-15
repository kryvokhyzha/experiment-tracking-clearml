from pathlib import Path
from typing import Iterable, Optional

import yaml


class Config(dict):
    def __init__(
        self,
        *args,
        init_paths: bool = True,
        init_dirs: bool = True,
        init_params: bool = True,
        config_filenames: Optional[Iterable[str]] = None,
        **kwargs
    ):
        super(Config, self).__init__()
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

        if init_paths:
            # self.path_to_root = Path(__file__).resolve().parent.parent
            self.path_to_root = Path(__file__).resolve().parent.parent
            self.path_to_data = self.path_to_root / "data"
            self.path_to_samples = self.path_to_data / "samples"
            self.path_to_prepared_data = self.path_to_data / "prepared"
            self.path_to_models = self.path_to_root / "models"

        if init_dirs:
            self._init_dirs()

        if config_filenames is not None:
            self.config_filenames = config_filenames
            self._read_configs()

        if init_params:
            pass

    def _read_configs(self):
        for config_filename in self.config_filenames:
            try:
                with open(self.path_to_configs / config_filename, "r") as f:
                    config_yaml = yaml.safe_load(f)

                self.update(Config.dict_to_map(config_yaml))
            except FileNotFoundError:
                pass

    def _init_dirs(self):
        self.path_to_data.mkdir(exist_ok=True, parents=True)
        self.path_to_samples.mkdir(exist_ok=True, parents=True)
        self.path_to_prepared_data.mkdir(exist_ok=True, parents=True)
        self.path_to_models.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def dict_to_map(obj):
        for k, v in obj.items():
            if isinstance(v, dict):
                obj[k] = Config(
                    Config.dict_to_map(v), init_paths=False, init_dirs=False, init_params=False, config_filenames=None
                )
        return obj

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Config, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Config, self).__delitem__(key)
        del self.__dict__[key]
