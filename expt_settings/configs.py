import os

import data_formatters.volatility
import data_formatters.citibike

class ExperimentConfig(object):
    """Defines experiment configs and paths to outputs."""

    default_experiments = ["volatility", "citibike"]

    def __init__(self, experiment="volatility", root_folder=None):
        if experiment not in self.default_experiments:
            raise ValueError(f"Unrecognised experiment={experiment}")

        if root_folder is None:
            root_folder = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "..", "outputs"
            )
            print(f"Using root folder {root_folder}")

        self.root_folder = root_folder
        self.experiment = experiment
        self.data_folder = os.path.join(root_folder, "data", experiment)
        self.model_folder = os.path.join(root_folder, "saved_models", experiment)
        self.results_folder = os.path.join(root_folder, "results", experiment)

        for d in [self.root_folder, self.data_folder, self.model_folder, self.results_folder]:
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

    @property
    def data_csv_path(self):
        csv_map = {
            "volatility": "formatted_omi_vol.csv",
            "citibike": "202507-citibike-tft-features-cleaned-500000-isnight.csv",
        }
        return os.path.join(self.data_folder, csv_map[self.experiment])

    @property
    def hyperparam_iterations(self):
        return 60

    def make_data_formatter(self):
        if self.experiment == "volatility":
            return data_formatters.volatility.VolatilityFormatter()
        elif self.experiment == "citibike":
            return data_formatters.citibike.CitibikeFormatter()
        else:
            raise ValueError(f"No formatter for experiment={self.experiment}")
