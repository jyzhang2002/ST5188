from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import shutil

import libs.utils as utils
import numpy as np
import pandas as pd

Deque = collections.deque

class HyperparamOptManager(object):
    """Manages hyperparameter optimisation using random search for a single worker.

    Attributes:
        param_ranges: dict[str, list]   -- discrete hyperparameter candidates
        results:    pd.DataFrame        -- columns = run_name, rows = {"loss","info"}
        fixed_params: dict              -- experiment-level fixed params
        saved_params: pd.DataFrame      -- columns = run_name, rows = param values
        best_score: float               -- lowest loss so far
        optimal_name: str               -- key of best run
        hyperparam_folder: str          -- folder to save artefacts
    """

    def __init__(self,
                 param_ranges,
                 fixed_params,
                 model_folder,
                 override_w_fixed_params=True):
        """Instantiates manager.

        Args:
            param_ranges: Discrete hyperparameter range for random search.
            fixed_params: Fixed model parameters per experiment.
            model_folder: Folder to store optimisation artifacts.
            override_w_fixed_params: Whether to override serialised fixed params
                                     with newly supplied values.
        """
        self.param_ranges = param_ranges
        self._max_tries = 1000

        self.results = pd.DataFrame()
        self.saved_params = pd.DataFrame()

        self.fixed_params = fixed_params
        self.best_score = np.inf
        self.optimal_name = ""

        self.hyperparam_folder = model_folder
        utils.create_folder_if_not_exist(self.hyperparam_folder)

        self._override_w_fixed_params = override_w_fixed_params

    # ------------------------------------------------------------------
    # basic utils
    # ------------------------------------------------------------------
    def load_results(self):
        """Loads results from previous hyperparameter optimisation (if any).

        Returns:
            bool: True if load is successful.
        """
        print("Loading results from", self.hyperparam_folder)

        results_file = os.path.join(self.hyperparam_folder, "results.csv")
        params_file = os.path.join(self.hyperparam_folder, "params.csv")

        if os.path.exists(results_file) and os.path.exists(params_file):
            self.results = pd.read_csv(results_file, index_col=0)
            self.saved_params = pd.read_csv(params_file, index_col=0)

            if not self.results.empty:
                # ensure 'loss' is float
                self.results.at["loss"] = self.results.loc["loss"].apply(float)
                self.best_score = self.results.loc["loss"].min()
                is_optimal = self.results.loc["loss"] == self.best_score
                # index of column where loss is best
                self.optimal_name = self.results.T[is_optimal].index[0]
                return True

        return False

    def _get_params_from_name(self, name):
        """Returns previously saved parameters given a key."""
        params_df = self.saved_params
        selected_params = dict(params_df[name])

        if self._override_w_fixed_params:
            for k in self.fixed_params:
                selected_params[k] = self.fixed_params[k]
        return selected_params

    def get_best_params(self):
        """Returns the optimal hyperparameters thus far."""
        return self._get_params_from_name(self.optimal_name)

    def clear(self):
        """Clears all previous results and saved parameters."""
        if os.path.exists(self.hyperparam_folder):
            shutil.rmtree(self.hyperparam_folder)
        os.makedirs(self.hyperparam_folder, exist_ok=True)
        self.results = pd.DataFrame()
        self.saved_params = pd.DataFrame()
        self.best_score = np.inf
        self.optimal_name = ""

    # ------------------------------------------------------------------
    # name / param sampling
    # ------------------------------------------------------------------
    def _check_params(self, params):
        """Checks that parameter map is properly defined."""
        valid_fields = list(self.param_ranges.keys()) + list(self.fixed_params.keys())
        invalid_fields = [k for k in params if k not in valid_fields]
        missing_fields = [k for k in valid_fields if k not in params]

        if invalid_fields:
            raise ValueError("Invalid Fields Found {} - Valid ones are {}".format(
                invalid_fields, valid_fields))
        if missing_fields:
            raise ValueError("Missing Fields Found {} - Valid ones are {}".format(
                missing_fields, valid_fields))

    def _get_name(self, params):
        """Returns a unique key for the supplied set of params."""
        self._check_params(params)
        fields = list(params.keys())
        fields.sort()
        return "_".join([str(params[k]) for k in fields])

    def get_next_parameters(self, ranges_to_skip=None):
        """Returns the next set of parameters to optimise.

        Args:
            ranges_to_skip: iterable of run names to skip.
        """
        if ranges_to_skip is None:
            # skip already-run configs (i.e. columns of results)
            ranges_to_skip = set(self.results.columns)

        if not isinstance(self.param_ranges, dict):
            raise ValueError("Only works for random search!")

        param_range_keys = list(self.param_ranges.keys())
        param_range_keys.sort()

        def _sample_once():
            params = {}
            for k in param_range_keys:
                val = self.param_ranges[k]

                try:
                    arr = np.asarray(val)
                except Exception as e:
                    raise ValueError(f"Cannot convert param '{k}' value {val} to array: {e}")

                if arr.ndim == 0:
                    arr = np.array([arr.item()])
                elif arr.ndim > 1:
                    arr = arr.flatten()

                choices = arr.tolist()

                if len(choices) == 0:
                    raise ValueError(f"Parameter '{k}' has empty search space!")

                params[k] = np.random.choice(choices)

            # add fixed params
            for fk in self.fixed_params:
                params[fk] = self.fixed_params[fk]
            return params

        for _ in range(self._max_tries):
            params = _sample_once()
            name = self._get_name(params)
            if name not in ranges_to_skip:
                return params

        raise ValueError("Exceeded max number of hyperparameter searches!!")

    # ------------------------------------------------------------------
    # result update
    # ------------------------------------------------------------------
    def update_score(self, parameters, loss, model, info=""):
        """Updates the results from last optimisation run.

        Args:
            parameters: dict of hyperparameters
            loss: float validation loss
            model: model object that has `.save(folder)` method
            info: extra info str

        Returns:
            bool: True if this run is the best so far
        """
        if np.isnan(loss):
            loss = np.inf

        if not os.path.isdir(self.hyperparam_folder):
            os.makedirs(self.hyperparam_folder, exist_ok=True)

        name = self._get_name(parameters)
        is_optimal = self.results.empty or loss < self.best_score

        if is_optimal and model is not None:
            print("Optimal model found, saving to", self.hyperparam_folder)
            model.save(self.hyperparam_folder)
            self.best_score = loss
            self.optimal_name = name

        # write into DataFrames
        self.results[name] = pd.Series({"loss": loss, "info": info})
        self.saved_params[name] = pd.Series(parameters)

        # persist to disk
        self.results.to_csv(os.path.join(self.hyperparam_folder, "results.csv"))
        self.saved_params.to_csv(os.path.join(self.hyperparam_folder, "params.csv"))

        return is_optimal


# ======================================================================
# Distributed version
# ======================================================================
class DistributedHyperparamOptManager(HyperparamOptManager):
    """Manages distributed hyperparameter optimisation across many workers/GPUs."""

    def __init__(self,
                 param_ranges,
                 fixed_params,
                 root_model_folder,
                 worker_number,
                 search_iterations=1000,
                 num_iterations_per_worker=5,
                 clear_serialised_params=False):
        """
        This version pre-generates `search_iterations` hyperparam combos
        and stores them under root_model_folder/hyperparams.
        Each worker picks its own subset according to worker_number.

        Args:
            param_ranges: dict[str, list]
            fixed_params: dict
            root_model_folder: str
            worker_number: int (1-based)
            search_iterations: int
            num_iterations_per_worker: int
            clear_serialised_params: bool
        """
        max_workers = int(np.ceil(search_iterations / num_iterations_per_worker))
        if worker_number > max_workers:
            raise ValueError(
                "Worker number ({}) cannot be larger than max workers ({})!".format(
                    worker_number, max_workers
                )
            )
        if worker_number > search_iterations:
            raise ValueError(
                "Worker number ({}) cannot be larger than search iterations ({})!".format(
                    worker_number, search_iterations
                )
            )

        print("*** Creating hyperparameter manager for worker {} ***".format(worker_number))

        hyperparam_folder = os.path.join(root_model_folder, str(worker_number))
        super(DistributedHyperparamOptManager, self).__init__(
            param_ranges,
            fixed_params,
            hyperparam_folder,
            override_w_fixed_params=True,
        )

        # for serialised hyperparam lists
        serialised_ranges_folder = os.path.join(root_model_folder, "hyperparams")
        self.serialised_ranges_folder = serialised_ranges_folder
        if clear_serialised_params:
            print("Regenerating hyperparameter list ...")
            if os.path.exists(serialised_ranges_folder):
                shutil.rmtree(serialised_ranges_folder)
        utils.create_folder_if_not_exist(serialised_ranges_folder)

        self.serialised_ranges_path = os.path.join(
            serialised_ranges_folder, "ranges_{}.csv".format(search_iterations)
        )

        self.hyperparam_folder = hyperparam_folder
        self.worker_num = worker_number
        self.total_search_iterations = search_iterations
        self.num_iterations_per_worker = num_iterations_per_worker

        self.global_hyperparam_df = self.load_serialised_hyperparam_df()
        self.worker_search_queue = self._get_worker_search_queue()

    # ------------------------------------------------------------------
    @property
    def optimisation_completed(self):
        return False if self.worker_search_queue else True

    def get_next_parameters(self):
        """Return next hyperparam dict for this worker."""
        if not self.worker_search_queue:
            raise ValueError("No more parameter combinations left for this worker.")
        param_name = self.worker_search_queue.pop()
        params = self.global_hyperparam_df.loc[param_name, :].to_dict()

        # Always override with fixed params
        for k in self.fixed_params:
            print("Overriding saved {} -> {}".format(k, self.fixed_params[k]))
            params[k] = self.fixed_params[k]
        return params

    # ------------------------------------------------------------------
    # serialised hyperparam list
    # ------------------------------------------------------------------
    def load_serialised_hyperparam_df(self):
        """Loads pre-generated hyperparam combinations from disk."""
        print("Loading params for {} search iterations from {}".format(
            self.total_search_iterations, self.serialised_ranges_path
        ))

        if os.path.exists(self.serialised_ranges_path):
            df = pd.read_csv(self.serialised_ranges_path, index_col=0)
        else:
            print("No serialised ranges found, regenerating ...")
            df = self.update_serialised_hyperparam_df()

        return df

    def update_serialised_hyperparam_df(self):
        """Regenerates and saves the full hyperparam combinations."""
        search_df = self._generate_full_hyperparam_df()
        print("Serialising params for {} search iterations to {}".format(
            self.total_search_iterations, self.serialised_ranges_path
        ))
        search_df.to_csv(self.serialised_ranges_path)
        return search_df

    def _generate_full_hyperparam_df(self):
        """Pre-generates all hyperparam combos for all workers."""
        np.random.seed(131)  # for reproducibility

        name_list = []
        param_list = []
        for _ in range(self.total_search_iterations):
            params = super(DistributedHyperparamOptManager, self).get_next_parameters(name_list)
            name = self._get_name(params)
            name_list.append(name)
            param_list.append(params)

        full_df = pd.DataFrame(param_list, index=name_list)
        return full_df

    # ------------------------------------------------------------------
    def clear(self):
        """Clears results for this worker and resets its queue."""
        super(DistributedHyperparamOptManager, self).clear()
        self.worker_search_queue = self._get_worker_search_queue()

    def load_results(self):
        """Load results from file and rebuild worker queue."""
        success = super(DistributedHyperparamOptManager, self).load_results()
        if success:
            self.worker_search_queue = self._get_worker_search_queue()
        return success

    # ------------------------------------------------------------------
    def _get_worker_search_queue(self):
        """Generates the queue of param combinations for current worker."""
        global_df = self.assign_worker_numbers(self.global_hyperparam_df)
        worker_df = global_df[global_df["worker"] == self.worker_num]
        # only those not yet finished (i.e. not in self.results)
        leftover = [name for name in worker_df.index if name not in self.results.columns]
        return Deque(leftover)

    def assign_worker_numbers(self, df):
        """Assigns worker numbers to each pre-generated param combo."""
        output = df.copy()

        n = self.total_search_iterations
        batch_size = self.num_iterations_per_worker
        max_worker_num = int(np.ceil(n / batch_size))

        worker_idx = np.concatenate([
            np.tile(i + 1, self.num_iterations_per_worker)
            for i in range(max_worker_num)
        ])
        output["worker"] = worker_idx[:len(output)]
        return output
