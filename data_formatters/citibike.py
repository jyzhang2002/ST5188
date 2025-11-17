import data_formatters.base
import data_formatters.volatility
import pandas as pd
import numpy as np

VolatilityFormatter = data_formatters.volatility.VolatilityFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class CitibikeFormatter(VolatilityFormatter):
    """Full-feature data formatter for Citi Bike + meteorological dataset."""

    # ----------------------------------------------------------------------
    # 📌 FULL COLUMN DEFINITION
    # ----------------------------------------------------------------------
    _column_definition = [
        # ==== Identifiers ====
        ('station_id', DataTypes.CATEGORICAL, InputTypes.ID),
        ('time', DataTypes.DATE, InputTypes.TIME),

        # ==== Target ====
        ('values', DataTypes.REAL_VALUED, InputTypes.TARGET),

        # ==== Core demand features ====
        ('arrivals', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('departures', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),

        # ==== Time known inputs ====
        ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('dow', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('is_weekend', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('sensor_day', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('is_night', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),

        # ==== Static station features ====
        ('latitude', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
        ('longitude', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
        ('station_name', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('station_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        
        # keep final_features_list
        ('spd_med_lag6', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('tt_med', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('wspd_obs_roll6h_mean', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pm25_mean_lag24', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pm25_mean_roll3h_mean', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('spd_med_roll24h_sum', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pm25_mean', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pm25_mean_lag3', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('temp_obs_roll24h_mean', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('wspd_obs_roll6h_sum', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pm25_mean_roll6h_mean', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pm25_mean_lag6', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pm25_mean_roll24h_mean', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pm25_mean_roll6h_sum', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pm25_mean_roll3h_sum', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('temp_obs_roll24h_sum', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('rh_obs', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pm25_mean_lag1', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('pm25_mean_roll24h_sum', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('wspd_obs_roll24h_mean', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
    ]

    # ----------------------------------------------------------------------
    # 📌 Derived feature construction
    # ----------------------------------------------------------------------
    def _ensure_values(self, df):
        """Ensure target variable 'values' = arrivals - departures."""
        if 'values' not in df.columns:
            if 'arrivals' not in df.columns or 'departures' not in df.columns:
                raise ValueError("Both 'arrivals' and 'departures' must exist to derive 'values'.")
            df = df.copy()
            df['values'] = df['arrivals'] - df['departures']
        return df

    def _ensure_sensor_day(self, df):
        """Derive integer 'sensor_day' based on date part of 'time'."""
        if 'sensor_day' in df.columns:
            return df
        if 'time' not in df.columns:
            raise ValueError("Column 'time' is required to derive sensor_day.")

        df = df.copy()
        dt = pd.to_datetime(df['time'], errors='coerce', utc=False)
        if dt.isna().all():
            raise ValueError("Unable to parse 'time' column to datetime.")
        base_day = dt.dt.normalize().min()
        df['sensor_day'] = (dt.dt.normalize() - base_day).dt.days.astype('int32')
        return df
    
    def _propose_boundaries(self, num_days: int):
        """Simple heuristic for short-range dataset."""
        # Our dataset ~33 days long (June 29 – July 31)
        if num_days <= 10:
            valid_b = max(3, num_days - 4)
            test_b = max(5, num_days - 2)
        else:
            valid_b = int(round(num_days * 0.80))
            test_b = int(round(num_days * 0.85))

        valid_b = max(3, min(valid_b, num_days - 4))
        test_b = max(valid_b + 1, min(num_days - 1, num_days))
        return int(valid_b), int(test_b)

    def _ensure_core_columns(self, df):
        """Ensure required helper columns exist."""
        df = df.copy()
        df = self._ensure_values(df)

        # ---- station_id -> id/categorical_id ----
        if 'id' not in df.columns or 'categorical_id' not in df.columns:
            codes, uniques = pd.factorize(df['station_id'], sort=True)
            df['id'] = codes.astype('int32')
            df['categorical_id'] = df['id']

        # ---- time parsing ----
        dt = pd.to_datetime(df['time'], errors='coerce', utc=False)
        if dt.isna().all():
            raise ValueError("Failed to parse 'time' column.")

        # ---- hours_from_start ----
        if 'hours_from_start' not in df.columns:
            base = dt.min()
            df['hours_from_start'] = (dt - base).dt.total_seconds() / 3600.0

        # ---- time_on_day ----
        df['time_on_day'] = dt.dt.hour + dt.dt.minute / 60.0

        # ---- day_of_week ----
        if 'dow' not in df.columns:
            df['dow'] = dt.dt.isocalendar().day.astype('int16')

        # ---- ensure sensor_day ----
        df = self._ensure_sensor_day(df)
        return df

    def split_data(self, df, valid_boundary=None, test_boundary=None):
        print('Formatting train-valid-test splits for full Citi Bike dataset.')
        df = self._ensure_core_columns(df)
        index = df['sensor_day']
        num_days = int(index.max() - index.min() + 1)

        valid_boundary = 22
        test_boundary = 26

        if valid_boundary in (None, 'auto') or test_boundary in (None, 'auto'):
            valid_boundary, test_boundary = self._propose_boundaries(num_days)
            print(f"[CitibikeFormatter] Auto boundaries -> valid={valid_boundary}, test={test_boundary}, num_days={num_days}")

        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary - 7) & (index < test_boundary)]
        test = df.loc[index >= test_boundary - 7]

        self.set_scalers(train)
        train_t, valid_t, test_t = (self.transform_inputs(x) for x in [train, valid, test])
        return train_t, valid_t, test_t

    # ----------------------------------------------------------------------
    # 📌 Model configs
    # ----------------------------------------------------------------------
    def get_fixed_params(self):
        return {
            'total_time_steps': 8 * 24,
            'num_encoder_steps': 7 * 24,
            'num_epochs': 10,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 4
        }
    
    # def get_experiment_params(self):
    #     return self.get_fixed_params()

    def get_default_model_params(self):
        return {
            'dropout_rate': 0.3,
            'hidden_layer_size': 64,
            'learning_rate': 0.001,
            'minibatch_size': 64,
            'max_gradient_norm': 1.00,
            'num_heads': 4,
            'stack_size': 1
        }

    def get_num_samples_for_calibration(self):
        # return 217924, 117859
        # return 300, 100
        # return 100000, 50000
        # return 30000, 10000
        # return 21726, 11661
        # return 201268, 67891
        return 217924, 51235
