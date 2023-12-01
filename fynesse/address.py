# This file contains code for supporting addressing questions in the data

from . import assess, access
from typing import Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
import geopandas as gpd
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


class WindowPredictor(ABC):
    # Price Predictor using only a small (temporal and spatial) data window
    def __init__(self, db: access.DBConn, bbox_size: float,
                 features=None, closeness_features=None, density_features=None, zones_features=False):
        """
        Abstract framework for fitting a predictor to a limited (temporal and spatial) data window
        :param: bbox_size specifies how far (in degrees) to go in each direction from the fitting point
        """
        self.db = db
        self.bbox_size = bbox_size

        self.features = ['year_of_transfer', 'month_of_transfer',
                         'detached', 'flat', 'other', 'semidetached', 'terraced'
                         ] if features is None else features
        self.closeness_features = ['school', 'healthcare', 'sustenance', 'public_transport',
                                   'nature', 'water', 'park', 'food_shop', 'commerce', 'tourism'
                                   ] if closeness_features is None else closeness_features
        self.density_features = ['school', 'healthcare', 'sustenance', 'public_transport',
                                 'nature', 'water', 'park', 'food_shop', 'commerce', 'tourism'
                                 ] if density_features is None else density_features
        self.zones_features = zones_features

        self.fitted_model = None
        self.train_idx, self.test_idx = None, None

        # For caching data we use to train the model
        self.gdf = None
        self.pois = None
        self.cache_lat, self.cache_long, self.cache_min_year, self.cache_max_year = None, None, None, None

    def fit(self, latitude: float, longitude: float, min_year: int, max_year: int, test_split_size=0.0):
        """
        Fit the model, centering the data bbox on (latitude, longitude), with dates in (min_year, max_year) inclusive
        :param test_split_size: how much of the data to reserve for testing
        """
        self._get_data(latitude, longitude, min_year, max_year)

        if test_split_size > 0.0:
            self.train_idx, self.test_idx = train_test_split(self.gdf.index, test_size=test_split_size)
        else:
            self.train_idx = self.gdf.index

        self.fit_raw(self.gdf.loc[self.train_idx])

    @abstractmethod
    def fit_raw(self, df):
        """
        Fitting procedure to implement for a specific model
        :param df: will be a GeoDF containing `self.features` columns
        """
        pass

    def predict(self, latitudes: list[float], longitudes: list[float], property_types: list[assess.PropertyType], dates,
                alpha=0.05):
        # Create features for prediction points
        features_df = pd.DataFrame([{'year_of_transfer': dates[i].year,
                                     'month_of_transfer': dates[i].month
                                     } | property_types[i].get_dummies() for i in range(len(property_types))])
        features_df = gpd.GeoDataFrame(features_df,
                                       geometry=gpd.points_from_xy(longitudes, latitudes))
        features_df.crs = "EPSG:4326"

        return self.predict_raw(features_df, alpha=alpha)

    @abstractmethod
    def predict_raw(self, df, alpha=0.05):
        """
        Prediction procedure to implement for a specific model
        :param df: will be a GeoDF containing `self.features` columns
        :param alpha: confidence interval
        """
        pass

    def predict_test(self):
        """
        return predictions on the test set after fitting with test_split_size > 0
        """
        if self.test_idx is None:
            raise "No test set. Fit model with test_split_size > 0 first."
        return self.predict_raw(self.gdf.loc[self.test_idx])

    def _get_data(self, latitude: float, longitude: float, min_year: int, max_year: int):
        if not (self.cache_lat is None) and not (self.cache_long is None) \
                and self.cache_lat == latitude and self.cache_long == longitude \
                and self.cache_min_year == min_year and self.cache_max_year == max_year:
            return

        north, south = latitude + self.bbox_size, latitude - self.bbox_size
        east, west = longitude + self.bbox_size, longitude - self.bbox_size

        # Fetch PP data in area
        self.gdf = assess.get_pc_bbox(self.db, north, south, east, west,
                                      min_year=min_year, max_year=max_year)

        # Fetch POIs in area
        self.pois = assess.get_pois_labels(north, south, east, west, buffer=0.01)

        self.cache_lat, self.cache_long = latitude, longitude
        self.cache_min_year, self.cache_max_year = min_year, max_year

    def prepare_features(self, df):
        """
        Add closeness, density and zones features to input DF. Also rescales some features.
        :param df: GeoDF with `self.features` cols
        """

        # Calculate closeness & densities
        closeness_df = assess.calculate_poi_closeness(df, self.pois, features=self.closeness_features)
        density_df = assess.calculate_poi_density(df, self.pois, features=self.density_features)
        density_df = np.log(density_df + 1)

        zones_df = assess.calculate_zones(df) if self.zones_features else pd.DataFrame()

        X = df[self.features].join(closeness_df).join(density_df).join(zones_df)

        if 'month_of_transfer' in X.columns:
            X['month_sin'] = np.sin((np.pi / 6.0) * X['month_of_transfer'])
            X['month_cos'] = np.cos((np.pi / 6.0) * X['month_of_transfer'])
            X = X.drop(columns=['month_of_transfer'])

        # We also rescale the year feature so that regularization doesn't misbehave
        if 'year_of_transfer' in X.columns:
            X['year_of_transfer'] = X['year_of_transfer'] - self.cache_min_year

        return X

    def get_feature_names(self):
        feature_names = self.features + [f'{x}_closeness' for x in self.closeness_features] + \
                        [f'{x}_density' for x in self.density_features] + \
                        (['commercial_retail_zone', 'farmland_zone', 'industrial_zone', 'residential_zone']
                         if self.zones_features else [])

        if 'month_of_transfer' in feature_names:
            feature_names.remove('month_of_transfer')
            feature_names = feature_names + ['month_sin', 'month_cos']

        return feature_names


class WindowPredictorLogLink(WindowPredictor):
    def __init__(self, db, bbox_size: float,
                 features=None, closeness_features=None, density_features=None, zones_features=False,
                 reg_alpha=0, L1_wt=0.5):
        """
        Gaussian GLM using log-link function
        :param reg_alpha: alpha parameter for regularisation
        :param L1_wt: L1 weight for regularisation
        """
        super().__init__(db, bbox_size, features, closeness_features, density_features, zones_features)
        self.reg_alpha = reg_alpha
        self.L1_wt = L1_wt

    def fit_raw(self, df):
        X = self.prepare_features(df)

        # Rescale price as otherwise regularizer can overflow
        y = df['price'] / 100_000.0

        model = sm.GLM(y.values, X.values, family=sm.families.Gaussian(link=sm.families.links.Log()))
        self.fitted_model = model.fit_regularized(alpha=self.reg_alpha, L1_wt=self.L1_wt, maxiter=1000) \
            if self.reg_alpha > 0.0 else model.fit()

    def predict_raw(self, df, alpha=0.05):
        X = self.prepare_features(df)

        preds = pd.Series(self.fitted_model.predict(X.values), name='mean').to_frame() if self.reg_alpha > 0.0 else \
            self.fitted_model.get_prediction(X.values).summary_frame(alpha=alpha)
        preds = preds * 100_000.0
        return preds


class WindowPredictorOLS(WindowPredictor):
    def __init__(self, db, bbox_size: float,
                 features=None, closeness_features=None, density_features=None, zones_features=False,
                 log_price=False):
        """
        OLS model
        :param log_price: whether to fit the logarithm of the price
        """
        super().__init__(db, bbox_size, features, closeness_features, density_features, zones_features)
        self.log_price = log_price

    def fit_raw(self, df):
        X = self.prepare_features(df)

        y = df['price'] / 100_000.0
        if self.log_price:
            y = np.log10(y)

        model = sm.OLS(y.values, X.values)
        self.fitted_model = model.fit()

    def predict_raw(self, df, alpha=0.05):
        X = self.prepare_features(df)

        preds = self.fitted_model.get_prediction(X.values).summary_frame(alpha=alpha)

        if self.log_price:
            preds = np.power(10.0, preds)

        preds = preds * 100_000.0
        return preds


class WindowPredictorRF(WindowPredictor):
    def __init__(self, db, bbox_size: float,
                 features=None, closeness_features=None, density_features=None, zones_features=False,
                 log_price=False, **rf_args):
        """
        RF model
        :param log_price: whether to fit the logarithm of the price
        :param rf_args: arguments to pass to RandomForestRegressor
        """
        super().__init__(db, bbox_size, features, closeness_features, density_features, zones_features)
        self.log_price = log_price
        self.rf_args = rf_args

    def fit_raw(self, df):
        X = self.prepare_features(df)

        y = df['price'] / 100_000.0
        if self.log_price:
            y = np.log10(y)

        self.fitted_model = RandomForestRegressor(**self.rf_args)
        self.fitted_model.fit(X, y)

    def predict_raw(self, df, alpha=0.05):
        X = self.prepare_features(df)

        preds = self.fitted_model.predict(X)

        if self.log_price:
            preds = np.power(10.0, preds)

        preds = preds * 100_000.0
        return preds


def predict_price(db, latitude, longitude, date, property_type):
    """Price prediction for UK housing."""
    m = WindowPredictorLogLink(db, bbox_size=0.05,
                               features=['year_of_transfer', 'detached', 'flat', 'other',
                                         'semidetached', 'terraced'],
                               closeness_features=['school', 'public_transport', 'nature', 'water',
                                                   'park'],
                               density_features=['sustenance', 'public_transport', 'food_shop',
                                                 'commerce', 'tourism'],
                               zones_features=False)

    m.fit(latitude=latitude, longitude=longitude,
          min_year=date.year - 1, max_year=date.year + 1)

    return m.predict([latitude], [longitude], [property_type], dates=[date])
