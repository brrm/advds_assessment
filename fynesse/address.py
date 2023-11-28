# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

from . import assess, access
from typing import Optional
import numpy as np
import pandas as pd 
import statsmodels.api as sm
import geopandas as gpd


class WindowPricePredictor:
    # Price Predictor using only a small (temporal and spatial) data window
    def __init__(self, db, bbox_size: float, n_years: int = 1, features = None):
        self.db = db
        self.bbox_size = bbox_size
        self.n_years = n_years

        self.features = ['year_of_transfer', 'detached', 'flat', 'other', 'semidetached', 'terraced',
                         'school_closeness', 'healthcare_closeness', 'sustenance_closeness', 
                         'public_transport_closeness', 'nature_closeness', 'water_closeness',
                         'park_closeness', 'food_shop_closeness', 'commerce_closeness', 'tourism_closeness'
                        ] if features is None else features
        

    def predict(self, latitude: float, longitude: float, property_type: assess.PropertyType, date = None):
        north, south = latitude + self.bbox_size, latitude - self.bbox_size
        east, west = longitude + self.bbox_size, longitude - self.bbox_size

        # TODO: could also use month as a feature
        year = access.UK_PP_DATA_RANGE[1] if date is None else date.year

        # Fetch PP data in area
        gdf = assess.get_pc_bbox(self.db, north, south, east, west, min_year=year-self.n_years, max_year=year+self.n_years)
        gdf = gdf.drop(['postcode', 'locality', 'town_city', 'district', 'county', 'country', 'date_of_transfer', 'db_id'], axis=1)
        
        # Fetch POIs in area
        pois = assess.get_pois_labels(north, south, east, west, buffer=0.0)

        # Calculate closeness
        closeness_df = assess.calculate_closeness(gdf, pois)

        # Fit
        y = gdf['price'].values
        X = gdf.join(closeness_df)[self.features].values
        m_lin = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    
        # Predict
        pred_df = pd.DataFrame({'year_of_transfer': year} | property_type.get_dummies(), index=[0])
        pred_df = gpd.GeoDataFrame(pred_df,
                                   geometry=gpd.points_from_xy([longitude], [latitude]))
        pred_df.crs = "EPSG:4326"
        
        pred_df = pred_df.join(assess.calculate_closeness(pred_df, pois))
        pred_df = pred_df[self.features]
        pred = m_lin.get_prediction(pred_df.values).summary_frame(alpha=0.05)
        return pred
        
    

