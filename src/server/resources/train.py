import pandas as pd
'''import numpy as np
import xgboost
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler
import shap'''
import json
from flask_restful import Resource
import warnings
warnings.filterwarnings('ignore')

def train_xgb(data_file):
     # import imputed dataset

     #df_imputed = pd.read_csv('master_data_chicago_imputed.csv')
     df_imputed = pd.read_csv(data_file)
    #  print(df_imputed.shape)

     # remove unnecessary columns and calculate % ratio between green and total area

     df_preprocessed = df_imputed.drop(columns=['community', 'GEOID', 'Longitude', 'Latitude'])
     df_preprocessed['Green area to total ratio'] = df_preprocessed['TOTAL_VEGETATED_SQFT'] / df_preprocessed['TOTAL_COM_AREA_SQFT'] * 100.0

     # scale the data (not necessary tbh)

     scaler = MinMaxScaler()

     #df_scaled = scaler.fit_transform(df_preprocessed)
     #df_preprocessed = pd.DataFrame(df_scaled, columns=df_preprocessed.columns)

     # prepare X and y

     dep_cols = ['HCSOBP_2015-2017', 
               'DIS_2015-2019', 
               'HCSOHSP_2016-2018',
               'HCSNSP_2016-2018', 
               'EDE_2015-2019', 
               'UMP_2015-2019',
               'HCSPAP_2015-2017',
               'LEQ_2015-2019', 
               'POP_2015-2019', 
               'VRALR_2015-2019',
               'VRDIDR_2015-2019', 
               'VRDO_2015-2019', 
               'VRSUR_2015-2019',
               'VRCAR_2015-2019', 
               'VRDI_2015-2019', 
               'VRLE_2019'
               ]

     # choose target variable

     #ind_cols = 'PARK_VEGETATED_SQFT' # -> total park area
     ind_cols = 'Green area to total ratio' # -> ratio

     X = df_preprocessed[dep_cols]
     y = df_preprocessed[ind_cols]

     # define own MSE (grid search maximizes negative MSE, so we do reverse for better interpretability)

     mse = make_scorer(mean_squared_error)#,greater_is_better=False)

     # split data into train and test sets

     X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.20, shuffle = True)

     xgbr = XGBRegressor()

     # define XGB parameter ranges for GS trials 

     xgb_params = {'nthread':[4], #when use hyperthread, xgboost may become slower
               'n_estimators': [100],
               'learning_rate': [.01, .03, .05, .07, .1], 
               'max_depth': [3, 5, 6, 7],
               'min_child_weight': [4],
               'subsample': [0.6, 0.7, 0.75, 0.8],
               'colsample_bytree': [0.7, 0.75, 0.8],
               'colsample_bylevel': [0.7, 0.75, 0.8],
               'random_state': [42]
               }

     gsXGB = GridSearchCV(xgbr, xgb_params, cv = 5, scoring=mse,
                         refit=True, n_jobs = 5, verbose=True)
     gsXGB.fit(X_train, y_train)

     # save best XGB model

     XGB_best = gsXGB.best_estimator_

    #  print("Best train score: ", gsXGB.best_score_)

     # check validation score

     y_valid_pred = XGB_best.predict(X_valid)
    #  print("Validation set r2 score: ", mean_squared_error(y_valid, y_valid_pred, squared=False))

     # train best XGB model on full train set and save to json

     rgsr = XGBRegressor(nthread=gsXGB.best_params_['nthread'],
                         n_estimators = 200,
                         max_depth=gsXGB.best_params_['max_depth'], 
                         min_child_weight = gsXGB.best_params_['min_child_weight'],
                         learning_rate=gsXGB.best_params_['learning_rate'], 
                         subsample=gsXGB.best_params_['subsample'], 
                         colsample_bytree=gsXGB.best_params_['colsample_bytree'],
                         random_state=42, seed=42)

     rgsr.fit(X, y)
     rgsr.save_model("src/server/data/model_xgb.json")
     model_file = "src/server/data/model_xgb.json"
     model = XGBRegressor()
     model.load_model("src/server/data/model_xgb.json")

     # use SHAP for XAI

     # explainer = shap.Explainer(model)

     # calculate shap values. This is what we will plot.

     # shap_values = explainer(X)

     # SHAP plots

     '''shap.initjs()

     shap.plots.force(shap_values[3])
     shap.plots.waterfall(shap_values[3])'''

     # how to save SHAP values and plot results for 1 prediction

     sample_idx = 36
     X_sample = X[sample_idx:sample_idx+1]
     y_sample = y[sample_idx:sample_idx+1]
     shap_values_list = model.feature_importances_
     #shap_values_list = explainer.shap_values(X_sample, y=y_sample)
     shap_values_df = pd.DataFrame(shap_values_list, columns=dep_cols)
     shap_values_df.to_json('src/server/data/shap_values.json')
     shap_values_file = 'src/server/data/shap_values.json'
          
     # shap.summary_plot(shap_values, X_sample, plot_type="bar")


     return model, model_file, shap_values_df


data_path = 'src/server/data/master_data_chicago_imputed.csv'
model_file = "src/server/data/model_xgb.json"
shap_values_file = 'src/server/data/shap_values.json'
#model, model_file, shap_values = train_xgb(data_path)


class ModelDataResource(Resource):
    """Resource to retrieve information about all features in the ames dataset."""

    def get(self):
        """GET request handler"""
        with open(model_file, 'r') as j:
          model = json.loads(j.read())
        return model
    
class ShapValuesDataResource(Resource):
    """Resource to retrieve information about all features in the ames dataset."""

    def get(self):
        """GET request handler"""
        with open(shap_values_file, 'r') as j:
          shap_values = json.loads(j.read())
        return shap_values