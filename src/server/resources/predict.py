import pandas as pd
import numpy as np
import json
import shap
from flask_restful import Resource
from xgboost import XGBRegressor, plot_importance
import warnings
warnings.filterwarnings('ignore')

def predict_new(input):
    
    inpt = input.split("@")
    x_inpt = inpt[1:]
    park = inpt[0]
    X = []
    
    cols = [
        "HCSOBP_2015-2017",
        "DIS_2015-2019",
        "HCSOHSP_2016-2018",
        "HCSNSP_2016-2018",
        "EDE_2015-2019",
        "UMP_2015-2019",
        "HCSPAP_2015-2017",
        "LEQ_2015-2019",
        "VRALR_2015-2019",
        "VRDIDR_2015-2019",
        "VRDO_2015-2019",
        "VRSUR_2015-2019",
        "VRCAR_2015-2019",
        "VRLE_2019",
        "POP_2015-2019",
        "VRDI_2015-2019",  
        "community"
    ]

    pos_cols = [
            'HCSOHSP_2016-2018',
            'HCSNSP_2016-2018', 
            'EDE_2015-2019',
            'LEQ_2015-2019', 
            'POP_2015-2019', 
            'VRLE_2019'
           ]
    
    orig_df = pd.read_csv('src/server/data/master_data_chicago_imputed.csv')
    
    orig_features = orig_df[cols]
    
    orig_X = orig_features[orig_features['community'] == park]
    orig_X.drop(columns='community', inplace=True)
    orig_X_np = np.array(orig_X.values)
    
    i = 0
    
    while i < len(x_inpt):
        feature = float(x_inpt[i])
        if orig_X.columns[i] in pos_cols:
            ft = orig_X_np[0][i] * (1.0 + feature)
        else:
            ft = orig_X_np[0][i] * (1.0 - feature)
        X.append(float(ft))
        i += 1
    
    model = XGBRegressor()
    model.load_model("src/server/data/model_xgb.json")
    
    X = np.array(X).reshape((1, 16))

    explainer = shap.TreeExplainer(model)
    
    y = model.predict(X)
    
    if y[0] < 0:
        y[0] = 0.0

    new_area = round(orig_df[orig_df['community'] == park]['TOTAL_COM_AREA_SQFT'].values[0] * float(y[0]) / 100.0, 2)
    area_diff = new_area - orig_df[orig_df['community'] == park]['TOTAL_VEGETATED_SQFT'].values[0]
    
    no_parks = int(round(area_diff / orig_df[orig_df['community'] == park]['AVG_PARK_SQFT'].values[0], 0))
    
    if no_parks < 0.0:
        no_parks = 0.0    
    
    output = {"value": [str(round(y[0], 2)), str(no_parks)]}
    

    shap_values_list = explainer.shap_values(X)
    shap_values_df = pd.DataFrame(shap_values_list, columns=cols[:-1])
    shap_values_df.to_json('src/server/data/shap_values_prediction.json', orient="index")
    
    return output

shap_values_file = 'src/server/data/shap_values_prediction.json'

class PredictionDataResource(Resource):
    """Resource to retrieve information about all features in the ames dataset."""

    def get(self, user_input):
        """GET request handler"""
        
        output = predict_new(user_input)
        json_output = json.dumps(output)
        
        return json.loads(json_output)
    
class ShapValuesPredictionDataResource(Resource):
    """Resource to retrieve information about all features in the ames dataset."""

    def get(self):
        """GET request handler"""
        with open(shap_values_file, 'r') as j:
          shap_values_prediction = json.load(j)['0']
        return shap_values_prediction
    