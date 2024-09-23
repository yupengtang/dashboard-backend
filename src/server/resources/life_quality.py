import os
import json
from flask_restful import Resource
import pandas as pd
from resources import plot_utils as pu
import json



def read_data(filename):
    df = pd.read_csv(filename)
    return df


def preprocess_data(filename):
    df_lq = pu.read_data(filename)
    df_lq, ind_cols = pu.preprocess_life_quality_data(df_lq)
    return df_lq, ind_cols


LQ_DESC_FILENAME = "src/server/data/master_chicago_data_imputed_clusters.csv"
lq_df, ind_cols = preprocess_data(LQ_DESC_FILENAME)
lf_json = json.loads(lq_df.to_json(orient='records'))


class LifeQualityResource(Resource):
    """Resource to retrieve information about all features in the ames dataset."""

    def get(self):
        """GET request handler"""
        return lf_json
    

class LifeQualityIndCols(Resource):
    """Resource to retrieve information about all features in the ames dataset."""

    def get(self):
        """GET request handler"""
        return json.dumps(list(ind_cols))


class LifeQualityResourceByFeature(Resource):
    """Resource to retrieve information about a feature by id."""

    def get(self, feature_name):
        """GET request handler"""
        return json.loads(lq_df[feature_name].to_json(orient='records'))
    