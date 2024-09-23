import os
import json
from flask_restful import Resource
import pandas as pd


def read_data(filename):
    df = pd.read_csv(filename)
    return df

# FIXME
TRAIN_DESC_FILENAME = "src/server/data/master_data_chicago_imputed.csv"
TRAIN_PREP_DESC_FILENAME = "src/server/data/master_data_chicago_imputed.csv"
train_df = read_data(TRAIN_PREP_DESC_FILENAME)
train_json = json.loads(train_df.to_json(orient='records'))

class TrainDataResource(Resource):
    """Resource to retrieve information about all features in the ames dataset."""

    def get(self):
        """GET request handler"""
        return train_json


class TrainDataResourceByFeature(Resource):
    """Resource to retrieve information about a feature by id."""

    def get(self, feature_name):
        """GET request handler"""
        return json.loads(train_df[feature_name].to_json(orient='records'))
