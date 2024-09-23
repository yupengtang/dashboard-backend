import os
import json
from flask_restful import Resource
import pandas as pd


def read_data(filename):
    df = pd.read_csv(filename)
    return df

# FIXME
PARKS_DESC_FILENAME = "src/server/data/parks_data.csv"
PARKS_PREP_DESC_FILENAME = "src/server/data/parks_df_preprocessed.csv"
parks_df = read_data(PARKS_PREP_DESC_FILENAME)
parks_json = json.loads(parks_df.to_json(orient='records'))

class ParksResource(Resource):
    """Resource to retrieve information about all features in the ames dataset."""

    def get(self):
        """GET request handler"""
        return parks_json


class ParksResourceByFeature(Resource):
    """Resource to retrieve information about a feature by id."""

    def get(self, feature_name):
        """GET request handler"""
        return json.loads(parks_df[feature_name].to_json(orient='records'))
