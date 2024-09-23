import os
import json
from flask_restful import Resource
import pandas as pd
from resources import plot_utils as pu
import json



def read_data(filename):
    f = open(filename)
    return json.load(f)


DESC_FILENAME = "src/server/data/features.json"
data = read_data(DESC_FILENAME)


class FeaturesResource(Resource):
    """Resource to retrieve information about all features in the ames dataset."""

    def get(self):
        """GET request handler"""
        return data
    