import os
import pandas as pd

from flask_restful import Resource


class DatasetResource(Resource):
    """dataset resource."""
    data_root = os.path.join("../", "data")

    def get(self, name):

        path_name = os.path.join(self.data_root, f"{name}.csv")
        data = pd.read_csv(path_name)

        return data.to_json()
