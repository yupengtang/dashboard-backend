import os
import json
import random
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


class ScatterResourceByFeature(Resource):
    attr_by_name = {
        'HCSRCP': ['HCSRCP_2016-2018', 'HCSRCP_2015-2017'],
        'CZD': ['CZD_2019', 'CZD_2018', 'CZD_2017', 'CZD_2016'],
        'CZM': ['CZM_2019', 'CZM_2018', 'CZM_2017', 'CZM_2016'],
        'CZR': ['CZR_2019', 'CZR_2018', 'CZR_2017', 'CZR_2016'],
        'CZS': ['CZS_2019', 'CZS_2018', 'CZS_2017', 'CZS_2016'],
        'CZV': ['CZV_2019', 'CZV_2018', 'CZV_2017', 'CZV_2016'],
        'TRF': ['TRF_2019', 'TRF_2018', 'TRF_2017', 'TRF_2016'],
        'PMC': ['PMC_2019', 'PMC_2018', 'PMC_2017', 'PMC_2016'],
        'MEODR': ['MEODR_2019', 'MEODR_2018', 'MEODR_2017', 'MEODR_2016'],
        'HCSOBP': ['HCSOBP_2016-2018', 'HCSOBP_2015-2017'],
        'HCSOHSP': ['HCSOHSP_2016-2018', 'HCSOHSP_2015-2017'],
        'HCSNSP': ['HCSNSP_2016-2018', 'HCSNSP_2015-2017'],
        'HCSPAP': ['HCSPAP_2016-2018', 'HCSPAP_2015-2017'],
        'HCSSP': ['HCSSP_2016-2018', 'HCSSP_2015-2017'],
        'VRLE': ['VRLE_2019', 'VRLE_2018', 'VRLE_2017', 'VRLE_2016', 'VRLE_2015']
    }

    # FIXME
    single_attr_by_name = { 
        'UNS_2015-2019': ['VRLE_2017','UNS_2015-2019'],
        'CCG_2015-2019': ['CCG_2015-2019','VRCAR_2015-2019'],
        'CCR_2015-2019': ['CCR_2015-2019','VRCAR_2015-2019'],
        'EDB_2015-2019':['EDB_2015-2019','EDE_2015-2019'],
        'EDE_2015-2019':['EDB_2015-2019','EDE_2015-2019'],
        'UMP_2015-2019': ['EDB_2015-2019','UMP_2015-2019'],
        'POV_2015-2019':['POV_2015-2019','UMP_2015-2019'],
        'LEQ_2015-2019':['LEQ_2015-2019','UMP_2015-2019'],
        'POP_2015-2019':['POP_2015-2019','EDB_2015-2019'],
        'VRDI_2015-2019':['VRDO_2015-2019','VRDI_2015-2019'],
        'VRDO_2015-2019':['VRDO_2015-2019','VRDI_2015-2019'],
        'VRALR_2015-2019':['VRALR_2015-2019','VRDIDR_2015-2019'],
        'VRDIDR_2015-2019':['VRALR_2015-2019','VRDIDR_2015-2019'],
        'VRSUR_2015-2019': ['VRSUR_2015-2019','VRCAR_2015-2019'],
        'VRCAR_2015-2019': ['VRSUR_2015-2019','VRCAR_2015-2019'],
        'VEGETATED_SQFT':['VEGETATED_SQFT','PARK_COUNT'],
        'PARK_COUNT':['PARK_COUNT','VEGETATED_SQFT']
    }

    def _get_normalized_by_index(self, df: pd.DataFrame, feature: str, index: int):
        norm_X = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
        return  norm_X.loc[index].item()

    """Resource to retrieve information about a feature by id."""
    def _get_data_years_park_attr(self, df, park_id:str, attr:str, park_name:str):
        # Get metrics
        attr_name = ''
        for key in self.attr_by_name.keys():
            if attr.split('_')[0] == key:
                attr_name = key

                # norm_X_VRLE = (df["VRLE_2017"] - df["VRLE_2017"].min()) / (df["VRLE_2017"].max() - df["VRLE_2017"].min())
                # norm_X_VRLE = (df["VRLE_2017"] - df["VRLE_2017"].min()) / (df["VRLE_2017"].max() - df["VRLE_2017"].min())
                # print(norm_X_VRLE.loc[park_id].item())

                # Get years 
                list_attr = []
                years = [int(i.split("_")[1].split("-")[0]) for i in self.attr_by_name[attr_name]]
                for cy, curr_attr in zip(years, self.attr_by_name[attr_name]):
                    vals = {
                        "value": df.loc[park_id, curr_attr].item(),
                        "years": cy,
                        "park": park_name,
                        "lifeExpectancy": self._get_normalized_by_index(df, "VRLE_2017", park_id),
                        "numberOfParks": self._get_normalized_by_index(df, "HCSOHSP_2015-2017", park_id),
                        "trafficIntensity": self._get_normalized_by_index(df, "TRF_2017", park_id),
                    }
                    list_attr.append(vals)
                return list_attr

        attr_name = attr
        list_attr = []
        
        for feature in self.single_attr_by_name.get(attr):
            vals = {
                "value": self._get_normalized_by_index(df, feature, park_id),
                "feature": feature,
                "park": park_name,
                "lifeExpectancy": self._get_normalized_by_index(df, "VRLE_2017", park_id),
                "numberOfParks": self._get_normalized_by_index(df, "HCSOHSP_2015-2017", park_id),
                "trafficIntensity": self._get_normalized_by_index(df, "TRF_2017", park_id),
            }

            list_attr.append(vals)
    
        return list_attr
        
        

    def get(self, feature_name, park_name):
        """GET request handler"""
        # Get parks
        ix_park = lq_df.index[lq_df["Community"] == park_name].tolist()[0]
        cluster_id = lq_df.loc[ix_park, "CLUSTER_LABEL"]
        parks_in_cluster = lq_df["Id"][lq_df["CLUSTER_LABEL"] == cluster_id].values.tolist()
        parks_to_plot = random.choices(parks_in_cluster, k=3)
        parks_to_plot.append(ix_park)
        parks_to_plot = list(set(parks_to_plot))
        parks_names = lq_df.loc[parks_to_plot, "Community"]

        list_parks = []
        for park, name in zip(parks_to_plot, parks_names):
            coordinates = self._get_data_years_park_attr(lq_df, park_id=park, park_name=name, attr=feature_name)
            list_parks.append(coordinates)

        print(list_parks)

        json_object = json.dumps(list_parks)
        print(json_object)

        return json.loads(json_object)
