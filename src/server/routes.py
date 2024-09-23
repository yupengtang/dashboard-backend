from flask_restful import Api
import resources as res

API = "/api/v1/"  # optional string



def add_routes(app):
    """Add URL mappings to the Flask instance"""
    api = Api(app, prefix=API)

    api.add_resource(res.TrainDataResource, "data/train_data")
    api.add_resource(res.TrainDataResourceByFeature, "data/train_data/<feature_name>")

    api.add_resource(res.ModelDataResource, "data/model")
    api.add_resource(res.ShapValuesDataResource, "data/model/shap_values")
    
    api.add_resource(res.PredictionDataResource, "data/prediction/<user_input>")
    api.add_resource(res.ShapValuesPredictionDataResource, "data/prediction/shap_values")

    api.add_resource(res.ParksResource, "data/parks")
    api.add_resource(res.ParksResourceByFeature, "data/parks/<feature_name>")

    api.add_resource(res.LifeQualityResource, "data/life_quality")

    api.add_resource(res.LifeQualityIndCols, "data/ind_cols")

    api.add_resource(res.LifeQualityResourceByFeature, "data/life_quality/<feature_name>")

    api.add_resource(res.FeaturesResource, "data/features")

    api.add_resource(res.ScatterResourceByFeature, "data/scatter_lq_year/<feature_name>/<park_name>")

    
    
    return api
