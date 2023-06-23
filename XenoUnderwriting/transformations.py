from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np

def perform_transformations(input_data, column_trans):
    transformed_features = column_trans.transform(input_data)
    return transformed_features

def get_column_transformer():
    log_scale_transformer = make_pipeline(FunctionTransformer(func=np.log1p, validate=True),StandardScaler())

    column_trans = ColumnTransformer( [(
                "binned_numeric_1", KBinsDiscretizer(n_bins=10, encode='onehot-dense'),["VehAge"],),
            ("binned_numeric_2",KBinsDiscretizer(n_bins=10, encode='onehot-dense'),["DrivAge"],),
            ("onehot_categorical",OneHotEncoder(),["VehBrand", "VehPower", "VehGas"],),
            ("log_scaled_numeric", log_scale_transformer, ["Density"]),
            ("passthrough_numeric", "passthrough", ["ClaimsNB", "Exposure", "ClaimAmount"]),],
        remainder="passthrough",)
    
    return column_trans

