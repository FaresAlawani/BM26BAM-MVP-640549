from django.shortcuts import render
from .transformations import perform_transformations, get_column_transformer
import pandas as pd
import numpy as np
import pickle

#####
def predictor(request):
    return render(request, 'main.html')
#####
def formInfo(request):
    ClaimsNB = request.GET["ClaimsNB"]
    Exposure = request.GET["Exposure"]
    VehPower = request.GET["VehPower"]
    VehAge = request.GET["VehAge"]
    DrivAge = request.GET["DrivAge"]
    VehBrand = request.GET["VehBrand"]
    VehGas = request.GET["VehGas"]
    Density = request.GET["Density"]
    ClaimAmount = request.GET["ClaimAmount"]
#######
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

#Dataframe and its transformations
    input_data = pd.DataFrame({
        "ClaimsNB": [ClaimsNB],
        "Exposure": [Exposure],
        "VehPower": [VehPower],
        "VehAge": [VehAge],
        "DrivAge": [DrivAge],
        "VehBrand": [VehBrand],
        "VehGas": [VehGas],
        "Density": [Density],
        "ClaimAmount": [ClaimAmount]})
#####
    column_trans = get_column_transformer()
    column_trans.fit(input_data)
    #######
    transformed_features = perform_transformations(input_data, column_trans)
    print(transformed_features)
#######
    y_pred = model.predict(transformed_features)
    #######
    context = {'prediction': y_pred}
    return render(request, 'results.html', context)
  
