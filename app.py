import streamlit as st
import numpy as np
import joblib
import torch 
#from classe.classes import *

st.header("Classification")


def get_data():
    col1, col2, col3, col4 = st.columns(4)
    with col1 :
        gp = st.number_input("GP   :", min_value= 0.,value = 36., key = "gp" )
        min = st.number_input("MIN :", min_value= 0.,value = 27.4, key = "min")
        pts = st.number_input("PTS :", min_value= 0.,value = 7.4, key = "pts")
        fgm = st.number_input("FGM :", min_value= 0.,value = 2.6, key = "fgm")
    with col2 : 
        fga = st.number_input("FGA :", min_value= 0.,value = 7.6, key = "fga")
        fg = st.number_input("FG%  :", min_value= 0.,value = 34.7, key = "fg")
        ftm = st.number_input("FTM :", min_value= 0.,value = 0.5, key = "ftm")
        fta = st.number_input("FTA :", min_value= 0.,value = 2.3, key = "fta")
    with col3:
        ft = st.number_input("FT%  :", min_value= 0.,value = 69.9, key = "ft")
        oreb = st.number_input("OREB :", min_value= 0.,value = 0.7, key = "oreb")
        dreb = st.number_input("DREB :", min_value= 0.,value = 3.4, key = "dreb")
        reb = st.number_input("REB :", min_value= 0.,value = 4.1, key = "reb")
    with col4:
        ast = st.number_input("AST :", min_value= 0.,value = 1.9, key = "ast")
        stl = st.number_input("STL :", min_value= 0.,value = 0.4, key = "stl")
        blk = st.number_input("BLK :", min_value= 0.,value = .4, key = "blk")
        tov = st.number_input("TOV :", min_value= 0.,value = 1.3, key = "tov")

    return np.array([[gp, min, pts, fgm, fga, fg, ftm, fta, ft, oreb, dreb, reb, ast, stl, blk, tov]], dtype=np.float64)

# Get data
""" Enter les informations du joueur : """

data = get_data()

#data

# importation des modèles 
scaler = joblib.load("modèles/scaler.pkl")
linear_regression = joblib.load("modèles/linear_regression.sav")

#model_cnn = BinaryClassifierCNN()
#model_cnn.load_state_dict(torch.load('modèles/model.pth'))
#model_cnn.eval()

#model_mlp = torch.load('model_mlp.pt')

# Data transformation
data = scaler.transform(data)

# Make prediction
"""Legendre :
- 0 pour NON
- 1 pour OUI"""
st.write("La prédiction est : ",linear_regression.predict(data)[0])

#model_cnn(data)