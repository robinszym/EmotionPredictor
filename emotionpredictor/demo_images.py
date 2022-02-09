import os
import random

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import torch
import torch.nn as nn
import numpy as np
import plotly.express as px
import random 
from PIL import Image
import torch
import requests
import pandas as pd

import clip



ARTEMIS_EMOTIONS = ['amusement',
 'awe',
 'contentment',
 'excitement',
 'anger',
 'disgust',
 'fear',
 'sadness',
 'something else']

class SLP(nn.Module):
    def __init__(self,input_size = 512, output_size = 9):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, output_size)
        )
        
    def forward(self, x):
        fp = self.layers(x.float())
        return fp

MODEL_PATH = "../../code/C-RN50x16"
emotion = SLP(768)
emotion.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
wiki_base_url = "https://uploads6.wikiart.org/images/"

test_set = pd.read_csv("../index2painting")
test_set = test_set["image_files"]
random_paintings = test_set

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

@torch.no_grad()
def get_text_encoding(texts, text_encoder):
    text = clip.tokenize(texts).to(device)
    text_features = text_encoder(text)
    return text_features

@torch.no_grad()
def get_img_encoding(img, img_encoder):
    return img_encoder(img)

def preprocess_img(img):
    return preprocess(img).unsqueeze(0)

app.layout = html.Div([
    html.H1(
        children='Emotion Predictor : Image',
        style={'textAlign': 'center'}
    ),
    html.Br(),
    html.Div([
        dcc.Graph(id='img'),
        dcc.Graph(id='graph')
    ],style={'display': 'flex', 'flex-direction': 'row', "border" : "solid", "border-color" : "#F8F9FD"}),
    html.Div([
        html.H6("Provide the url of an image :"),
        html.Div([
            dcc.Input(id='my-input', value='https://uploads5.wikiart.org/images/tivadar-kosztka-csontvary/castellamare-di-stabia-1902.jpg', type='url'),
            html.Button('random', id='btn-random', n_clicks=0)
        ])
    ], style = {'textAlign' : 'center'}),
], style = {"background-color" : "#F5F6F7"})

@app.callback(
    [Output(component_id='img', component_property='figure'),
    Output(component_id='graph', component_property='figure')],
    [Input(component_id='my-input', component_property='value'),
    Input(component_id='btn-random', component_property='n_clicks')]
)
def update_output_div(input_value, n_clicks):
    ctx = dash.callback_context
    if ctx.triggered[0]["prop_id"] == "btn-random.n_clicks" :
        painting = test_set[random.randint(0,len(test_set))]
        input_value = wiki_base_url + painting.replace("_", "/") + ".jpg"
        
    img = Image.open(requests.get(input_value, stream=True).raw)
    features = get_img_encoding(preprocess_img(img), clip_model.visual)
    fig_bars = px.bar_polar(r = emotion(features).softmax(1).tolist()[0], theta = ARTEMIS_EMOTIONS, color = ARTEMIS_EMOTIONS)
    fig_bars.update_layout(transition_duration=1000, transition = {'easing': 'cubic-in-out'})
    fig_img = px.imshow(img)
    fig_img.update_layout(transition_duration=1000, transition = {'easing': 'cubic-in-out'})
    return fig_img, fig_bars

clip_model, preprocess = clip.load("RN50x16")
clip_model.eval();

if __name__ == '__main__':
    app.run_server(debug=True, port = 8050)