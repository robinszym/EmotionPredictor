import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import torch
import torch.nn as nn
import numpy as np
import plotly.express as px
import os
import random 
from PIL import Image
import clip
import torch
from PIL import Image
import requests

device = "cpu"
def get_iterator(path, preprocess) :
    def iterator():
        for image in os.listdir(path):
            if image[-4:] != ".jpg":
                continue
            img_path = osp.join(path,image)
            yield preprocess(Image.open(img_path)).unsqueeze(0)
    return iterator()



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

ARTEMIS_EMOTIONS = ['amusement',
 'awe',
 'contentment',
 'excitement',
 'anger',
 'disgust',
 'fear',
 'sadness',
 'something else']
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
            dcc.Input(id='my-input', value='https://uploads5.wikiart.org/images/tivadar-kosztka-csontvary/castellamare-di-stabia-1902.jpg', type='url')
        ])
    ], style = {'textAlign' : 'center'}),
], style = {"background-color" : "#F5F6F7"})

@app.callback(
    [Output(component_id='img', component_property='figure'),
    Output(component_id='graph', component_property='figure')],
    Input(component_id='my-input', component_property='value')
)
def update_output_div(input_value):
    
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