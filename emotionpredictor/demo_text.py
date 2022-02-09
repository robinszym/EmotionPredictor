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

def get_text_encoding(texts, text_encoder):
    text = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = text_encoder(text)
    return text_features




app.layout = html.Div([
    html.H3("Emotion Predictor : Text"),
    html.H6("Predict the emotion reaction of this sentence"),
    html.Div([
        "Input: ",
        dcc.Input(id='my-input', value='initial value', type='text')
    ]),
    html.Br(),
    dcc.Graph(id='graph')
])

@app.callback(
    Output(component_id='graph', component_property='figure'),
    Input(component_id='my-input', component_property='value')
)
def update_output_div(input_value):
    
    
    text_features = get_text_encoding([input_value], clip_model.encode_text)
    fig_bars = px.bar(y = emotion(text_features).softmax(1).tolist()[0], x = ARTEMIS_EMOTIONS, color = ARTEMIS_EMOTIONS,
           title = f"Emotion prediction for {input_value}" )
    fig_bars.update_layout(transition_duration=1000, transition = {'easing': 'cubic-in-out'})
    
    return fig_bars

clip_model, preprocess = clip.load("RN50x16")
if __name__ == '__main__':
    app.run_server(debug=True, port = 8048)
