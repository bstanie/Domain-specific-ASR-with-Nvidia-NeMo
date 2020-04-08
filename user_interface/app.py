# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Dash front end for ASR app
import argparse
import dash
import os
import time
import subprocess
import sys
import webbrowser
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate



def get_ip(step):
    # Return ip or path where App Step is hosted
    port = os.environ['C_PORT']
    ip_addr_host = os.environ['UI_HOST_IP']
    ip_show = "http://" + str(ip_addr_host) + ":" + str(port) + "/lab/tree"
    domain = "/WSJ/"
    ip_full = ip_show + domain
    if step == 0 :
      ip_full = ip_full + "Step%200.%20Models%20and%20Data%20-%20Download%20and%20Preparation.ipynb"
    elif step == 1:
      ip_full = ip_full + "Step%201.%20The%20effect%20of%20Acoustic%20Model%20training.ipynb"
    elif step == 2:
      ip_full = ip_full + "Step%202.%20The%20effect%20of%20adding%20a%20Language%20Model.ipynb"
    elif step == 3:
      ip_full = ip_full + "Step%203.%20Performance%20Comparison%20with%20WER.ipynb"
    return ip_full

# Load the dash bootstrap template
app = dash.Dash(__name__)
app.title = 'Domain Specific - NeMo ASR app NEW'
navbar = dbc.NavbarSimple(
    brand="Domain Specific - NeMo ASR Application",
    sticky="top",
    light=True,
)

# Cards declarations
INSTRUCT = ([
  dbc.CardBody(
        [
            html.P( """This NeMo Automatic Speech Recognition (ASR) Application
            enables you to train or fine-tune pre-trained (acoustic and language)
            ASR models with your own data. Through the steps below, we empower you to create and
            evaluate (on-prem) your own ASR models built on your domain specific
            audio data. Now you can progressively create better performing ASR
            models specifically built for your data."""),
        ],
    ),
])
# Data Preparation
PREP_DATA = ([
    dbc.CardHeader(html.H4("Step 0. Download and Preparation of Models and Datasets")),
    dbc.CardBody(
        [
            html.P( "Download pre-trained models and pre-process the data for the example use-case."),
            html.P(html.A([dbc.Button("Notebook: Preparation", color="primary")], href=get_ip(0), target='_blank'))
        ]
    ),
])
# Acoustic Model
TRAIN_AM = ([
    dbc.CardHeader(html.H4("Step 1. The effect of Acoustic Model training"),
                   style={'background-color':'#5e9400', 'color':'#FFF'}),
    dbc.CardBody(
        [
          html.P( """Train an acoustic model on your own data and
            compare the performance of the baseline pre-trained model vs. the domain fine-tuned model."""),
          html.P(html.A([dbc.Button("Notebook: Acoustic Model", color="primary")], href=get_ip(1), target='_blank'))
        ]
    ),
])
# Language Model
TRAIN_LM = ([
    dbc.CardHeader(html.H4("Step 2. The effect of adding a Language Model"),
                   style={'background-color':'#76b900', 'color':'#FFF'}),
    dbc.CardBody(
        [
            html.P("""Train an N-gram language model on domain specific data and
            compare its performance with a language model trained on LiberiSpeech data."""),
            html.P(html.A([dbc.Button("Notebook: Language Model", color="primary")], href=get_ip(2), target='_blank'))
        ]
    ),
])
# Performance
PERF = ([
    dbc.CardHeader(html.H4("Step 3. Performance Comparison with WER"),
                   style={'background-color':'#7cd000', 'color':'#FFF'}),
    dbc.CardBody(
        [
            html.P("""Compare the Word Error Rate (WER) performance of the pre-trained and
            fine-tuned models created in the acoustic and language model workflows."""),
            html.P(html.A([dbc.Button("Notebook: Model Comparison", color="primary")], href=get_ip(3), target='_blank'))
        ]
    ),
])

# Layout
body = dbc.Container(
    [
      dbc.Row(
        dbc.Col(dbc.Card(INSTRUCT, style={"border":"None", "font-size":"1.2rem", "margin-bottom":"0"})),

      ),
      dbc.Row(
        dbc.Col(dbc.Card(PREP_DATA, color="light", inverse=False)),
      ),
      dbc.Row(
            [dbc.Col(dbc.Card(TRAIN_AM, color="light", inverse=False)),
             dbc.Col(dbc.Card(TRAIN_LM, color="light", inverse=False)),
             dbc.Col(dbc.Card(PERF, color="light", inverse=False))],
      ),
    ],
)
app.layout = html.Div([navbar, body])
if __name__ == "__main__":
    # Use host IP to access UI
    host_ip = os.environ['UI_HOST_IP']
    print("For remote access use ip: ", host_ip)
    app.run_server(debug=False, host='0.0.0.0', port=os.environ['UI_PORT'])