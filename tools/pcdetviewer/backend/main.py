# import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# Socket
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from utils import create_server_function
import fire

app = Flask("Visualizer")
CORS(app)
socketio = SocketIO(
    app, 
    async_mode='threading',
    max_http_buffer_size=1024 * 1,
    cors_allowed_origins=['*', 'http://127.0.0.1:8000', 'http://localhost:8000']
)



def main(port=16666):
    logger = common_utils.create_logger()
    
    create_server_function(app, logger)
    socketio.run(app, debug=True, host='127.0.0.1', port=port)

if __name__ == '__main__':
    fire.Fire()