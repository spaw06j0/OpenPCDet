import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import time

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from google.protobuf import text_format
from skimage import io
# from multiprocessing import Process, Value
import threading
import base64

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

class GlobalConfig:
    def __init__(self):
        self.root_path = None 
        self.config_path = None
        self.dataset = None
        self.model = None
        self.image_indices = None

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg
# python ./demo.py \
# --cfg ../output/kitti_models/centerpoint_pillar_svd/default/centerpoint_pillar_svd.yaml \
# --ckpt ../output/kitti_models/centerpoint_pillar_svd/default/ckpt/checkpoint_epoch_120.pth  \
# --data_path ../data/kitti/training/velodyne/000000.bin

BACKEND=GlobalConfig()


def create_server_function(my_app, logger):
    @my_app.route('/api/readinfo', methods=['POST'])            # finish
    def build_dataset():
        """
        Request
        --------
            config_path:    str
                OpenPCDet yaml file 
            root_path:      str
                point cloud bin file
        
        Response
        --------
            image_indices:  array=(n,)
                list of point cloud indices
        
        """
        global BACKEND
        response = {"status": "normal"}
        instance = request.json

        config_path = Path(instance["config_path"])
        data_path = Path(instance["root_path"])
        ext_path = '.bin'

        print('config_path', config_path)
        print('cfg', cfg)
        cfg_from_yaml_file(config_path, cfg)
        print(f'[Roger] start to build dataset')
        st_time = time.time()
        demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=data_path, ext=ext_path, logger=logger
        )
        BACKEND.dataset = demo_dataset
        BACKEND.image_indices = list(range(len(demo_dataset)))
        print(f'[Roger] finish build dataset: {time.time()-st_time:6.4f}')
        
        response["image_indexes"] = BACKEND.image_indices
        response = jsonify(results=[response])
        response.headers['Access-Control-Allow-Headers'] = '*'
        return response

    @my_app.route('/api/read_point_cloud', methods=['POST'])    # finish 
    def read_point_cloud():
        """
        Request
        --------
            num_of_features:    int
                number of feature in each points
            point_cloud_path:   str
                point cloud data path, either *.bin or *.npy file
            enable_int16:       bool
                whether quantilize the point cloud to int16 or not
            int16_factor
                quatnilize to certan extend, default 100
            
        Response
        --------
            num_of_feature:  int
                the number of feature in each point
            pointcloud:      str
                the point cloud encode to feature
        
        """
        global BACKEND
        instance = request.json
        num_of_features =  int(instance["num_of_features"])
        point_cloud_path = instance["point_cloud_path"]
        enable_int16 = instance["enable_int16"]
        response = {"status": "normal"}

        points = np.fromfile(point_cloud_path, dtype=np.float32, count=-1).reshape([-1, num_of_features])
        points = points[:, :3]
        if enable_int16:
            int16_factor = instance["int16_factor"]
            points *= int16_factor
            points = points.astype(np.int16)
        pc_str = base64.b64encode(points.tobytes())
        pc_str = pc_str.decode("utf-8")
        
        response["num_features"]=3
        response["pointcloud"] = pc_str
        response = jsonify(results=[response])
        response.headers['Access-Control-Allow-Headers'] = '*'
        return response

    @my_app.route('/api/get_pointcloud', methods=['POST'])      # partial finish
    def get_pointcloud():
        """
        Request
        --------
            image_idx:    int
                the index of point cloud to get
            enable_int16:       bool
                whether quantilize the point cloud to int16 or not
            int16_factor
                quatnilize to certan extend, default 100
            
        Response
        --------
            locs: array=(num_of_boxes, 3)
                the location of 3d bbox, (x,y,z)
                in kitti, x is front, y is left , z is top.
            dims: array=(num_of_boxes, 3)
                the length, width, height of 3d bbox
            rots: array=(num_of_boxes, 3)
                the x,y,z rotation of bbox, usually x,y is 0
            labels: array(num_of_bboxes,)
                the label of 3d bbox
            gt_2Dboxes: array(num_of_bboxes, 4)
                the 2d boxes coordiante on image, x1y1 x2y2
            num_features: int
                the number of feature in each points
            pointcloud:      str
                the point cloud encode to feature
            transfer_start: int
            correction_time: int
        """
        global BACKEND
        instance = request.json
        image_idx = instance["image_idx"]
        enable_int16 = instance["enable_int16"]
        is_inference = instance["is_inference"]
        response = { "status": "normal" }
        print('*'*200)
        print('is_inference', is_inference)
        print('enable_int16', enable_int16)
        print('image_idx', image_idx)

        prepare_time = time.time()
        data_dict = BACKEND.dataset[image_idx]
        if is_inference:
            if BACKEND.model is not None:
                infer_sample = BACKEND.dataset.collate_batch([data_dict])
                load_data_to_gpu(infer_sample)
                prepare_time = time.time() - prepare_time

                inference_time = time.time()
                with torch.no_grad():
                    pred_dicts, _ = BACKEND.model.forward(infer_sample)
                inference_time = time.time() - inference_time
        
                for key in pred_dicts[0].keys():
                    if isinstance(pred_dicts[0][key], torch.Tensor):
                        pred_dicts[0][key] = pred_dicts[0][key].cpu().numpy()

                rot = pred_dicts[0]['pred_boxes'][:, 6:7]
                rot_pad = np.zeros(rot.shape, rot.dtype)
                rot = np.concatenate([rot_pad, rot_pad, rot], axis=1)

                data_dict["dt_locs"] = pred_dicts[0]['pred_boxes'][:, 0:3].tolist()
                data_dict["dt_dims"] = pred_dicts[0]['pred_boxes'][:, 3:6].tolist()
                data_dict["dt_rots"] = rot.tolist()
                data_dict["dt_labels"] = pred_dicts[0]['pred_labels'].tolist()
                data_dict["dt_scores"] = pred_dicts[0]['pred_scores'].tolist()
                data_dict["inference_time"] = inference_time
                data_dict["processing_time"] = processing_time

            else:
                print('Model is empty')
                pass

        # Quantize data
        points = data_dict['points'][:, :3]
        if enable_int16:
            int16_factor = instance["int16_factor"]
            points *= int16_factor
            points = points.astype(np.int16)
        pc_str = base64.b64encode(points.tobytes())
        pc_str = pc_str.decode("utf-8")
    

        response["status"] = "normal"
        # response["locs"] = []
        # response["dims"] = []
        # response["rots"] = []
        # response["labels"] = []
        # response["gt_2Dboxes"] = []
        response["num_features"] = 3
        response["pointcloud"] = pc_st
        response["transfer_start"] = tim.time()
        response["correction_time"] = 0
        
        response = jsonify(results=[response])
        response.headers['Access-Control-Allow-Headers'] = '*'
        print("send response with size {}!".format(len(pc_str)))
        return response

    @my_app.route('/api/get_image', methods=['POST'])
    def get_image():
        global BACKEND
        instance = request.json
        response = {"status": "normal"}
        if BACKEND.root_path is None:
            return error_response("root path is not set")    
        image_idx = instance["image_idx"]
        idx = BACKEND.image_idxes.index(image_idx)
        query = {
            "lidar": {
                "idx": idx
            },
            "cam": {}
        }
        sensor_data = BACKEND.dataset.get_sensor_data(query)
        if "cam" in sensor_data and "data" in sensor_data["cam"] and sensor_data["cam"]["data"] is not None:
            image_str = sensor_data["cam"]["data"]
            response["image_b64"] = base64.b64encode(image_str).decode("utf-8")
            response["image_b64"] = 'data:image/{};base64,'.format(sensor_data["cam"]["datatype"]) + response["image_b64"]
            print("send an image with size {}!".format(len(response["image_b64"])))
        else:
            response["image_b64"] = ""
        response['transfer_start'] = time.time()
        response = jsonify(results=[response])
        response.headers['Access-Control-Allow-Headers'] = '*'
        return response

    @my_app.route('/api/build_network', methods=['POST'])       # finish
    def build_network_():
        """
        Request
        -------
        config_path: str
            the yaml config path
        checkpoint_path: str
            model checkpoint path
        
        Response
        --------
        None
        """
        global BACKEND
        instance = request.json
        cfg_path = Path(instance["config_path"])
        ckpt_path = Path(instance["checkpoint_path"])
        response = {"status": "normal"}
        if not cfg_path.exists():
            return error_response("config file not exist.")
        if not ckpt_path.exists():
            return error_response("ckpt file not exist.")

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=BACKEND.dataset)
        model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()
        BACKEND.model = model
        print('[Roger] successfully load network parameter: ', ckpt_path)

        response = jsonify(results=[response])
        response.headers['Access-Control-Allow-Headers'] = '*'
        return response

    @my_app.route('/api/inference_by_idx', methods=['POST'])    # finish
    def inference_by_idx():
        """
        Request
        -------
            image_idx:          int
                the data sample index

        Response
        --------
            dt_2Dboxes:         array=(num_of_bbox, 4) or Optional
            dt_locs:            array=(num_of_bbox, 3)
            dt_dims:            array=(num_of_bbox, 3)
            dt_rots:            array=(num_of_bbox, 3)
            dt_labels:          array=(num_of_bbox,)
            dt_scores:          array=(num_of_bbox,)
            processing_time:    int
            inference_time:     int
            transfer_time:      int
        """
        global BACKEND
        instance = request.json
        data_idx = instance["image_idx"]

        prepare_time = time.time()
        data_dict = BACKEND.dataset[data_idx]
        infer_sample = BACKEND.dataset.collate_batch([data_dict])
        load_data_to_gpu(infer_sample)
        prepare_time = time.time() - prepare_time


        inference_time = time.time()
        with torch.no_grad():
            pred_dicts, _ = BACKEND.model.forward(infer_sample)
        inference_time = time.time() - inference_time
        
        for key in pred_dicts[0].keys():
            if isinstance(pred_dicts[0][key], torch.Tensor):
                pred_dicts[0][key] = pred_dicts[0][key].cpu().numpy()

        rot = pred_dicts[0]['pred_boxes'][:, 6:7]
        rot_pad = np.zeros(rot.shape, rot.dtype)
        rot = np.concatenate([rot_pad, rot_pad, rot], axis=1)

        response = {
            "status": "normal",
            "dt_locs": pred_dicts[0]['pred_boxes'][:, 0:3].tolist(),
            "dt_dims": pred_dicts[0]['pred_boxes'][:, 3:6].tolist(),
            "dt_rots": rot.tolist(),
            "dt_labels": pred_dicts[0]['pred_labels'].tolist(),
            "dt_scores": pred_dicts[0]['pred_scores'].tolist(),
            "processing_time": prepare_time,
            "inference_time": inference_time,
            "transfer_start": 0,
        }
        response = jsonify(results=[response])
        response.headers['Access-Control-Allow-Headers'] = '*'
        return response        

    logger.info("Finish Creating All Response Function")