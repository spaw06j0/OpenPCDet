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
import pickle
import cv2
import copy
from posixpath import split
# =============================
from PIL import Image


class Calibration(object):
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            print('[Calibration] read file', calib_file)
            calib = self.get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

        # trans_lidar_to_cam and trans_cam_to_img'
        self.V2C_ = np.vstack(
            (self.V2C, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
        self.R0_ = np.hstack(
            (self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        self.R0_ = np.vstack(
            (self.R0_, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
        self.V2R = self.R0_ @ self.V2C_

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack(
            (pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack(
            (self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack(
            (R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack(
            (self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1

        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(
            np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        print('pts_img', pts_img.shape)
        pts_rect_depth = pts_2d_hom[:, 2] - \
            self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate(
            (x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate(
            (corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :,
                                          2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate(
            (x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate(
            (x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

    def get_calib_from_file(self, calib_file):
        with open(calib_file) as f:
            lines = f.readlines()

        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)
        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)

        return {'P2': P2.reshape(3, 4),
                'P3': P3.reshape(3, 4),
                'R0': R0.reshape(3, 3),
                'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin', info_path=None):
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
        self.info_path = info_path
        self.ext = ext
        self.image_list = []

        if self.info_path:
            print(f'[Roger] read info path: {self.info_path}')
            with open(self.info_path, 'rb') as f:
                data = pickle.load(f)
            # Read Point Cloud Path(.bin)
            if 'waymo' in str(self.info_path).split('/'):
                self.ext = '.npy'
                print(data[0].keys())
                print(data[0]['frame_id'])
                data_file_list = [
                    str(root_path / 'waymo_processed_data_v0_5_0' / f"{x['point_cloud']['lidar_sequence']}" / f"{x['point_cloud']['sample_idx']:04}{self.ext}") for x in data]
                # Read Image Path(.png)
                self.image_list = None
                # Read Calib Path(.txt)
                self.calib_list = None
                # Read Ground Truth
                print(data[0]['annos'].keys())
                self.gth = None
                # self.gth = [{
                #     'gt_2Dboxes':   None,
                #     'gt_dims':      x['annos']['dimensions'][x['annos']['name'] != 'DontCare'],
                #     'gt_locs':      x['annos']['location'][x['annos']['name'] != 'DontCare'],
                #     'gt_rots':      x['annos']['heading_angles'][x['annos']['name'] != 'DontCare'].reshape((-1, 1)),
                #     'gt_labels':    x['annos']['name'][x['annos']['name'] != 'DontCare'],
                # } for x in data if 'annos' in x]
            else:
                imgext = 'png' if 'kitti' in str(
                    self.info_path).split('/') else 'jpg'
                data_file_list = [
                    str(root_path / 'velodyne' / f"{x['point_cloud']['lidar_idx']}{self.ext}") for x in data]
                # Read Image Path(.png)
                self.image_list = [
                    str(root_path / 'image_2' / f"{x['image']['image_idx']}.{imgext}") for x in data]
                self.image_list.sort()
                # Read Calib Path(.txt)
                self.calib_list = [
                    str(root_path / 'calib' / f"{x['image']['image_idx']}.txt") for x in data]
                self.calib_list.sort()
                # Read Ground Truth
                self.gth = [{
                    'gt_2Dboxes':   x['annos']['bbox'][x['annos']['name'] != 'DontCare'],
                    'gt_dims':      x['annos']['dimensions'][x['annos']['name'] != 'DontCare'],
                    'gt_locs':      x['annos']['location'][x['annos']['name'] != 'DontCare'],
                    'gt_rots':      x['annos']['rotation_y'][x['annos']['name'] != 'DontCare'].reshape((-1, 1)),
                    'gt_labels':    x['annos']['name'][x['annos']['name'] != 'DontCare'],
                } for x in data if 'annos' in x]
        else:
            data_file_list = glob.glob(
                str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
            self.image_list = None
            self.calib_list = None
            self.gth = None
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        print(self.sample_file_list[index])
        if self.ext == '.bin':
            points = np.fromfile(
                self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        if self.image_list is not None:
            image = io.imread(self.image_list[index])
            image = image.astype(np.float32)
            image /= 255.0
            image_shape = np.array(image.shape[:2], dtype=np.int32)

        if self.calib_list is not None:
            calib = Calibration(self.calib_list[index])

        input_dict = {
            'points': points,
            'frame_id': index,
            'images': image,
            'image_shape': image_shape,
            'calib': calib,
            'trans_lidar_to_cam': calib.V2R,
            'trans_cam_to_img': calib.P2
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def get_sensor_data(self, query):
        idx = query['lidar']['idx']
        extension = '.jpg'
        query_keys = query.keys()

        if 'image' in query_keys and self.image_list is not None:
            # Read Image Data
            img = cv2.imread(self.image_list[idx])
            success, img = cv2.imencode(extension, img)
            image_str = img.tobytes()
            query['image']['data'] = image_str
            query["image"]["datatype"] = extension

        if 'image_shape' in query_keys and self.image_list is not None:
            im = Image.open(self.image_list[idx])
            width, height = im.size
            query["image_shape"] = (height, width, len(im.getbands()))

        if 'calib' in query_keys and self.calib_list is not None:
            # Read Project Matrix
            query["calib"] = Calibration(self.calib_list[idx])

        # Read 2D Ground Truth
        if '2D_gth' in query_keys and self.gth is not None:
            query['2D_gth']['gt_2Dboxes'] = self.gth[idx]['gt_2Dboxes']
            query['2D_gth']['gt_labels'] = self.gth[idx]['gt_labels']

        # Read 3D Ground Truth
        if '3D_gth' in query_keys and self.gth is not None:
            pad = np.zeros(self.gth[idx]['gt_rots'].shape)
            xyz_loc = np.concatenate([
                self.gth[idx]['gt_locs'][..., 2:3],
                -self.gth[idx]['gt_locs'][..., 0:1],
                -self.gth[idx]['gt_locs'][..., 1:2] +
                self.gth[idx]['gt_dims'][..., 1:2] * 0.5,
            ], axis=-1)
            # xyz_loc = self.gth[idx]['gt_locs']

            xyz_dim = np.concatenate([
                self.gth[idx]['gt_dims'][..., 2:3],
                self.gth[idx]['gt_dims'][..., 0:1],
                self.gth[idx]['gt_dims'][..., 1:2],
            ], axis=-1)
            # xyz_dim = self.gth[idx]['gt_dims']

            query['3D_gth']['gt_dims'] = xyz_dim
            query['3D_gth']['gt_locs'] = xyz_loc
            query['3D_gth']['gt_rots'] = np.concatenate(
                [pad, pad, -self.gth[idx]['gt_rots']], axis=-1)
            query['3D_gth']['gt_labels'] = self.gth[idx]['gt_labels']

        return query


class GlobalConfig:
    def __init__(self):
        self.root_path = None
        self.config_path = None
        self.dataset = None
        self.model = None
        self.image_indices = None


def boxes3d_to_corners3d_kitti_camera(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, ry] in camera coords, see the definition of ry in KITTI dataset
    :param bottom_center: whether y is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2.,
                         l / 2., l / 2., -l / 2., -l / 2], dtype=np.float32).T
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2.,
                         w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    if bottom_center:
        y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        y_corners[:, 4:8] = - \
            h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        y_corners = np.array([h / 2., h / 2., h / 2., h / 2., -
                             h / 2., -h / 2., -h / 2., -h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(
        ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                         [zeros, ones, zeros],
                         [np.sin(ry), zeros, np.cos(ry)]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :,
                                                      0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate(
        (x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)


def boxes3d_kitti_camera_to_imageboxes(boxes3d, calib, image_shape=None):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    :param calib:
    :return:
        box_2d_preds: (N, 4) [x1, y1, x2, y2]
    """
    corners3d = boxes3d_to_corners3d_kitti_camera(boxes3d)
    pts_img, _ = calib.rect_to_img(corners3d.reshape(-1, 3))
    corners_in_image = pts_img.reshape(-1, 8, 2)

    min_uv = np.min(corners_in_image, axis=1)  # (N, 2)
    max_uv = np.max(corners_in_image, axis=1)  # (N, 2)
    boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
    if image_shape is not None:
        boxes2d_image[:, 0] = np.clip(
            boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 1] = np.clip(
            boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
        boxes2d_image[:, 2] = np.clip(
            boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 3] = np.clip(
            boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

    return boxes2d_image


def boxes3d_lidar_to_kitti_camera(boxes3d_lidar, calib):
    """
    :param boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    :param calib:
    :return:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    """
    boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
    xyz_lidar = boxes3d_lidar_copy[:, 0:3]
    l, w, h = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:,
                                                             4:5], boxes3d_lidar_copy[:, 5:6]
    r = boxes3d_lidar_copy[:, 6:7]

    xyz_lidar[:, 2] -= h.reshape(-1) / 2
    xyz_cam = calib.lidar_to_rect(xyz_lidar)
    # xyz_cam[:, 1] += h.reshape(-1) / 2
    r = -r - np.pi / 2
    return np.concatenate([xyz_cam, l, h, w, r], axis=-1)

# =============================================================================


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin',
                        help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


BACKEND = GlobalConfig()


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
        info_path = Path(instance["info_path"])
        ext_path = '.bin'

        print('config_path', config_path)
        print('cfg', cfg)
        cfg_from_yaml_file(config_path, cfg)
        print(f'[Roger] start to build dataset')
        st_time = time.time()
        print('info_path', info_path, info_path != '.')
        if info_path != Path('.') and info_path != '' and info_path is not None:
            info_path = info_path
        else:
            info_path = None
        demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=data_path, ext=ext_path, logger=logger, info_path=info_path
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
        num_of_features = int(instance["num_of_features"])
        point_cloud_path = instance["point_cloud_path"]
        enable_int16 = instance["enable_int16"]
        response = {"status": "normal"}

        points = np.fromfile(point_cloud_path, dtype=np.float32,
                             count=-1).reshape([-1, num_of_features])
        points = points[:, :3]
        if enable_int16:
            int16_factor = instance["int16_factor"]
            points *= int16_factor
            points = points.astype(np.int16)
        pc_str = base64.b64encode(points.tobytes())
        pc_str = pc_str.decode("utf-8")

        response["num_features"] = 3
        response["pointcloud"] = pc_str
        response = jsonify(results=[response])
        response.headers['Access-Control-Allow-Headers'] = '*'
        return response

    @my_app.route('/api/get_pointcloud', methods=['POST'])      # finish
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
        response = {"status": "normal"}
        print('*'*200)
        print('is_inference', is_inference)
        print('enable_int16', enable_int16)
        print('image_idx', image_idx)

        processing_time = time.time()
        data_dict = BACKEND.dataset[image_idx]
        if is_inference:
            if BACKEND.model is not None:
                infer_sample = BACKEND.dataset.collate_batch([data_dict])
                load_data_to_gpu(infer_sample)
                processing_time = time.time() - processing_time

                inference_time = time.time()
                with torch.no_grad():
                    pred_dicts, _ = BACKEND.model.forward(infer_sample)
                inference_time = time.time() - inference_time
                for key in pred_dicts[0].keys():
                    if isinstance(pred_dicts[0][key], torch.Tensor):
                        pred_dicts[0][key] = pred_dicts[0][key].cpu().numpy()

                if BACKEND.dataset.calib_list is not None:
                    query = {
                        "lidar": {"idx": image_idx},
                        "calib": {},
                        "image_shape": {}
                    }
                    sensor_data = BACKEND.dataset.get_sensor_data(query)

                    pred_boxes_camera = boxes3d_lidar_to_kitti_camera(
                        pred_dicts[0]['pred_boxes'][:, :7],
                        sensor_data['calib'])
                    bboxes = boxes3d_kitti_camera_to_imageboxes(
                        pred_boxes_camera,
                        sensor_data['calib'],
                        image_shape=query['image_shape'])

                    bboxes = bboxes / np.concatenate([
                        query['image_shape'][1],
                        query['image_shape'][0],
                        query['image_shape'][1],
                        query['image_shape'][0]
                    ], axis=None)
                    response["dt_2Dboxes"] = bboxes.tolist()

                rot = pred_dicts[0]['pred_boxes'][:, 6:7]
                rot_pad = np.zeros(rot.shape, rot.dtype)
                rot = np.concatenate([rot_pad, rot_pad, rot], axis=1)
                print('pred_dicts[0]', pred_dicts[0].keys())
                response["dt_locs"] = pred_dicts[0]['pred_boxes'][:, 0:3].tolist()
                # x => 正->向前, 負->向後
                # y => 正->左邊, 負->右邊
                # z => 正->向上, 負->向下
                response["dt_dims"] = pred_dicts[0]['pred_boxes'][:, 3:6].tolist()
                response["dt_rots"] = rot.tolist()
                response["dt_labels"] = pred_dicts[0]['pred_labels'].tolist()
                response["dt_scores"] = pred_dicts[0]['pred_scores'].tolist()
                response["inference_time"] = inference_time
                response["processing_time"] = processing_time

                print('Model is inferencing')
            else:
                print('Model is empty')

        ###################################
        if BACKEND.dataset.info_path is not None and BACKEND.dataset.gth:
            query = {
                "lidar": {"idx": image_idx},
                "3D_gth": {},
                "2D_gth": {},
                "image_shape": {}
            }
            sensor_data = BACKEND.dataset.get_sensor_data(query)
            sensor_data['2D_gth']['gt_2Dboxes'] = sensor_data['2D_gth']['gt_2Dboxes'] / np.concatenate([
                query['image_shape'][1],
                query['image_shape'][0],
                query['image_shape'][1],
                query['image_shape'][0]
            ], axis=None)
        else:
            query = {
                "lidar": {"idx": image_idx},
                "image_shape": {}
            }
            sensor_data = BACKEND.dataset.get_sensor_data(query)

        ####################################
        ###  Point Cloud Quantize data    ##
        ####################################
        points = data_dict['points'][:, :3]
        if enable_int16:
            int16_factor = instance["int16_factor"]
            points *= int16_factor
            points = points.astype(np.int16)
        pc_str = base64.b64encode(points.tobytes())
        pc_str = pc_str.decode("utf-8")
        if BACKEND.dataset.info_path is not None and BACKEND.dataset.gth:
            response["gt_locs"] = sensor_data['3D_gth']['gt_locs'].tolist()
            response["gt_dims"] = sensor_data['3D_gth']['gt_dims'].tolist()
            response["gt_rots"] = sensor_data['3D_gth']['gt_rots'].tolist()
            response["gt_labels"] = sensor_data['3D_gth']['gt_labels'].tolist()
            response["gt_2Dboxes"] = sensor_data['2D_gth']['gt_2Dboxes'].tolist()

        response["status"] = "normal"
        response["num_features"] = 3
        response["pointcloud"] = pc_str
        response["transfer_start"] = time.time()
        response["correction_time"] = 0

        response = jsonify(results=[response])
        response.headers['Access-Control-Allow-Headers'] = '*'
        print("send response with size {}!".format(len(pc_str)))
        return response

    @my_app.route('/api/get_image', methods=['POST'])           # finsih
    def get_image():
        global BACKEND
        instance = request.json
        response = {"status": "normal"}
        # if BACKEND.root_path is None:
        #     return error_response("root path is not set")
        image_idx = instance["image_idx"]
        # idx = BACKEND.image_idxes.index(image_idx)
        query = {
            "lidar": {
                "idx": image_idx
            },
            "image": {}
        }
        sensor_data = BACKEND.dataset.get_sensor_data(query)
        # if "cam" in sensor_data and "data" in sensor_data["cam"] and sensor_data["cam"]["data"] is not None:
        if True:
            print('In get image')
            image_str = sensor_data["image"]["data"]
            response["image_b64"] = base64.b64encode(image_str).decode("utf-8")
            response["image_b64"] = f'data:image/{sensor_data["image"]["datatype"]};base64,' + \
                response["image_b64"]
            print("send an image with size {}!".format(
                len(response["image_b64"])))
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

        model = build_network(model_cfg=cfg.MODEL, num_class=len(
            cfg.CLASS_NAMES), dataset=BACKEND.dataset)
        model.load_params_from_file(
            filename=ckpt_path, logger=logger, to_cpu=True)
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
        print('success fully inference')
        return response

    logger.info("Finish Creating All Response Function")
