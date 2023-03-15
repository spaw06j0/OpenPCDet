var KittiViewer = function (pointCloud, logger, imageCanvas, socket) {
    this.socket = socket;
    this.rootPath = "/mnt/HDD7/meijin/Detection3D/OpenPCDet/data/kitti/training/";
    this.infoPath = "/mnt/HDD7/meijin/Detection3D/OpenPCDet/data/kitti/kitti_infos_val.pkl";
    this.single_pct_path = "";
    this.single_pct_path2 = "";
    this.num_of_features = 4;
    this.backend = "http://127.0.0.1:8888";
    this.checkpointPath = "/mnt/HDD7/meijin/Detection3D/OpenPCDet/output/kitti_models/voxel_rcnn_multi_center/default/ckpt/checkpoint_epoch_80.pth";
    this.datasetClassName = "KittiDataset"
    this.configPath = "/mnt/HDD7/meijin/Detection3D/OpenPCDet/tools/cfgs/kitti_models/voxel_rcnn_multi_center.yaml";
    this.isDrawImage = false;
    this.isDrawBEV = false;
    this.isDrawGT = true;
    this.isInference = false
    this.imageIndexes = [];
    this.imageIndex = 1;

    this.gtCSS3Dboxes = [];            // CSS ground truth for 3D bounding box
    this.gt_2Dboxes = []               // 2D bounding box on image
    this.gt_labels = []                  // Labels       for 3D GroundTruth bbox 
    this.gt_locs = []                    // Location     for 3D GroundTruth bbox
    this.gt_dims = []                    // W,H,L        for 3D GroundTruth bbox
    this.gt_rots = []                    // Angle        for 3D GroundTruth bbox
    this.dtCSS3Dboxes = [];            // CSS detection for 3D bounding box
    this.dt_2Dboxes = [];              // 2D bounding box on image
    this.dt_conf = []                    // Confidence   for 3D Detection bbox
    this.dt_locs = []                    // Location     for 3D Detection bbox
    this.dt_dims = []                    // W,H,L        for 3D Detection bbox
    this.dt_rots = []                    // Angle        for 3D Detection bbox

    this.pointCloud = pointCloud;
    this.maxPoints = 500000; // 500000; 30000
    // this.pointVertices = new Float32Array(this.maxPoints * 3);
    this.gtBoxColor = "#00ff00";
    this.dtBoxColor = "#ff0000";
    this.gtLabelColor = "#7fff00";
    this.dtLabelColor = "#ff7f00";
    this.logger = logger;
    this.imageCanvas = imageCanvas;
    this.image = '';
    this.confidence = 0.25;
    this.enableInt16 = true;
    this.int16Factor = 100;
    this.removeOutside = false;
    this.streaming = false;
};

function pathJoin(parts, sep) {
    let separator = sep || '/';
    let replace = new RegExp(separator + '{1,}', 'g');
    return parts.join(separator).replace(replace, separator);
}

KittiViewer.prototype = {
    readCookies: function () {
        if (CookiesKitti.get("kittiviewer_dataset_cname")) {
            this.datasetClassName = CookiesKitti.get("kittiviewer_dataset_cname");
        }
        if (CookiesKitti.get("kittiviewer_backend")) {
            this.backend = CookiesKitti.get("kittiviewer_backend");
        }
        if (CookiesKitti.get("kittiviewer_rootPath")) {
            this.rootPath = CookiesKitti.get("kittiviewer_rootPath");
        }
        if (CookiesKitti.get("kittiviewer_single_pct_path")) {
            this.single_pct_path = CookiesKitti.get("kittiviewer_single_pct_path");
        }
        if (CookiesKitti.get("kittiviewer_single_pct_path2")) {
            this.single_pct_path2 = CookiesKitti.get("kittiviewer_single_pct_path2");
        }
        if (CookiesKitti.get("kittiviewer_num_of_features")) {
            this.num_of_features = CookiesKitti.get("kittiviewer_num_of_features");
        }
        if (CookiesKitti.get("kittiviewer_checkpointPath")) {
            this.checkpointPath = CookiesKitti.get("kittiviewer_checkpointPath");
        }
        if (CookiesKitti.get("kittiviewer_configPath")) {
            this.configPath = CookiesKitti.get("kittiviewer_configPath");
        }
        if (CookiesKitti.get("kittiviewer_infoPath")) {
            this.infoPath = CookiesKitti.get("kittiviewer_infoPath");
        }
        if (CookiesKitti.get("kittiviewer_confidence")) {
            this.confidence = Number(CookiesKitti.get("kittiviewer_confidence"));
        }
        if (CookiesKitti.get("kittiviewer_drawImage")) {
            this.isDrawImage = Boolean(Number(CookiesKitti.get("kittiviewer_drawImage")))
        }
        if (CookiesKitti.get("kittiviewer_drawBEV")) {
            this.isDrawBEV = Boolean(Number(CookiesKitti.get("kittiviewer_drawBEV")))
        }
        if (CookiesKitti.get("kittiviewer_drawGT")) {
            this.isDrawGT = Boolean(Number(CookiesKitti.get("kittiviewer_drawGT")))
        }
        if (CookiesKitti.get("kittiviewer_isInference")) {
            this.isInference = Boolean(Number(CookiesKitti.get("kittiviewer_isInference")))
        }
    },
    load: function () {
        let self = this;
        let data = {};
        data["root_path"] = this.rootPath;
        data["info_path"] = this.infoPath;
        data['config_path'] = this.configPath;
        data["dataset_class_name"] = this.datasetClassName;
        return $.ajax({
            url: this.addhttp(this.backend) + '/api/readinfo',
            method: 'POST',
            contentType: "application/json",
            data: JSON.stringify(data),
            error: function (jqXHR, exception) {
                self.logger.error("load kitti info fail, please check your backend!");
                console.log("load kitti info fail, please check your backend!");
            },
            success: function (response) {
                let result = response["results"][0];
                self.imageIndexes = [];
                for (var i = 0; i < result["image_indexes"].length; ++i)
                    self.imageIndexes.push(result["image_indexes"][i]);
                self.logger.message("load kitti info success!");
            }
        });
    },
    addhttp: function (url) {
        if (!/^https?:\/\//i.test(url)) {
            url = 'http://' + url;
        }
        return url
    },

    load_pct: function () {
        let self = this;
        let data = {};
        data["point_cloud_path"] = pathJoin([this.single_pct_path, this.single_pct_path2]);
        data["num_of_features"] = this.num_of_features
        data["enable_int16"] = this.enableInt16;
        data["int16_factor"] = this.int16Factor;
        return $.ajax({
            url: this.addhttp(this.backend) + '/api/read_point_cloud',
            method: 'POST',
            contentType: "application/json",
            data: JSON.stringify(data),
            error: function (jqXHR, exception) {
                self.logger.error("load point cloud bin fail!");
                console.log("load point cloud bin fail!");
            },
            success: function (response) {
                self.logger.message("load point cloud bin success!");
                self._draw_response(response)
            }
        });
    },
    buildNet: function () {
        let self = this;
        let data = {};
        data["root_path"] = this.rootPath;
        data["info_path"] = this.infoPath;
        data['config_path'] = this.configPath;
        data["checkpoint_path"] = this.checkpointPath;

        return $.ajax({
            url: this.addhttp(this.backend) + '/api/build_network',
            method: 'POST',
            contentType: "application/json",
            data: JSON.stringify(data),
            error: function (jqXHR, exception) {
                self.logger.error("build kitti det fail!");
                console.log("build kitti det fail!");
            },
            success: function (response) {
                self.logger.message("build kitti det success!");
            }
        });
    },
    //
    inference: function () {
        // let self = this;
        // let data = {"image_idx": self.imageIndex, "remove_outside": self.removeOutside};
        // return $.ajax({
        //     url: this.addhttp(this.backend) + '/api/inference_by_idx',
        //     method: 'POST',
        //     contentType: "application/json",
        //     data: JSON.stringify(data),
        //     error: function (jqXHR, exception) {
        //         self.logger.error("inference fail!");
        //         console.log("inference fail!");
        //     },
        //     success: function (response) {
        //         self._draw_bbox(response)
        //         self.drawImage()
        //     }
        // });
    },
    plot: function () {
        return this._plot(this.imageIndex);
    },
    next: function () {
        this.imageIndex = (this.imageIndex + 1) % this.imageIndexes.length
        return this.plot();
    },
    prev: function () {
        this.imageIndex = (this.imageIndex - 1) % this.imageIndexes.length
        return this.plot();
    },
    clear: function () {
        for (var i = 0; i < this.gtCSS3Dboxes.length; ++i) {
            for (var j = this.gtCSS3Dboxes[i].children.length - 1; j >= 0; j--) {
                this.gtCSS3Dboxes[i].remove(this.gtCSS3Dboxes[i].children[j]);
            }
            scene.remove(this.gtCSS3Dboxes[i]);
            this.gtCSS3Dboxes[i].geometry.dispose();
            this.gtCSS3Dboxes[i].material.dispose();
        }

        for (var i = 0; i < this.dtCSS3Dboxes.length; ++i) {
            for (var j = this.dtCSS3Dboxes[i].children.length - 1; j >= 0; j--) {
                this.dtCSS3Dboxes[i].remove(this.dtCSS3Dboxes[i].children[j]);
            }
            scene.remove(this.dtCSS3Dboxes[i]);
            this.dtCSS3Dboxes[i].geometry.dispose();
            this.dtCSS3Dboxes[i].material.dispose();
        }
        this.gtCSS3Dboxes = [];
        this.dtCSS3Dboxes = [];
    },

    drawImage: function () {
        if (this.image === '') {
            console.log("no image to draw");
            return;
        } else {
            console.log('some image to draw')
        }
        let self = this;
        var image = new Image();
        // image.src = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNk+M9Qz0AEYBxVSF+FAAhKDveksOjmAAAAAElFTkSuQmCC";
        // console.log(response["image_b64"]);
        image.onload = function () {
            let aspect = image.width / image.height;
            let w = self.imageCanvas.width;
            self.imageCanvas.height = w / aspect;
            let h = self.imageCanvas.height;
            let ctx = self.imageCanvas.getContext("2d");
            // ctx.globalAlpha  = 0.5
            ctx.drawImage(image, 0, 0, w, h);

            if (self.isDrawGT) {
                self._add_2d_box_on_scene(ctx, self.gt_2Dboxes, w, h, 'green')
            }
            self._add_2d_box_on_scene(ctx, self.dt_2Dboxes, w, h, 'red', self.dt_conf, self.confidence)
        };
        image.src = this.image;
    },
    saveAsImage: function (renderer) {
    },
    saveFile: function (strData, filename) {
        // var link = document.createElement('a');
        // if (typeof link.download === 'string') {
        //     document.body.appendChild(link); //Firefox requires the link to be in the body
        //     link.download = filename;
        //     link.href = strData;
        //     link.click();
        //     document.body.removeChild(link); //remove the link when done
        // } else {
        //     location.replace(uri);
        // }
    },
    request_streaming: function () {
        this.streaming = True;
        for (idx = 0; idx < 3000; idx++)
            if (this.streaming)
                _plot(idx)
    },
    remove_streaming: function () {
        this.streaming = False;
    },
    _plot: function (image_idx) {
        if (this.imageIndexes.length != 0 && this.imageIndexes.includes(image_idx)) {
            let data = {};
            data["image_idx"] = image_idx;
            data["enable_int16"] = this.enableInt16;
            data["int16_factor"] = this.int16Factor;
            data["remove_outside"] = this.removeOutside;
            data["is_inference"] = this.isInference
            data["send_time"] = Date.now() / 1000.0;
            let self = this;
            let time1 = Date.now() / 1000.0;
            let start = Date.now() / 1000.0;

            send_point_cloud_request = () => {
                let true_of_time = Date.now() / 1000.0
                return new Promise((resolve, reject) => $.ajax({
                    url: this.addhttp(this.backend) + '/api/get_pointcloud',
                    method: 'POST',
                    contentType: "application/json",
                    data: JSON.stringify(data),
                    error: function (jqXHR, exception) {
                        self.logger.error("get point cloud fail!!");
                        reject();
                    },
                    success: (response) => {
                        self._draw_response(response)
                        resolve();
                        return
                    }
                }));
            }
            send_image_request = async () => {
                return new Promise((resolve, reject) => $.ajax({
                    url: this.addhttp(this.backend) + '/api/get_image',
                    method: 'POST',
                    contentType: "application/json",
                    data: JSON.stringify(data),
                    error: function (jqXHR, exception) {
                        self.logger.error("get image fail!!");
                        reject();
                    },
                    success: function (response) {
                        response = response["results"][0];
                        self.image = response["image_b64"];
                        resolve();
                    }
                }));
            }

            const ajax1 = send_point_cloud_request()
            if (this.isDrawImage) {
                const ajax2 = send_image_request()
                Promise.all([ajax1, ajax2]).then(() => {
                    // draw image, bbox
                    if (self.isDrawImage) {
                        console.log('send_point_cloud_request', ' is draw image')
                        self.drawImage();
                    }
                })
            }
        } else {
            if (this.imageIndexes.length == 0) {
                this.logger.error("image indexes isn't load, please click load button!");
            } else {
                this.logger.error("out of range!");
            }
        }
    },
    _draw_response: function (response) {
        // response["results"][0] = {
        //     'dims':             array(num_of_obj, 3), float
        //     'labels':           array(num_of_obj, 3), str
        //     'locs':             array(num_of_obj, 3), float
        //     'num_features':     int, example: either 3 or 4
        //     'pointcloud':       buffer object, approximately 150kb
        //     'rots':             array(num_of_obj, 3), float,
        //     'transfer_start': float, in second. the time right before server send the responese.
        // }
        let self = this;
        self.clear();
        response = response["results"][0];
        let points = [];
        console.log('[Roger] in _draw_response')
        console.log('[Roger] ', Object.keys(response))
        if ("pointcloud" in response) {
            console.log('[Roger] pointcloud in response')
            let points_buf = str2buffer(atob(response["pointcloud"]));
            if (self.enableInt16) {
                points = new Int16Array(points_buf);
            } else {
                points = new Float32Array(points_buf);
            }
            console.log('[Roger] points.length', points.length)
        }

        let numFeatures = response["num_features"];

        if ('image_b64' in response) {
            this.image = response['image_prefix'] + response['image_b64']
        }
        if ('gt_2Dboxes' in response) {
            this.gt_2Dboxes = response['gt_2Dboxes']
        }
        if ('dt_2Dboxes' in response) {
            this.dt_2Dboxes = response['dt_2Dboxes']
        } else {
            this.dt_2Dboxes = []
        }
        if ('gt_dims' in response && 'gt_locs' in response && 'gt_rots' in response && 'gt_labels' in response) {
            this.gt_dims = response['gt_dims']
            this.gt_locs = response['gt_locs']
            this.gt_rots = response['gt_rots']
            this.gt_labels = response['gt_labels']
        }
        if ('dt_dims' in response && 'dt_locs' in response && 'dt_rots' in response && 'dt_scores' in response) {
            this.dt_locs = response["dt_locs"];
            this.dt_dims = response["dt_dims"];
            this.dt_rots = response["dt_rots"];
            this.dt_conf = response['dt_scores'];
            // this.dt_cls = response['dt_cls']
        }
        if (self.isDrawGT) {
            self._add_gth_box_on_scene()
        }



        ///////////////////////////////////////
        // Processing Point Cloud       ///////
        ///////////////////////////////////////
        for (let i = 0; i < Math.min(points.length / numFeatures, self.maxPoints); i++) {
            for (let j = 0; j < numFeatures; ++j) {
                let idx_source = j
                let idx_target = j
                self.pointCloud.geometry.attributes.position.array[i * 3 + idx_source] =
                    points[i * numFeatures + idx_target];
            }
        }
        if (self.enableInt16) {
            for (let i = 0; i < self.pointCloud.geometry.attributes.position.array.length; i++) {
                self.pointCloud.geometry.attributes.position.array[i] /= self.int16Factor;
            }
        }
        self.pointCloud.geometry.setDrawRange(0, Math.min(points.length / numFeatures, self.maxPoints));
        self.pointCloud.geometry.attributes.position.needsUpdate = true;
        self.pointCloud.geometry.computeBoundingSphere();
        ///////////////////////////////////////
        // Refresh Bboxes                   ///
        ///////////////////////////////////////
        this._remove_boxes_on_scene(this.dtCSS3Dboxes)
        this._add_filtered_box_on_scene()
    },
    //
    _draw_bbox: function (response) {
        // response["results"][0] = {   
        //     'dt_dims':   array(num_of_box, 3),
        //     'dt_labels': array(num_of_box),
        //     'dt_locs':   array(num_of_box, 3),
        //     'dt_rots':   array(num_of_box, 3),
        //     'dt_scores': array(num_of_box,),
        //     'inference_time': float, in second.
        //     'transfer_start': float, in second. the time right before server send the responese.
        // }
        // console.log('[Roger] Calling _draw_bbox')
        // response = response["results"][0];

        // if ("dt_2Dboxes" in response) {
        //     console.log('[Roger] dt_2Dboxes is in response!!!!!!!!!!!!!!!!!')
        //     this.dt_2Dboxes = response["dt_2Dboxes"];
        // } else {
        //     console.log('[Roger] dt_2Dboxes not in response')
        //     this.dt_2Dboxes = []
        // }
        // this.dt_locs = response["dt_locs"];
        // this.dt_dims = response["dt_dims"];
        // this.dt_rots = response["dt_rots"];
        // this.dt_conf = response['dt_scores']

        // console.log('[Roger] ', response["dt_locs"].length)

        // this._remove_boxes_on_scene(this.dtCSS3Dboxes)
        // this._add_filtered_box_on_scene()
    },
    _remove_boxes_on_scene: function (boxes) {
        console.log('[Roger] Calling _remove_boxes_on_scene')
        // [Roger] remove all the bounding box in the scene
        for (var i = 0; i < boxes.length; ++i) {
            // [Roger] Guess: the label are child node of dtCSS3Dboxes[i], so we need to remove them from dtCSS3Dboxes[i] 
            // if we really need to change to whole new boxes
            for (var j = boxes[i].children.length - 1; j >= 0; j--) {
                boxes[i].remove(boxes[i].children[j]);
            }
            scene.remove(boxes[i]);
            // [Roger] Guess: destroy bounding box data structure
            boxes[i].geometry.dispose();
            boxes[i].material.dispose();
        }
    },
    _add_filtered_box_on_scene: function () {
        console.log('[Roger] Calling _add_filtered_box_on_scene confidence', this.confidence)
        let label_with_score = [];
        for (var i = 0; i < this.dt_locs.length; ++i) {
            label_with_score.push("score=" + this.dt_conf[i].toFixed(2).toString());
        }
        this.dtCSS3Dboxes = boxEdgeWithLabel(this.dt_dims, this.dt_locs, this.dt_rots, 2, this.dtBoxColor,
            label_with_score, this.dtLabelColor);
        for (var i = 0; i < this.dtCSS3Dboxes.length; ++i) {
            // do not render bounding box with low confidence
            if (this.confidence > this.dt_conf[i]) continue
            scene.add(this.dtCSS3Dboxes[i]);
        }
    },
    _add_gth_box_on_scene: function () {
        console.log('[Roger] Calling _add_gth_box_on_scene')
        this.gtCSS3Dboxes = boxEdgeWithLabel(this.gt_dims, this.gt_locs, this.gt_rots, 2, this.gtBoxColor,
            this.gt_labels, this.gtLabelColor);
        for (var i = 0; i < this.gtCSS3Dboxes.length; ++i) {
            // do not render bounding box with low confidence
            if (this.confidence > this.dt_conf[i]) continue
            scene.add(this.gtCSS3Dboxes[i]);
        }
    },
    _add_2d_box_on_scene: function (ctx, boxes, w, h, color, confs = null, threshold = null) {
        console.log('[Roger] Calling _add_2d_box_on_scene')
        for (let i = 0; i < boxes.length; ++i) {
            if (confs !== null && confs[i] < threshold) continue
            ctx.beginPath();
            x1 = boxes[i][0] * w;
            y1 = boxes[i][1] * h;
            x2 = boxes[i][2] * w;
            y2 = boxes[i][3] * h;

            ctx.rect(x1, y1, x2 - x1, y2 - y1);
            ctx.lineWidth = 1;
            ctx.strokeStyle = color;
            ctx.stroke();
        }
    }

}
