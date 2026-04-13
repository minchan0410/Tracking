#!/usr/bin/env python3
"""FCOS3D monocular 3D detection ROS node.

Subscribes to cam_front/raw (sensor_msgs/Image), runs FCOS3D inference,
publishes RViz markers and logs FPS / latency.
"""

import json
import os
import sys
import tempfile
import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HOME = os.path.expanduser("~")
CONFIG = os.path.join(
    _HOME,
    "Tracking/checkpoints/"
    "fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py",
)
CHECKPOINT = os.path.join(
    _HOME,
    "Tracking/checkpoints/"
    "fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_"
    "20210717_095645-8d806dc2.pth",
)

# NuScenes CAM_FRONT camera intrinsic (3x3)
CAM_FRONT_K = [
    [1266.417203046554, 0.0, 816.2679887771999],
    [0.0, 1266.417203046554, 491.50706579294757],
    [0.0, 0.0, 1.0],
]

SCORE_THRESH = 0.3
DEVICE = "cuda:0"
FPS_LOG_EVERY = 10  # log FPS every N frames


# ---------------------------------------------------------------------------
# mmdet3d inference helpers
# ---------------------------------------------------------------------------

def _load_mmdet3d():
    try:
        from mmdet3d.apis import init_model, inference_mono_3d_detector  # noqa: F401
        return init_model, inference_mono_3d_detector
    except ImportError as exc:
        rospy.logfatal(f"Cannot import mmdet3d: {exc}")
        sys.exit(1)


def _make_ann_file(img_path: str, k3x3: list) -> str:
    """Write a minimal mmdet3d mono3D annotation JSON and return its path."""
    cam2img = [k3x3[0] + [0.0], k3x3[1] + [0.0], k3x3[2] + [0.0], [0.0, 0.0, 0.0, 1.0]]
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    payload = {
        "data_list": [
            {
                "sample_idx": 0,
                "images": {
                    "CAM_FRONT": {
                        "img_path": img_path,
                        "cam2img": cam2img,
                        "lidar2cam": identity,
                        "lidar2img": cam2img,
                    }
                },
            }
        ]
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)
    return path


# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------

class Fcos3dNode:
    def __init__(self):
        rospy.init_node("fcos3d_node")

        init_model, self._infer = _load_mmdet3d()

        # Temp files reused each frame to avoid repeated disk alloc.
        self._tmp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
        self._ann_file = _make_ann_file(self._tmp_img, CAM_FRONT_K)

        rospy.loginfo("Loading FCOS3D model (this takes ~10 s) ...")
        self._model = init_model(CONFIG, CHECKPOINT, device=DEVICE)
        rospy.loginfo("FCOS3D model ready.")

        self._bridge = CvBridge()
        self._marker_pub = rospy.Publisher(
            "fcos3d/markers", MarkerArray, queue_size=1
        )
        self._frame = 0
        self._fps_t0: float | None = None

        self._sub = rospy.Subscriber(
            "cam_front/raw", Image, self._cb, queue_size=1, buff_size=2 ** 24
        )
        rospy.loginfo("Subscribed to cam_front/raw")

    # ------------------------------------------------------------------

    def _cb(self, msg: Image):
        t0 = time.perf_counter()

        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imwrite(self._tmp_img, img)

        try:
            results = self._infer(
                self._model, self._tmp_img, self._ann_file, cam_type="CAM_FRONT"
            )
            # inference_mono_3d_detector may return a list or a single sample
            sample = results[0] if isinstance(results, (list, tuple)) else results
            pred = getattr(sample, "pred_instances_3d", None)
        except Exception as exc:
            rospy.logwarn_throttle(5.0, f"Inference error: {exc}")
            return

        elapsed = time.perf_counter() - t0
        self._frame += 1

        # ------ FPS logging ------
        if self._fps_t0 is None:
            self._fps_t0 = time.perf_counter()
        if self._frame % FPS_LOG_EVERY == 0:
            fps = FPS_LOG_EVERY / (time.perf_counter() - self._fps_t0)
            n = len(pred.scores_3d) if pred is not None else 0
            rospy.loginfo(
                f"[FCOS3D] frame={self._frame:5d}  "
                f"fps={fps:5.1f}  "
                f"latency={elapsed * 1000:6.1f} ms  "
                f"dets={n}"
            )
            self._fps_t0 = time.perf_counter()

        # ------ Publish markers ------
        if pred is not None:
            self._publish_markers(pred, msg.header)

    # ------------------------------------------------------------------

    def _publish_markers(self, pred, header):
        import torch  # already imported by mmdet3d; just reference it locally

        arr = MarkerArray()
        scores = pred.scores_3d.cpu().numpy() if hasattr(pred.scores_3d, "cpu") else np.array(pred.scores_3d)
        boxes = pred.bboxes_3d  # CameraInstance3DBoxes

        tensor = boxes.tensor.cpu().numpy()  # (N, 9): x y z l h w rot vx vy

        for i, (row, score) in enumerate(zip(tensor, scores)):
            if float(score) < SCORE_THRESH:
                continue

            x, y, z, l, h, w = row[0], row[1], row[2], row[3], row[4], row[5]

            m = Marker()
            m.header = header
            m.ns = "fcos3d"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = float(z)
            m.pose.orientation.w = 1.0
            m.scale.x = float(l) if l > 0.1 else 0.1
            m.scale.y = float(w) if w > 0.1 else 0.1
            m.scale.z = float(h) if h > 0.1 else 0.1
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 0.4
            m.lifetime = rospy.Duration(0.5)
            arr.markers.append(m)

        self._marker_pub.publish(arr)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    node = Fcos3dNode()
    rospy.spin()
