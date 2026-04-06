#!/usr/bin/env python
"""Run FCOS3D on nuScenes mini surround cameras and save ego-frame detections.

If FCOS3D is unavailable, this script falls back to simulated detections
created from GT with position noise and distance-based confidence decay.
"""

import argparse
import glob
import json
import math
import os
import pickle
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from shapely.geometry import Polygon

CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

DEFAULT_CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

CLASS_ALIASES = {
    "ped": "pedestrian",
    "person": "pedestrian",
    "people": "pedestrian",
}

EDGE_IDS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FCOS3D surround inference on nuScenes mini -> ego-frame detections"
    )
    parser.add_argument("--dataroot", default="./data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument(
        "--config",
        default=(
            "./checkpoints/"
            "fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py"
        ),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--out-dir", default="./outputs/detections")
    parser.add_argument(
        "--stream-dir",
        default="./outputs/fcos_stream",
        help="Directory to save per-camera FCOS overlay images for visualization streaming.",
    )
    parser.add_argument(
        "--no-save-fcos-stream",
        action="store_true",
        help="Disable saving per-camera FCOS overlay images.",
    )
    parser.add_argument("--score-thresh", type=float, default=0.1)
    parser.add_argument(
        "--keep-classes",
        nargs="+",
        default=None,
        help=(
            "Keep only selected classes (e.g. --keep-classes car truck pedestrian). "
            "Alias: ped -> pedestrian."
        ),
    )
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--temp-ann-dir",
        default="./outputs/tmp_infos",
        help="Temporary directory for per-image mono3D annotation json.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scene-start",
        type=int,
        default=0,
        help="Start scene index (0-based).",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="Number of scenes to process. 0 means all scenes from scene-start.",
    )
    parser.add_argument(
        "--strict-fcos3d",
        action="store_true",
        help="Fail immediately if FCOS3D is unavailable. Do not fallback.",
    )
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="Ignore FCOS3D and always generate simulated detections.",
    )
    return parser.parse_args()


def find_checkpoint(args: argparse.Namespace) -> Optional[str]:
    if args.checkpoint:
        return args.checkpoint

    candidates = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pth")))
    return candidates[0] if candidates else None


def normalize_keep_classes(classes: Optional[List[str]]) -> Optional[set]:
    if not classes:
        return None
    normalized = set()
    for name in classes:
        key = str(name).strip().lower()
        if not key:
            continue
        normalized.add(CLASS_ALIASES.get(key, key))
    return normalized if normalized else None


def to_numpy(x: Any) -> np.ndarray:
    if x is None:
        return np.array([])
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.array(x)


def build_detector(
    config_path: str, checkpoint_path: str, device: str
) -> Dict[str, Any]:
    from mmdet3d import apis as mmdet3d_apis  # pylint: disable=import-error
    from mmdet3d.apis import init_model  # pylint: disable=import-error

    model = init_model(config_path, checkpoint_path, device=device)

    # Prefer mono infer API for FCOS3D/mono3D models.
    # Fallback to generic inference only when mono API is unavailable.
    infer_fns = []
    if hasattr(mmdet3d_apis, "inference_mono_3d_detector"):
        infer_fns = [mmdet3d_apis.inference_mono_3d_detector]
    elif hasattr(mmdet3d_apis, "inference_detector"):
        infer_fns = [mmdet3d_apis.inference_detector]

    if not infer_fns:
        raise RuntimeError(
            "No supported inference API found in mmdet3d.apis "
            "(expected inference_mono_3d_detector or inference_detector)."
        )

    dataset_meta = getattr(model, "dataset_meta", {}) or {}
    class_names = (
        dataset_meta.get("classes")
        or dataset_meta.get("CLASSES")
        or getattr(model, "CLASSES", None)
        or DEFAULT_CLASSES
    )

    return {
        "model": model,
        "infer_fns": infer_fns,
        "class_names": list(class_names),
    }


def extract_pred_instances(pred_instances: Any) -> Optional[Dict[str, np.ndarray]]:
    if pred_instances is None:
        return None

    bboxes_3d = getattr(pred_instances, "bboxes_3d", None)
    scores_3d = getattr(pred_instances, "scores_3d", None)
    labels_3d = getattr(pred_instances, "labels_3d", None)

    if bboxes_3d is None and isinstance(pred_instances, dict):
        bboxes_3d = pred_instances.get("bboxes_3d")
        scores_3d = pred_instances.get("scores_3d")
        labels_3d = pred_instances.get("labels_3d")

    if bboxes_3d is None:
        return None

    boxes_np = to_numpy(getattr(bboxes_3d, "tensor", bboxes_3d))
    scores_np = to_numpy(scores_3d)
    labels_np = to_numpy(labels_3d).astype(np.int64) if labels_3d is not None else None

    corners_np = None
    if hasattr(bboxes_3d, "corners"):
        try:
            corners_np = to_numpy(bboxes_3d.corners)
        except Exception:
            corners_np = None

    if scores_np.size == 0:
        scores_np = np.zeros((boxes_np.shape[0],), dtype=np.float32)
    if labels_np is None or labels_np.size == 0:
        labels_np = np.zeros((boxes_np.shape[0],), dtype=np.int64)

    return {
        "boxes": boxes_np,
        "scores": scores_np,
        "labels": labels_np,
        "corners": corners_np,
    }


def normalize_inference_output(output: Any) -> Optional[Dict[str, np.ndarray]]:
    candidates = []

    if isinstance(output, (list, tuple)):
        candidates.extend(list(output))
    else:
        candidates.append(output)

    for candidate in candidates:
        if hasattr(candidate, "pred_instances_3d"):
            parsed = extract_pred_instances(candidate.pred_instances_3d)
            if parsed is not None:
                return parsed

        if isinstance(candidate, dict):
            if "pred_instances_3d" in candidate:
                parsed = extract_pred_instances(candidate["pred_instances_3d"])
                if parsed is not None:
                    return parsed
            parsed = extract_pred_instances(candidate)
            if parsed is not None:
                return parsed

    return None


def run_single_image_inference(
    model: Any,
    infer_fns: List[Any],
    image_path: str,
    ann_file: Optional[str] = None,
    cam_type: str = "CAM_FRONT",
) -> Dict[str, np.ndarray]:
    last_error = None

    for infer_fn in infer_fns:
        try:
            if infer_fn.__name__ == "inference_mono_3d_detector":
                if ann_file is None:
                    raise RuntimeError(
                        "inference_mono_3d_detector requires ann_file, but got None."
                    )
                output = infer_fn(model, image_path, ann_file, cam_type=cam_type)
            else:
                output = infer_fn(model, image_path)
            parsed = normalize_inference_output(output)
            if parsed is not None:
                return parsed
        except Exception as exc:  # pragma: no cover - runtime API mismatch fallback
            last_error = exc

    raise RuntimeError(
        f"All inference methods failed for image: {image_path}. Last error: {last_error}"
    )


def warmup_detector(
    nusc: NuScenes, detector_bundle: Dict[str, Any], temp_ann_path: str
) -> Optional[str]:
    for scene in nusc.scene:
        sample_token = scene["first_sample_token"]
        if not sample_token:
            continue
        sample = nusc.get("sample", sample_token)
        for cam_name in CAMERAS:
            sd_token = sample["data"].get(cam_name)
            if sd_token is None:
                continue
            sd = nusc.get("sample_data", sd_token)
            image_path = os.path.join(nusc.dataroot, sd["filename"])
            write_mono3d_ann_file(
                nusc=nusc,
                sample=sample,
                cam_name=cam_name,
                image_path=image_path,
                out_path=temp_ann_path,
            )
            try:
                run_single_image_inference(
                    detector_bundle["model"],
                    detector_bundle["infer_fns"],
                    image_path,
                    ann_file=temp_ann_path,
                    cam_type=cam_name,
                )
                return None
            except Exception as exc:
                return str(exc)
    return "No camera sample was found for warmup."


def principal_axis_yaw(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 2:
        return 0.0
    centered = points_xy - points_xy.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    axis = eig_vecs[:, np.argmax(eig_vals)]
    return float(math.atan2(axis[1], axis[0]))


def camera_detections_to_ego(
    dets: Dict[str, np.ndarray],
    calibrated_sensor: Dict[str, Any],
    class_names: List[str],
    score_thresh: float,
    camera_name: str,
) -> List[Dict[str, Any]]:
    # FCOS3D (CameraInstance3DBoxes) uses camera box size order:
    # (x_size, y_size, z_size) with yaw around camera y-axis.
    # In practice for nuScenes mono3d, x_size behaves as vehicle length and
    # z_size behaves as width after ego conversion, so we map as:
    # length <- dims[0], width <- dims[2], height <- dims[1].
    rotation = Quaternion(calibrated_sensor["rotation"]).rotation_matrix
    translation = np.asarray(calibrated_sensor["translation"], dtype=np.float32)

    boxes = dets["boxes"]
    scores = dets["scores"]
    labels = dets["labels"]
    corners = dets.get("corners")

    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)

    converted = []
    for idx, box in enumerate(boxes):
        if box.shape[0] < 7:
            continue

        score = float(scores[idx]) if idx < len(scores) else 0.0
        if score <= score_thresh:
            continue

        label = int(labels[idx]) if idx < len(labels) else -1
        cls_name = class_names[label] if 0 <= label < len(class_names) else f"class_{label}"

        center_sensor = box[:3].astype(np.float32)
        dims = np.abs(box[3:6].astype(np.float32))
        length = float(dims[0])
        height = float(dims[1])
        width = float(dims[2])

        if corners is not None and idx < len(corners):
            corners_sensor = np.asarray(corners[idx], dtype=np.float32)
            corners_ego = (rotation @ corners_sensor.T).T + translation
            center_ego = corners_ego.mean(axis=0)
            yaw_ego = principal_axis_yaw(corners_ego[:, :2])
        else:
            # Camera boxes are commonly encoded with bottom-center origin.
            # Shift to geometric center before converting to ego for consistent
            # box reconstruction in BEV and projection.
            center_sensor_geom = center_sensor + np.array(
                [0.0, -0.5 * height, 0.0], dtype=np.float32
            )
            center_ego = rotation @ center_sensor_geom + translation

            yaw_sensor = float(box[6])
            # Camera yaw is around y-axis (down). Heading in camera x-z plane:
            # yaw=0 -> +x, yaw=-pi/2 -> +z.
            heading_sensor = np.array(
                [math.cos(yaw_sensor), 0.0, -math.sin(yaw_sensor)],
                dtype=np.float32,
            )
            heading_ego = rotation @ heading_sensor
            yaw_ego = float(math.atan2(heading_ego[1], heading_ego[0]))

        converted.append(
            {
                "bbox_3d": [
                    float(center_ego[0]),
                    float(center_ego[1]),
                    float(center_ego[2]),
                    width,
                    height,
                    length,
                    yaw_ego,
                ],
                "confidence": score,
                "class": cls_name,
                "camera": camera_name,
            }
        )

    return converted


def conf_to_blue(conf: float) -> tuple:
    conf = float(np.clip(conf, 0.0, 1.0))
    fade = int(round(190 - 160 * conf))
    fade = int(np.clip(fade, 20, 190))
    return (255, fade, fade)  # BGR


def stream_overlay_image_path(
    stream_dir: str, scene_idx: int, sample_token: str, cam_name: str
) -> str:
    return os.path.join(
        stream_dir,
        f"scene_{scene_idx:04d}",
        sample_token,
        f"{cam_name}.jpg",
    )


def project_camera_points(points_cam: np.ndarray, cam_intrinsic: np.ndarray) -> tuple:
    uvw = points_cam @ cam_intrinsic.T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-6)
    z = points_cam[:, 2]
    return uv, z


def draw_projected_camera_box(
    image: np.ndarray,
    corners_cam: np.ndarray,
    cam_intrinsic: np.ndarray,
    color: tuple,
    label: Optional[str] = None,
) -> bool:
    corners_cam = np.asarray(corners_cam, dtype=np.float32)
    if corners_cam.shape != (8, 3):
        return False

    uv, z = project_camera_points(corners_cam, cam_intrinsic)
    if float(np.min(z)) <= 0.1:
        return False

    pts = uv.astype(np.int32)
    h, w = image.shape[:2]
    if np.all((pts[:, 0] < 0) | (pts[:, 0] >= w) | (pts[:, 1] < 0) | (pts[:, 1] >= h)):
        return False

    for i0, i1 in EDGE_IDS:
        p0 = (int(pts[i0, 0]), int(pts[i0, 1]))
        p1 = (int(pts[i1, 0]), int(pts[i1, 1]))
        cv2.line(image, p0, p1, color, 2)

    if label:
        anchor = (int(pts[0, 0]), int(pts[0, 1]))
        cv2.putText(
            image,
            label,
            (anchor[0] + 3, anchor[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return True


def save_fcos_stream_image(
    image_path: str,
    out_path: str,
    cam_name: str,
    camera_intrinsic: List[List[float]],
    dets: Dict[str, np.ndarray],
    class_names: List[str],
    score_thresh: float,
    keep_classes: Optional[set],
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        return

    boxes = dets.get("boxes", np.array([]))
    if isinstance(boxes, np.ndarray) and boxes.ndim == 1 and boxes.size > 0:
        boxes = boxes.reshape(1, -1)

    num_boxes = int(boxes.shape[0]) if isinstance(boxes, np.ndarray) and boxes.ndim == 2 else 0
    scores = dets.get("scores", np.zeros((num_boxes,), dtype=np.float32))
    labels = dets.get("labels", np.zeros((num_boxes,), dtype=np.int64))
    corners = dets.get("corners")
    cam_intrinsic = np.asarray(camera_intrinsic, dtype=np.float32)

    drawn = 0
    for idx in range(num_boxes):
        score = float(scores[idx]) if idx < len(scores) else 0.0
        if score <= score_thresh:
            continue

        label_idx = int(labels[idx]) if idx < len(labels) else -1
        cls_name = (
            class_names[label_idx]
            if 0 <= label_idx < len(class_names)
            else f"class_{label_idx}"
        )
        cls_key = str(cls_name).lower()
        if keep_classes is not None and cls_key not in keep_classes:
            continue

        if corners is None or idx >= len(corners):
            continue

        box_corners = np.asarray(corners[idx], dtype=np.float32)
        ok = draw_projected_camera_box(
            image=image,
            corners_cam=box_corners,
            cam_intrinsic=cam_intrinsic,
            color=conf_to_blue(score),
            label=f"{cls_name}:{score:.2f}",
        )
        if ok:
            drawn += 1

    cv2.rectangle(image, (8, 8), (640, 46), (20, 20, 20), -1)
    cv2.putText(
        image,
        f"{cam_name} FCOS3D  boxes={drawn}",
        (16, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, image)


def bbox_bev_polygon(bbox_3d: List[float]) -> Polygon:
    x, y, _, w, _, l, yaw = bbox_3d

    dx = l * 0.5
    dy = w * 0.5

    corners_local = np.array(
        [[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]], dtype=np.float32
    )
    rot = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float32,
    )
    corners = corners_local @ rot.T + np.array([x, y], dtype=np.float32)

    poly = Polygon(corners)
    return poly.buffer(0) if not poly.is_valid else poly


def transform_matrix(rotation: List[float], translation: List[float]) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = Quaternion(rotation).rotation_matrix.astype(np.float32)
    mat[:3, 3] = np.asarray(translation, dtype=np.float32)
    return mat


def write_mono3d_ann_file(
    nusc: NuScenes,
    sample: Dict[str, Any],
    cam_name: str,
    image_path: str,
    out_path: str,
) -> None:
    cam_sd = nusc.get("sample_data", sample["data"][cam_name])
    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    cam_ep = nusc.get("ego_pose", cam_sd["ego_pose_token"])

    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    lidar_ep = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

    lidar_sensor2ego = transform_matrix(lidar_cs["rotation"], lidar_cs["translation"])
    lidar_ego2global = transform_matrix(lidar_ep["rotation"], lidar_ep["translation"])
    cam_sensor2ego = transform_matrix(cam_cs["rotation"], cam_cs["translation"])
    cam_ego2global = transform_matrix(cam_ep["rotation"], cam_ep["translation"])

    lidar2cam = (
        np.linalg.inv(cam_sensor2ego)
        @ np.linalg.inv(cam_ego2global)
        @ lidar_ego2global
        @ lidar_sensor2ego
    ).astype(np.float32)

    cam2img = np.eye(4, dtype=np.float32)
    cam2img[:3, :3] = np.asarray(cam_cs["camera_intrinsic"], dtype=np.float32)
    lidar2img = (cam2img @ lidar2cam).astype(np.float32)

    info = {
        "data_list": [
            {
                "sample_idx": sample["token"],
                "images": {
                    cam_name: {
                        "img_path": image_path,
                        "cam2img": cam2img.tolist(),
                        "lidar2cam": lidar2cam.tolist(),
                        "lidar2img": lidar2img.tolist(),
                    }
                },
            }
        ]
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f)


def bev_iou(poly_a: Polygon, poly_b: Polygon) -> float:
    if not poly_a.is_valid or not poly_b.is_valid:
        return 0.0
    inter = poly_a.intersection(poly_b).area
    if inter <= 0.0:
        return 0.0
    union = poly_a.area + poly_b.area - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def rotated_nms_ego(
    detections: List[Dict[str, Any]], iou_thresh: float
) -> List[Dict[str, Any]]:
    if not detections:
        return []

    ordered = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    for det in ordered:
        det["_bev_poly"] = bbox_bev_polygon(det["bbox_3d"])

    kept: List[Dict[str, Any]] = []
    for det in ordered:
        suppress = False
        for existing in kept:
            if det["class"] != existing["class"]:
                continue
            if bev_iou(det["_bev_poly"], existing["_bev_poly"]) > iou_thresh:
                suppress = True
                break
        if not suppress:
            kept.append(det)

    for det in kept:
        det.pop("_bev_poly", None)

    return kept


def get_sample_ego_pose(nusc: NuScenes, sample: Dict[str, Any]) -> Dict[str, Any]:
    if "LIDAR_TOP" in sample["data"]:
        ref_sd_token = sample["data"]["LIDAR_TOP"]
    else:
        ref_sd_token = next(iter(sample["data"].values()))

    ref_sd = nusc.get("sample_data", ref_sd_token)
    return nusc.get("ego_pose", ref_sd["ego_pose_token"])


def global_box_to_ego(
    translation: List[float],
    size: List[float],
    rotation: List[float],
    ego_pose: Dict[str, Any],
) -> Box:
    box = Box(
        center=np.asarray(translation, dtype=np.float32),
        size=np.asarray(size, dtype=np.float32),
        orientation=Quaternion(rotation),
    )
    box.translate(-np.asarray(ego_pose["translation"], dtype=np.float32))
    box.rotate(Quaternion(ego_pose["rotation"]).inverse)
    return box


def simulated_detections_from_gt(
    nusc: NuScenes,
    sample: Dict[str, Any],
    ego_pose: Dict[str, Any],
    rng: np.random.Generator,
    score_thresh: float,
) -> List[Dict[str, Any]]:
    simulated = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        cls_name = category_to_detection_name(ann["category_name"])
        if cls_name is None:
            continue

        box_ego = global_box_to_ego(
            ann["translation"], ann["size"], ann["rotation"], ego_pose
        )

        x, y, z = box_ego.center
        x += float(rng.normal(0.0, 0.5))
        y += float(rng.normal(0.0, 0.5))
        z += float(rng.normal(0.0, 0.2))

        width, length, height = box_ego.wlh
        yaw = float(box_ego.orientation.yaw_pitch_roll[0])

        distance = math.hypot(x, y)
        confidence = math.exp(-distance / 40.0) * float(rng.uniform(0.75, 1.0))
        confidence = float(np.clip(confidence, 0.01, 0.99))

        if confidence <= score_thresh:
            continue

        simulated.append(
            {
                "bbox_3d": [
                    float(x),
                    float(y),
                    float(z),
                    float(width),
                    float(height),
                    float(length),
                    yaw,
                ],
                "confidence": confidence,
                "class": cls_name,
                "camera": "SIM_FALLBACK",
            }
        )

    return simulated


def scene_output_path(out_dir: str, scene_idx: int) -> str:
    return os.path.join(out_dir, f"scene_{scene_idx:04d}.pkl")


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.temp_ann_dir, exist_ok=True)
    temp_ann_path = os.path.join(args.temp_ann_dir, "mono3d_single.json")
    keep_classes = normalize_keep_classes(args.keep_classes)
    if keep_classes is not None:
        print(f"[INFO] Class filter enabled: {sorted(keep_classes)}")

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    detector_bundle = None
    if not args.force_fallback:
        ckpt_path = find_checkpoint(args)
        if ckpt_path is None:
            msg = "Checkpoint not found."
            if args.strict_fcos3d:
                raise RuntimeError(msg)
            print(f"[WARN] {msg} Switching to fallback simulation.")
        elif not os.path.isfile(args.config):
            msg = "Config file not found."
            if args.strict_fcos3d:
                raise RuntimeError(msg)
            print(f"[WARN] {msg} Switching to fallback simulation.")
        else:
            try:
                detector_bundle = build_detector(args.config, ckpt_path, args.device)
                warmup_error = warmup_detector(nusc, detector_bundle, temp_ann_path)
                if warmup_error is None:
                    print(
                        "[INFO] FCOS3D loaded.",
                        f"config={args.config}",
                        f"checkpoint={ckpt_path}",
                    )
                else:
                    msg = f"FCOS3D warmup failed: {warmup_error}"
                    if args.strict_fcos3d:
                        raise RuntimeError(msg)
                    detector_bundle = None
                    print("[WARN] FCOS3D warmup failed. Switching to fallback simulation.")
                    print(f"[WARN] reason: {warmup_error}")
            except Exception as exc:
                msg = f"FCOS3D init failed: {exc}"
                if args.strict_fcos3d:
                    raise RuntimeError(msg) from exc
                print("[WARN] FCOS3D init failed. Switching to fallback simulation.")
                print(f"[WARN] reason: {exc}")

    use_fallback = detector_bundle is None
    if use_fallback and args.strict_fcos3d:
        raise RuntimeError("strict-fcos3d enabled, but detector is unavailable.")
    if use_fallback:
        print("[INFO] Using GT-based simulated detections fallback mode.")

    save_fcos_stream = (not args.no_save_fcos_stream) and (not use_fallback)
    if save_fcos_stream:
        os.makedirs(args.stream_dir, exist_ok=True)
        print(f"[INFO] Saving FCOS camera overlays to: {args.stream_dir}")

    rng = np.random.default_rng(args.seed)

    scene_start = max(args.scene_start, 0)
    if scene_start >= len(nusc.scene):
        raise ValueError(
            f"--scene-start={scene_start} is out of range for {len(nusc.scene)} scenes."
        )

    if args.max_scenes and args.max_scenes > 0:
        scene_end = min(len(nusc.scene), scene_start + args.max_scenes)
    else:
        scene_end = len(nusc.scene)

    selected_scenes = list(enumerate(nusc.scene))[scene_start:scene_end]

    for scene_idx, scene in selected_scenes:
        scene_name = scene.get("name", f"scene_{scene_idx:04d}")
        progress_idx = scene_idx - scene_start + 1
        progress_total = len(selected_scenes)
        print(f"[INFO] Processing {scene_name} ({progress_idx}/{progress_total})")

        sample_results: Dict[str, List[Dict[str, Any]]] = {}

        sample_token = scene["first_sample_token"]
        frame_idx = 0
        while sample_token:
            sample = nusc.get("sample", sample_token)

            if use_fallback:
                ego_pose = get_sample_ego_pose(nusc, sample)
                merged = simulated_detections_from_gt(
                    nusc=nusc,
                    sample=sample,
                    ego_pose=ego_pose,
                    rng=rng,
                    score_thresh=args.score_thresh,
                )
            else:
                merged = []
                for cam_name in CAMERAS:
                    sd_token = sample["data"].get(cam_name)
                    if sd_token is None:
                        continue

                    sd = nusc.get("sample_data", sd_token)
                    image_path = os.path.join(nusc.dataroot, sd["filename"])
                    calibrated_sensor = nusc.get(
                        "calibrated_sensor", sd["calibrated_sensor_token"]
                    )
                    write_mono3d_ann_file(
                        nusc=nusc,
                        sample=sample,
                        cam_name=cam_name,
                        image_path=image_path,
                        out_path=temp_ann_path,
                    )

                    try:
                        dets_raw = run_single_image_inference(
                            detector_bundle["model"],
                            detector_bundle["infer_fns"],
                            image_path,
                            ann_file=temp_ann_path,
                            cam_type=cam_name,
                        )
                    except Exception as exc:
                        print(
                            f"[WARN] Inference failed for {scene_name} frame {frame_idx} "
                            f"camera {cam_name}: {exc}"
                        )
                        continue

                    if save_fcos_stream:
                        stream_img_path = stream_overlay_image_path(
                            stream_dir=args.stream_dir,
                            scene_idx=scene_idx,
                            sample_token=sample_token,
                            cam_name=cam_name,
                        )
                        save_fcos_stream_image(
                            image_path=image_path,
                            out_path=stream_img_path,
                            cam_name=cam_name,
                            camera_intrinsic=calibrated_sensor["camera_intrinsic"],
                            dets=dets_raw,
                            class_names=detector_bundle["class_names"],
                            score_thresh=args.score_thresh,
                            keep_classes=keep_classes,
                        )

                    merged.extend(
                        camera_detections_to_ego(
                            dets=dets_raw,
                            calibrated_sensor=calibrated_sensor,
                            class_names=detector_bundle["class_names"],
                            score_thresh=args.score_thresh,
                            camera_name=cam_name,
                        )
                    )

            if keep_classes is not None:
                merged = [
                    d for d in merged if str(d.get("class", "")).lower() in keep_classes
                ]

            merged = rotated_nms_ego(merged, iou_thresh=args.nms_iou)
            merged = [d for d in merged if d["confidence"] > args.score_thresh]

            sample_results[sample_token] = merged

            sample_token = sample["next"] if sample["next"] else None
            frame_idx += 1

        out_path = scene_output_path(args.out_dir, scene_idx)
        with open(out_path, "wb") as f:
            pickle.dump(sample_results, f)

        print(f"[INFO] Saved: {out_path}")

    print("[DONE] Detection export complete.")


if __name__ == "__main__":
    main()
