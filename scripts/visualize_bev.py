#!/usr/bin/env python
"""Visualize ego-frame detections in BEV with optional surround-camera overlay."""

import argparse
import math
import os
import pickle
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from nuscenes.nuscenes import NuScenes
except Exception:
    NuScenes = None

try:
    from pyquaternion import Quaternion
except Exception:
    Quaternion = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None


CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

CAMERA_GRID = [
    ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
    ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
]

CAMERA_SHORT_LABEL = {
    "CAM_FRONT": "Front",
    "CAM_FRONT_LEFT": "Front Left",
    "CAM_FRONT_RIGHT": "Front Right",
    "CAM_BACK": "Back",
    "CAM_BACK_LEFT": "Back Left",
    "CAM_BACK_RIGHT": "Back Right",
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

CLASS_ALIASES = {
    "ped": "pedestrian",
    "person": "pedestrian",
    "people": "pedestrian",
}

_FONT_CACHE: Dict[int, object] = {}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BEV visualization for nuScenes scene outputs")
    parser.add_argument("--scene", type=int, default=0, help="Scene index (e.g., 0 -> scene_0000.pkl)")
    parser.add_argument("--det-dir", default="./outputs/detections")
    parser.add_argument("--gt-dir", default="./outputs/gt")
    parser.add_argument("--conf_thresh", type=float, default=0.1)
    parser.add_argument("--range_m", type=float, default=50.0)
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--window-name", default="nuScenes BEV Viewer")

    parser.add_argument("--with-overlay", action="store_true", help="Show camera-image overlay panel")
    parser.add_argument(
        "--overlay-source",
        choices=["stream", "project"],
        default="stream",
        help="Overlay source: stream=pre-saved FCOS images, project=project ego boxes on-the-fly",
    )
    parser.add_argument(
        "--overlay-stream-dir",
        default="./outputs/fcos_stream",
        help="Directory containing pre-saved FCOS camera overlay images.",
    )
    parser.add_argument(
        "--overlay-mode",
        choices=["all", "single"],
        default="all",
        help="Overlay layout: all=6 surround cameras, single=one selected camera",
    )
    parser.add_argument("--overlay-camera", default="CAM_FRONT", help="Camera channel for single overlay mode")
    parser.add_argument("--overlay-max-boxes", type=int, default=120, help="Max 3D boxes drawn on overlay")
    parser.add_argument("--dataroot", default="./data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--show-gt", action="store_true", help="Show GT boxes/text (default: off)")
    parser.add_argument(
        "--keep-classes",
        nargs="+",
        default=None,
        help=(
            "Show only selected detection classes "
            "(e.g. --keep-classes car truck pedestrian). Alias: ped -> pedestrian."
        ),
    )
    return parser.parse_args()


def scene_file_path(base_dir: str, scene_idx: int) -> str:
    return os.path.join(base_dir, f"scene_{scene_idx:04d}.pkl")


def load_pickle(path: str) -> Dict[str, List[dict]]:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_korean_font(size: int):
    if ImageFont is None:
        return None
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]

    candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/malgunbd.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                font = ImageFont.truetype(path, size=size)
                _FONT_CACHE[size] = font
                return font
            except Exception:
                continue

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    _FONT_CACHE[size] = font
    return font


def draw_korean_lines(
    img: np.ndarray,
    lines: List[str],
    x: int,
    y: int,
    color: Tuple[int, int, int] = (230, 230, 230),
    size: int = 18,
    line_gap: int = 24,
) -> None:
    font = get_korean_font(size)
    if Image is None or ImageDraw is None or font is None:
        for i, line in enumerate(lines):
            cv2.putText(
                img,
                line.encode("ascii", "ignore").decode("ascii"),
                (x, y + i * line_gap),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    drawer = ImageDraw.Draw(pil_img)
    r, g, b = int(color[2]), int(color[1]), int(color[0])

    for i, line in enumerate(lines):
        drawer.text((x, y + i * line_gap), line, font=font, fill=(r, g, b))

    img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def ego_to_pixel(x: float, y: float, img_size: int, resolution: float) -> Tuple[int, int]:
    center = img_size // 2
    px = int(round(center - y / resolution))
    py = int(round(center - x / resolution))
    return px, py


def bbox_corners_xy(bbox_3d: List[float]) -> np.ndarray:
    x, y, _, w, _, l, yaw = bbox_3d
    dx = l * 0.5
    dy = w * 0.5

    local = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]], dtype=np.float32)

    rot = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float32,
    )

    return local @ rot.T + np.array([x, y], dtype=np.float32)


def bbox_corners_xyz_ego(bbox_3d: List[float]) -> np.ndarray:
    x, y, z, w, h, l, yaw = bbox_3d
    dx = l * 0.5
    dy = w * 0.5
    dz = h * 0.5

    local = np.array(
        [
            [dx, dy, -dz],
            [dx, -dy, -dz],
            [-dx, -dy, -dz],
            [-dx, dy, -dz],
            [dx, dy, dz],
            [dx, -dy, dz],
            [-dx, -dy, dz],
            [-dx, dy, dz],
        ],
        dtype=np.float32,
    )

    rot = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return local @ rot.T + np.array([x, y, z], dtype=np.float32)


def conf_to_blue(conf: float) -> Tuple[int, int, int]:
    conf = float(np.clip(conf, 0.0, 1.0))
    fade = int(round(190 - 160 * conf))
    fade = int(np.clip(fade, 20, 190))
    return (255, fade, fade)  # BGR


def draw_grid(img: np.ndarray, resolution: float, spacing_m: float = 10.0) -> None:
    h, w = img.shape[:2]
    center = w // 2
    step = int(round(spacing_m / resolution))
    color = (35, 35, 35)

    for offset in range(step, center + 1, step):
        cv2.line(img, (center - offset, 0), (center - offset, h - 1), color, 1)
        cv2.line(img, (center + offset, 0), (center + offset, h - 1), color, 1)
        cv2.line(img, (0, center - offset), (w - 1, center - offset), color, 1)
        cv2.line(img, (0, center + offset), (w - 1, center + offset), color, 1)

    cv2.line(img, (center, 0), (center, h - 1), (55, 55, 55), 1)
    cv2.line(img, (0, center), (w - 1, center), (55, 55, 55), 1)


def draw_ego_vehicle(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    tri = np.array([[cx, cy - 14], [cx - 10, cy + 10], [cx + 10, cy + 10]], np.int32)
    cv2.fillConvexPoly(img, tri, (255, 255, 255))


def draw_rotated_box(
    img: np.ndarray,
    bbox_3d: List[float],
    color: Tuple[int, int, int],
    resolution: float,
    thickness: int = 2,
) -> Tuple[int, int]:
    corners = bbox_corners_xy(bbox_3d)
    _, w = img.shape[:2]
    pix = [ego_to_pixel(float(p[0]), float(p[1]), w, resolution) for p in corners]
    pts = np.array(pix, dtype=np.int32)

    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

    x, y, _, _, _, l, yaw = bbox_3d
    fx = x + (l * 0.5) * math.cos(yaw)
    fy = y + (l * 0.5) * math.sin(yaw)
    c0 = ego_to_pixel(x, y, w, resolution)
    c1 = ego_to_pixel(fx, fy, w, resolution)
    cv2.line(img, c0, c1, color, thickness)

    return c0


def draw_bev_legend(
    img: np.ndarray,
    conf_thresh: float,
    det_count: int,
    gt_count: int,
    overlay_desc: str,
    show_gt: bool,
) -> None:
    _, w = img.shape[:2]
    box_w = 470
    box_h = 170 if show_gt else 145
    x0 = max(8, w - box_w - 8)
    y0 = 8

    panel = img.copy()
    cv2.rectangle(panel, (x0, y0), (x0 + box_w, y0 + box_h), (15, 15, 15), -1)
    cv2.addWeighted(panel, 0.78, img, 0.22, 0.0, dst=img)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (85, 85, 85), 1)

    cv2.rectangle(img, (x0 + 12, y0 + 40), (x0 + 34, y0 + 54), conf_to_blue(0.95), 2)
    if show_gt:
        cv2.rectangle(img, (x0 + 12, y0 + 66), (x0 + 34, y0 + 80), (0, 255, 0), 2)

    lines = [
        "화면 안내",
        "파란 박스: FCOS 검출 결과",
        "박스 옆 숫자: 신뢰도(0~1), 높을수록 진한 파랑",
    ]
    if show_gt:
        lines.append("초록 박스: GT")

    count_line = f"현재 프레임 검출: {det_count}개 (conf >= {conf_thresh:.2f})"
    if show_gt:
        count_line = f"현재 프레임 GT: {gt_count}개 / 검출: {det_count}개"
    lines.append(count_line)
    lines.append(f"오버레이: {overlay_desc}")

    draw_korean_lines(
        img,
        lines,
        x=x0 + 44,
        y=y0 + 12,
        color=(230, 230, 230),
        size=18,
        line_gap=24,
    )


def stream_overlay_image_path(
    overlay_stream_dir: str, scene_idx: int, sample_token: str, cam_channel: str
) -> str:
    return os.path.join(
        overlay_stream_dir,
        f"scene_{scene_idx:04d}",
        sample_token,
        f"{cam_channel}.jpg",
    )


def load_stream_camera_image(
    overlay_stream_dir: str,
    scene_idx: int,
    sample_token: str,
    cam_channel: str,
) -> np.ndarray:
    img_path = stream_overlay_image_path(
        overlay_stream_dir=overlay_stream_dir,
        scene_idx=scene_idx,
        sample_token=sample_token,
        cam_channel=cam_channel,
    )
    image = cv2.imread(img_path)
    if image is None:
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        blank[:] = (22, 22, 22)
        cv2.putText(
            blank,
            f"Missing stream image: {cam_channel}",
            (24, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            blank,
            img_path[-120:],
            (24, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
        image = blank

    cv2.rectangle(image, (8, 8), (430, 46), (20, 20, 20), -1)
    cv2.putText(
        image,
        f"{cam_channel} ({CAMERA_SHORT_LABEL.get(cam_channel, cam_channel)})",
        (16, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    return image


def project_ego_points_to_image(
    points_ego: np.ndarray,
    cam_translation: np.ndarray,
    cam_rotation: np.ndarray,
    cam_intrinsic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if Quaternion is None:
        raise RuntimeError("pyquaternion is required for overlay projection.")

    rot_cam_to_ego = Quaternion(cam_rotation).rotation_matrix.astype(np.float32)
    points_cam = (points_ego - cam_translation.reshape(1, 3)) @ rot_cam_to_ego
    z = points_cam[:, 2]

    uvw = points_cam @ cam_intrinsic.T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-6)
    return uv, z


def draw_projected_box(
    image: np.ndarray,
    bbox_3d: List[float],
    cam_translation: np.ndarray,
    cam_rotation: np.ndarray,
    cam_intrinsic: np.ndarray,
    color: Tuple[int, int, int],
    label: Optional[str] = None,
) -> bool:
    corners_ego = bbox_corners_xyz_ego(bbox_3d)
    corners_uv, z = project_ego_points_to_image(corners_ego, cam_translation, cam_rotation, cam_intrinsic)

    if float(np.min(z)) <= 0.1:
        return False

    pts = corners_uv.astype(np.int32)
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


def render_camera_overlay(
    nusc: Optional[NuScenes],
    dataroot: str,
    sample_token: str,
    cam_channel: str,
    detections: List[dict],
    gt_list: List[dict],
    conf_thresh: float,
    overlay_max_boxes: int,
    show_gt: bool,
    keep_classes: Optional[set],
) -> np.ndarray:
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    blank[:] = (20, 20, 20)

    if nusc is None:
        cv2.putText(blank, "Overlay disabled: NuScenes not loaded", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
        return blank

    sample = nusc.get("sample", sample_token)
    if cam_channel not in sample["data"]:
        cv2.putText(blank, f"No camera channel: {cam_channel}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
        return blank

    sd_token = sample["data"][cam_channel]
    sd = nusc.get("sample_data", sd_token)
    calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    image_path = os.path.join(dataroot, sd["filename"])
    image = cv2.imread(image_path)
    if image is None:
        cv2.putText(blank, f"Image load failed: {sd['filename']}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)
        return blank

    cam_translation = np.array(calib["translation"], dtype=np.float32)
    cam_rotation = np.array(calib["rotation"], dtype=np.float32)
    cam_intrinsic = np.array(calib["camera_intrinsic"], dtype=np.float32)

    if show_gt:
        shown_gt = 0
        for gt in gt_list:
            bbox = gt.get("bbox_3d")
            if bbox is None:
                continue
            label = str(gt.get("track_id", ""))[:8]
            ok = draw_projected_box(
                image,
                bbox,
                cam_translation,
                cam_rotation,
                cam_intrinsic,
                color=(0, 255, 0),
                label=label if label else None,
            )
            if ok:
                shown_gt += 1
            if shown_gt >= overlay_max_boxes:
                break

    shown_det = 0
    for det in detections:
        conf = float(det.get("confidence", 0.0))
        if conf < conf_thresh:
            continue
        det_cls = str(det.get("class", "")).lower()
        if keep_classes is not None and det_cls not in keep_classes:
            continue

        det_cam = str(det.get("camera", ""))
        if det_cam and det_cam != cam_channel:
            continue

        bbox = det.get("bbox_3d")
        if bbox is None:
            continue

        label = f"{conf:.2f}"
        ok = draw_projected_box(
            image,
            bbox,
            cam_translation,
            cam_rotation,
            cam_intrinsic,
            color=conf_to_blue(conf),
            label=label,
        )
        if ok:
            shown_det += 1
        if shown_det >= overlay_max_boxes:
            break

    cv2.rectangle(image, (8, 8), (430, 46), (20, 20, 20), -1)
    cv2.putText(
        image,
        f"{cam_channel} ({CAMERA_SHORT_LABEL.get(cam_channel, cam_channel)})",
        (16, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    return image


def render_overlay_panel(
    nusc: Optional[NuScenes],
    dataroot: str,
    scene_idx: int,
    sample_token: str,
    overlay_source: str,
    overlay_stream_dir: str,
    overlay_mode: str,
    overlay_camera: str,
    detections: List[dict],
    gt_list: List[dict],
    conf_thresh: float,
    overlay_max_boxes: int,
    show_gt: bool,
    keep_classes: Optional[set],
) -> np.ndarray:
    if overlay_source == "stream":
        if overlay_mode == "single":
            return load_stream_camera_image(
                overlay_stream_dir=overlay_stream_dir,
                scene_idx=scene_idx,
                sample_token=sample_token,
                cam_channel=overlay_camera,
            )
    else:
        if overlay_mode == "single":
            return render_camera_overlay(
                nusc=nusc,
                dataroot=dataroot,
                sample_token=sample_token,
                cam_channel=overlay_camera,
                detections=detections,
                gt_list=gt_list,
                conf_thresh=conf_thresh,
                overlay_max_boxes=overlay_max_boxes,
                show_gt=show_gt,
                keep_classes=keep_classes,
            )

    tile_w = 480
    tile_h = 270
    gap = 8
    title_h = 54
    canvas_h = title_h + gap + (tile_h * 2) + (gap * 2)
    canvas_w = (tile_w * 3) + (gap * 4)

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 18)

    if overlay_source == "stream":
        title = "Surround Cameras (Saved FCOS Stream)"
    else:
        title = "Surround Cameras (Projected from ego boxes)"
    cv2.putText(
        canvas,
        title,
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    for r, row in enumerate(CAMERA_GRID):
        for c, cam in enumerate(row):
            if overlay_source == "stream":
                cam_img = load_stream_camera_image(
                    overlay_stream_dir=overlay_stream_dir,
                    scene_idx=scene_idx,
                    sample_token=sample_token,
                    cam_channel=cam,
                )
            else:
                cam_img = render_camera_overlay(
                    nusc=nusc,
                    dataroot=dataroot,
                    sample_token=sample_token,
                    cam_channel=cam,
                    detections=detections,
                    gt_list=gt_list,
                    conf_thresh=conf_thresh,
                    overlay_max_boxes=overlay_max_boxes,
                    show_gt=show_gt,
                    keep_classes=keep_classes,
                )

            tile = cv2.resize(cam_img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            x0 = gap + c * (tile_w + gap)
            y0 = title_h + gap + r * (tile_h + gap)
            canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
            cv2.rectangle(canvas, (x0, y0), (x0 + tile_w, y0 + tile_h), (80, 80, 80), 1)

    return canvas


def compose_bev_frame(
    scene_idx: int,
    scene_name: str,
    frame_idx: int,
    total_frames: int,
    sample_token: str,
    detections: List[dict],
    gt_list: List[dict],
    conf_thresh: float,
    img_size: int,
    resolution: float,
    overlay_desc: str,
    show_gt: bool,
    keep_classes: Optional[set],
) -> np.ndarray:
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img[:] = (12, 12, 12)

    draw_grid(img, resolution)
    draw_ego_vehicle(img)

    gt_count = 0
    if show_gt:
        for gt in gt_list:
            bbox = gt.get("bbox_3d")
            if bbox is None:
                continue
            anchor = draw_rotated_box(img, bbox, (0, 255, 0), resolution, thickness=2)
            gt_count += 1
            track_id = gt.get("track_id", "")
            if track_id:
                label = str(track_id)[:8]
                cv2.putText(
                    img,
                    label,
                    (anchor[0] + 4, anchor[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (140, 255, 140),
                    1,
                    cv2.LINE_AA,
                )

    det_count = 0
    for det in detections:
        conf = float(det.get("confidence", 0.0))
        if conf < conf_thresh:
            continue
        det_cls = str(det.get("class", "")).lower()
        if keep_classes is not None and det_cls not in keep_classes:
            continue
        bbox = det.get("bbox_3d")
        if bbox is None:
            continue

        color = conf_to_blue(conf)
        anchor = draw_rotated_box(img, bbox, color, resolution, thickness=2)
        det_count += 1
        cv2.putText(
            img,
            f"{conf:.2f}",
            (anchor[0] + 4, anchor[1] + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    info_text = f"scene_idx={scene_idx:04d} ({scene_name}) frame={frame_idx + 1}/{total_frames}"
    cv2.putText(img, info_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (230, 230, 230), 1, cv2.LINE_AA)

    token_text = f"sample={sample_token[:16]}..."
    cv2.putText(img, token_text, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)

    key_hint = "[space] pause  [n] next  [p] prev  [q] quit"
    cv2.putText(img, key_hint, (10, img_size - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    draw_bev_legend(
        img=img,
        conf_thresh=conf_thresh,
        det_count=det_count,
        gt_count=gt_count,
        overlay_desc=overlay_desc,
        show_gt=show_gt,
    )
    return img


def main() -> None:
    args = parse_args()

    keep_classes = normalize_keep_classes(args.keep_classes)
    if keep_classes is not None:
        print(f"[INFO] Visualization class filter enabled: {sorted(keep_classes)}")
    print(f"[INFO] Show GT: {bool(args.show_gt)}")

    det_path = scene_file_path(args.det_dir, args.scene)
    gt_path = scene_file_path(args.gt_dir, args.scene)

    if not os.path.isfile(det_path):
        raise FileNotFoundError(f"Detection file not found: {det_path}")

    det_data = load_pickle(det_path)
    gt_data: Dict[str, List[dict]] = {}
    if args.show_gt:
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"GT file not found: {gt_path}")
        gt_data = load_pickle(gt_path)

    sample_tokens = list(det_data.keys()) if len(det_data) > 0 else list(gt_data.keys())
    if not sample_tokens:
        raise RuntimeError("No frames found in selected scene files.")

    img_size = int(round((args.range_m * 2.0) / args.resolution))
    if img_size <= 0:
        raise ValueError("Invalid BEV size. Check --range_m and --resolution.")

    nusc = None
    scene_name = f"scene-{args.scene:04d}"
    overlay_enabled = bool(args.with_overlay)

    if overlay_enabled:
        if args.overlay_source == "project":
            if NuScenes is None or Quaternion is None:
                print("[WARN] nuscenes-devkit/pyquaternion import failed. Overlay disabled.")
                overlay_enabled = False
            else:
                nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
                if 0 <= args.scene < len(nusc.scene):
                    scene_name = nusc.scene[args.scene]["name"]
        else:
            if not os.path.isdir(args.overlay_stream_dir):
                print(
                    f"[WARN] overlay stream dir not found: {args.overlay_stream_dir}. "
                    "Overlay panel may show missing-image placeholders."
                )

    if args.overlay_mode not in {"all", "single"}:
        raise ValueError("overlay-mode must be one of: all, single")

    if args.overlay_mode == "single" and args.overlay_camera not in CAMERAS:
        raise ValueError(f"overlay-camera must be one of {CAMERAS}")

    overlay_desc = "끄기"
    if overlay_enabled:
        if args.overlay_mode == "all":
            if args.overlay_source == "stream":
                overlay_desc = "서라운드 6카메라(저장 스트림)"
            else:
                overlay_desc = "서라운드 6카메라(실시간 투영)"
        else:
            if args.overlay_source == "stream":
                overlay_desc = f"단일 카메라({args.overlay_camera}, 저장 스트림)"
            else:
                overlay_desc = f"단일 카메라({args.overlay_camera}, 실시간 투영)"

    # WINDOW_NORMAL lets you resize the viewer by dragging window borders.
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(args.window_name, 1800 if overlay_enabled else 1000, 1000)

    idx = 0
    paused = False
    total = len(sample_tokens)
    wait_ms = max(1, int(round(1000.0 / max(args.fps, 0.1))))

    while True:
        token = sample_tokens[idx]
        bev_frame = compose_bev_frame(
            scene_idx=args.scene,
            scene_name=scene_name,
            frame_idx=idx,
            total_frames=total,
            sample_token=token,
            detections=det_data.get(token, []),
            gt_list=gt_data.get(token, []),
            conf_thresh=args.conf_thresh,
            img_size=img_size,
            resolution=args.resolution,
            overlay_desc=overlay_desc,
            show_gt=bool(args.show_gt),
            keep_classes=keep_classes,
        )

        frame = bev_frame
        if overlay_enabled:
            overlay = render_overlay_panel(
                nusc=nusc,
                dataroot=args.dataroot,
                scene_idx=args.scene,
                sample_token=token,
                overlay_source=args.overlay_source,
                overlay_stream_dir=args.overlay_stream_dir,
                overlay_mode=args.overlay_mode,
                overlay_camera=args.overlay_camera,
                detections=det_data.get(token, []),
                gt_list=gt_data.get(token, []),
                conf_thresh=args.conf_thresh,
                overlay_max_boxes=args.overlay_max_boxes,
                show_gt=bool(args.show_gt),
                keep_classes=keep_classes,
            )

            overlay_resized = cv2.resize(
                overlay,
                (int(round(overlay.shape[1] * (img_size / max(overlay.shape[0], 1)))), img_size),
                interpolation=cv2.INTER_AREA,
            )
            divider = np.zeros((img_size, 4, 3), dtype=np.uint8)
            divider[:] = (45, 45, 45)
            frame = np.concatenate([bev_frame, divider, overlay_resized], axis=1)

        cv2.imshow(args.window_name, frame)

        key = cv2.waitKey(0 if paused else wait_ms) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
            continue
        if key == ord("n"):
            idx = min(idx + 1, total - 1)
            paused = True
            continue
        if key == ord("p"):
            idx = max(idx - 1, 0)
            paused = True
            continue

        if not paused:
            if idx < total - 1:
                idx += 1
            else:
                paused = True

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
