#!/usr/bin/env python
"""Prepare nuScenes GT boxes in ego frame for each scene/sample."""

import argparse
import os
import pickle
from typing import Any, Dict, List

import numpy as np
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert nuScenes sample_annotation global boxes to ego frame"
    )
    parser.add_argument("--dataroot", default="./data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--out-dir", default="./outputs/gt")
    return parser.parse_args()


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


def scene_output_path(out_dir: str, scene_idx: int) -> str:
    return os.path.join(out_dir, f"scene_{scene_idx:04d}.pkl")


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    for scene_idx, scene in enumerate(nusc.scene):
        scene_name = scene.get("name", f"scene_{scene_idx:04d}")
        print(f"[INFO] Processing {scene_name} ({scene_idx + 1}/{len(nusc.scene)})")

        scene_gt: Dict[str, List[Dict[str, Any]]] = {}

        sample_token = scene["first_sample_token"]
        frame_idx = 0
        while sample_token:
            sample = nusc.get("sample", sample_token)
            ego_pose = get_sample_ego_pose(nusc, sample)

            frame_gt: List[Dict[str, Any]] = []
            for ann_token in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                cls_name = category_to_detection_name(ann["category_name"])
                if cls_name is None:
                    continue

                box_ego = global_box_to_ego(
                    ann["translation"],
                    ann["size"],
                    ann["rotation"],
                    ego_pose,
                )

                x, y, z = box_ego.center
                width, length, height = box_ego.wlh
                yaw = float(box_ego.orientation.yaw_pitch_roll[0])

                frame_gt.append(
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
                        "track_id": ann["instance_token"],
                        "class": cls_name,
                    }
                )

            scene_gt[sample_token] = frame_gt
            sample_token = sample["next"] if sample["next"] else None
            frame_idx += 1

        out_path = scene_output_path(args.out_dir, scene_idx)
        with open(out_path, "wb") as f:
            pickle.dump(scene_gt, f)

        print(f"[INFO] Saved: {out_path}")

    print("[DONE] GT export complete.")


if __name__ == "__main__":
    main()
