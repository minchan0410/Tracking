#!/bin/bash
# Run the ROS node using the yolo3d_env venv Python.
# PYTHONPATH order: venv packages first (highest priority), then ROS, then system.
export PYTHONPATH=/home/minchan0410/Tracking/yolo3d_env/lib/python3.8/site-packages:/opt/ros/noetic/lib/python3/dist-packages:/usr/lib/python3/dist-packages:/usr/local/lib/python3.8/dist-packages

exec /home/minchan0410/Tracking/yolo3d_env/bin/python3 "$@"
