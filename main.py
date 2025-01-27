import supervision as sv
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.image as mpimg
import scipy.linalg
import sys
import os

from config.field_keypoints import fieldKeypoints
from models.tracking_manager import TrackingManager
from models.byte_tracker import EnhancedByteTracker
from models.homography_smoother import HomographySmoother
from utils.image_processing import process_frame_with_yolo, find_jersey_color
from utils.homography import calculate_homography
from utils.print_detections import print_object_detections

def is_hex_color(color_str):
    color_str = color_str.lstrip('#')
    return len(color_str) == 6 and all(c in '0123456789ABCDEFabcdef' for c in color_str)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))





if len(sys.argv) < 7:
    print("Usage args: HighlightId, Home_team_color_hex, Away_team_color_hex, Gk_Home_team_color_hex, GK_Away_team_color_hex, referee_color")
    sys.exit(1)

video_path = "./assets/videos/" + sys.argv[1] + ".mp4" # print(sys.argv[1])
if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' does not exist")
    sys.exit(1)

if(not (is_hex_color(sys.argv[2]) and is_hex_color(sys.argv[3]) and is_hex_color(sys.argv[4])  and is_hex_color(sys.argv[5])  and is_hex_color(sys.argv[6])) ):
    print("Wrong hex values")
    sys.exit(1)

home_team_color = hex_to_rgb(sys.argv[2])   
away_team_color = hex_to_rgb(sys.argv[3])      
home_team_goalkeeper_color = hex_to_rgb(sys.argv[4])   
away_team_goalkeeper_color = hex_to_rgb(sys.argv[5])   
referee_color = hex_to_rgb(sys.argv[6])    
tolerance = 40


model = YOLO('./AI_models/gitplayer.pt')
keypoint_detection_model = YOLO('./AI_models/train6/weights/best.pt')
field_image = mpimg.imread('./assets/images/tacticalmap3.jpg')


cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

tracker = EnhancedByteTracker()
smoother = HomographySmoother(smoothing_factor=0.8)
label_annotator = sv.LabelAnnotator()
tracking_manager = TrackingManager()
last_field_positions = {}
prev_valid_keypoints = None
max_movement = 20

frame_count = 1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    pipeline_results = process_frame_with_yolo(frame, keypoint_detection_model, conf_threshold=0.8)
    pose_detections = pipeline_results['detections']

    raw_homography, prev_valid_keypoints = calculate_homography(prev_valid_keypoints, fieldKeypoints, pose_detections, max_movement)

    if raw_homography is not None:
        homography_matrix = smoother.update(raw_homography)
    else:
        continue


    results = model.predict(source=frame, verbose=False, conf=0.25)[0]

    ball_class_idx = None
    for idx, class_name in model.names.items():
        if class_name == 'ball':
            ball_class_idx = int(idx)
            break


    mask = results.boxes.cls != ball_class_idx
    results.boxes = results.boxes[mask]

    jersey_colors = find_jersey_color(frame, results, model.names, home_team_color, away_team_color, home_team_goalkeeper_color, away_team_goalkeeper_color, referee_color, tolerance)

    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections, homography_matrix, jersey_colors)

    colors = []
    for i in range(len(detections.class_id)):
        track_id = detections.tracker_id[i]
        color = tracker.get_prevalent_jersey_color(track_id)
        if color is not None:
            # Convert from RGB to BGR
            colors.append((int(color[2]), int(color[1]), int(color[0])))
        else:
            colors.append((255, 255, 255))  

    print(f"{frame_count}")
    frame_count += 1

    last_field_positions = print_object_detections(
            detections,
            colors,
            homography_matrix,
            home_team_color, 
            away_team_color, 
            home_team_goalkeeper_color, 
            away_team_goalkeeper_color, 
            referee_color,
            last_field_positions,
            tracking_manager
        )
    