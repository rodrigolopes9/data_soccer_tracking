import numpy as np
import cv2

def calculate_homography(prev_valid_keypoints, fieldKeypoints, detections, max_movement):
    raw_homography_matrix = None
    if detections:  
        for detection in detections: 
            keypoints = detection['keypoints']
            confidences = detection['confidence_scores']


            valid_indices = np.where(confidences >= 0.8)[0]
            valid_keypoints = keypoints[valid_indices]

           

            if len(valid_indices) >= 4:
                if prev_valid_keypoints is not None and len(valid_keypoints) == len(prev_valid_keypoints):
                    movement = np.max(np.sqrt(np.sum((valid_keypoints[:, :2] - prev_valid_keypoints[:, :2])**2, axis=1)))
                    if movement > max_movement:
                        valid_keypoints = prev_valid_keypoints
                
                prev_valid_keypoints = valid_keypoints.copy()

                source_points = valid_keypoints[:, :2]
                destination_points = np.array(
                    [fieldKeypoints[idx] for idx in valid_indices],
                    dtype=np.float32
                )
                source_points = np.array(source_points, dtype=np.float32)

                if len(valid_indices) < 10:
                    raw_homography_matrix, _ = cv2.findHomography(
                        source_points,
                        destination_points,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=5.0
                    )
                else:
                    raw_homography_matrix, _ = cv2.findHomography(
                        source_points,
                        destination_points,
                        method=cv2.LMEDS,
                        ransacReprojThreshold=5.0
                    )
    return raw_homography_matrix, prev_valid_keypoints