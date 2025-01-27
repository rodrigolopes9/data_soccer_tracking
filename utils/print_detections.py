import numpy as np
import supervision as sv
import cv2
from config.field_keypoints import fieldKeypoints

MIN_X = fieldKeypoints[0][0]
MIN_Y = fieldKeypoints[0][1]
MAX_X = fieldKeypoints[29][0]
MAX_Y = fieldKeypoints[29][1]

HOME_TEAM = "home_team"
AWAY_TEAM = "away_team"
REFEREE = "referee"
GOALKEEPER = "goalkeeper"
PLAYER = "player"





def print_object_detections(detections, colors, homography_matrix, home_team_color, 
            away_team_color, 
            home_team_goalkeeper_color, 
            away_team_goalkeeper_color, 
            referee_color, last_field_positions={}, tracking_manager={}):
    current_ids = []

    print("ID,Coordinate_X,Coordinate_Y,Type,Team")

   
    for i in range(len(detections.class_id)):
        track_id = detections.tracker_id[i]
        class_id = detections.class_id[i]

        
        bbox = detections.xyxy[i]
        ground_point = ((bbox[0] + bbox[2]) / 2, bbox[3])
        point_homogeneous = np.array([[ground_point[0]], [ground_point[1]], [1]])
        transformed_point = np.dot(homography_matrix, point_homogeneous)
        transformed_point = transformed_point / transformed_point[2]
        map_x = int(transformed_point[0][0])
        map_y = int(transformed_point[1][0])

  
        if class_id == 1: 
            if not tracking_manager.should_track_id(class_id, track_id, (map_x, map_y)):
                continue
            tracking_manager.update_goalkeeper_position(track_id, (map_x, map_y))
        else:
            if not tracking_manager.should_track_id(class_id, track_id):
                continue

        current_ids.append(track_id)

        
        last_field_positions[track_id] = {
            'position': (map_x, map_y),
            'color': colors[i],
            'class_id': class_id
        }

        type = getType(class_id)
        team = getTeam(colors[i], away_team_color, home_team_color, home_team_goalkeeper_color, away_team_goalkeeper_color, referee_color )
        normalized_x, normalized_y = transform_coordinates(MIN_X, MIN_Y, MAX_X, MAX_Y, map_x, map_y)
        print(f"{track_id},{normalized_x},{normalized_y},{type},{team}")

    # Handle disappeared detections
    for track_id, data in list(last_field_positions.items()):
        if track_id not in current_ids:
            class_id = data.get('class_id')
            pos = data['position']

            if class_id == 1:  # Goalkeeper
                if not tracking_manager.should_track_id(class_id, track_id, pos):
                    continue
            else:
                if not tracking_manager.should_track_id(class_id, track_id):
                    continue
            
            type = getType(class_id)
            team = getTeam(data['color'], away_team_color, home_team_color, home_team_goalkeeper_color, away_team_goalkeeper_color, referee_color )
            normalized_x, normalized_y = transform_coordinates(MIN_X, MIN_Y, MAX_X, MAX_Y, pos[0], pos[1])
            print(f"{track_id},{normalized_x},{normalized_y},{type},{team}")

    return last_field_positions

def transform_coordinates(MIN_X, MIN_Y, MAX_X, MAX_Y, coordinate_x, coordinate_y):

   FIELD_WIDTH = MAX_X - MIN_X 
   FIELD_HEIGHT = MAX_Y - MIN_Y

   normalized_x = round((coordinate_x - MIN_X) / FIELD_WIDTH, 3)
   normalized_y = round((coordinate_y - MIN_Y) / FIELD_HEIGHT, 3)

   # Clamp values to ensure they're between 0 and 1
   normalized_x = max(0, min(1, normalized_x))
   normalized_y = max(0, min(1, normalized_y))

   return normalized_x, normalized_y

def getType(class_id):
    if class_id == 1:
        return GOALKEEPER
    elif class_id == 2:
        return PLAYER
    elif class_id == 3:
        return REFEREE
    else:
        return None
    
def getTeam(color, away_team_color, home_team_color, home_team_goalkeeper_color, away_team_goalkeeper_color, referee_color ):
    if np.array_equal((color[2],color[1],color[0]), (away_team_color)):
        return AWAY_TEAM
    if np.array_equal((color[2],color[1],color[0]), home_team_color):
        return HOME_TEAM
    elif np.array_equal((color[2],color[1],color[0]), home_team_goalkeeper_color):
        return HOME_TEAM
    elif np.array_equal((color[2],color[1],color[0]), away_team_goalkeeper_color):
        return AWAY_TEAM
    elif np.array_equal((color[2],color[1],color[0]), referee_color):
        return REFEREE
    else:
        return None