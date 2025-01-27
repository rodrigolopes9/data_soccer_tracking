import numpy as np
from collections import defaultdict
import supervision as sv

class EnhancedByteTracker(sv.ByteTrack):
    def __init__(self):
        super().__init__()
        self.track_history = defaultdict(list)
        self.track_metadata = defaultdict(dict)
        self.id_mapping = {} 
        self.all_disappeared_ids = set()
        self.all_tracked_ids = set()

    def update_with_detections(self, detections, homography_matrix, jersey_colors):
        previous_ids = set(self.track_metadata.keys())

        if len(detections) > 0:
            modified_class_ids = detections.class_id.copy()


            detections = sv.Detections(
                xyxy=detections.xyxy,
                confidence=detections.confidence,
                class_id=modified_class_ids,
                tracker_id=detections.tracker_id if hasattr(detections, 'tracker_id') else None
            )

           
            if detections.tracker_id is not None:
                for i, original_id in enumerate(detections.tracker_id):
                    if original_id in self.id_mapping:
                        detections.tracker_id[i] = self.id_mapping[original_id]

            tracked_detections = super().update_with_detections(detections)

            if len(tracked_detections) == 0:
                self.all_disappeared_ids.update(previous_ids)
                return tracked_detections

            current_ids = set(tracked_detections.tracker_id)
            disappeared_ids = previous_ids - current_ids
            self.all_disappeared_ids.update(disappeared_ids)
            new_ids = current_ids - self.all_tracked_ids
            reappeared_ids = current_ids.intersection(self.all_disappeared_ids)
            self.all_disappeared_ids -= reappeared_ids
            self.all_tracked_ids.update(current_ids)


            id_reassignments = {}

            for new_id in new_ids:
                new_id_idx = list(tracked_detections.tracker_id).index(new_id)
                new_id_jerseycolor = jersey_colors[new_id_idx]
                new_id_bbox = tracked_detections.xyxy[new_id_idx]
                new_ground_point = self._calculate_ground_point(new_id_bbox)
                new_field_position = self._transform_point(new_ground_point, homography_matrix)


               
                best_match_id = None
                best_match_score = 0.5


                for disappeared_id in self.all_disappeared_ids:
                    last_positions = [frame['field_position'] for frame in self.track_history[disappeared_id][-5:]]
                    avg_position = tuple(np.mean(last_positions, axis=0))
                    distance_score = self.calculate_distance_score(new_field_position, avg_position)

                  
                    jersey_colors_history = [frame['jersey_color'] for frame in self.track_history[disappeared_id]]
                    color_counts = {}
                    for color in jersey_colors_history:
                        if color is not None:
                            color_str = str(color)
                            color_counts[color_str] = color_counts.get(color_str, 0) + 1

                    calculate_color_similarity_score = self.calculate_color_similarity(new_id_jerseycolor, color_counts)


                    if distance_score*calculate_color_similarity_score > best_match_score:
                        best_match_id = disappeared_id
                        best_match_score = distance_score

                   

                if best_match_id is not None:
                    id_reassignments[new_id] = best_match_id


            if len(id_reassignments) > 0 :
                for new_id, old_id in id_reassignments.items():
                    self.id_mapping[new_id] = old_id
                    self.all_disappeared_ids.remove(old_id)
                    new_ids.remove(new_id)
                    self.all_tracked_ids.add(old_id)


            modified_tracker_ids = tracked_detections.tracker_id.copy()
            for i, tracker_id in enumerate(modified_tracker_ids):
                if tracker_id in self.id_mapping:
                    modified_tracker_ids[i] = self.id_mapping[tracker_id]

            tracked_detections = sv.Detections(
                xyxy=tracked_detections.xyxy,
                confidence=tracked_detections.confidence,
                class_id=tracked_detections.class_id,
                tracker_id=modified_tracker_ids
            )


            for i, original_id in enumerate(tracked_detections.tracker_id):
                ground_point = self._calculate_ground_point(tracked_detections.xyxy[i])
                field_position = self._transform_point(ground_point, homography_matrix)

                self.track_history[original_id].append({
                    'frame_number': len(self.track_history[original_id]),
                    'confidence': tracked_detections.confidence[i],
                    'class_id': tracked_detections.class_id[i],
                    'field_position': field_position,
                    'jersey_color': jersey_colors[i] if i < len(jersey_colors) else None
                })

                self.track_metadata[original_id].update({
                    'lifetime': len(self.track_history[original_id]),
                    'average_confidence': self._calculate_avg_confidence(original_id),
                    'last_field_position': field_position,
                    'last_jersey_color': jersey_colors[i] if i < len(jersey_colors) else None
                })

            return tracked_detections

        return sv.Detections.empty()

    def get_prevalent_jersey_color(self, track_id, window_size=None):

        if track_id not in self.track_history:
            return None


        history = self.track_history[track_id]
        if window_size is not None:
            history = history[-window_size:]

        jersey_colors_history = [frame['jersey_color'] for frame in history]


        color_counts = {}
        for color in jersey_colors_history:
            if color is not None:
                color_str = str(color)
                color_counts[color_str] = color_counts.get(color_str, 0) + 1

        if not color_counts:
            return None


        most_common_color_str = max(color_counts.items(), key=lambda x: x[1])[0]


        color_values = most_common_color_str.strip('()').split(',')
        most_common_color = tuple(int(x.strip()) for x in color_values)

        return most_common_color

    def get_all_prevalent_colors(self):

        prevalent_colors = {}
        for track_id in self.track_metadata.keys():
            color = self.get_prevalent_jersey_color(track_id)
            if color is not None:
                prevalent_colors[track_id] = color
        return prevalent_colors


    def calculate_distance_score(self, pos1, pos2, max_distance=300):

        distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        return max(0, 1 - (distance / max_distance))

    def calculate_color_similarity(self, new_color, color_counts):

        if not color_counts or new_color is None:
            return 0

        new_color_str = str(new_color)


        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)


        if new_color_str not in color_counts:
            return 0


        for i, (color, count) in enumerate(sorted_colors):
            if color == new_color_str:
                if i == 0:  # Most frequent color
                    return 1.5
                elif i == 1:  # Second most frequent
                    return 0.5
                else:  # Least frequent
                    return 0.1

        return 0

    def _calculate_ground_point(self, bbox):
        """Calculate the point where the player touches the ground"""
        x = (bbox[0] + bbox[2]) / 2  
        y = bbox[3]                 
        return (x, y)

    def _transform_point(self, point, homography_matrix):
        """Transform a point using the homography matrix"""

        point_homogeneous = np.array([[point[0]], [point[1]], [1]])


        transformed_point = np.dot(homography_matrix, point_homogeneous)


        transformed_point = transformed_point / transformed_point[2]
        return (transformed_point[0][0], transformed_point[1][0])

    def _calculate_center(self, bbox):
        """Calculate center point of bounding box"""
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        return (x, y)

    def _calculate_avg_confidence(self, track_id):
        """Calculate average confidence for a track"""
        confidences = [frame['confidence'] for frame in self.track_history[track_id]]
        return sum(confidences) / len(confidences) if confidences else 0

    def get_track_info(self, track_id):
        """Get complete information about a track"""
        return {
            'history': self.track_history[track_id],
            'metadata': self.track_metadata[track_id]
        }

    def filter_tracks(self, min_lifetime=0, min_confidence=0):
        """Filter tracks based on criteria"""
        filtered_ids = []
        for track_id in self.track_metadata:
            if (self.track_metadata[track_id]['lifetime'] >= min_lifetime and
                self.track_metadata[track_id]['average_confidence'] >= min_confidence):
                filtered_ids.append(track_id)
        return filtered_ids