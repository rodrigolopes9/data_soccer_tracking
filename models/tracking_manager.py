class TrackingManager:
    def __init__(self):
       
        self.tracked_ids = {
            1: set(),  # Goalkeepers
            2: set(),  # Players
            3: set()   # Referees
        }
        self.class_limits = {
            1: 2,   # Goalkeepers
            2: 20,  # Players
            3: 3    # Referees
        }

        self.goalkeeper_sides = {}  
        self.side_goalkeeper = {    
            'left': None,
            'right': None
        }

    def get_field_side(self, x_position):
        """Determine which side of the field a position is on"""
        field_middle = 675
        return 'left' if x_position < field_middle else 'right'

    def should_track_id(self, class_id, track_id, position=None):
        """Check if this ID should be tracked"""
        if class_id != 1:
            if class_id not in self.class_limits:
                return True
            if track_id in self.tracked_ids[class_id]:
                return True
            if len(self.tracked_ids[class_id]) < self.class_limits[class_id]:
                self.tracked_ids[class_id].add(track_id)
                return True
            return False


        if track_id in self.tracked_ids[1]:
            return True


        if position is not None:
            current_side = self.get_field_side(position[0])


            if self.side_goalkeeper[current_side] is None:
                self.tracked_ids[1].add(track_id)
                self.goalkeeper_sides[track_id] = current_side
                self.side_goalkeeper[current_side] = track_id
                print(f"Adding new goalkeeper {track_id} on {current_side} side")
                return True
            else:
                print(f"Side {current_side} already has goalkeeper {self.side_goalkeeper[current_side]}")

        return False

    def update_goalkeeper_position(self, track_id, position):
        """Update goalkeeper position and side"""
        if track_id in self.tracked_ids[1]:
            current_side = self.get_field_side(position[0])
            old_side = self.goalkeeper_sides.get(track_id)

            if old_side != current_side:
                if old_side:
                    self.side_goalkeeper[old_side] = None
                self.goalkeeper_sides[track_id] = current_side
                self.side_goalkeeper[current_side] = track_id