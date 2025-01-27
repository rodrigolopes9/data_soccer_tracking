class HomographySmoother:
    def __init__(self, smoothing_factor=0.8):
        self.prev_homography = None
        self.smoothing_factor = smoothing_factor
        
    def update(self, current_homography):
        if current_homography is None:
            return self.prev_homography
            
        if self.prev_homography is None:
            self.prev_homography = current_homography
            return current_homography
            
        # Smooth the homography matrix
        smoothed = self.smoothing_factor * self.prev_homography + \
                  (1 - self.smoothing_factor) * current_homography
                  
        self.prev_homography = smoothed
        return smoothed
