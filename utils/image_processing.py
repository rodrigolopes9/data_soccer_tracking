import cv2
import numpy as np

def process_frame_with_yolo(frame, model, target_size=(1280, 1280), conf_threshold=0.25):

    original_frame = frame.copy()

    def preprocess_frame_yolo(frame, target_size=(1280, 1280)):
        """
        Preprocess a frame following YOLO requirements:
        1. Stretch resize to target size
        2. Apply histogram equalization
        3. Convert to RGB (YOLO uses RGB format)
        """

        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

        
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)


        l, a, b = cv2.split(lab)
        l_eq = cv2.equalizeHist(l)


        lab_eq = cv2.merge([l_eq, a, b])


        processed = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        return processed

    def transform_keypoints_to_original(keypoints, original_frame, target_size=(1280, 1280)):
        """
        Transform YOLO keypoints from preprocessed frame back to original coordinates.
        """

        orig_height, orig_width = original_frame.shape[:2]
        target_width, target_height = target_size

        transformed_keypoints = keypoints.copy()

        valid_points = transformed_keypoints[:, 2] > 0

        transformed_keypoints[valid_points, 0] *= (orig_width / target_width)

        transformed_keypoints[valid_points, 1] *= (orig_height / target_height)

        return transformed_keypoints


    preprocessed_frame = preprocess_frame_yolo(frame, target_size)

 
    results = model.predict(source=preprocessed_frame, verbose=False, conf=conf_threshold)[0]

    processed_detections = []

    if hasattr(results, 'keypoints') and results.keypoints is not None:
        for person in results.keypoints.data:

            keypoints = person.cpu().numpy() if hasattr(person, 'cpu') else person

            transformed_keypoints = transform_keypoints_to_original(
                keypoints,
                original_frame,
                target_size
            )

            processed_detections.append({
                'keypoints': transformed_keypoints,
                'confidence_scores': transformed_keypoints[:, 2]
            })

    return {
        'original_frame': original_frame,
        'preprocessed_frame': preprocessed_frame,
        'detections': processed_detections
    }


def find_jersey_color(frame, results, class_names, target_color1, target_color2, target_color3, target_color4, target_color5, tolerance):
  target_color_rgb1 = np.array(target_color1, dtype=np.int16)
  target_color_rgb2 = np.array(target_color2, dtype=np.int16)
  target_color_rgb3 = np.array(target_color3, dtype=np.int16)
  target_color_rgb4 = np.array(target_color4, dtype=np.int16)
  target_color_rgb5 = np.array(target_color5, dtype=np.int16)

  color_atribution = []
  for result in results:
      boxes = result.boxes.xyxy.cpu().numpy()  
      classes = result.boxes.cls.cpu().numpy()  
      scores = result.boxes.conf.cpu().numpy()  

      for box, cls_id, score in zip(boxes, classes, scores):
          x1, y1, x2, y2 = box


          class_name = class_names[int(cls_id)]

          ycenter = int((y1 + y2)/2)
          yNoHead = int(((y1-y2)/3)+ycenter)

          x1New = int(x1 - ((x1-x2)/5))
          x2New = int(x2 + ((x1-x2)/5))

          if class_name != 'ball':

              roi = frame[yNoHead:ycenter, x1New:x2New]

              if roi.size == 0:
                  continue

    
              roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)


              lower_bound1 = np.clip(target_color_rgb1 - tolerance, 0, 255).astype(np.uint8)
              upper_bound1 = np.clip(target_color_rgb1 + tolerance, 0, 255).astype(np.uint8)
              lower_bound2 = np.clip(target_color_rgb2 - tolerance, 0, 255).astype(np.uint8)
              upper_bound2 = np.clip(target_color_rgb2 + tolerance, 0, 255).astype(np.uint8)
              lower_bound3 = np.clip(target_color_rgb3 - tolerance, 0, 255).astype(np.uint8)
              upper_bound3 = np.clip(target_color_rgb3 + tolerance, 0, 255).astype(np.uint8)
              lower_bound4 = np.clip(target_color_rgb4 - tolerance, 0, 255).astype(np.uint8)
              upper_bound4 = np.clip(target_color_rgb4 + tolerance, 0, 255).astype(np.uint8)
              lower_bound5 = np.clip(target_color_rgb5 - tolerance, 0, 255).astype(np.uint8)
              upper_bound5 = np.clip(target_color_rgb5 + tolerance, 0, 255).astype(np.uint8)


              mask1 = cv2.inRange(roi_rgb, lower_bound1, upper_bound1)
              mask2 = cv2.inRange(roi_rgb, lower_bound2, upper_bound2)
              mask3 = cv2.inRange(roi_rgb, lower_bound3, upper_bound3)
              mask4 = cv2.inRange(roi_rgb, lower_bound4, upper_bound4)
              mask5 = cv2.inRange(roi_rgb, lower_bound5, upper_bound5)


              matching_pixels1 = cv2.countNonZero(mask1)
              matching_pixels2 = cv2.countNonZero(mask2)
              matching_pixels3 = cv2.countNonZero(mask3)
              matching_pixels4 = cv2.countNonZero(mask4)
              matching_pixels5 = cv2.countNonZero(mask5)

              total_pixels = roi_rgb.shape[0] * roi_rgb.shape[1]

              jerseyColor = (0,0,0)
              jersey_color = max(matching_pixels1, matching_pixels2, matching_pixels3, matching_pixels4, matching_pixels5)

              if jersey_color == matching_pixels1:
                  jerseyColor = target_color1
              elif jersey_color == matching_pixels2:
                  jerseyColor = target_color2
              elif jersey_color == matching_pixels3:
                  jerseyColor = target_color3
              elif jersey_color == matching_pixels4:
                  jerseyColor = target_color4
              elif jersey_color == matching_pixels5:
                  jerseyColor = target_color5

              color_atribution.append(jerseyColor)

          else:
            color_atribution.append(None)

  return color_atribution