#installing everything
import cv2
import mediapipe as mp
import time
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

#line connections for hand
hand_connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]


options = HandLandmarkerOptions(base_options = BaseOptions(model_asset_path = 'hand_landmarker.task'), running_mode = RunningMode.VIDEO, num_hands = 1)

Landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

timestamp_ms = 0

#main code
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)   # 1 = horizontal flip (mirror effect)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = imgRGB)
    results = Landmarker.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 1
    if results.hand_landmarks:
      for hand_landmarks in results.hand_landmarks:
          points = []
          for landmark in hand_landmarks:
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                points.append((x, y))

          for start_idx, end_idx in hand_connections:
                 cv2.line(img, points[start_idx], points[end_idx], (0, 255, 0), 1)

         #for point in points: 
             # cv2.circle(img, (x, y), 4, (255, 0, 0), cv2.FILLED)
          for point in points:
                 cv2.circle(img, point, 3, (255, 0, 0), cv2.FILLED)

          for i, point in enumerate(points):
             if i == 8:  # index fingertip
                   cv2.circle(img, point, 6, (255, 0, 255), -1)   
          
                

          

    
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

