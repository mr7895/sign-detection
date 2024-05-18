import cv2
import numpy as np
import os
from mediapipe import solutions as mp

# Set up directories for data saving
DATA_PATH = 'MP_Data'
actions = np.array([chr(i) for i in range(ord('A'), ord('Z')+1)])  # Actions A to Z
no_sequences = 30
sequence_length = 30

# Create directories for storing data if they don't exist
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

# Initialize MediaPipe drawing and hand modules
mp_drawing = mp.drawing_utils
mp_hands = mp.hands

# Import custom functions from a separate Python file
from function import mediapipe_detection, draw_styled_landmarks, extract_keypoints

# Set up MediaPipe Hands model
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # Iterate through actions and sequences
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                # Read image
                image_path = os.path.join('Image', action, f'{sequence}.png')
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"Image not found: {image_path}")
                    continue  # Skip if frame not found

                # Process image and extract keypoints
                image, results = mediapipe_detection(frame, hands)
                draw_styled_landmarks(image, results)

                # Display feedback for collecting frames
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(200) if frame_num == 0 else cv2.waitKey(10)

                # Save keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                np.save(npy_path, keypoints)

                # Allow for exit on demand
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
