import os
import cv2

# Directory for storing images
directory = 'Image'
subdirectories = [chr(i) for i in range(ord('A'), ord('Z')+1)]  # Extend this list as needed

# Create necessary directories if they don't exist
for sub in subdirectories:
    os.makedirs(os.path.join(directory, sub), exist_ok=True)

# Initialize the video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame could not be read.")
            break

        # Count the number of files in each directory
        count = {}
        for sub in subdirectories:
            path = os.path.join(directory, sub)
            files = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
            count[sub.lower()] = len(files)

        # Example of placing text on the frame for each category
        y_offset = 100
        for key, value in count.items():
            cv2.putText(frame, f"{key}: {value}", (10, y_offset), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
            y_offset += 20

        # Displaying the frame
        cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
        cv2.imshow("data", frame)
        cv2.imshow("ROI", frame[40:400, 0:300])

        # Handling keyboard input for each category
        interrupt = cv2.waitKey(10)
       
        for sub in subdirectories:
            if interrupt & 0xFF == ord(sub.lower()):
                cv2.imwrite(os.path.join(directory, sub, f'{count[sub.lower()]}.png'), frame[40:400, 0:300])

finally:
    cap.release()
    cv2.destroyAllWindows()
