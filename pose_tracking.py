import cv2
import mediapipe as mp
import time

# Initialize MediaPipe pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Set up webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_clip.avi', fourcc, 20.0, (640, 480))

# FPS tracking
p_time = 0

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        # Mirror and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = pose.process(rgb)

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # FPS calculation
        c_time = time.time()
        fps = int(1 / (c_time - p_time)) if c_time != p_time else 0
        p_time = c_time
        cv2.putText(frame, f'FPS: {fps}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('Pose Estimation', frame)

        # Save to video file
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()
