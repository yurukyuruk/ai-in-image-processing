import cv2
import numpy as np

video_path = "output_clip.avi"  
cap = cv2.VideoCapture(video_path)

# Output writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_dense = cv2.VideoWriter('dense_flow.avi', fourcc, 20.0, (640, 480))
out_sparse = cv2.VideoWriter('sparse_flow.avi', fourcc, 20.0, (640, 480))

# Params for Lucas–Kanade (Sparse)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read first frame
ret, first_frame = cap.read()
first_frame = cv2.resize(first_frame, (640, 480))
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Initialize points for sparse optical flow
prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)

hsv_mask = np.zeros_like(first_frame)
hsv_mask[..., 1] = 255  # For dense optical flow coloring

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- DENSE OPTICAL FLOW (Farneback) ---
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv_mask[..., 0] = ang * 180 / np.pi / 2
    hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    dense_flow = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
    out_dense.write(dense_flow)
    cv2.imshow("Dense Optical Flow", dense_flow)

    # --- SPARSE OPTICAL FLOW (Lucas–Kanade) ---
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

    # Draw the tracks
    mask = np.zeros_like(frame)
    if next_pts is not None and prev_pts is not None:
        for i, (new, old) in enumerate(zip(next_pts[status == 1], prev_pts[status == 1])):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)
        sparse_output = cv2.add(frame, mask)
        out_sparse.write(sparse_output)
        cv2.imshow("Sparse Optical Flow", sparse_output)

    # Update previous frame/points
    prev_gray = gray.copy()
    if next_pts is not None:
        prev_pts = next_pts[status == 1].reshape(-1, 1, 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_dense.release()
out_sparse.release()
cv2.destroyAllWindows()
