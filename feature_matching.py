import cv2
import numpy as np

# Load two related images
img1 = cv2.imread('image4.jpg')  # Change filenames as needed
img2 = cv2.imread('image5.jpg')

# Resize both images to the same, smaller size
target_size = (800, 600)  # width, height
img1 = cv2.resize(img1, target_size)
img2 = cv2.resize(img2, target_size)
 
# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# --- ORB Detector ---
orb = cv2.ORB_create()
# Resize to same height (optional but helps with visualization)
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
kp1_orb, des1_orb = orb.detectAndCompute(gray1, None)
kp2_orb, des2_orb = orb.detectAndCompute(gray2, None)

# --- SIFT Detector ---
sift = cv2.SIFT_create()
kp1_sift, des1_sift = sift.detectAndCompute(gray1, None)
kp2_sift, des2_sift = sift.detectAndCompute(gray2, None)

# --- Matcher (BFMatcher for both) ---
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # For ORB
matches_orb = bf.match(des1_orb, des2_orb)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)

bf_sift = cv2.BFMatcher()  # For SIFT (uses L2 norm by default)
matches_sift = bf_sift.match(des1_sift, des2_sift)
matches_sift = sorted(matches_sift, key=lambda x: x.distance)

# Draw and show ORB matches
img_matches_orb = cv2.drawMatches(img1, kp1_orb, img2, kp2_orb, matches_orb[:15], None, flags=2)
cv2.imshow('ORB Matches', img_matches_orb)
cv2.imwrite('orb_matches.jpg', img_matches_orb)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Draw and show SIFT matches
img_matches_sift = cv2.drawMatches(img1, kp1_sift, img2, kp2_sift, matches_sift[:30], None, flags=2)
cv2.imshow('SIFT Matches', img_matches_sift)
cv2.imwrite('sift_matches.jpg', img_matches_sift)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save output
cv2.imwrite('orb_matches.jpg', img_matches_orb)
cv2.imwrite('sift_matches.jpg', img_matches_sift)
print("[INFO] Match visualizations saved as 'orb_matches.jpg' and 'sift_matches.jpg'")
