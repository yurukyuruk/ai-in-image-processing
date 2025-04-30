import cv2
import numpy as np
import os

# List of images with different formats
image_files = ['image1.jpg', 'image2.png', 'image3.tiff']

# Store loaded images
loaded_images = []

# Load images and check for successful loading
for file in image_files:
    if not os.path.exists(file):
        print(f"[ERROR] File not found: {file}")
        continue
    img = cv2.imread(file)
    if img is None:
        print(f"[ERROR] Failed to load image: {file}")
    else:
        print(f"[INFO] Loaded: {file} | Shape: {img.shape}")
        loaded_images.append((file, img))

# Process images
for file_name, image in loaded_images:
    # Display the original image in a resizable window
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Original Image', image)
    print(f"[INFO] Showing: {file_name} — Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Color Space Conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    cv2.imwrite(f"{file_name}_gray.png", gray)
    cv2.imwrite(f"{file_name}_hsv.png", hsv)
    print(f"[INFO] Saved grayscale and HSV versions of {file_name}")

    # Resize image
    resized = cv2.resize(image, (640, 480))
    cv2.imwrite(f"{file_name}_resized.png", resized)
    print(f"[INFO] Saved resized image of {file_name}")

    # Crop 100x100 ROI from center
    h, w = resized.shape[:2]
    center_y, center_x = h // 2, w // 2
    roi = resized[center_y-50:center_y+50, center_x-50:center_x+50]
    cv2.imwrite(f"{file_name}_roi.png", roi)
    print(f"[INFO] Saved 100x100 center ROI of {file_name}")

print("[DONE] All images processed.")
