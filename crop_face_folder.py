#!/usr/bin/env python3

import cv2
import os
import sys
import argparse
from PIL import Image

def detect_and_crop_face(input_image_path, output_image_path, crop_size):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Unable to read image {input_image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        center_x, center_y = x + w // 2, y + h // 2
        
        # Calculate crop coordinates
        left = max(0, center_x - crop_size // 2)
        top = max(0, center_y - crop_size // 2)
        right = min(img.shape[1], left + crop_size)
        bottom = min(img.shape[0], top + crop_size)

        # Adjust crop area if it goes out of bounds
        if right - left < crop_size:
            left = max(0, right - crop_size)
        if bottom - top < crop_size:
            top = max(0, bottom - crop_size)

        cropped_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cropped_img = cropped_img.crop((left, top, left + crop_size, top + crop_size))
        cropped_img.save(output_image_path)
        print(f"Face detected and cropped: {output_image_path}")
    else:
        print(f"No face detected in {input_image_path}")

def process_folder(input_folder, output_folder, crop_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(image_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"cropped_{filename}")
            detect_and_crop_face(input_path, output_path, crop_size)

def main():
    parser = argparse.ArgumentParser(description="Crop faces in images to squares of specified size.")
    parser.add_argument("input_folder", help="Path to the folder containing input images")
    parser.add_argument("output_folder", help="Path to the folder where cropped images will be saved")
    parser.add_argument("-s", "--size", type=int, default=256, help="Size of the square crop in pixels (default: 256)")
    
    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder, args.size)

if __name__ == "__main__":
    main()