from modules import images
from modules.shared import opts

from PIL import Image, ImageOps, ImageChops

import mediapipe as mp
import numpy as np
import math
import cv2
import os

def getFacialLandmarks(image):
    height, width, _ = image.shape
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True,max_num_faces=4,min_detection_confidence=0.5) as face_mesh:
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(image_rgb)

        facelandmarks = []
        if result.multi_face_landmarks is not None:
            for facial_landmarks in result.multi_face_landmarks:
                landmarks = []
                for i in range(0, 468):
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * width)
                    y = int(pt1.y * height)
                    landmarks.append([x, y])
                    #cv2.circle(image, (x, y), 2, (100,100,0), -1)
                #cv2.imshow("Cropped", image)
                facelandmarks.append(np.array(landmarks, np.int32))

        return facelandmarks
    
def apply_overlay(image, paste_loc, imageOriginal, mask):
    x, y, w, h = paste_loc
    base_image = Image.new('RGBA', (imageOriginal.width, imageOriginal.height))
    image = images.resize_image(1, image, w, h)
    base_image.paste(image, (x, y))
    face = base_image
    new_mask = ImageChops.multiply(face.getchannel("A"), mask)
    face.putalpha(new_mask)

    imageOriginal = imageOriginal.convert('RGBA')
    image = Image.alpha_composite(imageOriginal, face)

    return image

def composite(image1, image2, mask, visualizationOpacity):
    mask_np = np.array(mask)
    mask_np = np.where(mask_np == 255, visualizationOpacity, 0)
    mask = Image.fromarray(mask_np).convert('L')
    image = image2.copy()
    image.paste(image1, None, mask)

    return image

def findBiggestFace(inputImage):
    # Store a copy of the input image:
    biggestFace = inputImage.copy()
    # Set initial values for the
    # largest contour:
    largestArea = 0
    largestContourIndex = 0

    # Find the contours on the binary image:
    contours, hierarchy = cv2.findContours(inputImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour in the contours list:
    for i, cc in enumerate(contours):
        # Find the area of the contour:
        area = cv2.contourArea(cc)
        # Store the index of the largest contour:
        if area > largestArea:
            largestArea = area
            largestContourIndex = i

    # Once we get the biggest face, paint it black:
    tempMat = inputImage.copy()
    cv2.drawContours(tempMat, contours, largestContourIndex, (0, 0, 0), -1, 8, hierarchy)
    # Erase smaller faces:
    biggestFace = biggestFace - tempMat

    return biggestFace

def maskResize(mask, maskSize, height):
    size = maskSize * (0.011*height)/5
    kernel = np.ones((int(math.ceil((0.011*height)+abs(size))),int(math.ceil((0.011*height)+abs(size)))),'uint8')
    if size > 0:
        mask = cv2.dilate(mask,kernel,iterations=1)
    else:
        mask = cv2.erode(mask,kernel,iterations=1)

    return mask

def listFiles(path, searchSubdir, allFiles):
    if searchSubdir:
        for root, _, files in os.walk(os.path.abspath(path)):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP')):
                    allFiles.append(os.path.join(root, file))
    else:
        allFiles = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP'))]

    return allFiles

