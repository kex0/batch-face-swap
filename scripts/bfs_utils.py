from modules import images
from modules.shared import opts

from PIL import Image, ImageOps, ImageChops

import mediapipe as mp
import numpy as np
import math
import cv2
import os

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

def custom_save_image(p, image, pathToSave, forced_filename, suffix, info):
    if pathToSave != "":
        if opts.samples_format == "png":
            images.save_image(image, pathToSave, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)
        elif image.mode != 'RGB':
            image = image.convert('RGB')
            images.save_image(image, pathToSave, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)
        else:
            images.save_image(image, pathToSave, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)

    elif pathToSave == "":
        if opts.samples_format == "png":
            images.save_image(image, opts.outdir_img2img_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)
        elif image.mode != 'RGB':
            image = image.convert('RGB')
            images.save_image(image, opts.outdir_img2img_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)
        else:
            images.save_image(image, opts.outdir_img2img_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)

def debugsave(image):
    images.save_image(image, os.getenv("AUTO1111_DEBUGDIR", "outputs"), "", "", "", "jpg", "", None)
