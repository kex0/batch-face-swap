from modules import images, sd_samplers
from modules.shared import opts
from modules.processing import create_infotext

from PIL import Image, ImageOps, ImageChops

import traceback
import piexif
import json

import mediapipe as mp
import numpy as np
import math
import cv2
import sys
import os

def image_channels(image):
    return image.shape[2] if image.ndim == 3 else 1

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

def maskResize(mask, maskWidth, maskHeight):
    maskOriginal = mask
    try:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except IndexError:
        return mask
    x,y,w,h = cv2.boundingRect(contours[0])
    center_x = x + w // 2
    center_y = y + h // 2

    # Get the bounding box of the contour
    x,y,w,h = cv2.boundingRect(contours[0])

    # Crop the image
    cropped = maskOriginal[y:y+h,x:x+w]

    # Resize the cropped image by percentage
    height, width = cropped.shape[:2]
    new_height = max(int(height * maskWidth / 100), 1)
    new_width = max(int(width * maskHeight / 100), 1)
    resized = cv2.resize(cropped, (new_width, new_height))

    # Create black image with same resolution as original image
    black_image = Image.fromarray(np.zeros((maskOriginal.shape[0], maskOriginal.shape[1]), np.uint8))

    # Paste cropped image back to original image so that center of white part on new image matches center of white part on original image
    height, width = resized.shape[:2]
    x_offset = center_x - width // 2
    y_offset = center_y - height // 2
    black_image.paste(Image.fromarray(resized), (x_offset, y_offset))
    mask = np.array(black_image)

    return mask

def listFiles(input_path, searchSubdir, allFiles):
    try:
        if searchSubdir:
            for root, _, files in os.walk(os.path.abspath(input_path)):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP')):
                        allFiles.append(os.path.join(root, file))
        else:
            allFiles = [os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP'))]
    except FileNotFoundError:
        if input_path != "":
            print(f'Directory "{input_path}" not found!')

    return allFiles

def custom_save_image(p, image, output_path, forced_filename, suffix, info):
    if output_path != "":
        if opts.samples_format == "png":
            images.save_image(image, output_path, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)
        elif image.mode != 'RGB':
            image = image.convert('RGB')
            images.save_image(image, output_path, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)
        else:
            images.save_image(image, output_path, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)

    elif output_path == "":
        if opts.samples_format == "png":
            images.save_image(image, opts.outdir_img2img_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)
        elif image.mode != 'RGB':
            image = image.convert('RGB')
            images.save_image(image, opts.outdir_img2img_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)
        else:
            images.save_image(image, opts.outdir_img2img_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)

def debugsave(image):
    images.save_image(image, os.getenv("AUTO1111_DEBUGDIR", "outputs"), "", "", "", "jpg", "", None)

def read_info_from_image(image):
    items = image.info or {}

    geninfo = items.pop('parameters', None)

    if "exif" in items:
        exif = piexif.load(items["exif"])
        exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
        try:
            exif_comment = piexif.helper.UserComment.load(exif_comment)
        except ValueError:
            exif_comment = exif_comment.decode('utf8', errors="ignore")

        if exif_comment:
            items['exif comment'] = exif_comment
            geninfo = exif_comment

        for field in ['jfif', 'jfif_version', 'jfif_unit', 'jfif_density', 'dpi', 'exif',
                      'loop', 'background', 'timestamp', 'duration']:
            items.pop(field, None)

    if items.get("Software", None) == "NovelAI":
        try:
            json_info = json.loads(items["Comment"])
            sampler = sd_samplers.samplers_map.get(json_info["sampler"], "Euler a")

            geninfo = f"""{items["Description"]}
Negative prompt: {json_info["uc"]}
Steps: {json_info["steps"]}, Sampler: {sampler}, CFG scale: {json_info["scale"]}, Seed: {json_info["seed"]}, Size: {image.width}x{image.height}, Clip skip: 2, ENSD: 31337"""
        except Exception:
            print("Error parsing NovelAI image generation parameters:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

    return geninfo, items

def infotext(p, iteration=0, position_in_batch=0, comments={}):
        if p.all_prompts == None:
            p.all_prompts = [p.prompt]
        p.all_prompts = [p.prompt]
        if p.all_negative_prompts == None:
            p.all_negative_prompts = [p.negative_prompt]
        p.all_negative_prompts = [p.negative_prompt]
        if p.all_seeds == None:
            p.all_seeds = [p.seed]
        if p.all_subseeds == None:
            p.all_subseeds = [p.subseed]
        return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch)
