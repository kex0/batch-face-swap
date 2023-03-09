import sys
import os

cwd = os.getcwd()
utils_dir = os.path.join(cwd, 'extensions', 'batch-face-swap', 'scripts')
sys.path.extend([utils_dir])

from bfs_utils import *
from face_detect import *

import modules.scripts as scripts
import gradio as gr
import time

from modules import images, masking, generation_parameters_copypaste
from modules.processing import process_images, create_infotext, Processed
from modules.shared import opts, cmd_opts, state

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter, UnidentifiedImageError
import math

def findFaces(facecfg, image, width, height, divider, onlyHorizontal, onlyVertical, file, totalNumberOfFaces, singleMaskPerImage, countFaces, maskSize, skip):
    rejected = 0
    masks = []
    faces_info = []
    imageOriginal = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    heightOriginal = height
    widthOriginal = width

    # Calculate the size of each small image
    if onlyVertical == True:
        small_width = math.ceil(width / divider)
        small_height = height
    elif onlyHorizontal == True:
        small_width = width
        small_height = math.ceil(height / divider)
    else:
        small_width = math.ceil(width / divider)
        small_height = math.ceil(height / divider)


    # Divide the large image into a list of small images
    small_images = []
    for i in range(0, height, small_height):
        for j in range(0, width, small_width):
            small_images.append(image.crop((j, i, j + small_width, i + small_height)))

    # Process each small image
    processed_images = []
    facesInImage = 0

    for i, small_image in enumerate(small_images):
        small_image_index = i
        small_image = cv2.cvtColor(np.array(small_image), cv2.COLOR_RGB2BGR)

        faces = []

        if facecfg.faceMode != FaceMode.ORIGINAL:
            # use OpenCV2 multi-scale face detector to find all the faces

            known_face_rects = []

            # first find the faces the old way, since OpenCV is BAD at faces near the camera
            # save the convex hulls, but also getting bounding boxes so OpenCV can skip those
            landmarks = getFacialLandmarks(small_image, facecfg)
            for landmark in landmarks:
                face_info = {}
                convexhull = cv2.convexHull(landmark)
                faces.append(convexhull)
                bounds = cv2.boundingRect(convexhull)
                known_face_rects.append(list(bounds)) # convert tuple to array for consistency

                x_chin = landmark[152][0]
                y_chin = -landmark[152][1]
                x_forehead = landmark[10][0]
                y_forehead = -landmark[10][1]

                deltaX = x_forehead - x_chin
                deltaY = y_forehead - y_chin
                
                face_angle = math.atan2(deltaY, deltaX) * 180 / math.pi
                if onlyHorizontal == True:
                    x = (i // (divider) * small_width) + landmark[0][0]
                    y = (i % (divider) * small_height) + landmark[0][1]
                else:
                    x = (i % (divider) * small_width) + landmark[0][0]
                    y = (i // (divider) * small_height) + landmark[0][1]
                face_center = (x, y)
                face_info["angle"] = face_angle
                face_info["center"] = face_center
                faces_info.append(face_info)

            faceRects = getFaceRectangles(small_image, known_face_rects, facecfg)

            for rect in faceRects:
                landmarkHull, faces_info = getFacialLandmarkConvexHull(image, rect, faces_info, onlyHorizontal, divider, small_width, small_height, small_image_index, facecfg)
                if landmarkHull is not None:
                    faces.append(landmarkHull)
                else:
                    rejected += 1

            numberOfFaces = int(len(faces))
            totalNumberOfFaces += numberOfFaces
            if countFaces:
                continue

        else:
            landmarks = []
            landmarks = getFacialLandmarks(small_image, facecfg)
            numberOfFaces = int(len(landmarks))
            totalNumberOfFaces += numberOfFaces

            if countFaces:
                continue

            faces = []
            for landmark in landmarks:
                face_info = {}
                convexhull = cv2.convexHull(landmark)
                faces.append(convexhull)

                x_chin = landmark[152][0]
                y_chin = -landmark[152][1]
                x_forehead = landmark[10][0]
                y_forehead = -landmark[10][1]

                deltaX = x_forehead - x_chin
                deltaY = y_forehead - y_chin
                
                face_angle = math.atan2(deltaY, deltaX) * 180 / math.pi
                if onlyHorizontal == True:
                    x = (i // (divider) * small_width) + landmark[0][0]
                    y = (i % (divider) * small_height) + landmark[0][1]
                else:
                    x = (i % (divider) * small_width) + landmark[0][0]
                    y = (i // (divider) * small_height) + landmark[0][1]
                face_center = (x, y)
                face_info["angle"] = face_angle
                face_info["center"] = face_center
                faces_info.append(face_info)

        if len(faces) == 0:
            small_image[:] = (0, 0, 0)

        if numberOfFaces > 0:
            facesInImage += numberOfFaces
        if facesInImage == 0 and i == len(small_images) - 1:
            skip = 1

        mask = np.zeros((small_height, small_width), np.uint8)
        for i in range(len(faces)):
            small_image = cv2.fillConvexPoly(mask, faces[i], 255)
        processed_image = Image.fromarray(small_image)
        processed_images.append(processed_image)

    if countFaces:
        return totalNumberOfFaces

    if file != None:
        if FaceDetectDevelopment:
            print(f"Found {facesInImage} face(s) in {str(file)} (rejected {rejected} from OpenCV)")
        else:
            print(f"Found {facesInImage} face(s) in {str(file)}")
    # else:
    #     print(f"Found {facesInImage} face(s)")

    # Create a new image with the same size as the original large image
    new_image = Image.new('RGB', (width, height))

    # Paste the processed small images into the new image
    if onlyHorizontal == True:
        for i, processed_image in enumerate(processed_images):
            x = i // (divider) * small_width
            y = i % (divider) * small_height
            new_image.paste(processed_image, (x, y))
    else:
        for i, processed_image in enumerate(processed_images):
            x = i % (divider) * small_width
            y = i // (divider) * small_height
            new_image.paste(processed_image, (x, y))

    image = cv2.cvtColor(np.array(new_image), cv2.COLOR_RGB2BGR)
    imageOriginal[:] = (0, 0, 0)
    imageOriginal[0:heightOriginal, 0:widthOriginal] = image[0:height, 0:width]

    # convert to grayscale
    imageOriginal = cv2.cvtColor(imageOriginal, cv2.COLOR_RGB2GRAY)
    # convert grayscale to binary
    thresh = 100
    imageOriginal = cv2.threshold(imageOriginal,thresh,255,cv2.THRESH_BINARY)[1]
    binary_image = cv2.convertScaleAbs(imageOriginal)

    try:
        kernel = np.ones((int(math.ceil(0.011*height)),int(math.ceil(0.011*height))),'uint8')
        dilated = cv2.dilate(binary_image,kernel,iterations=1)
        kernel = np.ones((int(math.ceil(0.0045*height)),int(math.ceil(0.0025*height))),'uint8')
        dilated = cv2.dilate(dilated,kernel,iterations=1,anchor=(1, -1))
        kernel = np.ones((int(math.ceil(0.014*height)),int(math.ceil(0.0025*height))),'uint8')
        dilated = cv2.dilate(dilated,kernel,iterations=1,anchor=(-1, 1))
        mask = dilated
    except cv2.error:
        mask = dilated

    if maskSize != 0:
        mask = maskResize(mask, maskSize, height)

    if not singleMaskPerImage:
        if facesInImage > 1:
            segmentFaces = True
            while (segmentFaces):
                currentBiggest = findBiggestFace(mask)
                masks.append(currentBiggest)
                mask = mask - currentBiggest

                whitePixels = cv2.countNonZero(mask)
                whitePixelThreshold = 0.0005 * (widthOriginal * heightOriginal)
                if (whitePixels < whitePixelThreshold):
                    segmentFaces = False

            return masks, totalNumberOfFaces, faces_info, skip

    masks.append(mask)

    return masks, totalNumberOfFaces, faces_info, skip

# generate debug image
def faceDebug(p, masks, image, finishedImages, invertMask, forced_filename, pathToSave, info):
    generatedImages = []
    paste_to = []
    imageOriginal = image
    overlay_image = image
    print( f"here, {len(masks)}" )

    for n, mask in enumerate(masks):
        mask = Image.fromarray(masks[n])
        if invertMask:
            mask = ImageOps.invert(mask)

        image_masked = Image.new('RGBa', (image.width, image.height))
        image_masked.paste(overlay_image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert('L')))

        overlay_image = image_masked.convert('RGBA')

    debugsave(overlay_image)

def faceSwap(p, masks, image, finishedImages, invertMask, forced_filename, pathToSave, info, selectedTab, geninfo, faces_info, rotation_threshold):
    p.do_not_save_samples = True
    generatedImages = []
    paste_to = []
    imageOriginal = image
    overlay_image = image

    for n, mask in enumerate(masks):
        rotate = False
        mask = Image.fromarray(masks[n])
        if invertMask:
            mask = ImageOps.invert(mask)

        image_masked = Image.new('RGBa', (image.width, image.height))
        image_masked.paste(overlay_image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert('L')))

        overlay_image = image_masked.convert('RGBA')

        crop_region = masking.get_crop_region(np.array(mask), p.inpaint_full_res_padding)
        crop_region = masking.expand_crop_region(crop_region, p.width, p.height, mask.width, mask.height)
        x1, y1, x2, y2 = crop_region
        paste_to.append((x1, y1, x2-x1, y2-y1))

        for i in range(len(faces_info)):
            pixel_color = mask.getpixel((faces_info[i]["center"][0],faces_info[i]["center"][1]))
            if pixel_color == 255:
                index = i
                break

        mask = mask.crop(crop_region)
        image_mask = images.resize_image(2, mask, p.width, p.height)

        image = image.crop(crop_region)
        image = images.resize_image(2, image, p.width, p.height)
        image_cropped = image

        rotation_threshold = rotation_threshold
        if 90+rotation_threshold > faces_info[index]["angle"] and 90-rotation_threshold < faces_info[index]["angle"]:
            pass
        else:
            angle_difference = (90-int(faces_info[index]["angle"]) + 360) % 360
            image = image.rotate(angle_difference, expand=True)
            image_mask = image_mask.rotate(angle_difference, expand=True)
            rotate = True

        p.init_images = [image]
        p.image_mask = image_mask

        if geninfo != "":
            p.prompt = str(geninfo.get("Prompt"))
            p.negative_prompt = str(geninfo.get("Negative prompt"))
            p.sampler_name = str(geninfo.get("Sampler"))
            p.cfg_scale = float(geninfo.get("CFG scale"))
            p.width = int(geninfo.get("Size-1"))
            p.height = int(geninfo.get("Size-2"))

        proc = process_images(p)
        if rotate:
            for i in range(len(proc.images)):
                image_copy = image_cropped.copy()
                proc.images[i] = proc.images[i].rotate(int(faces_info[index]["angle"])-90)
                w1, h1 = image_cropped.size
                w2, h2 = proc.images[i].size
                x = (w1 - w2) // 2
                y = (h1 - h2) // 2
                image_copy.paste(proc.images[i], (x, y))
                proc.images[i] = image_copy
        generatedImages.append(proc.images)

        image = imageOriginal

    for j in range(p.n_iter * p.batch_size):
        image = imageOriginal
        for k in range(len(generatedImages)):
            mask = Image.fromarray(masks[k])
            mask = mask.filter(ImageFilter.GaussianBlur(p.mask_blur))
            image = apply_overlay(generatedImages[k][j], paste_to[k], image, mask)

        info = infotext(p)

        if pathToSave != "":
            if opts.samples_format == "png":
                images.save_image(image, pathToSave, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename+"_"+str(j+1) if forced_filename != None and (p.batch_size > 1 or p.n_iter > 1) else forced_filename)
            elif image.mode != 'RGB':
                image = image.convert('RGB')
                images.save_image(image, pathToSave, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename+"_"+str(j+1) if forced_filename != None and (p.batch_size > 1 or p.n_iter > 1) else forced_filename)
            else:
                images.save_image(image, pathToSave, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename+"_"+str(j+1) if forced_filename != None and (p.batch_size > 1 or p.n_iter > 1) else forced_filename)
        else:
            if opts.samples_format == "png":
                images.save_image(image, opts.outdir_img2img_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename+"_"+str(j+1) if forced_filename != None and (p.batch_size > 1 or p.n_iter > 1) else forced_filename)
            elif image.mode != 'RGB':
                image = image.convert('RGB')
                images.save_image(image, opts.outdir_img2img_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename+"_"+str(j+1) if forced_filename != None and (p.batch_size > 1 or p.n_iter > 1) else forced_filename)
            else:
                images.save_image(image, opts.outdir_img2img_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename+"_"+str(j+1) if forced_filename != None and (p.batch_size > 1 or p.n_iter > 1) else forced_filename)

        finishedImages.append(image)

    p.do_not_save_samples = False

    return finishedImages

def generateImages(p, facecfg, path, searchSubdir, viewResults, divider, howSplit, saveMask, pathToSave, onlyMask, saveNoFace, overrideDenoising, overrideMaskBlur, invertMask, singleMaskPerImage, countFaces, maskSize, keepOriginalName, pathExisting, pathMasksExisting, pathToSaveExisting, selectedTab, loadGenParams, rotation_threshold):
    suffix = ''
    info = infotext(p)
    if selectedTab == "generateMasksTab":
        wasCountFaces = False
        finishedImages = []
        totalNumberOfFaces = 0
        allFiles = []
        geninfo = ""

        if howSplit == "Horizontal only ▤":
            onlyHorizontal = True
            onlyVertical = False
        elif howSplit == "Vertical only ▥":
            onlyHorizontal = False
            onlyVertical = True
        elif howSplit == "Both ▦":
            onlyHorizontal = False
            onlyVertical = False

    # RUN IF PATH IS INSERTED
        if path != '':
            allFiles = listFiles(path, searchSubdir, allFiles)
            start_time = time.thread_time()

            if countFaces:
                print("\nCounting faces...")
                for i, file in enumerate(allFiles):
                    skip = 0
                    image = Image.open(file)
                    width, height = image.size
                    masks, totalNumberOfFaces, faces_info, skip = findFaces(facecfg, image, width, height, divider, onlyHorizontal, onlyVertical, file, totalNumberOfFaces, singleMaskPerImage, countFaces, maskSize, skip)

            if not onlyMask and countFaces:
                print(f"\nWill process {len(allFiles)} images, found {totalNumberOfFaces} faces, generating {p.n_iter * p.batch_size} new images for each.")
                state.job_count = totalNumberOfFaces * p.n_iter
            elif not onlyMask and not countFaces:
                print(f"\nWill process {len(allFiles)} images, generating {p.n_iter * p.batch_size} new images for each.")
                state.job_count = len(allFiles) * p.n_iter

            for i, file in enumerate(allFiles):
                if keepOriginalName:
                    forced_filename = os.path.splitext(os.path.basename(file))[0]
                else:
                    forced_filename = None

                if countFaces:
                    state.job = f"{i+1} out of {totalNumberOfFaces}"
                    totalNumberOfFaces = 0
                    wasCountFaces = True
                    countFaces = False
                else:
                    state.job = f"{i+1} out of {len(allFiles)}"

                if state.skipped:
                    state.skipped = False
                if state.interrupted and onlyMask:
                    state.interrupted = False
                elif state.interrupted:
                    break

                try:
                    image = Image.open(file)
                    width, height = image.size

                    if loadGenParams:
                        geninfo, _ = read_info_from_image(image)
                        geninfo = generation_parameters_copypaste.parse_generation_parameters(geninfo)

                except UnidentifiedImageError:
                    print(f"\nUnable to open {file}, skipping")
                    continue

                if not onlyMask:
                    if overrideDenoising == True:
                        p.denoising_strength = 0.5
                    if overrideMaskBlur == True:
                        p.mask_blur = int(math.ceil(0.01*height))

                skip = 0
                masks, totalNumberOfFaces, faces_info, skip = findFaces(facecfg, image, width, height, divider, onlyHorizontal, onlyVertical, file, totalNumberOfFaces, singleMaskPerImage, countFaces, maskSize, skip)

                if facecfg.debugSave:
                    faceDebug(p, masks, image, finishedImages, invertMask, forced_filename, pathToSave, info)

                # Only generate mask
                if onlyMask:
                    suffix = '_mask'

                    # Load mask
                    for i, mask in enumerate(masks):
                        mask = Image.fromarray(mask)

                        # Invert mask if needed
                        if invertMask:
                            mask = ImageOps.invert(mask)
                        finishedImages.append(mask)

                        if saveMask and skip != 1:
                            custom_save_image(p, mask, pathToSave, forced_filename, suffix, info)
                        elif saveMask and skip == 1 and saveNoFace:
                            custom_save_image(p, mask, pathToSave, forced_filename, suffix, info)

                # If face was not found but user wants to save images without face
                if skip == 1 and saveNoFace and not onlyMask:
                    custom_save_image(p, image, pathToSave, forced_filename, suffix, info)

                    finishedImages.append(image)
                    state.skipped = True
                    continue

                # If face was not found, just skip
                if skip == 1:
                    state.skipped = True
                    continue

                if not onlyMask:
                    finishedImages = faceSwap(p, masks, image, finishedImages, invertMask, forced_filename, pathToSave, info, selectedTab, geninfo, faces_info, rotation_threshold)

                if not viewResults:
                    finishedImages = []

            if wasCountFaces == True:
                countFaces = True

            timing = time.thread_time() - start_time
            print(f"Found {totalNumberOfFaces} faces in {len(allFiles)} images in {timing} seconds.")

    # RUN IF PATH IS NOT INSERTED AND IMAGE IS
        if path == '' and p.init_images[0] != None:
            forced_filename = None
            image = p.init_images[0]
            width, height = image.size

            if loadGenParams:
                geninfo, _ = read_info_from_image(image)
                geninfo = generation_parameters_copypaste.parse_generation_parameters(geninfo)

            if countFaces:
                print("\nCounting faces...")
                skip = 0
                masks, totalNumberOfFaces, faces_info, skip = findFaces(image, width, height, divider, onlyHorizontal, onlyVertical, None, totalNumberOfFaces, singleMaskPerImage, countFaces, maskSize, skip)

            if not onlyMask and countFaces:
                print(f"\nWill process {len(p.init_images)} images, found {totalNumberOfFaces} faces, generating {p.n_iter * p.batch_size} new images for each.")
                state.job_count = totalNumberOfFaces * p.n_iter
            elif not onlyMask and not countFaces:
                print(f"\nWill process {len(p.init_images)} images, creating {p.n_iter * p.batch_size} new images for each.")
                state.job_count = len(p.init_images) * p.n_iter

            if countFaces:
                state.job = f"{1} out of {totalNumberOfFaces}"
                totalNumberOfFaces = 0
                wasCountFaces = True
                countFaces = False
            else:
                state.job = f"{1} out of {len(p.init_images)}"

            if not onlyMask:
                if overrideDenoising == True:
                    p.denoising_strength = 0.5
                if overrideMaskBlur == True:
                    p.mask_blur = int(math.ceil(0.01*height))

            skip = 0
            masks, totalNumberOfFaces, faces_info, skip = findFaces(facecfg, image, width, height, divider, onlyHorizontal, onlyVertical, None, totalNumberOfFaces, singleMaskPerImage, countFaces, maskSize, skip)
            if facecfg.debugSave:
                faceDebug(p, masks, image, finishedImages, invertMask, forced_filename, pathToSave, info)

            # Only generate mask
            if onlyMask:

                suffix = '_mask'

                # Load mask
                for i, mask in enumerate(masks):
                    mask = Image.fromarray(mask)

                    # Invert mask if needed
                    if invertMask:
                        mask = ImageOps.invert(mask)
                    finishedImages.append(mask)

                    if saveMask and skip != 1:
                        custom_save_image(p, mask, pathToSave, forced_filename, suffix, info)
                    elif saveMask and skip == 1 and saveNoFace:
                        custom_save_image(p, mask, pathToSave, forced_filename, suffix, info)


            # If face was not found but user wants to save images without face
            if skip == 1 and saveNoFace and not onlyMask:
                custom_save_image(p, image, pathToSave, forced_filename, suffix, info)

                finishedImages.append(image)
                state.skipped = True

            # If face was not found, just skip
            if skip == 1:
                state.skipped = True

            if not onlyMask:
                finishedImages = faceSwap(p, masks, image, finishedImages, invertMask, forced_filename, pathToSave, info, selectedTab, geninfo, faces_info, rotation_threshold)

            if wasCountFaces == True:
                countFaces = True

            print(f"Found {totalNumberOfFaces} faces in {len(p.init_images)} images.")

# EXISTING MASKS
    elif selectedTab == "existingMasksTab":
        finishedImages = []
        allImages = []
        allMasks = []
        searchSubdir = False

        if pathExisting != '' and pathMasksExisting != '':
            allImages = listFiles(pathExisting, searchSubdir, allImages)
            allMasks = listFiles(pathMasksExisting, searchSubdir, allMasks)

            print(f"\nWill process {len(allImages)} images, generating {p.n_iter * p.batch_size} new images for each.")
            state.job_count = len(allImages) * p.n_iter
            for i, file in enumerate(allImages):
                forced_filename = os.path.splitext(os.path.basename(file))[0]

                state.job = f"{i+1} out of {len(allImages)}"

                if state.skipped:
                    state.skipped = False
                elif state.interrupted:
                    break

                try:
                    image = Image.open(file)
                    width, height = image.size

                    masks = []
                    masks.append(Image.open(os.path.join(pathMasksExisting, os.path.splitext(os.path.basename(file))[0])+os.path.splitext(allMasks[i])[1]))
                except UnidentifiedImageError:
                    print(f"\nUnable to open {file}, skipping")
                    continue

                if overrideDenoising == True:
                    p.denoising_strength = 0.5
                if overrideMaskBlur == True:
                    p.mask_blur = int(math.ceil(0.01*height))

                finishedImages = faceSwap(p, masks, image, finishedImages, invertMask, forced_filename, pathToSaveExisting, info, selectedTab, faces_info, rotation_threshold)

                if not viewResults:
                    finishedImages = []

    return finishedImages

class Script(scripts.Script):
    def title(self):
        return "Batch Face Swap"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):

        def updateVisualizer(searchSubdir: bool, howSplit: str, divider: int, maskSize: int, path: str, visualizationOpacity: int, faceMode: int):
            # this is a huge pain to patch through so don't bother for now

            facecfg = FaceDetectConfig(faceMode)
            allFiles = []
            totalNumberOfFaces = 0

            if path != '':
                allFiles = listFiles(path, searchSubdir, allFiles)

            if len(allFiles) > 0:
                imgPath = allFiles[0]
                try:
                    image = Image.open(imgPath)
                    maxsize = (1000, 500)
                    image.thumbnail(maxsize,Image.ANTIALIAS)
                except UnidentifiedImageError:
                    allFiles = []

            visualizationOpacity = (visualizationOpacity/100)*255
            color = "white"
            thickness = 5

            if "Both" in howSplit:
                onlyHorizontal = False
                onlyVertical = False
                if len(allFiles) == 0:
                    image = Image.open("./extensions/batch-face-swap/images/exampleB.jpg")
                width, height = image.size

                # if len(masks)==0 and path != '':
                masks, totalNumberOfFaces, faces_info, skip = findFaces(facecfg, image, width, height, divider, onlyHorizontal, onlyVertical, file=None, totalNumberOfFaces=totalNumberOfFaces, singleMaskPerImage=True, countFaces=False, maskSize=maskSize, skip=0)

                mask = masks[0]

                mask = maskResize(mask, maskSize, height)

                mask = Image.fromarray(mask)
                redImage = Image.new("RGB", (width, height), (255, 0, 0))

                mask.convert("L")
                draw = ImageDraw.Draw(mask, "L")

                if divider > 1:
                    for i in range(divider-1):
                        start_point = (0, int((height/divider)*(i+1)))
                        end_point = (int(width), int((height/divider)*(i+1)))
                        draw.line([start_point, end_point], fill=color, width=thickness)

                    for i in range(divider-1):
                        start_point = (int((width/divider)*(i+1)), 0)
                        end_point = (int((width/divider)*(i+1)), int(height))
                        draw.line([start_point, end_point], fill=color, width=thickness)

                image = composite(redImage, image, mask, visualizationOpacity)

            elif "Vertical" in howSplit:
                onlyHorizontal = False
                onlyVertical = True
                if len(allFiles) == 0:
                    image = Image.open("./extensions/batch-face-swap/images/exampleV.jpg")
                    # mask = Image.open("./extensions/batch-face-swap/images/exampleV_mask.jpg")
                    # mask = np.array(mask)
                width, height = image.size

                # if len(masks)==0 and path != '':
                masks, totalNumberOfFaces, faces_info, skip = findFaces(facecfg, image, width, height, divider, onlyHorizontal, onlyVertical, file=None, totalNumberOfFaces=totalNumberOfFaces, singleMaskPerImage=True, countFaces=False, maskSize=maskSize, skip=0)

                mask = masks[0]

                mask = maskResize(mask, maskSize, height)

                mask = Image.fromarray(mask)
                redImage = Image.new("RGB", (width, height), (255, 0, 0))

                mask.convert("L")
                draw = ImageDraw.Draw(mask, "L")

                if divider > 1:
                    for i in range(divider-1):
                        start_point = (int((width/divider)*(i+1)), 0)
                        end_point = (int((width/divider)*(i+1)), int(height))
                        draw.line([start_point, end_point], fill=color, width=thickness)

                image = composite(redImage, image, mask, visualizationOpacity)

            else:
                onlyHorizontal = True
                onlyVertical = False
                if len(allFiles) == 0:
                    image = Image.open("./extensions/batch-face-swap/images/exampleH.jpg")
                width, height = image.size

                # if len(masks)==0 and path != '':
                masks, totalNumberOfFaces, faces_info, skip = findFaces(facecfg, image, width, height, divider, onlyHorizontal, onlyVertical, file=None, totalNumberOfFaces=totalNumberOfFaces, singleMaskPerImage=True, countFaces=False, maskSize=maskSize, skip=0)

                mask = masks[0]

                mask = maskResize(mask, maskSize, height)

                mask = Image.fromarray(mask)
                redImage = Image.new("RGB", (width, height), (255, 0, 0))

                mask.convert("L")
                draw = ImageDraw.Draw(mask, "L")

                if divider > 1:
                    for i in range(divider-1):
                        start_point = (0, int((height/divider)*(i+1)))
                        end_point = (int(width), int((height/divider)*(i+1)))
                        draw.line([start_point, end_point], fill=color, width=thickness)

                image = composite(redImage, image, mask, visualizationOpacity)

            update = gr.Image.update(value=image)
            return update

        def switchSaveMaskInteractivity(onlyMask: bool):
            return gr.Checkbox.update(interactive=bool(onlyMask))
        def switchSaveMask(onlyMask: bool):
            if onlyMask == False:
                return gr.Checkbox.update(value=bool(onlyMask))
        def switchTipsVisibility(showTips: bool):
            return gr.HTML.update(visible=bool(showTips))
        def switchInvertMask(invertMask: bool):
            return gr.Checkbox.update(value=bool(invertMask))

        with gr.Column(variant='panel'):
            gr.HTML("<p style=\"margin-bottom:0.75em;margin-top:0.75em;font-size:1.5em;color:red\">Make sure you're in the \"Inpaint upload\" tab!</p>") 
        
        with gr.Box():
            # Overrides
            with gr.Column(variant='panel'):
                gr.HTML("<p style=\"margin-top:0.75em;font-size:1.25em\">Overrides:</p>")
                with gr.Row():
                    overrideDenoising = gr.Checkbox(value=True, label="""Override "Denoising strength" to 0.5""")
                    overrideMaskBlur = gr.Checkbox(value=True, label="""Override "Mask blur" to automatic""")

        with gr.Column(variant='panel'):
            with gr.Tab("Generate masks") as generateMasksTab:
                # Face detection
                with gr.Column(variant='compact'):
                    gr.HTML("<p style=\"margin-top:0.10em;margin-bottom:0.75em;font-size:1.5em\"><strong>Face detection:</strong></p>")
                    with gr.Row():
                        faceDetectMode = gr.Dropdown(label="Detector", choices=face_mode_names, value=face_mode_names[FaceMode.DEFAULT], type="index", elem_id=self.elem_id("z_type"))
                        minFace = gr.Slider(minimum=10, maximum=200, step=1  , value=30, label="Minimum face size in pixels")

                with gr.Column(variant='panel'):
                    htmlTip1 = gr.HTML("<p>Activate the 'Masks only' checkbox to see how many faces do your current settings detect without generating SD image. (check console)</p><p>You can also save generated masks to disk. Only possible with 'Masks only' (if you leave path empty, it will save the masks to your default webui outputs directory)</p><p>'Single mask per image' is only recommended with 'Invert mask' or if you want to save one mask per image, not per face. If you activate it without inverting mask, and try to process an image with multiple faces, it will generate only one image for all faces, producing bad results.</p><p>'Rotation threshold', if the face is rotated at an angle higher than this value, it will be automatically rotated so it's upright before generating, producing much better results.</p>",visible=False)
                    # Settings
                    with gr.Column(variant='panel'):
                        gr.HTML("<p style=\"margin-top:0.10em;font-size:1.5em\">Settings:</p>")
                        with gr.Column(variant='compact'):
                            with gr.Row():
                                onlyMask = gr.Checkbox(value=False, label="Masks only", visible=True)
                                saveMask = gr.Checkbox(value=False, label="Save masks to disk", interactive=False)
                            with gr.Row():
                                invertMask = gr.Checkbox(value=False, label="Invert mask", visible=True)
                                singleMaskPerImage = gr.Checkbox(value=False, label="Single mask per image", visible=True)
                            with gr.Row(variant='panel'):
                                rotation_threshold = gr.Slider(minimum=0, maximum=180, step=1, value=20, label="Rotation threshold")

                # Path to images
                with gr.Column(variant='panel'):
                    gr.HTML("<p style=\"margin-top:0.10em;font-size:1.5em\"><strong>Path to images:</strong></p>")
                    with gr.Column(variant='panel'):
                        htmlTip2 = gr.HTML("<p>'Load from subdirectories' will include all images in all subdirectories.</p>",visible=False)
                        with gr.Row():
                            path = gr.Textbox(label="Images directory",placeholder=r"C:\Users\dude\Desktop\images")
                            pathToSave = gr.Textbox(label="Output directory (OPTIONAL)",placeholder=r"Leave empty to save to default directory")
                        searchSubdir = gr.Checkbox(value=False, label="Load from subdirectories")
                        keepOriginalName = gr.Checkbox(value=False, label="Keep original file name (OVERWRITES FILES WITH THE SAME NAME)")
                        loadGenParams = gr.Checkbox(value=False, label="Load generation parameters from images")

                # Image splitter
                with gr.Column(variant='panel'):
                    gr.HTML("<p style=\"margin-top:0.10em;font-size:1.5em\"><strong>Image splitter:</strong></p>")
                    with gr.Column(variant='panel'):
                        htmlTip3 = gr.HTML("<p>This divides image to smaller images and tries to find a face in the individual smaller images.</p><p>Useful when faces are small in relation to the size of the whole picture and are not being detected.</p><p>(may result in mask that only covers a part of a face or no detection if the division goes right through the face)</p><p>Open 'Split visualizer' to see how it works.</p>",visible=False)
                        with gr.Row():
                            divider = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="How many images to divide into")
                            maskSize = gr.Slider(minimum=-10, maximum=10, step=1, value=0, label="Mask size")
                        howSplit = gr.Radio(["Horizontal only ▤", "Vertical only ▥", "Both ▦"], value = "Both ▦", label = "How to divide")
                        with gr.Accordion(label="Visualizer", open=False):
                            exampleImage = gr.Image(value=Image.open("./extensions/batch-face-swap/images/exampleB.jpg"), label="Split visualizer", show_label=False, type="pil", visible=True).style(height=500)
                            with gr.Row(variant='compact'):
                                with gr.Column(variant='panel'):
                                    gr.HTML("", visible=False)
                                with gr.Column(variant='compact'):
                                    visualizationOpacity = gr.Slider(minimum=0, maximum=100, step=1, value=75, label="Opacity")

                # Other
                with gr.Column(variant='panel'):
                    gr.HTML("<p style=\"margin-top:0.10em;font-size:1.5em\">Other:</p>")
                    with gr.Column(variant='panel'):
                        htmlTip4 = gr.HTML("<p>'Count faces before generating' is required to see accurate progress bar (not recommended when processing a large number of images). Because without knowing the number of faces, the webui can't know how many images it will generate. Activating it means you will search for faces twice.</p>",visible=False)
                        saveNoFace = gr.Checkbox(value=True, label="Save image even if face was not found")
                        countFaces = gr.Checkbox(value=False, label="Count faces before generating (accurate progress bar but NOT recommended)")

            with gr.Tab("Existing masks",) as existingMasksTab:
                with gr.Column(variant='panel'):
                    htmlTip5 = gr.HTML("<p style=\"margin-bottom:0.75em\">Image name and it's corresponding mask must have exactly the same name (if image is called `abc.jpg` then it's mask must also be called `abc.jpg`)</p>",visible=False)
                    pathExisting = gr.Textbox(label="Images directory",placeholder=r"C:\Users\dude\Desktop\images")
                    pathMasksExisting = gr.Textbox(label="Masks directory",placeholder=r"C:\Users\dude\Desktop\masks")
                    pathToSaveExisting = gr.Textbox(label="Output directory (OPTIONAL)",placeholder=r"Leave empty to save to default directory")

        # General
        with gr.Box():
            with gr.Column(variant='panel'):
                gr.HTML("<p style=\"margin-top:0.10em;font-size:1.5em\">General:</p>")
                htmlTip6 = gr.HTML("<p>Activate 'Show results in WebUI' checkbox to see results in the WebUI at the end (not recommended when processing a large number of images)</p>",visible=False)
                with gr.Row():
                    viewResults = gr.Checkbox(value=True, label="Show results in WebUI")
                    showTips = gr.Checkbox(value=False, label="Show tips")

        # Face detect internals
        with gr.Column(variant='panel', visible = FaceDetectDevelopment):
            gr.HTML("<p style=\"margin-top:0.75em;margin-bottom:0.5em;font-size:1.5em\"><strong>Debug internal config:</strong></p>")
            with gr.Column(variant='panel'):
                with gr.Row():
                    debugSave     = gr.Checkbox(value=False, label="Save debug images")
                    optimizeDetect= gr.Checkbox(value=True, label="Used optimized detector")
                face_x_scale = gr.Slider(minimum=1 , maximum=  6, step=0.1, value=4, label="Face x-scaleX")
                face_y_scale = gr.Slider(minimum=1 , maximum=  6, step=0.1, value=2.5, label="Face y-scaleX")

                multiScale   = gr.Slider(minimum=1.0, maximum=200, step=0.001, value=1.03, label="Multiscale search stepsizess")
                multiScale2  = gr.Slider(minimum=0.8, maximum=200, step=0.001, value=1.0 , label="Multiscale search secondary scalar")
                multiScale3  = gr.Slider(minimum=0.8, maximum=2.0, step=0.001, value=1.0 , label="Multiscale search tertiary scale")

                minNeighbors = gr.Slider(minimum=1 , maximum = 10, step=1  , value=5, label="minNeighbors")
                mpconfidence = gr.Slider(minimum=0.01, maximum = 2.0, step=0.01, value=0.5, label="FaceMesh confidence threshold")
                mpcount      = gr.Slider(minimum=1, maximum = 20, step=1, value=5, label="FaceMesh maximum faces")

        selectedTab = gr.Textbox(value="generateMasksTab", visible=False)
        generateMasksTab.select(lambda: "generateMasksTab", inputs=None, outputs=selectedTab)
        existingMasksTab.select(lambda: "existingMasksTab", inputs=None, outputs=selectedTab)

        faceDetectMode.change(fn=None, _js="gradioApp().getElementById('mode_img2img').querySelectorAll('button')[4].click()", inputs=None, outputs=None)
        minFace.change(fn=None, _js="gradioApp().getElementById('mode_img2img').querySelectorAll('button')[4].click()", inputs=None, outputs=None)
        
        pathExisting.change(fn=None, _js="gradioApp().getElementById('mode_img2img').querySelectorAll('button')[4].click()", inputs=None, outputs=None)
        pathMasksExisting.change(fn=None, _js="gradioApp().getElementById('mode_img2img').querySelectorAll('button')[4].click()", inputs=None, outputs=None)
        pathToSaveExisting.change(fn=None, _js="gradioApp().getElementById('mode_img2img').querySelectorAll('button')[4].click()", inputs=None, outputs=None)

        path.change(fn=None, _js="gradioApp().getElementById('mode_img2img').querySelectorAll('button')[4].click()", inputs=None, outputs=None)
        pathToSave.change(fn=None, _js="gradioApp().getElementById('mode_img2img').querySelectorAll('button')[4].click()", inputs=None, outputs=None)
        onlyMask.change(switchSaveMaskInteractivity, onlyMask, saveMask)
        onlyMask.change(switchSaveMask, onlyMask, saveMask)
        invertMask.change(switchInvertMask, invertMask, singleMaskPerImage)

        faceDetectMode.change(updateVisualizer, [searchSubdir, howSplit, divider, maskSize, path, visualizationOpacity, faceDetectMode], exampleImage)
        minFace.change(updateVisualizer, [searchSubdir, howSplit, divider, maskSize, path, visualizationOpacity, faceDetectMode], exampleImage)
        visualizationOpacity.change(updateVisualizer, [searchSubdir, howSplit, divider, maskSize, path, visualizationOpacity, faceDetectMode], exampleImage)
        searchSubdir.change(updateVisualizer, [searchSubdir, howSplit, divider, maskSize, path, visualizationOpacity, faceDetectMode], exampleImage)
        howSplit.change(updateVisualizer, [searchSubdir, howSplit, divider, maskSize, path, visualizationOpacity, faceDetectMode], exampleImage)
        divider.change(updateVisualizer, [searchSubdir, howSplit, divider, maskSize, path, visualizationOpacity, faceDetectMode], exampleImage)
        maskSize.change(updateVisualizer, [searchSubdir, howSplit, divider, maskSize, path, visualizationOpacity, faceDetectMode], exampleImage)
        path.change(updateVisualizer, [searchSubdir, howSplit, divider, maskSize, path, visualizationOpacity, faceDetectMode], exampleImage)

        showTips.change(switchTipsVisibility, showTips, htmlTip1)
        showTips.change(switchTipsVisibility, showTips, htmlTip2)
        showTips.change(switchTipsVisibility, showTips, htmlTip3)
        showTips.change(switchTipsVisibility, showTips, htmlTip4)
        showTips.change(switchTipsVisibility, showTips, htmlTip5)
        showTips.change(switchTipsVisibility, showTips, htmlTip6)

        return [overrideDenoising, overrideMaskBlur, path, searchSubdir, divider, howSplit, saveMask, pathToSave, viewResults, saveNoFace, onlyMask, invertMask, singleMaskPerImage, countFaces, maskSize, keepOriginalName, pathExisting, pathMasksExisting, pathToSaveExisting, selectedTab, faceDetectMode, face_x_scale, face_y_scale, minFace, multiScale, multiScale2, multiScale3, minNeighbors, mpconfidence, mpcount, debugSave, optimizeDetect, loadGenParams, rotation_threshold]

    def run(self, p, overrideDenoising, overrideMaskBlur, path, searchSubdir, divider, howSplit, saveMask, pathToSave, viewResults, saveNoFace, onlyMask, invertMask, singleMaskPerImage, countFaces, maskSize, keepOriginalName, pathExisting, pathMasksExisting, pathToSaveExisting, selectedTab, faceDetectMode, face_x_scale, face_y_scale, minFace, multiScale, multiScale2, multiScale3, minNeighbors, mpconfidence, mpcount, debugSave, optimizeDetect, loadGenParams, rotation_threshold):
        wasGrid = p.do_not_save_grid
        wasInpaintFullRes = p.inpaint_full_res

        p.inpaint_full_res = 1
        p.do_not_save_grid = True

        all_images = []

        facecfg = FaceDetectConfig(faceDetectMode, face_x_scale, face_y_scale, minFace, multiScale, multiScale2, multiScale3, minNeighbors, mpconfidence, mpcount, debugSave, optimizeDetect)
        finishedImages = generateImages(p, facecfg, path, searchSubdir, viewResults, int(divider), howSplit, saveMask, pathToSave, onlyMask, saveNoFace, overrideDenoising, overrideMaskBlur, invertMask, singleMaskPerImage, countFaces, maskSize, keepOriginalName, pathExisting, pathMasksExisting, pathToSaveExisting, selectedTab, loadGenParams, rotation_threshold)

        if not viewResults:
            finishedImages = []

        all_images += finishedImages
        proc = Processed(p, all_images)

        p.do_not_save_grid = wasGrid
        p.inpaint_full_res = wasInpaintFullRes

        return proc
