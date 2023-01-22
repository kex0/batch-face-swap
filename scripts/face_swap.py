import modules.scripts as scripts
import gradio as gr
import os

from modules import images, masking
from modules.processing import process_images, create_infotext, Processed
from modules.shared import opts, cmd_opts, state

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import math

def apply_overlay(image, paste_loc, overlay):
    if paste_loc is not None:
        x, y, w, h = paste_loc
        base_image = Image.new('RGBA', (overlay.width, overlay.height))
        image = images.resize_image(1, image, w, h)
        base_image.paste(image, (x, y))
        image = base_image

    image = image.convert('RGBA')
    image.alpha_composite(overlay)

    return image

def composite(image1, image2, mask, visualizationOpacity):
    mask_np = np.array(mask)
    mask_np = np.where(mask_np == 255, visualizationOpacity, 0)
    mask = Image.fromarray(mask_np).convert('L')
    image = image2.copy()
    image.paste(image1, None, mask)
    return image

def findBiggestBlob(inputImage):
    # Store a copy of the input image:
    biggestBlob = inputImage.copy()
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

    # Once we get the biggest blob, paint it black:
    tempMat = inputImage.copy()
    cv2.drawContours(tempMat, contours, largestContourIndex, (0, 0, 0), -1, 8, hierarchy)
    # Erase smaller blobs:
    biggestBlob = biggestBlob - tempMat

    return biggestBlob

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

def findFaceDivide(image, width, height, divider, onlyHorizontal, onlyVertical, file, totalNumberOfFaces, singleMaskPerImage, countFaces, maskSize, skip):
    masks = []
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
        small_image = cv2.cvtColor(np.array(small_image), cv2.COLOR_RGB2BGR)
        landmarks = []
        landmarks = getFacialLandmarks(small_image)
        numberOfFaces = int(len(landmarks))
        totalNumberOfFaces += numberOfFaces
        if countFaces:
            return totalNumberOfFaces

        faces = []
        for landmark in landmarks:
            convexhull = cv2.convexHull(landmark)
            faces.append(convexhull)

        if len(landmarks) == 0:
            small_image[:] = (0, 0, 0)

        if numberOfFaces > 0:
            facesInImage += numberOfFaces
        if facesInImage == 0 and i == len(small_images) - 1:
            skip = 1

        mask = np.zeros((small_height, small_width), np.uint8)
        for i in range(len(landmarks)):
            small_image = cv2.fillConvexPoly(mask, faces[i], 255)
        processed_image = Image.fromarray(small_image)
        processed_images.append(processed_image)

    if file != None:
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

    # define kernel
    kernel = np.ones((int(math.ceil(0.011*height)),int(math.ceil(0.011*height))),'uint8')
    dilated = cv2.dilate(binary_image,kernel,iterations=1)
    kernel = np.ones((int(math.ceil(0.0045*height)),int(math.ceil(0.0025*height))),'uint8')
    dilated = cv2.dilate(dilated,kernel,iterations=1,anchor=(1, -1))
    kernel = np.ones((int(math.ceil(0.014*height)),int(math.ceil(0.0025*height))),'uint8')
    dilated = cv2.dilate(dilated,kernel,iterations=1,anchor=(-1, 1))
    mask = dilated

    if maskSize != 0:
        size = maskSize * (0.011*height)/5
        kernel = np.ones((int(math.ceil((0.011*height)+abs(size))),int(math.ceil((0.011*height)+abs(size)))),'uint8')
        if size > 0:
            mask = cv2.dilate(mask,kernel,iterations=1)
        else:
            mask = cv2.erode(mask,kernel,iterations=1)

    if not singleMaskPerImage:
        if facesInImage > 1:
            segmentFaces = True
            while (segmentFaces):
                currentBiggest = findBiggestBlob(mask)
                masks.append(currentBiggest)
                mask = mask - currentBiggest

                whitePixels = cv2.countNonZero(mask)
                whitePixelThreshold = 0.0005 * (widthOriginal * heightOriginal)
                if (whitePixels < whitePixelThreshold):
                    segmentFaces = False
            return masks, totalNumberOfFaces, skip

    masks.append(mask)

    return masks, totalNumberOfFaces, skip
     

def generateMasks(p, path, searchSubdir, divider, howSplit, saveMask, pathToSave, onlyMask, saveNoFace, overrideDenoising, overrideMaskBlur, invertMask, singleMaskPerImage, countFaces, maskSize):

    def infotext(iteration=0, position_in_batch=0):
        if p.all_prompts == None:
            p.all_prompts = [p.prompt]
        if p.all_negative_prompts == None:
            p.all_negative_prompts = [p.negative_prompt]
        if p.all_seeds == None:
            p.all_seeds = [p.seed]
        if p.all_subseeds == None:
            p.all_subseeds = [p.subseed]
        return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch)

    wasCountFaces = False
    finishedImages = []
    comments = {}
    totalNumberOfFaces = 0
    allFiles = []

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
        if searchSubdir:
            for root, _, files in os.walk(os.path.abspath(path)):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP')):
                        allFiles.append(os.path.join(root, file))
    
        else:
            allFiles = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP'))]

        if countFaces:
            print("\nCounting faces...")
            for i, file in enumerate(allFiles):
                skip = 0
                image = Image.open(file)
                width, height = image.size
                totalNumberOfFaces = findFaceDivide(image, width, height, divider, onlyHorizontal, onlyVertical, file, totalNumberOfFaces, singleMaskPerImage, countFaces, maskSize, skip)

                    
        if not onlyMask and countFaces:
            print(f"\nWill process {len(allFiles)} images, found {totalNumberOfFaces} faces, generating {p.n_iter * p.batch_size} new images for each.")
            state.job_count = totalNumberOfFaces * p.n_iter  
        elif not onlyMask and not countFaces:
            print(f"\nWill process {len(allFiles)} images, generating {p.n_iter * p.batch_size} new images for each.")
            state.job_count = len(allFiles) * p.n_iter

        for i, file in enumerate(allFiles):
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

            image = Image.open(file)
            width, height = image.size
            
            if not onlyMask:
                if overrideDenoising == True:
                    p.denoising_strength = 0.5
                if overrideMaskBlur == True:
                    p.mask_blur = int(math.ceil(0.01*height))

            skip = 0
            masks, totalNumberOfFaces, skip = findFaceDivide(image, width, height, divider, onlyHorizontal, onlyVertical, file, totalNumberOfFaces, singleMaskPerImage, countFaces, maskSize, skip)                   
        
            if onlyMask:
                suffix = '_mask'
                if pathToSave != "":
                    for i, mask in enumerate(masks):
                        mask = Image.fromarray(mask)
                        if invertMask:
                            mask = ImageOps.invert(mask)
                        finishedImages.append(mask)
                        if saveMask == True:
                            images.save_image(mask, pathToSave, "", p.seed, p.prompt, opts.samples_format, info=infotext(), p=p, suffix=suffix)
                    
                elif pathToSave == "":
                    for i, mask in enumerate(masks):
                        mask = Image.fromarray(mask)
                        if invertMask:
                            mask = ImageOps.invert(mask)
                        finishedImages.append(mask)
                        if saveMask == True:
                            images.save_image(mask, opts.outdir_img2img_samples, "", p.seed, p.prompt, opts.samples_format, info=infotext(), p=p, suffix=suffix)

            if skip == 1 and saveNoFace and not onlyMask:
                images.save_image(image, p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, info=infotext(), p=p)
                finishedImages.append(image)
                state.skipped = True
                continue
            
            if skip == 1:
                state.skipped = True
                continue

            if not onlyMask:
                # Single face
                if len(masks) == 1:
                    mask = Image.fromarray(masks[0])
                    if invertMask:
                            mask = ImageOps.invert(mask)

                    p.init_images = [image]
                    p.image_mask = mask

                    proc = process_images(p)

                    for n in range(p.batch_size):
                        finishedImages.append(proc.images[n])
                else:
                    # Multi face
                    generatedImages = []
                    paste_to = []
                    imageOriginal = image
                    overlay_image = image
                    p.do_not_save_samples = True

                    for n, mask in enumerate(masks):
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

                        mask = mask.crop(crop_region)
                        image_mask = images.resize_image(2, mask, p.width, p.height)

                        
                        image = image.crop(crop_region)
                        image = images.resize_image(2, image, p.width, p.height)

                        p.init_images = [image]
                        p.image_mask = image_mask
                        proc = process_images(p)
                        generatedImages.append(proc.images)

                        image = imageOriginal

                    for j in range(p.batch_size):
                        image = overlay_image
                            
                        for k in range(len(generatedImages)):
                            image = apply_overlay(generatedImages[k][j], paste_to[k], image)
                        images.save_image(image, p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, info=infotext(), p=p)
                        finishedImages.append(image)

                    p.do_not_save_samples = False

        if wasCountFaces == True:
            countFaces = True

        print(f"Found {totalNumberOfFaces} faces in {len(allFiles)} images.") 

# RUN IF PATH IS NOT INSERTED AND IMAGE IS   
    if path == '' and p.init_images[0] != None:
        print(f"\nWill process {len(p.init_images)} images, creating {p.n_iter * p.batch_size} new images for each.")
        state.job_count = len(p.init_images) * p.n_iter
        state.job = f"{1} out of {len(p.init_images)}"

        image = p.init_images[0]
        width, height = image.size

        if not onlyMask:
            if overrideDenoising == True:
                p.denoising_strength = 0.5
            if overrideMaskBlur == True:
                p.mask_blur = int(math.ceil(0.01*height))

        skip = 0
        masks, totalNumberOfFaces, skip = findFaceDivide(image, width, height, divider, onlyHorizontal, onlyVertical, None, totalNumberOfFaces, singleMaskPerImage, countFaces, maskSize, skip)
        
        if onlyMask:
            suffix = '_mask'
            if pathToSave != "":
                for i, mask in enumerate(masks):
                    mask = Image.fromarray(mask)
                    if invertMask:
                        mask = ImageOps.invert(mask)
                    finishedImages.append(mask)
                    if saveMask == True:
                        images.save_image(mask, pathToSave, "", p.seed, p.prompt, opts.samples_format, info=infotext(), p=p, suffix=suffix)
                
            elif pathToSave == "":
                for i, mask in enumerate(masks):
                    mask = Image.fromarray(mask)
                    if invertMask:
                        mask = ImageOps.invert(mask)
                    finishedImages.append(mask)
                    if saveMask == True:
                        images.save_image(mask, opts.outdir_img2img_samples, "", p.seed, p.prompt, opts.samples_format, info=infotext(), p=p, suffix=suffix)

        if skip == 1 and saveNoFace and not onlyMask:
            images.save_image(image, p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, info=infotext(), p=p)
            finishedImages.append(image)
            state.skipped = True
        
        if skip == 1:
            state.skipped = True
        
        if not onlyMask:
            if len(masks) == 1:
                mask = Image.fromarray(masks[0])
                if invertMask:
                    mask = ImageOps.invert(mask)

                p.init_images = [image]
                p.image_mask = mask

                proc = process_images(p)

                for n in range(p.batch_size):
                    finishedImages.append(proc.images[n])
            else:
                generatedImages = []
                paste_to = []
                imageOriginal = image
                overlay_image = image
                p.do_not_save_samples = True

                for n, mask in enumerate(masks):
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

                    mask = mask.crop(crop_region)
                    image_mask = images.resize_image(2, mask, p.width, p.height)

                    
                    image = image.crop(crop_region)
                    image = images.resize_image(2, image, p.width, p.height)

                    p.init_images = [image]
                    p.image_mask = image_mask
                    proc = process_images(p)
                    generatedImages.append(proc.images)

                    image = imageOriginal

                for j in range(p.batch_size):
                    image = overlay_image
                        
                    for k in range(len(generatedImages)):
                        image = apply_overlay(generatedImages[k][j], paste_to[k], image)
                    images.save_image(image, p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, info=infotext(), p=p)
                    finishedImages.append(image)

                p.do_not_save_samples = False

        print(f"Found {totalNumberOfFaces} faces in {len(p.init_images)} images.")

    return finishedImages

class Script(scripts.Script):  
    def title(self):
        return "Batch Face Swap"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        def switchMaskExample(searchSubdir: bool, howSplit: str, divider: int, maskSize: int, path: str, visualizationOpacity: int):
            allFiles = []
            totalNumberOfFaces = 0
            if path != '':
                if searchSubdir:
                    for root, _, files in os.walk(os.path.abspath(path)):
                        for file in files:
                            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP')):
                                allFiles.append(os.path.join(root, file))
                else:
                    allFiles = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP'))]

            if len(allFiles) > 0:
                imgPath = allFiles[0]
                image = Image.open(imgPath)
                maxsize = (1000, 500)
                image.thumbnail(maxsize,Image.ANTIALIAS)

            visualizationOpacity = (visualizationOpacity/100)*255
            color = "white"
            thickness = 5

            if "Both" in howSplit:
                onlyHorizontal = False
                onlyVertical = False
                if len(allFiles) == 0:
                    image = Image.open("./extensions/batch-face-swap/images/exampleB.jpg")
                    # mask = Image.open("./extensions/batch-face-swap/images/exampleB_mask.jpg")
                    # mask = np.array(mask)
                width, height = image.size

                # if len(masks)==0 and path != '':
                masks, totalNumberOfFaces, skip = findFaceDivide(image, width, height, divider, onlyHorizontal, onlyVertical, file=None, totalNumberOfFaces=totalNumberOfFaces, singleMaskPerImage=True, countFaces=False, maskSize=maskSize, skip=0)
                mask = masks[0]

                size = maskSize * (0.011*height)/5
                kernel = np.ones((int(math.ceil((0.011*height)+abs(size))),int(math.ceil((0.011*height)+abs(size)))),'uint8')
                if size > 0:
                    mask = cv2.dilate(mask,kernel,iterations=1)
                else:
                    mask = cv2.erode(mask,kernel,iterations=1)

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
                masks, totalNumberOfFaces, skip = findFaceDivide(image, width, height, divider, onlyHorizontal, onlyVertical, file=None, totalNumberOfFaces=totalNumberOfFaces, singleMaskPerImage=True, countFaces=False, maskSize=maskSize, skip=0)
                mask = masks[0]

                size = maskSize * (0.011*height)/5
                kernel = np.ones((int(math.ceil((0.011*height)+abs(size))),int(math.ceil((0.011*height)+abs(size)))),'uint8')
                if size > 0:
                    mask = cv2.dilate(mask,kernel,iterations=1)
                else:
                    mask = cv2.erode(mask,kernel,iterations=1)

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
                    # mask = Image.open("./extensions/batch-face-swap/images/exampleH_mask.jpg")
                    # mask = np.array(mask)
                width, height = image.size

                # if len(masks)==0 and path != '':
                masks, totalNumberOfFaces, skip = findFaceDivide(image, width, height, divider, onlyHorizontal, onlyVertical, file=None, totalNumberOfFaces=totalNumberOfFaces, singleMaskPerImage=True, countFaces=False, maskSize=maskSize, skip=0)
                mask = masks[0]

                size = maskSize * (0.011*height)/5
                kernel = np.ones((int(math.ceil((0.011*height)+abs(size))),int(math.ceil((0.011*height)+abs(size)))),'uint8')
                if size > 0:
                    mask = cv2.dilate(mask,kernel,iterations=1)
                else:
                    mask = cv2.erode(mask,kernel,iterations=1)

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

        def switchMaskPathVisibility(saveMask: bool, onlyMask: bool):
            if onlyMask == False:
                return gr.Row.update(visible=bool(onlyMask))
            else:
                return gr.Row.update(visible=bool(saveMask))

        def switchTipsVisibility(showTips: bool):
            return gr.HTML.update(visible=bool(showTips))
        def switchInvertMask(invertMask: bool):
            return gr.Checkbox.update(value=bool(invertMask))


        gr.HTML("<p style=\"margin-bottom:0.75em;margin-top:0.75em;font-size:1.5em;color:red\">Make sure you're in the \"Inpaint upload\" tab!</p>") 
        with gr.Row():
            gr.HTML("<p style=\"margin-top:0.75em;font-size:1.25em\">Settings:</p>")
            gr.HTML("<p style=\"margin-top:0.75em;font-size:1.25em\">Overrides:</p>")
        with gr.Column(variant='panel'):
            htmlTip1 = gr.HTML("<p>Activate the 'Masks only' checkbox to see how many faces do your current settings detect without generating SD image. (check console)</p><p>You can also save generated masks to disk. Only possible with 'Masks only' (if you leave path empty, it will save the masks to your default webui outputs directory)</p><p>'Single mask per image' is only recommended with 'Invert mask' or if you want to save one mask per image, not per face. If you activate it without inverting mask, and try to process an image with multiple faces, it will generate only one image for all faces, producing bad results.</p>",visible=False)
            with gr.Row():
                # Settings
                with gr.Column():
                    with gr.Row():
                        onlyMask = gr.Checkbox(value=False, label="Masks only", visible=True)
                        saveMask = gr.Checkbox(value=False, label="Save masks to disk", interactive=False)
                    with gr.Row(visible=False) as maskPathRow:     
                        pathToSave = gr.Textbox(label="Mask save directory (OPTIONAL)",placeholder=r"C:\Users\dude\Desktop\masks (OPTIONAL)",visible=True)
                    with gr.Row():
                        invertMask = gr.Checkbox(value=False, label="Invert mask", visible=True)
                        singleMaskPerImage = gr.Checkbox(value=False, label="Single mask per image", visible=True)
                # Overrides
                with gr.Column():        
                    with gr.Row():
                        overrideDenoising = gr.Checkbox(value=True, label="""Override "Denoising strength" to 0.5""")
                    with gr.Row():
                        overrideMaskBlur = gr.Checkbox(value=True, label="""Override "Mask blur" to automatic""")

        # Path to images
        gr.HTML("<p style=\"margin-top:0.75em;margin-bottom:0.5em;font-size:1.5em\"><strong>Path to images:</strong></p>")
        with gr.Column(variant='panel'):
            htmlTip2 = gr.HTML("<p>'Load from subdirectories' will include all images in all subdirectories.</p>",visible=False)
            path = gr.Textbox(label="Images directory",placeholder=r"C:\Users\dude\Desktop\images")
            searchSubdir = gr.Checkbox(value=False, label="Load from subdirectories")

        # Image splitter
        gr.HTML("<p style=\"margin-top:0.75em;margin-bottom:0.5em;font-size:1.5em\"><strong>Image splitter:</strong></p>")
        with gr.Column(variant='panel'): 
            htmlTip3 = gr.HTML("<p>This divides image to smaller images and tries to find a face in the individual smaller images.</p><p>Useful when faces are small in relation to the size of the whole picture and are not being detected.</p><p>(may result in mask that only covers a part of a face or no detection if the division goes right through the face)</p><p>Open 'Split visualizer' to see how it works.</p>",visible=False)
            with gr.Row():
                divider = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="How many images to divide into")
                maskSize = gr.Slider(minimum=-10, maximum=10, step=1, value=0, label="Mask size")
            howSplit = gr.Radio(["Horizontal only ▤", "Vertical only ▥", "Both ▦"], value = "Both ▦", label = "How to divide")
            with gr.Accordion(label="Visualizer", open=False):  
                exampleImage = gr.Image(value=Image.open("./extensions/batch-face-swap/images/exampleB.jpg"), label="Split visualizer", show_label=False, type="pil", visible=True).style(height=500)
                with gr.Row(variant='compact'):
                    with gr.Column(variant='compact'):
                        gr.HTML("", visible=False)
                    with gr.Column(variant='compact'):
                        visualizationOpacity = gr.Slider(minimum=0, maximum=100, step=1, value=75, label="Opacity")
        
        # Other
        gr.HTML("<p style=\"margin-top:0.75em;font-size:1.25em\">Other:</p>")
        with gr.Column(variant='panel'):
            htmlTip4 = gr.HTML("<p>Activate 'Show results in WebUI' checkbox to see results in the WebUI at the end (not recommended when processing a large number of images)</p><p>'Count faces before generating' is required to see accurate progress bar (not recommended when processing a large number of images). Because without knowing the number of faces, the webui can't know how many images it will generate. Activating it means you will search for faces twice.</p>",visible=False)
            with gr.Row():
                saveNoFace = gr.Checkbox(value=True, label="Save image even if face was not found")
                viewResults = gr.Checkbox(value=False, label="Show results in WebUI")
            with gr.Row():
                countFaces = gr.Checkbox(value=False, label="Count faces before generating (accurate progress bar but NOT recommended)")
            with gr.Row():
                showTips = gr.Checkbox(value=False, label="Show tips")

        path.change(fn=None, _js="gradioApp().getElementById('mode_img2img').querySelectorAll('button')[4].click()", inputs=None, outputs=None)
        onlyMask.change(switchSaveMaskInteractivity, onlyMask, saveMask)
        onlyMask.change(switchSaveMask, onlyMask, saveMask)
        saveMask.change(switchMaskPathVisibility, [saveMask, onlyMask], maskPathRow)
        onlyMask.change(switchMaskPathVisibility, [saveMask, onlyMask], maskPathRow)

        showTips.change(switchTipsVisibility, showTips, htmlTip1)
        showTips.change(switchTipsVisibility, showTips, htmlTip2)
        showTips.change(switchTipsVisibility, showTips, htmlTip3)
        showTips.change(switchTipsVisibility, showTips, htmlTip4)

        invertMask.change(switchInvertMask, invertMask, singleMaskPerImage)
        
        # howSplit.change(switchSplitExample, [howSplit, divider, path], exampleImage)
        # divider.change(switchSplitExample, [howSplit, divider, path], exampleImage)
        # path.change(switchSplitExample, [howSplit, divider, path], exampleImage)

        visualizationOpacity.change(switchMaskExample, [searchSubdir, howSplit, divider, maskSize, path, visualizationOpacity], exampleImage)
        searchSubdir.change(switchMaskExample, [searchSubdir, howSplit, divider, maskSize, path, visualizationOpacity], exampleImage)
        howSplit.change(switchMaskExample, [searchSubdir, howSplit, divider, maskSize, path, visualizationOpacity], exampleImage)
        divider.change(switchMaskExample, [searchSubdir, howSplit, divider, maskSize, path, visualizationOpacity], exampleImage)
        maskSize.change(switchMaskExample, [searchSubdir, howSplit, divider, maskSize, path, visualizationOpacity], exampleImage)
        path.change(switchMaskExample, [searchSubdir, howSplit, divider, maskSize, path, visualizationOpacity], exampleImage)



        return [overrideDenoising, overrideMaskBlur, path, searchSubdir, divider, howSplit, saveMask, pathToSave, viewResults, saveNoFace, onlyMask, invertMask, singleMaskPerImage, countFaces, maskSize]

    def run(self, p, overrideDenoising, overrideMaskBlur, path, searchSubdir, divider, howSplit, saveMask, pathToSave, viewResults, saveNoFace, onlyMask, invertMask, singleMaskPerImage, countFaces, maskSize):
        all_images = []
        divider = int(divider)

        finishedImages = generateMasks(p, path, searchSubdir, divider, howSplit, saveMask, pathToSave, onlyMask, saveNoFace, overrideDenoising, overrideMaskBlur, invertMask, singleMaskPerImage, countFaces, maskSize)

        if not viewResults:
            finishedImages = []

        all_images += finishedImages   
        proc = Processed(p, all_images)

        return proc
