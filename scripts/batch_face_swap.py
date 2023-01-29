import sys
import os

cwd = os.getcwd()
utils_dir = os.path.join(cwd, 'extensions', 'batch-face-swap', 'scripts')
sys.path.extend([utils_dir])

from bfs_utils import *

import modules.scripts as scripts
import gradio as gr

from modules import images, masking
from modules.processing import process_images, create_infotext, Processed
from modules.shared import opts, cmd_opts, state

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter, UnidentifiedImageError
import math


def findFaces(image, width, height, divider, onlyHorizontal, onlyVertical, file, totalNumberOfFaces, singleMaskPerImage, countFaces, maskSize, skip):
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
            return masks, totalNumberOfFaces, skip

    masks.append(mask)

    return masks, totalNumberOfFaces, skip

def faceSwap(p, masks, image, finishedImages, invertMask, forced_filename, pathToSave, info, selectedTab):
    p.do_not_save_samples = True

    if len(masks) == 1:
        if selectedTab == "existingMasksTab":
            mask = masks[0]
        else:
            mask = Image.fromarray(masks[0])
        if invertMask:
            mask = ImageOps.invert(mask)

        p.init_images = [image]
        p.image_mask = mask

        proc = process_images(p)
        
        if pathToSave != "":
            for n in range(p.n_iter * p.batch_size):
                if opts.samples_format == "png":
                    images.save_image(proc.images[n], pathToSave, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename+"_"+str(n+1) if forced_filename != None and (p.batch_size > 1 or p.n_iter > 1) else forced_filename)
                elif image.mode != 'RGB':
                    image = image.convert('RGB')
                    images.save_image(proc.images[n], pathToSave, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename+"_"+str(n+1) if forced_filename != None and (p.batch_size > 1 or p.n_iter > 1) else forced_filename)
                else:
                    images.save_image(proc.images[n], pathToSave, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename+"_"+str(n+1) if forced_filename != None and (p.batch_size > 1 or p.n_iter > 1) else forced_filename)
        else:
            for n in range(p.n_iter * p.batch_size):
                if opts.samples_format == "png":
                    images.save_image(proc.images[n], p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename+"_"+str(n+1) if forced_filename != None and (p.batch_size > 1 or p.n_iter > 1) else forced_filename)
                elif image.mode != 'RGB':
                    image = image.convert('RGB')
                    images.save_image(proc.images[n], p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename+"_"+str(n+1) if forced_filename != None and (p.batch_size > 1 or p.n_iter > 1) else forced_filename)
                else:
                    images.save_image(proc.images[n], p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename+"_"+str(n+1) if forced_filename != None and (p.batch_size > 1 or p.n_iter > 1) else forced_filename)

            finishedImages.append(proc.images[n])
    else:
        generatedImages = []
        paste_to = []
        imageOriginal = image
        overlay_image = image

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
            image = imageOriginal             
            for k in range(len(generatedImages)):
                mask = Image.fromarray(masks[k])
                mask = mask.filter(ImageFilter.GaussianBlur(p.mask_blur))
                image = apply_overlay(generatedImages[k][j], paste_to[k], image, mask)

            if pathToSave != "":
                if opts.samples_format == "png":
                    images.save_image(image, pathToSave, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename+"_"+str(j+1) if forced_filename != None and p.batch_size > 1 else forced_filename)
                elif image.mode != 'RGB':
                    image = image.convert('RGB')
                    images.save_image(image, pathToSave, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename+"_"+str(j+1) if forced_filename != None and p.batch_size > 1 else forced_filename)
                else:
                    images.save_image(image, pathToSave, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename+"_"+str(j+1) if forced_filename != None and p.batch_size > 1 else forced_filename)
            else:
                if opts.samples_format == "png":
                    images.save_image(image, p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename+"_"+str(j+1) if forced_filename != None and p.batch_size > 1 else forced_filename)
                elif image.mode != 'RGB':
                    image = image.convert('RGB')
                    images.save_image(image, p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename+"_"+str(j+1) if forced_filename != None and p.batch_size > 1 else forced_filename)
                else:
                    images.save_image(image, p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename+"_"+str(j+1) if forced_filename != None and p.batch_size > 1 else forced_filename)

            finishedImages.append(image)

        p.do_not_save_samples = False

    return finishedImages
     

def generateImages(p, path, searchSubdir, viewResults, divider, howSplit, saveMask, pathToSave, onlyMask, saveNoFace, overrideDenoising, overrideMaskBlur, invertMask, singleMaskPerImage, countFaces, maskSize, keepOriginalName, info, pathExisting, pathMasksExisting, pathToSaveExisting, selectedTab):
    if selectedTab == "generateMasksTab":
        wasCountFaces = False
        finishedImages = []
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
            allFiles = listFiles(path, searchSubdir, allFiles)

            if countFaces:
                print("\nCounting faces...")
                for i, file in enumerate(allFiles):
                    skip = 0
                    image = Image.open(file)
                    width, height = image.size
                    totalNumberOfFaces = findFaces(image, width, height, divider, onlyHorizontal, onlyVertical, file, totalNumberOfFaces, singleMaskPerImage, countFaces, maskSize, skip)
                        
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
                except UnidentifiedImageError:
                    print(f"\nUnable to open {file}, skipping")
                    continue
                
                if not onlyMask:
                    if overrideDenoising == True:
                        p.denoising_strength = 0.5
                    if overrideMaskBlur == True:
                        p.mask_blur = int(math.ceil(0.01*height))

                skip = 0
                masks, totalNumberOfFaces, skip = findFaces(image, width, height, divider, onlyHorizontal, onlyVertical, file, totalNumberOfFaces, singleMaskPerImage, countFaces, maskSize, skip)                   
            
                # Only generate mask
                if onlyMask:
                    suffix = '_mask'

                    # If path to save mask was provided
                    if pathToSave != "":
                        for i, mask in enumerate(masks):
                            mask = Image.fromarray(mask)

                            # Invert mask if needed
                            if invertMask:
                                mask = ImageOps.invert(mask)
                            finishedImages.append(mask)

                            # Save mask
                            if saveMask == True:
                                if opts.samples_format == "png":
                                    images.save_image(mask, pathToSave, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)
                                elif mask.mode != 'RGB':
                                    mask = mask.convert('RGB')
                                    images.save_image(mask, pathToSave, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)
                                else:
                                    images.save_image(mask, pathToSave, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)

                    # If path to save mask was NOT provided   
                    elif pathToSave == "":
                        for i, mask in enumerate(masks):
                            mask = Image.fromarray(mask)

                            # Invert mask if needed
                            if invertMask:
                                mask = ImageOps.invert(mask)
                            finishedImages.append(mask)

                            # Save mask
                            if saveMask == True:
                                if opts.samples_format == "png":
                                    images.save_image(mask, opts.outdir_img2img_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)
                                elif mask.mode != 'RGB':
                                    mask = mask.convert('RGB')
                                    images.save_image(mask, opts.outdir_img2img_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)
                                else:
                                    images.save_image(mask, opts.outdir_img2img_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)

                # If face was not found but user wants to save images without face
                if skip == 1 and saveNoFace and not onlyMask:
                    if opts.samples_format == "png":
                        images.save_image(image, p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename)
                    elif image.mode != 'RGB':
                        image = image.convert('RGB')
                        images.save_image(image, p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename)
                    else:
                        images.save_image(image, p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename)  
                        
                    finishedImages.append(image)
                    state.skipped = True
                    continue

                # If face was not found, just skip
                if skip == 1:
                    state.skipped = True
                    continue
                
                if not onlyMask:
                    finishedImages = faceSwap(p, masks, image, finishedImages, invertMask, forced_filename, pathToSave, info, selectedTab)

                if not viewResults:
                    finishedImages = []

            if wasCountFaces == True:
                countFaces = True

            print(f"Found {totalNumberOfFaces} faces in {len(allFiles)} images.") 

    # RUN IF PATH IS NOT INSERTED AND IMAGE IS   
        if path == '' and p.init_images[0] != None:
            forced_filename = None
            image = p.init_images[0]
            width, height = image.size

            if countFaces:
                print("\nCounting faces...")
                skip = 0
                totalNumberOfFaces = findFaces(image, width, height, divider, onlyHorizontal, onlyVertical, None, totalNumberOfFaces, singleMaskPerImage, countFaces, maskSize, skip)

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
            masks, totalNumberOfFaces, skip = findFaces(image, width, height, divider, onlyHorizontal, onlyVertical, None, totalNumberOfFaces, singleMaskPerImage, countFaces, maskSize, skip)
            
            # Only generate mask
            if onlyMask:
                suffix = '_mask'

                # If path to save mask was provided
                if pathToSave != "":
                    for i, mask in enumerate(masks):
                        mask = Image.fromarray(mask)

                        # Invert mask if needed
                        if invertMask:
                            mask = ImageOps.invert(mask)
                        finishedImages.append(mask)

                        # Save mask
                        if saveMask == True:
                            if opts.samples_format == "png":
                                images.save_image(mask, pathToSave, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)
                            elif mask.mode != 'RGB':
                                mask = mask.convert('RGB')
                                images.save_image(mask, pathToSave, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)
                            else:
                                images.save_image(mask, pathToSave, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)
                    
                # If path to save mask was NOT provided
                elif pathToSave == "":
                    for i, mask in enumerate(masks):
                        mask = Image.fromarray(mask)

                        # Invert mask if needed
                        if invertMask:
                            mask = ImageOps.invert(mask)
                        finishedImages.append(mask)

                        # Save mask
                        if saveMask == True:
                            if opts.samples_format == "png":
                                images.save_image(mask, opts.outdir_img2img_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)
                            elif mask.mode != 'RGB':
                                mask = mask.convert('RGB')
                                images.save_image(mask, opts.outdir_img2img_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)
                            else:
                                images.save_image(mask, opts.outdir_img2img_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename, suffix=suffix)

            # If face was not found but user wants to save images without face
            if skip == 1 and saveNoFace and not onlyMask:
                if opts.samples_format == "png":
                    images.save_image(image, p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename)
                elif image.mode != 'RGB':
                    image = image.convert('RGB')
                    images.save_image(image, p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename)
                else:
                    images.save_image(image, p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=forced_filename)
                    
                finishedImages.append(image)
                state.skipped = True
            
            # If face was not found, just skip
            if skip == 1:
                state.skipped = True
            
            if not onlyMask:
                finishedImages = faceSwap(p, masks, image, finishedImages, invertMask, forced_filename, pathToSave, info, selectedTab)

            if wasCountFaces == True:
                countFaces = True

            print(f"Found {totalNumberOfFaces} faces in {len(p.init_images)} images.")

    elif selectedTab == "existingMasksTab":
        finishedImages = []
        allImages = []
        searchSubdir = False

        if pathExisting != '' and pathMasksExisting != '':
            allImages = listFiles(pathExisting, searchSubdir, allImages)

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
                    masks.append(Image.open(os.path.join(pathMasksExisting, os.path.basename(file))))
                except UnidentifiedImageError:
                    print(f"\nUnable to open {file}, skipping")
                    continue
                
                if overrideDenoising == True:
                    p.denoising_strength = 0.5
                if overrideMaskBlur == True:
                    p.mask_blur = int(math.ceil(0.01*height))

                finishedImages = faceSwap(p, masks, image, finishedImages, invertMask, forced_filename, pathToSaveExisting, info, selectedTab)

                if not viewResults:
                    finishedImages = []

    return finishedImages

class Script(scripts.Script):  
    def title(self):
        return "Batch Face Swap"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):

        def updateVisualizer(searchSubdir: bool, howSplit: str, divider: int, maskSize: int, path: str, visualizationOpacity: int):
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
                    # mask = Image.open("./extensions/batch-face-swap/images/exampleB_mask.jpg")
                    # mask = np.array(mask)
                width, height = image.size

                # if len(masks)==0 and path != '':
                masks, totalNumberOfFaces, skip = findFaces(image, width, height, divider, onlyHorizontal, onlyVertical, file=None, totalNumberOfFaces=totalNumberOfFaces, singleMaskPerImage=True, countFaces=False, maskSize=maskSize, skip=0)
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
                masks, totalNumberOfFaces, skip = findFaces(image, width, height, divider, onlyHorizontal, onlyVertical, file=None, totalNumberOfFaces=totalNumberOfFaces, singleMaskPerImage=True, countFaces=False, maskSize=maskSize, skip=0)
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
                    # mask = Image.open("./extensions/batch-face-swap/images/exampleH_mask.jpg")
                    # mask = np.array(mask)
                width, height = image.size

                # if len(masks)==0 and path != '':
                masks, totalNumberOfFaces, skip = findFaces(image, width, height, divider, onlyHorizontal, onlyVertical, file=None, totalNumberOfFaces=totalNumberOfFaces, singleMaskPerImage=True, countFaces=False, maskSize=maskSize, skip=0)
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
                with gr.Row():
                    overrideMaskBlur = gr.Checkbox(value=True, label="""Override "Mask blur" to automatic""")

        with gr.Column(variant='panel'):
            with gr.Tab("Generate masks") as generateMasksTab:
                with gr.Column(variant='panel'):
                    htmlTip1 = gr.HTML("<p>Activate the 'Masks only' checkbox to see how many faces do your current settings detect without generating SD image. (check console)</p><p>You can also save generated masks to disk. Only possible with 'Masks only' (if you leave path empty, it will save the masks to your default webui outputs directory)</p><p>'Single mask per image' is only recommended with 'Invert mask' or if you want to save one mask per image, not per face. If you activate it without inverting mask, and try to process an image with multiple faces, it will generate only one image for all faces, producing bad results.</p>",visible=False)
                    # Settings
                    with gr.Column(variant='panel'):
                        gr.HTML("<p style=\"margin-top:0.75em;font-size:1.25em\">Settings:</p>")
                        with gr.Column():
                            with gr.Row():
                                onlyMask = gr.Checkbox(value=False, label="Masks only", visible=True)
                                saveMask = gr.Checkbox(value=False, label="Save masks to disk", interactive=False)
                            with gr.Row():
                                invertMask = gr.Checkbox(value=False, label="Invert mask", visible=True)
                                singleMaskPerImage = gr.Checkbox(value=False, label="Single mask per image", visible=True)

                # Path to images
                with gr.Column(variant='panel'):
                    gr.HTML("<p style=\"margin-top:0.75em;margin-bottom:0.5em;font-size:1.5em\"><strong>Path to images:</strong></p>")
                    with gr.Column(variant='panel'):
                        htmlTip2 = gr.HTML("<p>'Load from subdirectories' will include all images in all subdirectories.</p>",visible=False)
                        with gr.Row():
                            path = gr.Textbox(label="Images directory",placeholder=r"C:\Users\dude\Desktop\images")
                            pathToSave = gr.Textbox(label="Output directory (OPTIONAL)",placeholder=r"Leave empty to save to default directory")
                        searchSubdir = gr.Checkbox(value=False, label="Load from subdirectories")
                        keepOriginalName = gr.Checkbox(value=False, label="Keep original file name (OVERWRITES FILES WITH THE SAME NAME)")

                # Image splitter
                with gr.Column(variant='panel'):
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
                                with gr.Column(variant='panel'):
                                    gr.HTML("", visible=False)
                                with gr.Column(variant='compact'):
                                    visualizationOpacity = gr.Slider(minimum=0, maximum=100, step=1, value=75, label="Opacity")
                
                # Other
                with gr.Column(variant='panel'):
                    gr.HTML("<p style=\"margin-top:0.75em;font-size:1.5em\">Other:</p>")
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
                gr.HTML("<p style=\"margin-top:0.75em;font-size:1.5em\">General:</p>")
                with gr.Column(variant='panel'):
                    htmlTip6 = gr.HTML("<p>Activate 'Show results in WebUI' checkbox to see results in the WebUI at the end (not recommended when processing a large number of images)</p>",visible=False)
                    with gr.Row():
                        viewResults = gr.Checkbox(value=False, label="Show results in WebUI")
                        showTips = gr.Checkbox(value=False, label="Show tips")

        selectedTab = gr.Textbox(value="generateMasksTab", visible=False)
        generateMasksTab.select(lambda: "generateMasksTab", inputs=None, outputs=selectedTab)
        existingMasksTab.select(lambda: "existingMasksTab", inputs=None, outputs=selectedTab)
        
        pathExisting.change(fn=None, _js="gradioApp().getElementById('mode_img2img').querySelectorAll('button')[4].click()", inputs=None, outputs=None)
        pathMasksExisting.change(fn=None, _js="gradioApp().getElementById('mode_img2img').querySelectorAll('button')[4].click()", inputs=None, outputs=None)
        pathToSaveExisting.change(fn=None, _js="gradioApp().getElementById('mode_img2img').querySelectorAll('button')[4].click()", inputs=None, outputs=None)
        path.change(fn=None, _js="gradioApp().getElementById('mode_img2img').querySelectorAll('button')[4].click()", inputs=None, outputs=None)
        pathToSave.change(fn=None, _js="gradioApp().getElementById('mode_img2img').querySelectorAll('button')[4].click()", inputs=None, outputs=None)
        onlyMask.change(switchSaveMaskInteractivity, onlyMask, saveMask)
        onlyMask.change(switchSaveMask, onlyMask, saveMask)
        invertMask.change(switchInvertMask, invertMask, singleMaskPerImage)

        visualizationOpacity.change(updateVisualizer, [searchSubdir, howSplit, divider, maskSize, path, visualizationOpacity], exampleImage)
        searchSubdir.change(updateVisualizer, [searchSubdir, howSplit, divider, maskSize, path, visualizationOpacity], exampleImage)
        howSplit.change(updateVisualizer, [searchSubdir, howSplit, divider, maskSize, path, visualizationOpacity], exampleImage)
        divider.change(updateVisualizer, [searchSubdir, howSplit, divider, maskSize, path, visualizationOpacity], exampleImage)
        maskSize.change(updateVisualizer, [searchSubdir, howSplit, divider, maskSize, path, visualizationOpacity], exampleImage)
        path.change(updateVisualizer, [searchSubdir, howSplit, divider, maskSize, path, visualizationOpacity], exampleImage)

        showTips.change(switchTipsVisibility, showTips, htmlTip1)
        showTips.change(switchTipsVisibility, showTips, htmlTip2)
        showTips.change(switchTipsVisibility, showTips, htmlTip3)
        showTips.change(switchTipsVisibility, showTips, htmlTip4)
        showTips.change(switchTipsVisibility, showTips, htmlTip5)
        showTips.change(switchTipsVisibility, showTips, htmlTip6)

        return [overrideDenoising, overrideMaskBlur, path, searchSubdir, divider, howSplit, saveMask, pathToSave, viewResults, saveNoFace, onlyMask, invertMask, singleMaskPerImage, countFaces, maskSize, keepOriginalName, pathExisting, pathMasksExisting, pathToSaveExisting, selectedTab]

    def run(self, p, overrideDenoising, overrideMaskBlur, path, searchSubdir, divider, howSplit, saveMask, pathToSave, viewResults, saveNoFace, onlyMask, invertMask, singleMaskPerImage, countFaces, maskSize, keepOriginalName, pathExisting, pathMasksExisting, pathToSaveExisting, selectedTab):
        comments = {}
        wasGrid = p.do_not_save_grid
        p.do_not_save_grid = True
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

        info = infotext()
        all_images = []


        finishedImages = generateImages(p, path, searchSubdir, viewResults, int(divider), howSplit, saveMask, pathToSave, onlyMask, saveNoFace, overrideDenoising, overrideMaskBlur, invertMask, singleMaskPerImage, countFaces, maskSize, keepOriginalName, info, pathExisting, pathMasksExisting, pathToSaveExisting, selectedTab)
        
        
        if not viewResults:
            finishedImages = []

        all_images += finishedImages   
        proc = Processed(p, all_images)
        p.do_not_save_grid = wasGrid

        return proc
