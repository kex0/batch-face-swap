import sys
import os

cwd = os.getcwd()
utils_dir = os.path.join(cwd, 'extensions', 'batch-face-swap', 'scripts')
sys.path.extend([utils_dir])

from bfs_utils import *
from face_detect import *
from sd_helpers import renderImg2Img, renderTxt2Img

import modules.scripts as scripts
import gradio as gr
import time
import random

from modules import images, masking, generation_parameters_copypaste, script_callbacks
from modules.processing import process_images, create_infotext, Processed
from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.sd_samplers

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter, UnidentifiedImageError
import math

def apply_checkpoint(x):
    info = modules.sd_models.get_closet_checkpoint_match(x)
    if info is None:
        raise RuntimeError(f"Unknown checkpoint: {x}")
    modules.sd_models.reload_model_weights(shared.sd_model, info)

def findFaces(facecfg, image, width, height, divider, onlyHorizontal, onlyVertical, file, totalNumberOfFaces, singleMaskPerImage, countFaces, maskWidth, maskHeight, skip):
    rejected = 0
    masks = []
    faces_info = []
    imageOriginal = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    heightOriginal = height
    widthOriginal = width

    # Calculate the size of each small image
    small_width  = width  if onlyHorizontal else math.ceil(width  / divider)
    small_height = height if onlyVertical   else math.ceil(height / divider)

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

        if facecfg.faceMode == FaceMode.ORIGINAL:
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
                faces_info.append(computeFaceInfo(landmark, onlyHorizontal, divider, small_width, small_height, small_image_index))

        elif facecfg.faceMode == FaceMode.YUNET:
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

                faces_info.append(computeFaceInfo(landmark, onlyHorizontal, divider, small_width, small_height, small_image_index))

            faceRects = getFaceRectanglesYuNet(small_image, known_face_rects)

            for rect in faceRects:
                landmarkHull, face_info = getFacialLandmarkConvexHull(image, rect, onlyHorizontal, divider, small_width, small_height, small_image_index, facecfg)
                if landmarkHull is not None:
                    faces.append(landmarkHull)
                    faces_info.append(face_info)
                else:
                    rejected += 1

            numberOfFaces = int(len(faces))
            totalNumberOfFaces += numberOfFaces
            if countFaces:
                continue

        else:
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

                faces_info.append(computeFaceInfo(landmark, onlyHorizontal, divider, small_width, small_height, small_image_index))

            faceRects = getFaceRectangles(small_image, known_face_rects, facecfg)

            for rect in faceRects:
                landmarkHull, face_info = getFacialLandmarkConvexHull(image, rect, onlyHorizontal, divider, small_width, small_height, small_image_index, facecfg)
                if landmarkHull is not None:
                    faces.append(landmarkHull)
                    faces_info.append(face_info)
                else:
                    rejected += 1

            numberOfFaces = int(len(faces))
            totalNumberOfFaces += numberOfFaces
            if countFaces:
                continue


        if len(faces) == 0:
            small_image[:] = (0, 0, 0)

        if numberOfFaces > 0:
            facesInImage += numberOfFaces
        if facesInImage == 0 and i == len(small_images) - 1:
            skip = 1

        for i in range(len(faces)):
            processed_images = []
            for k in range(len(small_images)):
                mask = np.zeros((small_height, small_width), np.uint8)
                if k == small_image_index:
                    small_image = cv2.fillConvexPoly(mask, faces[i], 255)
                    processed_image = Image.fromarray(small_image)
                    processed_images.append(processed_image)
                else:
                    processed_image = Image.fromarray(mask)
                    processed_images.append(processed_image)

            # Create a new image with the same size as the original large image
            new_image = Image.new('RGB', (width, height))

            # Paste the processed small images into the new image
            if onlyHorizontal == True:
                for i, processed_image in enumerate(processed_images):
                    x = (i // divider) * small_width
                    y = (i %  divider) * small_height
                    new_image.paste(processed_image, (x, y))
            else:
                for i, processed_image in enumerate(processed_images):
                    x = (i %  divider) * small_width
                    y = (i // divider) * small_height
                    new_image.paste(processed_image, (x, y))                 
            masks.append(new_image)

    # if countFaces:
    #     return totalNumberOfFaces

    if file != None:
        if FaceDetectDevelopment:
            print(f"Found {facesInImage} face(s) in {str(file)} (rejected {rejected} from OpenCV)")
        else:
            print(f"Found {facesInImage} face(s) in {str(file)}")
    # else:
    #     print(f"Found {facesInImage} face(s)")

    binary_masks = []
    for i, mask in enumerate(masks):
        gray_image = mask.convert('L')
        numpy_array = np.array(gray_image)
        binary_mask = cv2.threshold(numpy_array, 200, 255, cv2.THRESH_BINARY)[1]
        if maskWidth != 100 or maskHeight != 100:
            binary_mask = maskResize(binary_mask, maskWidth, maskHeight)
        binary_masks.append(binary_mask)

    # try:
    #     kernel = np.ones((int(math.ceil(0.011*height)),int(math.ceil(0.011*height))),'uint8')
    #     dilated = cv2.dilate(binary_mask,kernel,iterations=1)
    #     kernel = np.ones((int(math.ceil(0.0045*height)),int(math.ceil(0.0025*height))),'uint8')
    #     dilated = cv2.dilate(dilated,kernel,iterations=1,anchor=(1, -1))
    #     kernel = np.ones((int(math.ceil(0.014*height)),int(math.ceil(0.0025*height))),'uint8')
    #     dilated = cv2.dilate(dilated,kernel,iterations=1,anchor=(-1, 1))
    #     mask = dilated
    # except cv2.error:
    #     mask = dilated

    if singleMaskPerImage and len(binary_masks) > 0:
        result = []
        h, w = binary_masks[0].shape
        result = np.full((h,w), 0, dtype=np.uint8)
        for mask in binary_masks:
            result = cv2.add(result, mask)
        masks = [ result ]
        return masks, totalNumberOfFaces, faces_info, skip
    else:
        masks = binary_masks
        return masks, totalNumberOfFaces, faces_info, skip

# generate debug image
def faceDebug(p, masks, image, finishedImages, invertMask, forced_filename, output_path, info):
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

    debugsave(overlay_image)

def faceSwap(p, masks, image, finishedImages, invertMask, forced_filename, output_path, info, selectedTab,mainTab, geninfo, faces_info, rotation_threshold, overridePrompt, bfs_prompt, bfs_nprompt, overrideSampler, sd_sampler, overrideModel, sd_model, overrideDenoising, denoising_strength, overrideMaskBlur, mask_blur, overridePadding, inpaint_full_res_padding, overrideSeed, overrideSteps, steps, overrideCfgScale, cfg_scale, overrideSize, bfs_width, bfs_height):

    bfs_prompt = bfs_prompt if overridePrompt else p.prompt
    bfs_nprompt = bfs_nprompt if overridePrompt else p.negative_prompt
    original_model = modules.sd_models.select_checkpoint()
    if overrideModel:
        apply_checkpoint(sd_model)
    sd_sampler = sd_sampler if overrideSampler else p.sampler_name
    batch_size = 1 if mainTab == "txt2img" else p.batch_size
    n_iter = 1 if mainTab == "txt2img" else p.n_iter
    denoising_strength = 0.5 if overrideDenoising else denoising_strength
    seed = int(random.randrange(4294967294)) if overrideSeed else p.seed
    steps = steps if overrideSteps else p.steps
    cfg_scale = cfg_scale if overrideCfgScale else p.cfg_scale
    bfs_width = bfs_width if overrideSize else p.width
    bfs_height = bfs_height if overrideSize else p.height

    # automatically adjust mask_blur based on the size of the image but don't make it higher than 30
    mask_blur = np.clip(int(math.ceil(0.01*image.height if image.height > image.width else 0.01*image.width)), None, 30) if overrideMaskBlur else mask_blur

    # automatically adjust inpaint_full_res_padding based on the size of the image
    inpaint_full_res_padding = int(math.ceil(0.03*image.height if image.height > image.width else 0.03*image.width))  if overridePadding else inpaint_full_res_padding

    inpainting_full_res = 1
    inpainting_fill = 1
    
    wasGrid = p.do_not_save_grid
    wasReturnGrid = opts.return_grid
    opts.return_grid = False
    p.do_not_save_grid = True
    p.do_not_save_samples = True
    index = 0
    generatedImages = []
    paste_to = []
    imageOriginal = image
    overlay_image = image

    for n, mask in enumerate(masks):
        rotate = False
        mask = Image.fromarray(masks[n])
        if invertMask:
            image_mask = ImageOps.invert(mask)
        else:
            image_masked = Image.new('RGBa', (image.width, image.height))
            image_masked.paste(overlay_image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert('L')))

            overlay_image = image_masked.convert('RGBA')

            crop_region = masking.get_crop_region(np.array(mask), inpaint_full_res_padding)
            crop_region = masking.expand_crop_region(crop_region, p.width, p.height, mask.width, mask.height)
            x1, y1, x2, y2 = crop_region
            paste_to.append((x1, y1, x2-x1, y2-y1))

            for i in range(len(faces_info)):
                try:
                    pixel_color = mask.getpixel((faces_info[i]["center"][0],faces_info[i]["center"][1]))
                except IndexError:
                    pixel_color = 0
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

        if geninfo != "":
            bfs_prompt = str(geninfo.get("Prompt"))
            bfs_nprompt = str(geninfo.get("Negative prompt"))
            sd_sampler = str(geninfo.get("Sampler"))
            cfg_scale = float(geninfo.get("CFG scale"))
            bfs_width = int(geninfo.get("Size-1"))
            bfs_height = int(geninfo.get("Size-2"))

        proc = renderImg2Img(
            bfs_prompt,
            bfs_nprompt,
            sd_sampler,
            steps,
            cfg_scale,
            seed,
            bfs_width,
            bfs_height,
            image,
            image_mask,
            batch_size,
            n_iter,
            denoising_strength,
            mask_blur,
            inpainting_fill,
            inpainting_full_res,
            inpaint_full_res_padding,
            do_not_save_samples = True,
        )
        apply_checkpoint(original_model.title)

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

    for j in range(n_iter * batch_size):
        if not invertMask:
            image = imageOriginal
            for k in range(len(generatedImages)):
                mask = Image.fromarray(masks[k])
                mask = mask.filter(ImageFilter.GaussianBlur(mask_blur))
                image = apply_overlay(generatedImages[k][j], paste_to[k], image, mask)
        else:
            image = proc.images[j]

        info = infotext(p)

        final_forced_filename = forced_filename+"_"+str(j+1) if forced_filename != None and (batch_size > 1 or n_iter > 1) else forced_filename
        if opts.samples_format != "png" and image.mode != 'RGB':
            image = image.convert('RGB')
        images.save_image(image, output_path if output_path !="" else opts.outdir_img2img_samples, "", p.seed, p.prompt, opts.samples_format, info=info, p=p, forced_filename=final_forced_filename)

        finishedImages.append(image)

    p.do_not_save_samples = False
    p.do_not_save_grid = wasGrid
    opts.return_grid = wasReturnGrid

    return finishedImages

def generateImages(p, facecfg, input_image, input_path, searchSubdir, viewResults, divider, howSplit, saveMask, output_path, saveToOriginalFolder, onlyMask, saveNoFace, overridePrompt, bfs_prompt, bfs_nprompt, overrideSampler, sd_sampler, overrideModel, sd_model, overrideDenoising, denoising_strength, overrideMaskBlur, mask_blur, overridePadding, inpaint_full_res_padding, overrideSeed, overrideSteps, steps, overrideCfgScale, cfg_scale, overrideSize, bfs_width, bfs_height, invertMask, singleMaskPerImage, countFaces, maskWidth, maskHeight, keepOriginalName, pathExisting, pathMasksExisting, output_pathExisting, selectedTab,mainTab, loadGenParams, rotation_threshold):
    suffix = ''
    info = infotext(p)
    if selectedTab == "generateMasksTab":
        finishedImages = []
        wasCountFaces = False
        totalNumberOfFaces = 0
        allFiles = []
        geninfo = ""

        onlyHorizontal = ("Horizontal" in howSplit)
        onlyVertical   = ("Vertical" in howSplit)

        # if neither path nor image, we're done
        if input_path == '' and input_image is None:
           return finishedImages

        # flag whether we're processing a directory or a specified image
        # (the code after this supports multiple images in an array, but the UI only allows a single image)
        usingFilenames = (input_path != '')
        if usingFilenames:
            allFiles = listFiles(input_path, searchSubdir, allFiles) 
        else:
            allFiles += input_image

        start_time = time.thread_time()

        if countFaces:
            print("\nCounting faces...")
            for i, file in enumerate(allFiles):
                skip = 0
                image = Image.open(file) if usingFilenames else file
                width, height = image.size
                masks, totalNumberOfFaces, faces_info, skip = findFaces(facecfg, image, width, height, divider, onlyHorizontal, onlyVertical, file, totalNumberOfFaces, singleMaskPerImage, countFaces, maskWidth, maskHeight, skip)

        if not onlyMask and countFaces:
            print(f"\nWill process {len(allFiles)} images, found {totalNumberOfFaces} faces, generating {p.n_iter * p.batch_size} new images for each.")
            state.job_count = totalNumberOfFaces * p.n_iter
        elif not onlyMask and not countFaces:
            print(f"\nWill process {len(allFiles)} images, generating {p.n_iter * p.batch_size} new images for each.")
            state.job_count = len(allFiles) * p.n_iter

        for i, file in enumerate(allFiles):
            if usingFilenames and keepOriginalName:
                forced_filename = os.path.splitext(os.path.basename(file))[0]
            else:
                forced_filename = None

            if usingFilenames and saveToOriginalFolder:
                output_path = os.path.dirname(file)

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
                image = Image.open(file) if usingFilenames else file
                width, height = image.size

                if loadGenParams:
                    geninfo, _ = read_info_from_image(image)
                    geninfo = generation_parameters_copypaste.parse_generation_parameters(geninfo)

            except UnidentifiedImageError:
                print(f"\nUnable to open {file}, skipping")
                continue

            skip = 0
            masks, totalNumberOfFaces, faces_info, skip = findFaces(facecfg, image, width, height, divider, onlyHorizontal, onlyVertical, file, totalNumberOfFaces, singleMaskPerImage, countFaces, maskWidth, maskHeight, skip)

            if facecfg.debugSave:
                faceDebug(p, masks, image, finishedImages, invertMask, forced_filename, output_path, info)

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
                        custom_save_image(p, mask, output_path, forced_filename, suffix, info)
                    elif saveMask and skip == 1 and saveNoFace:
                        custom_save_image(p, mask, output_path, forced_filename, suffix, info)

            # If face was not found but user wants to save images without face
            if skip == 1 and saveNoFace and not onlyMask:
                custom_save_image(p, image, output_path, forced_filename, suffix, info)

                finishedImages.append(image)
                state.skipped = True
                continue

            # If face was not found, just skip
            if skip == 1:
                state.skipped = True
                continue

            if not onlyMask:
                finishedImages = faceSwap(p, masks, image, finishedImages, invertMask, forced_filename, output_path, info, selectedTab, mainTab, geninfo, faces_info, rotation_threshold, overridePrompt, bfs_prompt, bfs_nprompt, overrideSampler, sd_sampler, overrideModel, sd_model, overrideDenoising, denoising_strength, overrideMaskBlur, mask_blur, overridePadding, inpaint_full_res_padding, overrideSeed, overrideSteps, steps, overrideCfgScale, cfg_scale, overrideSize, bfs_width, bfs_height,)

            if usingFilenames and not viewResults:
                finishedImages = []

        if wasCountFaces == True:
            countFaces = True

        timing = time.thread_time() - start_time
        print(f"Found {totalNumberOfFaces} faces in {len(allFiles)} images in {timing} seconds.")

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


                finishedImages = faceSwap(p, masks, image, finishedImages, invertMask, forced_filename, output_pathExisting, info, selectedTab, mainTab, faces_info, rotation_threshold, overridePrompt, bfs_prompt, bfs_nprompt, overrideSampler, sd_sampler, overrideModel, sd_model, overrideDenoising, denoising_strength, overrideMaskBlur, mask_blur, overridePadding, inpaint_full_res_padding, overrideSeed, overrideSteps, steps, overrideCfgScale, cfg_scale, overrideSize, bfs_width, bfs_height,)

                if not viewResults:
                    finishedImages = []

    return finishedImages

class Script(scripts.Script):
    def title(self):
        return "Batch Face Swap"

    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        def updateVisualizer(searchSubdir: bool, howSplit: str, divider: int, maskWidth: int, maskHeight: int, input_path: str, visualizationOpacity: int, faceMode: int):
                    facecfg = FaceDetectConfig(faceMode) # this is a huge pain to patch through so don't bother
                    allFiles = []
                    totalNumberOfFaces = 0

                    usingFilenames = (input_path != '')
                    if usingFilenames:
                        allFiles = listFiles(input_path, searchSubdir, allFiles)                    

                    if len(allFiles) > 0:
                        file = allFiles[0]
                        try:
                            image = Image.open(file)
                            maxsize = (1000, 500)
                            image.thumbnail(maxsize,Image.ANTIALIAS)
                        except (UnidentifiedImageError, AttributeError):
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
                        masks, totalNumberOfFaces, faces_info, skip = findFaces(facecfg, image, width, height, divider, onlyHorizontal, onlyVertical, file=None, totalNumberOfFaces=totalNumberOfFaces, singleMaskPerImage=True, countFaces=False, maskWidth=maskWidth, maskHeight=maskHeight, skip=0)

                        if len(masks) > 0:
                            mask = masks[0]
                        else:
                            mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)

                        # mask = maskResize(mask, maskSize, height)

                        mask = Image.fromarray(mask)
                        redImage = Image.new("RGB", (width, height), (255, 0, 0))

                        mask = mask.convert("L")
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
                        masks, totalNumberOfFaces, faces_info, skip = findFaces(facecfg, image, width, height, divider, onlyHorizontal, onlyVertical, file=None, totalNumberOfFaces=totalNumberOfFaces, singleMaskPerImage=True, countFaces=False, maskWidth=maskWidth, maskHeight=maskHeight, skip=0)

                        if len(masks) > 0:
                            mask = masks[0]
                        else:
                            mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)

                        # mask = maskResize(mask, maskSize, height)

                        mask = Image.fromarray(mask)
                        redImage = Image.new("RGB", (width, height), (255, 0, 0))

                        mask = mask.convert("L")
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
                        masks, totalNumberOfFaces, faces_info, skip = findFaces(facecfg, image, width, height, divider, onlyHorizontal, onlyVertical, file=None, totalNumberOfFaces=totalNumberOfFaces, singleMaskPerImage=True, countFaces=False, maskWidth=maskWidth, maskHeight=maskHeight, skip=0)

                        if len(masks) > 0:
                            mask = masks[0]
                        else:
                            mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)

                        # mask = maskResize(mask, maskSize, height)

                        mask = Image.fromarray(mask)
                        redImage = Image.new("RGB", (width, height), (255, 0, 0))

                        mask = mask.convert("L")
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
        def switchColumnVisibility(switch_element: bool):
            return gr.Column.update(visible=bool(switch_element))
        def switchColumnVisibilityInverted(switch_element: bool):
            return gr.Column.update(visible=bool(not switch_element))
        def switchEnableLabel(enabled: bool):
            if enabled == True:
                return gr.Checkbox.update(label=str("Enabled ✅"))
            else:
                return gr.Checkbox.update(label=str("Disabled ❌"))
        
        with gr.Accordion("🎭 Batch Face Swap 🎭", open = False, elem_id="batch_face_swap"):
            with gr.Row():
                enabled = gr.Checkbox(label='Disabled ❌', value=False)
                if not is_img2img:
                    with gr.Column():
                        regen_btn = gr.Button(value="Swap 🎭", variant="primary", interactive=True)
                        gr.HTML("<p style=\"margin-bottom:3em;font-size:0.7em\">Check your output folder (this won't show results in webui)</p>",visible=True)
                else:
                    regen_btn = gr.Button(value="Swap 🎭", variant="primary", visible=False)
            with gr.Accordion("♻ Overrides ♻", open = True):
                with gr.Box():
                    # Overrides
                    with gr.Column():
                        with gr.Column(variant='panel'):
                            overridePrompt = gr.Checkbox(value=False, label="""Override "Prompt" """)
                            with gr.Column(visible=False) as override_prompt_col:
                                bfs_prompt = gr.Textbox(label="Prompt", show_label=False, lines=2, placeholder="Prompt")
                                bfs_nprompt = gr.Textbox(label="Negative prompt", show_label=False, lines=2, placeholder="Negative prompt")
                        with gr.Row():
                            with gr.Column():
                                with gr.Column(variant='panel', scale=2):
                                    overrideSeed = gr.Checkbox(value=True, label="""Override "Seed" to random""", interactive=True)
                                with gr.Column(variant='panel'):
                                    overrideSampler = gr.Checkbox(value=False, label="""Override "Sampling method" """)
                                    with gr.Column(visible=False) as override_sampler_col:
                                        available_samplers = [s.name for s in modules.sd_samplers.samplers_for_img2img]
                                        sd_sampler = gr.Dropdown(label="Sampling Method", choices=available_samplers, value="Euler a", type="value", interactive=True)
                                with gr.Column(variant='panel'):
                                    overrideSteps = gr.Checkbox(value=False, label="""Override "Sampling steps" """, interactive=True)
                                    with gr.Column(visible=False) as override_steps_col:
                                        steps = gr.Slider(minimum=1, maximum=150, step=1 , value=30, label="Sampling Steps", interactive=True)
                                with gr.Column(variant='panel'):
                                    overrideDenoising = gr.Checkbox(value=True, label="""Override "Denoising strength" to 0.5""")
                                    with gr.Column(visible=False) as override_denoising_col:
                                        denoising_strength = gr.Slider(minimum=0, maximum=1, step=0.01 , value=0.5, label="Denoising Strength", interactive=True)

                            with gr.Column():
                                with gr.Column(variant='panel', scale=2):
                                    overrideSize = gr.Checkbox(value=False, label="""Override "Resolution" """, interactive=True)
                                    with gr.Column(visible=False) as override_size_col:
                                        with gr.Row():
                                            bfs_width = gr.Slider(minimum=64, maximum=2048, step=4 , value=512, label="Width", interactive=True)
                                            bfs_height = gr.Slider(minimum=64, maximum=2048, step=4 , value=512, label="Height", interactive=True)
                                with gr.Column(variant='panel'):
                                    overrideModel = gr.Checkbox(value=False, label="""Override "Stable Diffusion checkpoint" """)
                                    with gr.Column(visible=False) as override_model_col:
                                        with gr.Row():
                                            available_models = modules.sd_models.checkpoint_tiles()
                                            sd_model = gr.Dropdown(label="SD Model", choices=available_models, value=shared.sd_model.sd_checkpoint_info.title, type="value", interactive=True)
                                            modules.ui.create_refresh_button(sd_model, modules.sd_models.list_models, lambda: {"choices": modules.sd_models.checkpoint_tiles()}, "refresh_sd_checkpoint")
                                with gr.Column(variant='panel'):
                                    overrideCfgScale = gr.Checkbox(value=False, label="""Override "CFG Scale" """, interactive=True)
                                    with gr.Column(visible=False) as override_cfg_col:
                                        cfg_scale = gr.Slider(minimum=1, maximum=30, step=1 , value=6, label="CFG Scale", interactive=True)
                                with gr.Column(variant='panel'):
                                    overrideMaskBlur = gr.Checkbox(value=True, label="""Override "Mask blur" to automatic""")
                                    with gr.Column(visible=False) as override_maskBlur_col:
                                        mask_blur = gr.Slider(minimum=0, maximum=64, step=1  , value=4, label="Mask Blur", interactive=True)
            
                        with gr.Column(variant='panel'):
                            with gr.Row():
                                overridePadding = gr.Checkbox(value=True, label="""Override "Only masked padding, pixels" to automatic""")
                            with gr.Row():
                                with gr.Column(visible=False) as override_padding_col:
                                    inpaint_full_res_padding = gr.Slider(minimum=0, maximum=256, step=4 , value=32, label="Only masked padding, pixels", interactive=True)

            if is_img2img:
                # Path to images
                gr.HTML("<p style=\"margin-top:0.10em;font-size:1.5em\"><strong>Input:</strong></p>")
                with gr.Column(variant='panel'):
                    htmlTip1 = gr.HTML("<p>'Load from subdirectories' will include all images in all subdirectories.</p>",visible=False)
                    with gr.Row():
                        input_path = gr.Textbox(label="Images directory",placeholder=r"C:\Users\dude\Desktop\images", visible=True)
                        output_path = gr.Textbox(label="Output directory (OPTIONAL)",placeholder=r"Leave empty to save to default directory")
                    with gr.Row():
                        searchSubdir = gr.Checkbox(value=False, label="Load from subdirectories")
                        saveToOriginalFolder = gr.Checkbox(value=False, label="Save to original folder")
                    keepOriginalName = gr.Checkbox(value=False, label="Keep original file name (OVERWRITES FILES WITH THE SAME NAME)")
                    loadGenParams = gr.Checkbox(value=False, label="Load generation parameters from images")

            else:
                htmlTip1 = gr.HTML("<p></p>",visible=False)
                input_path = gr.Textbox(label="Images directory", visible=False)
                output_path = gr.Textbox(label="Output directory (OPTIONAL)", visible=False)
                searchSubdir = gr.Checkbox(value=False, label="Load from subdirectories", visible=False)
                saveToOriginalFolder = gr.Checkbox(value=False, label="Save to original folder", visible=False)
                keepOriginalName = gr.Checkbox(value=False, label="Keep original file name (OVERWRITES FILES WITH THE SAME NAME)", visible=False)
                loadGenParams = gr.Checkbox(value=False, label="Load generation parameters from images", visible=False)

            with gr.Accordion("⚙️ Settings ⚙️", open = False):
                with gr.Column(variant='panel'):
                    with gr.Tab("Generate masks") as generateMasksTab:
                        # Face detection
                        with gr.Column(variant='compact'):
                            gr.HTML("<p style=\"margin-top:0.10em;margin-bottom:0.75em;font-size:1.5em\"><strong>Face detection:</strong></p>")
                            with gr.Row():
                                faceDetectMode = gr.Dropdown(label="Detector", choices=face_mode_names, value=face_mode_names[FaceMode.DEFAULT], type="index", elem_id="z_type")
                                minFace = gr.Slider(minimum=10, maximum=200, step=1  , value=30, label="Minimum face size in pixels")

                        with gr.Column(variant='panel'):
                            htmlTip2 = gr.HTML("<p>Activate the 'Masks only' checkbox to see how many faces do your current settings detect without generating SD image. (check console)</p><p>You can also save generated masks to disk. Only possible with 'Masks only' (if you leave path empty, it will save the masks to your default webui outputs directory)</p><p>'Single mask per image' is only recommended with 'Invert mask' or if you want to save one mask per image, not per face. If you activate it without inverting mask, and try to process an image with multiple faces, it will generate only one image for all faces, producing bad results.</p><p>'Rotation threshold', if the face is rotated at an angle higher than this value, it will be automatically rotated so it's upright before generating, producing much better results.</p>",visible=False)
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

                        # Image splitter
                        with gr.Column(variant='panel'):
                            gr.HTML("<p style=\"margin-top:0.10em;font-size:1.5em\"><strong>Image splitter:</strong></p>")
                            with gr.Column(variant='panel'):
                                htmlTip3 = gr.HTML("<p>This divides image to smaller images and tries to find a face in the individual smaller images.</p><p>Useful when faces are small in relation to the size of the whole picture and are not being detected.</p><p>(may result in mask that only covers a part of a face or no detection if the division goes right through the face)</p><p>Open 'Split visualizer' to see how it works.</p>",visible=False)
                                with gr.Row():
                                    divider = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="How many images to divide into")
                                    maskWidth = gr.Slider(minimum=0, maximum=300, step=1, value=100, label="Mask width")
                                with gr.Row():
                                    howSplit = gr.Radio(["Horizontal only ▤", "Vertical only ▥", "Both ▦"], value = "Both ▦", label = "How to divide")
                                    maskHeight = gr.Slider(minimum=0, maximum=300, step=1, value=100, label="Mask height")
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
                            output_pathExisting = gr.Textbox(label="Output directory (OPTIONAL)",placeholder=r"Leave empty to save to default directory")

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
                
                # def retriveP(getp: bool):

                # getp = gr.Checkbox(value=True, label="get p", visible=False)
                
                mainTab  = gr.Textbox(value=f"""{"img2img" if is_img2img else "txt2img"}""", visible=False)
                selectedTab = gr.Textbox(value="generateMasksTab", visible=False)
                generateMasksTab.select(lambda: "generateMasksTab", inputs=None, outputs=selectedTab)
                existingMasksTab.select(lambda: "existingMasksTab", inputs=None, outputs=selectedTab)

                # make sure user is in the "Inpaint upload" tab
                input_path.change(fn=None, _js="gradioApp().getElementById('mode_img2img').querySelectorAll('button')[4].click()", inputs=None, outputs=None)
                output_path.change(fn=None, _js="gradioApp().getElementById('mode_img2img').querySelectorAll('button')[4].click()", inputs=None, outputs=None)
                searchSubdir.change(fn=None, _js="gradioApp().getElementById('mode_img2img').querySelectorAll('button')[4].click()", inputs=None, outputs=None)
                saveToOriginalFolder.change(fn=None, _js="gradioApp().getElementById('mode_img2img').querySelectorAll('button')[4].click()", inputs=None, outputs=None)
                keepOriginalName.change(fn=None, _js="gradioApp().getElementById('mode_img2img').querySelectorAll('button')[4].click()", inputs=None, outputs=None)
                loadGenParams.change(fn=None, _js="gradioApp().getElementById('mode_img2img').querySelectorAll('button')[4].click()", inputs=None, outputs=None)


                def regen(input_path: str, searchSubdir: bool, viewResults: bool, divider: int, howSplit: str, saveMask: bool, output_path: str, saveToOriginalFolder: bool, onlyMask: bool, saveNoFace: bool, overridePrompt: bool, bfs_prompt: str, bfs_nprompt: str, overrideSampler: bool, sd_sampler: str, overrideModel: bool, sd_model: str, overrideDenoising: bool, denoising_strength: float, overrideMaskBlur: bool, mask_blur: float, overridePadding: bool, inpaint_full_res_padding: float, overrideSeed: bool, overrideSteps: bool, steps: float, overrideCfgScale: bool, cfg_scale: float, overrideSize: bool, bfs_width: float, bfs_height: float, invertMask: bool, singleMaskPerImage: bool, countFaces: bool, maskWidth: float, maskHeight: float, keepOriginalName: bool, pathExisting: str, pathMasksExisting: str, output_pathExisting: str, selectedTab: str, mainTab: str, loadGenParams: bool, rotation_threshold: float, faceDetectMode: str, face_x_scale: float, face_y_scale: float, minFace: float, multiScale: float, multiScale2: float, multiScale3: float, minNeighbors: float, mpconfidence: float, mpcount: float, debugSave: bool, optimizeDetect: bool):
                    try:
                        p=original_p
                        image = input_image
                    except NameError:
                        print("Make sure you generated an image first!")
                        return

                    facecfg = FaceDetectConfig(faceDetectMode, face_x_scale, face_y_scale, minFace, multiScale, multiScale2, multiScale3, minNeighbors, mpconfidence, mpcount, debugSave, optimizeDetect)

                    finishedImages = generateImages(p, facecfg, image, input_path, searchSubdir, viewResults, int(divider), howSplit, saveMask, output_path, saveToOriginalFolder, onlyMask, saveNoFace, overridePrompt, bfs_prompt, bfs_nprompt, overrideSampler, sd_sampler, overrideModel, sd_model, overrideDenoising, denoising_strength, overrideMaskBlur, mask_blur, overridePadding, inpaint_full_res_padding, overrideSeed, overrideSteps, steps, overrideCfgScale, cfg_scale, overrideSize, bfs_width, bfs_height, invertMask, singleMaskPerImage, countFaces, maskWidth, maskHeight, keepOriginalName, pathExisting, pathMasksExisting, output_pathExisting, selectedTab, mainTab, loadGenParams, rotation_threshold)

                regen_btn.click(fn=regen, inputs=[input_path, searchSubdir, viewResults, divider, howSplit, saveMask, output_path, saveToOriginalFolder, onlyMask, saveNoFace, overridePrompt, bfs_prompt, bfs_nprompt, overrideSampler, sd_sampler, overrideModel, sd_model, overrideDenoising, denoising_strength, overrideMaskBlur, mask_blur, overridePadding, inpaint_full_res_padding, overrideSeed, overrideSteps, steps, overrideCfgScale, cfg_scale, overrideSize, bfs_width, bfs_height, invertMask, singleMaskPerImage, countFaces, maskWidth, maskHeight, keepOriginalName, pathExisting, pathMasksExisting, output_pathExisting, selectedTab, mainTab, loadGenParams, rotation_threshold, faceDetectMode, face_x_scale, face_y_scale, minFace, multiScale, multiScale2, multiScale3, minNeighbors, mpconfidence, mpcount, debugSave, optimizeDetect], outputs=None)
                
                enabled.change(switchEnableLabel, enabled, enabled)
                
                onlyMask.change(switchSaveMaskInteractivity, onlyMask, saveMask)
                onlyMask.change(switchSaveMask, onlyMask, saveMask)
                invertMask.change(switchInvertMask, invertMask, singleMaskPerImage)

                faceDetectMode.change(updateVisualizer, [searchSubdir, howSplit, divider, maskWidth, maskHeight, input_path, visualizationOpacity, faceDetectMode], exampleImage)
                minFace.change(updateVisualizer, [searchSubdir, howSplit, divider, maskWidth, maskHeight, input_path, visualizationOpacity, faceDetectMode], exampleImage)
                visualizationOpacity.change(updateVisualizer, [searchSubdir, howSplit, divider, maskWidth, maskHeight, input_path, visualizationOpacity, faceDetectMode], exampleImage)
                searchSubdir.change(updateVisualizer, [searchSubdir, howSplit, divider, maskWidth, maskHeight, input_path, visualizationOpacity, faceDetectMode], exampleImage)
                howSplit.change(updateVisualizer, [searchSubdir, howSplit, divider, maskWidth, maskHeight, input_path, visualizationOpacity, faceDetectMode], exampleImage)
                divider.change(updateVisualizer, [searchSubdir, howSplit, divider, maskWidth, maskHeight, input_path, visualizationOpacity, faceDetectMode], exampleImage)
                maskWidth.change(updateVisualizer, [searchSubdir, howSplit, divider, maskWidth, maskHeight, input_path, visualizationOpacity, faceDetectMode], exampleImage)
                maskHeight.change(updateVisualizer, [searchSubdir, howSplit, divider, maskWidth, maskHeight, input_path, visualizationOpacity, faceDetectMode], exampleImage)
                input_path.change(updateVisualizer, [searchSubdir, howSplit, divider, maskWidth, maskHeight, input_path, visualizationOpacity, faceDetectMode], exampleImage)

                overridePrompt.change(switchColumnVisibility, overridePrompt, override_prompt_col)
                overrideSize.change(switchColumnVisibility, overrideSize, override_size_col)
                overrideSteps.change(switchColumnVisibility, overrideSteps, override_steps_col)
                overrideCfgScale.change(switchColumnVisibility, overrideCfgScale, override_cfg_col)
                overrideSampler.change(switchColumnVisibility, overrideSampler, override_sampler_col)
                overrideModel.change(switchColumnVisibility, overrideModel, override_model_col)
                overrideDenoising.change(switchColumnVisibilityInverted, overrideDenoising, override_denoising_col)
                overrideMaskBlur.change(switchColumnVisibilityInverted, overrideMaskBlur, override_maskBlur_col)
                overridePadding.change(switchColumnVisibilityInverted, overridePadding, override_padding_col)

                showTips.change(switchTipsVisibility, showTips, htmlTip1)
                showTips.change(switchTipsVisibility, showTips, htmlTip2)
                showTips.change(switchTipsVisibility, showTips, htmlTip3)
                showTips.change(switchTipsVisibility, showTips, htmlTip4)
                showTips.change(switchTipsVisibility, showTips, htmlTip5)
                showTips.change(switchTipsVisibility, showTips, htmlTip6)

        return [enabled, mainTab, overridePrompt, bfs_prompt, bfs_nprompt, overrideSampler, sd_sampler, overrideModel, sd_model, overrideDenoising, denoising_strength, overrideMaskBlur, mask_blur, overridePadding, inpaint_full_res_padding, overrideSeed, overrideSteps, steps, overrideCfgScale, cfg_scale, overrideSize, bfs_width, bfs_height, input_path, searchSubdir, divider, howSplit, saveMask, output_path, saveToOriginalFolder, viewResults, saveNoFace, onlyMask, invertMask, singleMaskPerImage, countFaces, maskWidth, maskHeight, keepOriginalName, pathExisting, pathMasksExisting, output_pathExisting, selectedTab, faceDetectMode, face_x_scale, face_y_scale, minFace, multiScale, multiScale2, multiScale3, minNeighbors, mpconfidence, mpcount, debugSave, optimizeDetect, loadGenParams, rotation_threshold]

    def process(self, p, enabled, mainTab, overridePrompt, bfs_prompt, bfs_nprompt, overrideSampler, sd_sampler, overrideModel, sd_model, overrideDenoising, denoising_strength, overrideMaskBlur, mask_blur, overridePadding, inpaint_full_res_padding, overrideSeed, overrideSteps, steps, overrideCfgScale, cfg_scale, overrideSize, bfs_width, bfs_height, input_path, searchSubdir, divider, howSplit, saveMask, output_path, saveToOriginalFolder, viewResults, saveNoFace, onlyMask, invertMask, singleMaskPerImage, countFaces, maskWidth, maskHeight, keepOriginalName, pathExisting, pathMasksExisting, output_pathExisting, selectedTab, faceDetectMode, face_x_scale, face_y_scale, minFace, multiScale, multiScale2, multiScale3, minNeighbors, mpconfidence, mpcount, debugSave, optimizeDetect, loadGenParams, rotation_threshold):
        
        global original_p
        global all_images
        original_p = p

        if enabled and mainTab == "img2img":
            wasGrid = p.do_not_save_grid
            p.do_not_save_grid = True

            all_images = []

            facecfg = FaceDetectConfig(faceDetectMode, face_x_scale, face_y_scale, minFace, multiScale, multiScale2, multiScale3, minNeighbors, mpconfidence, mpcount, debugSave, optimizeDetect)

            if input_path == '':
                input_image = [ p.init_images[0] ]
            else:
                input_image = None

            finishedImages = generateImages(p, facecfg, input_image, input_path, searchSubdir, viewResults, int(divider), howSplit, saveMask, output_path, saveToOriginalFolder, onlyMask, saveNoFace, overridePrompt, bfs_prompt, bfs_nprompt, overrideSampler, sd_sampler, overrideModel, sd_model, overrideDenoising, denoising_strength, overrideMaskBlur, mask_blur, overridePadding, inpaint_full_res_padding, overrideSeed, overrideSteps, steps, overrideCfgScale, cfg_scale, overrideSize, bfs_width, bfs_height, invertMask, singleMaskPerImage, countFaces, maskWidth, maskHeight, keepOriginalName, pathExisting, pathMasksExisting, output_pathExisting, selectedTab, mainTab, loadGenParams, rotation_threshold)

            if not viewResults:
                finishedImages = []

            all_images += finishedImages

            proc = Processed(p, all_images)

            # doing this to prevent starting another img2img generation
            p.batch_size = 1
            p.n_iter = 0
            p.init_images[0] = all_images[0]
            
            p.do_not_save_grid = wasGrid
            return proc
        else:
            pass

    def postprocess(self, p, processed, enabled, mainTab, overridePrompt, bfs_prompt, bfs_nprompt, overrideSampler, sd_sampler, overrideModel, sd_model, overrideDenoising, denoising_strength, overrideMaskBlur, mask_blur, overridePadding, inpaint_full_res_padding, overrideSeed, overrideSteps, steps, overrideCfgScale, cfg_scale, overrideSize, bfs_width, bfs_height, input_path, searchSubdir, divider, howSplit, saveMask, output_path, saveToOriginalFolder, viewResults, saveNoFace, onlyMask, invertMask, singleMaskPerImage, countFaces, maskWidth, maskHeight, keepOriginalName, pathExisting, pathMasksExisting, output_pathExisting, selectedTab, faceDetectMode, face_x_scale, face_y_scale, minFace, multiScale, multiScale2, multiScale3, minNeighbors, mpconfidence, mpcount, debugSave, optimizeDetect, loadGenParams, rotation_threshold):

        global all_images
        global input_image

        if input_path == '':
            input_image = []
            input_image += processed.images

        if enabled and mainTab == "txt2img":
            
            wasGrid = p.do_not_save_grid
            p.do_not_save_grid = True
            

            all_images = []

            facecfg = FaceDetectConfig(faceDetectMode, face_x_scale, face_y_scale, minFace, multiScale, multiScale2, multiScale3, minNeighbors, mpconfidence, mpcount, debugSave, optimizeDetect)


            finishedImages = generateImages(p, facecfg, input_image, input_path, searchSubdir, viewResults, int(divider), howSplit, saveMask, output_path, saveToOriginalFolder, onlyMask, saveNoFace, overridePrompt, bfs_prompt, bfs_nprompt, overrideSampler, sd_sampler, overrideModel, sd_model, overrideDenoising, denoising_strength, overrideMaskBlur, mask_blur, overridePadding, inpaint_full_res_padding, overrideSeed, overrideSteps, steps, overrideCfgScale, cfg_scale, overrideSize, bfs_width, bfs_height, invertMask, singleMaskPerImage, countFaces, maskWidth, maskHeight, keepOriginalName, pathExisting, pathMasksExisting, output_pathExisting, selectedTab, mainTab, loadGenParams, rotation_threshold)

            if not viewResults:
                finishedImages = []

            all_images += finishedImages

            p.do_not_save_grid = wasGrid

            processed.images = all_images

        elif enabled and mainTab == "img2img":
            processed.images = all_images
        else:
            pass

# def on_ui_settings():
#     section = ('bfs', "BatchFaceSwap")
#     shared.opts.add_option("bfs_override_prompt", shared.OptionInfo(
#         False, "Default state of Override Prompt", gr.Checkbox, {"interactive": True}, section=section))
#     shared.opts.add_option("bfs_prompt", shared.OptionInfo(
#         "", "Default Prompt", section=section))
#     shared.opts.add_option("bfs_nprompt", shared.OptionInfo(
#         "", "Default Negative Prompt", section=section))


# script_callbacks.on_ui_settings(on_ui_settings)
