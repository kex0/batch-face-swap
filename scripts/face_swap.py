import modules.scripts as scripts
import gradio as gr
import os

from modules import images, masking
from modules.processing import process_images, Processed, StableDiffusionProcessingImg2Img, StableDiffusionProcessing
from modules.shared import opts, cmd_opts, state

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, UnidentifiedImageError, ImageOps
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

def findFaceDivide(image, width, height, divider, onlyHorizontal, onlyVertical, file, totalNumberOfFaces, skip):
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

    print(f"Found {facesInImage} face(s) in {str(file)}")
    
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
    image = dilated

    if facesInImage > 1:
        segmentFaces = True
        while (segmentFaces):
            currentBiggest = findBiggestBlob(image)
            masks.append(currentBiggest)
            image = image - currentBiggest

            whitePixels = cv2.countNonZero(image)
            whitePixelThreshold = 0.005 * (widthOriginal * heightOriginal)
            if (whitePixels < whitePixelThreshold):
                segmentFaces = False
    else:
        masks.append(dilated)

    return masks, totalNumberOfFaces, skip 

def generateMasks(path, divider, howSplit, saveMask, pathToSave):
    p = StableDiffusionProcessingImg2Img(StableDiffusionProcessing)
    if howSplit == "Horizontal only ▤":
        onlyHorizontal = True
        onlyVertical = False
    elif howSplit == "Vertical only ▥":
        onlyHorizontal = False
        onlyVertical = True
    elif howSplit == "Both ▦":
        onlyHorizontal = False
        onlyVertical = False

    dirPath = path
    divider = int(divider)
    files = os.listdir(dirPath)
    totalNumberOfFaces = 0

    for i, file in enumerate(files):
        state.job = f"{i+1} out of {len(files)}"
        if state.skipped:
            state.skipped = False
        if state.interrupted:
            state.interrupted = False

        imgPath = os.path.join(dirPath, file)
        try:
            image = Image.open(imgPath)
            width, height = image.size
        except UnidentifiedImageError:
            print(f"{file} is not an image.")
            continue
        
        try:
            skip = 0
            masks, totalNumberOfFaces, skip = findFaceDivide(image, width, height, divider, onlyHorizontal, onlyVertical, file, totalNumberOfFaces, skip)
            if skip == 1:
                state.skipped = True
                continue
            if saveMask == True:
                suffix = '_mask'
                if pathToSave != "":
                    for i, mask in enumerate(masks):
                        mask = Image.fromarray(mask)
                        images.save_image(mask, pathToSave, "", p.seed, p.prompt, opts.samples_format, p=p, suffix=suffix)
                    
                elif pathToSave == "":
                    for i, mask in enumerate(masks):
                        mask = Image.fromarray(mask)
                        images.save_image(mask, opts.outdir_img2img_samples, "", p.seed, p.prompt, opts.samples_format, p=p, suffix=suffix)

        except cv2.error as e:
            print(e)

    print(f"Found {totalNumberOfFaces} faces in {len(files)} images.") 
    return gr.HTML.update(value=f"<p style=\"font-size:1.25em\">Found {totalNumberOfFaces} faces in {len(files)} images.</p>",visible=True)

class Script(scripts.Script):  
    def title(self):
        return "Batch Face Swap"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        def switchExample(howSplit: str, divider: int, path: str):
            if path != "":
                files = os.listdir(path)
                imgPath = os.path.join(path, files[0])
                image = Image.open(imgPath)
                maxsize = (1000, 500)
                image.thumbnail(maxsize,Image.ANTIALIAS)

            if "Both" in howSplit:
                if path == "":
                    image = Image.open("./extensions/batch-face-swap/images/exampleB.jpg")
                width, height = image.size
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
                if divider > 1:
                    for i in range(divider-1):
                        start_point = (0, int((height/divider)*(i+1)))
                        end_point = (int(width), int((height/divider)*(i+1)))
                        color = (255, 0, 0)
                        thickness = 4
                        image = cv2.line(image, start_point, end_point, color, thickness)

                    for i in range(divider-1):
                        start_point = (int((width/divider)*(i+1)), 0)
                        end_point = (int((width/divider)*(i+1)), int(height))
                        color = (255, 0, 0)
                        thickness = 4
                        image = cv2.line(image, start_point, end_point, color, thickness)

            elif "Vertical" in howSplit:
                if path == "":
                    image = Image.open("./extensions/batch-face-swap/images/exampleV.jpg")
                width, height = image.size
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
                if divider > 1:
                    for i in range(divider-1):
                        start_point = (int((width/divider)*(i+1)), 0)
                        end_point = (int((width/divider)*(i+1)), int(height))
                        color = (255, 0, 0)
                        thickness = 4
                        image = cv2.line(image, start_point, end_point, color, thickness)

            else:
                if path == "":
                    image = Image.open("./extensions/batch-face-swap/images/exampleH.jpg")
                width, height = image.size
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
                if divider > 1:
                    for i in range(divider-1):
                        start_point = (0, int((height/divider)*(i+1)))
                        end_point = (int(width), int((height/divider)*(i+1)))
                        color = (255, 0, 0)
                        thickness = 4
                        image = cv2.line(image, start_point, end_point, color, thickness)

            image = Image.fromarray(image)
            update = gr.Image.update(value=image)
            return update

        def switchTextbox(saveMask: bool):
            return gr.Textbox.update(visible=bool(saveMask))
        def switchHTML(showTips: bool):
            return gr.HTML.update(visible=bool(showTips))


        gr.HTML("<p style=\"margin-bottom:0.75em;margin-top:0.75em;font-size:1.5em;color:red\">Make sure you're in the \"Inpaint upload\" tab!</p>") 
        with gr.Column():
            gr.HTML("<p style=\"margin-top:0.75em;font-size:1.25em\">Overrides:</p>")
            overrideDenoising = gr.Checkbox(value=True, label="""Override "Denoising strength" to 0.5 (values between 0.4-0.6 usually give great results)""")
            overrideMaskBlur = gr.Checkbox(value=True, label="""Override "Mask blur" (it will automatically adjust based on the image size)""")
        with gr.Column():
            gr.HTML("<p style=\"margin-top:0.75em;font-size:1.25em\"><strong>Step 1:</strong> Images:</p>")
            htmlTip1 = gr.HTML("<p>Path to a folder containing images.</p>",visible=False)
            path = gr.Textbox(label="Images directory",placeholder=r"C:\Users\dude\Desktop\images")
        with gr.Column():
            gr.HTML("<p style=\"margin-top:0.75em;font-size:1.25em\"><strong>Step 2:</strong> Image splitter:</p>")
            htmlTip2 = gr.HTML("<p>This divides image to smaller images and tries to find a face in the individual smaller images.</p><p>Useful when faces are small in relation to the size of the whole picture and not being detected.</p><p>(may result in mask that only covers a part of a face or no detection if the division goes right through the face)</p>",visible=False)
            divider = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="How many images to divide into")
            howSplit = gr.Radio(["Horizontal only ▤", "Vertical only ▥", "Both ▦"], value = "Both ▦", label = "How to divide")
            with gr.Accordion(label="Split visualizer", open=False):    
                exampleImage = gr.Image(value=Image.open("./extensions/batch-face-swap/images/exampleB.jpg"), label="Split visualizer", type="pil", visible=True).style(height=500)
        with gr.Column():
            gr.HTML("<p style=\"margin-top:0.75em;font-size:1.25em\">Other:</p>")
            htmlTip3 = gr.HTML("<p>Press 'Generate masks' button to see how many faces do your current settings detect without generating SD image.</p><p>You can also save generated masks to disk. (if you leave path empty, it will save the masks to your default webui outputs directory)</p><p>Activate 'View all results' checkbox to see results in the WebUI at the end (not recommended when processing a large number of images)</p>",visible=False)
            showTips = gr.Checkbox(value=False, label="Show tips")
            viewResults = gr.Checkbox(value=False, label="View all results")
        with gr.Column():
            saveMask = gr.Checkbox(value=False, label="Save masks to disk")  
            pathToSave = gr.Textbox(label="Mask save directory (OPTIONAL)",placeholder=r"C:\Users\dude\Desktop\masks (OPTIONAL)",visible=False)
            testMask = gr.Button(value="Generate masks",variant="primary")
            testMaskOut = gr.HTML(value="",visible=False)
            testMask.click(fn=generateMasks,inputs=[path, divider, howSplit, saveMask, pathToSave],outputs=[testMaskOut])

        path.change(fn=None, _js="gradioApp().getElementById('mode_img2img').querySelectorAll('button')[4].click()", inputs=None, outputs=None)
        saveMask.change(switchTextbox, saveMask, pathToSave)
        showTips.change(switchHTML, showTips, htmlTip1)
        showTips.change(switchHTML, showTips, htmlTip2)
        showTips.change(switchHTML, showTips, htmlTip3)
        
        howSplit.change(switchExample, [howSplit, divider, path], exampleImage)
        divider.change(switchExample, [howSplit, divider, path], exampleImage)
        path.change(switchExample, [howSplit, divider, path], exampleImage)



        return [overrideDenoising, overrideMaskBlur, path, divider, howSplit, testMask, saveMask, pathToSave, viewResults]

    def run(self, p, overrideDenoising, overrideMaskBlur, path, divider, howSplit, testMask, saveMask, pathToSave, viewResults):
        if howSplit == "Horizontal only ▤":
            onlyHorizontal = True
            onlyVertical = False
        elif howSplit == "Vertical only ▥":
            onlyHorizontal = False
            onlyVertical = True
        elif howSplit == "Both ▦":
            onlyHorizontal = False
            onlyVertical = False

        finishedImages = []
        all_images = []
        totalNumberOfFaces = 0
        dirPath = path
        divider = int(divider)
        files = os.listdir(dirPath)

        print(f"\nWill process {len(files)} images, creating {p.n_iter * p.batch_size} new images for each.")
        state.job_count = len(files) * p.n_iter

        for i, file in enumerate(files):
            state.job = f"{i+1} out of {len(files)}"
            if state.skipped:
                state.skipped = False
            if state.interrupted:
                break

            imgPath = os.path.join(dirPath, file)
            try:
                image = Image.open(imgPath)
                width, height = image.size
            except UnidentifiedImageError:
                print(f"{file} is not an image.")
                continue
            
            if overrideDenoising == True:
                p.denoising_strength = 0.5
            if overrideMaskBlur == True:
                p.mask_blur = int(math.ceil(0.01*height))

            try:
                skip = 0
                masks, totalNumberOfFaces, skip = findFaceDivide(image, width, height, divider, onlyHorizontal, onlyVertical, file, totalNumberOfFaces, skip)
                if skip == 1:
                    state.skipped = True
                    continue
                
                if len(masks) == 1:
                    if not viewResults:
                        finishedImages = []

                    mask = Image.fromarray(masks[0])

                    p.init_images = [image]
                    p.image_mask = mask

                    proc = process_images(p)

                    for n in range(p.batch_size):
                        finishedImages.append(proc.images[n])
                else:
                    if not viewResults:
                        finishedImages = []

                    generatedImages = []
                    paste_to = []
                    imageOriginal = image
                    overlay_image = image
                    p.do_not_save_samples = True

                    for n, mask in enumerate(masks):
                        mask = Image.fromarray(masks[n])

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
                        images.save_image(image, p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, p=p)
                        finishedImages.append(image)

                    p.do_not_save_samples = False

            except cv2.error as e:
                print(e)

        print(f"Found {totalNumberOfFaces} faces in {len(files)} images.") 

        all_images += finishedImages   
        proc = Processed(p, all_images)

        return proc
