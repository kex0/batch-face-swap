import modules.scripts as scripts
import gradio as gr
import os

from modules import images
from modules.ui import plaintext_to_html
from modules.processing import process_images, Processed, StableDiffusionProcessing
from modules.shared import opts, cmd_opts, state

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, UnidentifiedImageError
import math

global skip
global totalNumberOfFaces

def getFacialLandmarks(image):
    height, width, _ = image.shape
    #image = cv2.resize(image,(int(width/5), int(height/5)))
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

def findFaceDivide(image, width, height, divider, onlyHorizontal, onlyVertical, file):
    global skip
    global totalNumberOfFaces

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
        # small_image.show()
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
            facesInImage += 1
        if facesInImage == 0 and i == len(small_images) - 1:
            skip = 1

        mask = np.zeros((small_height, small_width), np.uint8)
        for i in range(len(landmarks)):
            small_image = cv2.fillConvexPoly(mask, faces[i], 255)
        processed_image = Image.fromarray(small_image)
        # processed_image.show()
        processed_images.append(processed_image)

    print(f"Found {facesInImage} face(s) in {str(file)}")
    
    # Create a new image with the same size as the original large image
    new_image = Image.new('RGB', (width, height))

    # Paste the processed small images into the new image
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
    # cv2.imwrite("binary_image.jpg", binary_image)
    # define kernel
    kernel = np.ones((int(math.ceil(0.011*height)),int(math.ceil(0.011*height))),'uint8')
    dilated = cv2.dilate(binary_image,kernel,iterations=1)
    kernel = np.ones((int(math.ceil(0.0045*height)),int(math.ceil(0.0025*height))),'uint8')
    dilated = cv2.dilate(dilated,kernel,iterations=1,anchor=(1, -1))
    kernel = np.ones((int(math.ceil(0.014*height)),int(math.ceil(0.0025*height))),'uint8')
    dilated = cv2.dilate(dilated,kernel,iterations=1,anchor=(-1, 1))
    # cv2.imwrite("dilated.jpg", dilated)
    image = dilated

    image = Image.fromarray(image) 

    return image

def generateMasks(path, divider, howSplit, saveMask, pathToSave):
    global skip
    global totalNumberOfFaces
    p = StableDiffusionProcessing
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

    fileNumber = 0
    totalNumberOfFaces = 0
    print(files)
    for i, file in enumerate(files):
        fileNumber += 1
        state.job = f"{i+1} out of {len(files)}"
        if state.skipped:
            state.skipped = False

        if state.interrupted:
            state.interrupted = False
        imgPath = os.path.join(dirPath, file)
        try:
            image = Image.open(imgPath)
        except UnidentifiedImageError:
            print(f"{file} is not an image.")
            continue
        width, height = image.size
        try:
            skip = 0
            mask = findFaceDivide(image, width, height, divider, onlyHorizontal, onlyVertical, file)
            if skip == 1:
                state.skipped = True
                continue
            if saveMask == True:
                suffix = '_mask'
                if pathToSave != "":
                    images.save_image(mask, pathToSave, f"{suffix.join(os.path.splitext(file))}", p=p)
                elif pathToSave == "":
                    images.save_image(mask, opts.outdir_img2img_samples, f"{suffix.join(os.path.splitext(file))}", p=p)

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
        def switch(saveMask: bool):
            return gr.Textbox.update(visible=bool(saveMask))

        gr.HTML("<p style=\"margin-bottom:0.75em;margin-top:0.75em;font-size:1.25em;color:red\">If you're in the \"Inpaint\" tab, set \"Mask source\" to \"Upload mask\"!</p>")
        with gr.Column():
            gr.HTML("<p style=\"margin-top:0.75em;font-size:1.25em\">Overrides:</p>")
            overrideDenoising = gr.Checkbox(value=True, label="""Override "Denoising strength" to 0.5 (values between 0.4-0.6 usually give great results)""")
            overrideMaskBlur = gr.Checkbox(value=True, label="""Override "Mask blur" (it will automatically adjust based on the image size)""")
        with gr.Column():
            gr.HTML("<p style=\"margin-top:0.75em;font-size:1.25em\"><strong>Step 1:</strong> Images:</p>")
            gr.HTML("<p>(path to a folder containing images.)</p>")
            path = gr.Textbox(label="Images directory",placeholder=r"C:\Users\dude\Desktop\images")
        with gr.Column():
            gr.HTML("<p style=\"margin-top:0.75em;font-size:1.25em\"><strong>Step 2:</strong> Image splitter:</p>")
            gr.HTML("<p>This divides image to smaller images and tries to find a face in the individual smaller images.</p>")
            gr.HTML("<p>Useful when faces are small in relation to the size of the whole picture.</p>")
            gr.HTML("<p>(may result in mask that only covers a part of a face if the division goes right through the face)</p>")
            divider = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="How many times to divide image")
            howSplit = gr.Radio(["Horizontal only ▤", "Vertical only ▥", "Both ▦"], value = "Both ▦", label = "How to divide")
        with gr.Column():
            gr.HTML("<p style=\"margin-top:0.75em;font-size:1.25em\">Other:</p>")
            gr.HTML("<p>Simple utility to see how many faces do your current settings detect without generating SD image.</p>")
            gr.HTML("<p>You can also save generated masks to disk. (if you leave path empty, it will save the masks to your default webui outputs directory)</p>")
            saveMask = gr.Checkbox(value=False, label="Save masks to disk") 
            pathToSave = gr.Textbox(label="Mask save directory (OPTIONAL)",placeholder=r"C:\Users\dude\Desktop\masks (OPTIONAL)",visible=False)
            testMask = gr.Button(value="Generate masks",variant="primary")
            testMaskOut = gr.HTML(value="",visible=False)
            testMask.click(fn=generateMasks,inputs=[path, divider, howSplit, saveMask, pathToSave],outputs=[testMaskOut])

        saveMask.change(switch, saveMask, pathToSave)

        return [overrideDenoising, overrideMaskBlur, path, divider, howSplit, testMask, saveMask, pathToSave]

    def run(self, p, overrideDenoising, overrideMaskBlur, path, divider, howSplit, testMask, saveMask, pathToSave):
        global skip
        global totalNumberOfFaces

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
        print(f"Will process {len(files)} images, creating {p.n_iter * p.batch_size} new images for each.")
        
        state.job_count = len(files) * p.n_iter

        fileNumber = 0
        history = []
        all_images = []
        totalNumberOfFaces = 0
        print(files)
        for i, file in enumerate(files):
            fileNumber += 1
            state.job = f"{i+1} out of {len(files)}"
            if state.skipped:
                state.skipped = False

            if state.interrupted:
                break
            imgPath = os.path.join(dirPath, file)
            try:
                image = Image.open(imgPath)
            except UnidentifiedImageError:
                print(f"{file} is not an image.")
                continue
            width, height = image.size
            p.init_images = [image]
            if overrideDenoising == True:
                p.denoising_strength = 0.5
            if overrideMaskBlur == True:
                p.mask_blur = int(math.ceil(0.01*height))

            try:
                skip = 0
                mask = findFaceDivide(image, width, height, divider, onlyHorizontal, onlyVertical, file)
                if skip == 1:
                    state.skipped = True
                    continue
                p.image_mask = mask
                proc = process_images(p)
                for i in range(p.batch_size):
                    history.append(proc.images[i])

            except cv2.error as e:
                print(e)

        print(f"Found {totalNumberOfFaces} faces in {len(files)} images.") 

        all_images += history   
        proc = Processed(p, all_images)

        return proc