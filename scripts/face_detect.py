FaceDetectDevelopment = False # set to True enable the development panel UI

from modules import images
from modules.shared import opts
from modules.paths import models_path
from modules.textual_inversion import autocrop

from PIL import Image, ImageOps

import mediapipe as mp
import numpy as np
import math
import cv2
import sys
import os
from enum import IntEnum

class FaceMode(IntEnum):
    # these can be in any order, but they must range from 0..N with no gaps
    ORIGINAL=0,
    OPENCV_NORMAL=1
    OPENCV_SLOW=2
    OPENCV_SLOWEST=3
    YUNET=4
    DEVELOPMENT=5 # must be highest numbered to avoid breaking UI

    DEFAULT=4     # the one the UI defaults to

# for the UI dropdown, we want an array that's indexed by the above numbers, and we don't want
# bugs if you tweak them and don't keep the array in sync, so initialize the array explicitly
# using them as indices:

face_mode_init = [
    (FaceMode.ORIGINAL      , "Fastest (mediapipe, max 5 faces)"),
    (FaceMode.OPENCV_NORMAL , "Normal (OpenCV + mediapipe)"),
    (FaceMode.OPENCV_SLOW   , "Slow (OpenCV + mediapipe)"),
    (FaceMode.OPENCV_SLOWEST, "Extremely slow (OpenCV + mediapipe)"),
    (FaceMode.YUNET, "YuNet (YuNet + mediapipe)")
]
if FaceDetectDevelopment:
    face_mode_init.append((FaceMode.DEVELOPMENT, "Development testing"))

face_mode_names = [None] * len(face_mode_init)
for index,name in face_mode_init:
    face_mode_names[index] = name

class FaceDetectConfig:

    def __init__(self, faceMode, face_x_scale=None, face_y_scale=0, minFace=0, multiScale=0, multiScale2=0, multiScale3=0, minNeighbors=0, mpconfidence=0, mpcount=0, debugSave=0, optimizeDetect=0):
        self.faceMode       = faceMode
        self.face_x_scale   = face_x_scale
        self.face_y_scale   = face_y_scale
        self.minFaceSize    = int(minFace)
        self.multiScale     = multiScale
        self.multiScale2    = multiScale2
        self.multiScale3    = multiScale3
        self.minNeighbors   = int(minNeighbors)
        self.mpconfidence   = mpconfidence
        self.mpcount        = mpcount
        self.debugSave      = debugSave
        self.optimizeDetect = optimizeDetect

        # allow this function to be called with just faceMode (used by updateVisualizer)
        # or with an explicit list of values (from the main UI) but throw away those values
        # if we're not in development
        #
        # for the "just faceMode" version we could put most of these as default arguments,
        # but then we'd have to maintain the defaults both above and below, so easier on
        # development to only put the correct defaults below:

        if not FaceDetectDevelopment or (face_x_scale is None):
            # If not in development mode, override all passed-in parameters to defaults
            # also, if called from UpdateVisualizer, all the arguments will be default values
            # we use None/0 in the parameter list above so we don't have to update the default values in two places in the code
            self.face_x_scale=4.0
            self.face_y_scale=2.5
            self.minFace=30
            self.multiScale=1.03
            self.multiScale2=1
            self.multiScale3=1
            self.minNeighbors=5
            self.mpconfidence=0.5
            self.mpcount=5
            self.debugSave=False
            self.optimizeDetect=False

        if True: # disable this to alter these modes specifically
            if   faceMode == FaceMode.OPENCV_NORMAL:
                self.multiScale = 1.1
            elif faceMode == FaceMode.OPENCV_SLOW:
                self.multiScale = 1.01
            elif faceMode == FaceMode.OPENCV_SLOWEST:
                self.multiScale = 1.003

def getFacialLandmarks(image, facecfg):
    height, width, _ = image.shape
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True,max_num_faces=facecfg.mpcount,min_detection_confidence=facecfg.mpconfidence) as face_mesh:
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

def computeFaceInfo(landmark, onlyHorizontal, divider, small_width, small_height, small_image_index):
    x_chin = landmark[152][0]
    y_chin = -landmark[152][1]
    x_forehead = landmark[10][0]
    y_forehead = -landmark[10][1]

    deltaX = x_forehead - x_chin
    deltaY = y_forehead - y_chin
    
    face_angle = math.atan2(deltaY, deltaX) * 180 / math.pi

    # compute center in global coordinates in case the image was split
    if onlyHorizontal == True:
        x = ((small_image_index // divider) * small_width ) + landmark[0][0]
        y = ((small_image_index %  divider) * small_height) + landmark[0][1]
    else:
        x = ((small_image_index %  divider) * small_width ) + landmark[0][0]
        y = ((small_image_index // divider) * small_height) + landmark[0][1]

    return { "angle": face_angle, "center": (x,y) }

# try to get landmarks for a face located at rect
def getFacialLandmarkConvexHull(image, rect, onlyHorizontal, divider, small_width, small_height, small_image_index, facecfg):
    image = np.array(image)
    height, width, channels = image.shape

    # make a subimage to hand to FaceMesh
    (x,y,w,h) = rect

    face_center_x      = (x + w//2)
    face_center_y      = (y + h//2)

    # the new image is just 2x the size of the face
    subrect_width  = int(float(w) * facecfg.face_x_scale)
    subrect_height = int(float(h) * facecfg.face_y_scale)
    subrect_halfwidth  = subrect_width      // 2
    subrect_halfheight = subrect_height     // 2
    subrect_width      = subrect_halfwidth  * 2;
    subrect_height     = subrect_halfheight * 2;
    subrect_x_center = face_center_x
    subrect_y_center = face_center_y

    subimage = np.zeros((subrect_height, subrect_width, channels), np.uint8)

    # this is the coordinates of the top left of the subimage relative to the original image
    subrect_x0 = subrect_x_center - subrect_halfwidth
    subrect_y0 = subrect_y_center - subrect_halfheight

    # we allow room for up to 1/2 of a face adjacent
    crop_face_x0 = face_center_x - w
    crop_face_x1 = face_center_x + w
    crop_face_y0 = face_center_y - h
    crop_face_y1 = face_center_y + h

    crop_face_x0 = max(crop_face_x0, 0 )
    crop_face_y0 = max(crop_face_y0, 0)
    crop_face_x1 = min(crop_face_x1, width )
    crop_face_y1 = min(crop_face_y1, height)

    # now crop the face coordinates down to the subrect as well
    crop_face_x0 = max(crop_face_x0, subrect_x0)
    crop_face_y0 = max(crop_face_y0, subrect_y0)
    crop_face_x1 = min(crop_face_x1, subrect_x0 + subrect_width );
    crop_face_y1 = min(crop_face_y1, subrect_y0 + subrect_height);

    face_image = image[crop_face_y0:crop_face_y1, crop_face_x0:crop_face_x1]

    # by construction the face image can't be larger than the subrect, but it can be smaller
    subimage[crop_face_y0-subrect_y0:crop_face_y1-subrect_y0, crop_face_x0-subrect_x0:crop_face_x1-subrect_x0] = face_image

    # store the face box in these coordinates for later use
    face_rect = (x - subrect_x0,
                 y - subrect_y0,
                 w,
                 h)

    # final face bounding rect must overlap at least 1/4 of pixels that CV2 expected, or it's not a match
    min_match_area = w * h // 4;

    landmarks = getFacialLandmarks(subimage, facecfg)
    mp_face_mesh = mp.solutions.face_mesh

    best_hull = None
    best_landmark = None
    best_area = min_match_area

    for landmark in landmarks:
        face_info = {}
        convexhull = cv2.convexHull(landmark)
        bounds = cv2.boundingRect(convexhull)
        # compute intersection with face_rect
        x0 = max(face_rect[0], bounds[0])
        y0 = max(face_rect[1], bounds[1])
        x1 = min(face_rect[0] + face_rect[2], bounds[0] + bounds[2])
        y1 = min(face_rect[1] + face_rect[3], bounds[1] + bounds[3])
        area = (x1-x0) * (y1-y0)

        if area > best_area:
            best_area = area
            best_hull = convexhull
            best_landmark = landmark

    face_info = None
    if best_hull is not None:
        # translate the convex hull back into the coordinate space of the passed-in image
        for i in range(len(best_hull)):
            best_hull[i][0][0] += subrect_x0
            best_hull[i][0][1] += subrect_y0

        # compute face_info and translate it back into the coordinate space
        face_info = computeFaceInfo(best_landmark, onlyHorizontal, divider, small_width, small_height, small_image_index)
        face_info["center"] = (face_info["center"][0] + subrect_x0, face_info["center"][1] + subrect_y0)

    return best_hull, face_info

def contractRect(r):
    (x0,y0,w,h) = r
    hw,hh = w/2, h/2
    xc,yc = x0+hw, y0+hh
    x0 = xc - hw*0.85
    y0 = yc - hh*0.85
    x1 = xc + hw*0.85
    y1 = yc + hh*0.85
    return (x0,y0,x1,y1)

def rectangleListOverlap(rlist, rect):
    # contract rect slightly
    rx0,ry0,rx1,ry1 = contractRect(rect)

    for r in rlist:
        x0,y0,x1,y1 = contractRect(r)
        if x0 < rx1 and x1 > rx0 and y0 < ry1 and y1 > ry0:
            return r
    return None

# use cv2 detectMultiScale directly
def getFaceRectanglesSimple(image, known_face_rects, facecfg):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("extensions/batch-face-swap/scripts/haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=facecfg.multiScale, minNeighbors=facecfg.minNeighbors, minSize=(facecfg.minFaceSize,facecfg.minFaceSize))

    all_faces = []
    for r in faces:
        if rectangleListOverlap(known_face_rects, r) is None:
            all_faces.append(r)
    return all_faces


# use cv2 detectMultiScale at multiple scales
def getFaceRectangles(image, known_face_rects, facecfg):
    if not facecfg.optimizeDetect:
        return getFaceRectanglesSimple(image, known_face_rects, facecfg)

    height, width, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("extensions/batch-face-swap/scripts/haarcascade_frontalface_default.xml")
    all_faces = []

    minsize = facecfg.minFaceSize
    # naiveScale is the scale between successive detections
    naiveScale = facecfg.multiScale
    partition_count = int(facecfg.multiScale2+0.5)
    partition_count = max(partition_count, 1)
    effectiveScale = math.pow(naiveScale, partition_count)

    current = gray
    total_scale = 1
    resize_scale = naiveScale

    for i in range(0,partition_count):
        faces = face_cascade.detectMultiScale(current, scaleFactor=effectiveScale, minNeighbors=facecfg.minNeighbors, minSize=(minsize,minsize))
        new_faces = []
        for rorig in faces:
            r = [ int(rorig[0] * total_scale), int(rorig[1] * total_scale), int(rorig[2] * total_scale), int(rorig[3] * total_scale) ]
            if rectangleListOverlap(known_face_rects, r) is None:
                if rectangleListOverlap(all_faces, r) is None:
                    new_faces.append(r)

        all_faces.extend(new_faces)

        total_scale *= resize_scale
        width = int(width / total_scale)
        height = int(height / total_scale)
        current = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LANCZOS4)

    return all_faces

def getFaceRectanglesYuNet(img_array, known_face_rects):
    new_faces = []
    dnn_model_path = autocrop.download_and_cache_models()
    face_detector = cv2.FaceDetectorYN.create(dnn_model_path, "", (0, 0))
    
    face_detector.setInputSize((img_array.shape[1], img_array.shape[0]))
    _, faces = face_detector.detect(img_array)

    if faces is None:
        return new_faces

    face_coords = []
    for face in faces:
        if math.isinf(face[0]):
            continue
        x = int(face[0])
        y = int(face[1])
        w = int(face[2])
        h = int(face[3])
        if w == 0 or h == 0:
            print("ignore w,h = 0 face")
            continue

        face_coords.append( [ x, y, w, h ] )
    
    for r in face_coords:
        if rectangleListOverlap(known_face_rects, r) is None:
            new_faces.append(r)
    
    return new_faces

#####################################################################################################################
#
# various attempts at optimizing. some of them were better, but a lot more
# work is needed to get something good
#
# note that these are not fully compatible with the rest of the code,
# as they've changed since it was written
#
#


def getFaceRectangles4(image, facecfg):
    if not facecfg.optimizeDetect:
        return getFaceRectanglesSimple(image, facecfg)

    height, width, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("extensions/batch-face-swap/scripts/haarcascade_frontalface_default.xml")
    all_faces = []

    resize_scale = facecfg.multiScale
    trueScale = facecfg.multiScale2
    overlapScale = facecfg.multiScale3

    size = min(width,height)
    minsize = facecfg.minFaceSize

    # run the smallest detectors from 30..60
    current = gray
    total_scale = 1.0
    #trueScale = trueScale * trueScale
    while current.shape[0] > minsize and current.shape[1] > minsize:
        height, width = current.shape[0], current.shape[1]
        maxsize = int(minsize * resize_scale * overlapScale)
        faces = face_cascade.detectMultiScale(current, scaleFactor=trueScale, minNeighbors=facecfg.minNeighbors, minSize=(minsize,minsize), maxSize=(maxsize,maxsize))
        new_faces = []
        for rorig in faces:
            r = [ int(rorig[0] * total_scale), int(rorig[1] * total_scale), int(rorig[2] * total_scale), int(rorig[3] * total_scale) ]

            # clamp detected shape back to range, this shou'dn't be necessary
            orig = r.copy()
            if r[0]+r[2] > gray.shape[1]:
               excess = r[0] + r[2] - gray.shape[1]
               r[2] -= excess//2
               r[0] = gray.shape[1] - r[2]
            if r[1]+r[3] > gray.shape[0]:
               excess = r[1] + r[3] - gray.shape[0]
               r[3] -= excess//2
               r[1] = gray.shape[0] - r[3]
            if orig[0] != r[0] or orig[1] != r[1] or orig[2] != r[2] or orig[3] != r[3]:
               print( "Clamped bad rect from " + str(orig) + " to " + str(r) + " for image size " + str((gray.shape[1],gray.shape[0])))
               overlap = rectangleListOverlap(all_faces, r)
            if overlap is None:
               new_faces.append((r, (minsize*total_scale,maxsize*total_scale)))

        all_faces.extend(new_faces)

        width = int(width / resize_scale)
        height = int(height / resize_scale)
        current = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        total_scale *= resize_scale

        if width < 600 and height < 600:
            resize_scale = 10
            trueScale = math.pow(facecfg.multiScale2,0.5)
        else:
            trueScale = facecfg.multiScale2

    for i in range(0,len(all_faces)):
        r,_ = all_faces[i]
        all_faces[i] = r

    return all_faces

# use cv2 detectMultiScale at multiple scales
def getFaceRectangles3(image, facecfg):
    if facecfg.multiScale <= 1.1:
        return getFaceRectanglesSimple(image, facecfg)

    height, width, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("extensions/batch-face-swap/scripts/haarcascade_frontalface_default.xml")
    all_faces = []

    size = min(width,height)

    while size >= facecfg.minFaceSize:
        maxsize = size
        minsize = int(size / facecfg.multiScale)

        # make sure there's some overlap
        maxsize = int(maxsize*1.1 + 1)
        minsiez = int(minsize/1.1 - 1)

        ratio = float(maxsize) / float(minsize)
        stepCount = (maxsize - minsize) * facecfg.multiScale2
        scale = math.exp(math.log(ratio) / (stepCount+1))

        print( f"Compute scale factor {scale} to go from {minsize} to {maxsize} in {stepCount} steps" )
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=facecfg.minNeighbors, minSize=(minsize,minsize), maxSize=(maxsize,maxsize))
        new_faces = []
        for r in faces:
            overlap = rectangleListOverlap(all_faces, r)
            if overlap is None:
                new_faces.append((r, (minsize,maxsize)))
        all_faces.extend(new_faces)
        size = minsize

    for i in range(0,len(all_faces)):
        r,_ = all_faces[i]
        all_faces[i] = r
    return all_faces

# use cv2 detectMultiScale at multiple scales
def getFaceRectangles2(image, facecfg):
    if facecfg.multiScale < 1.5:
        return getFaceRectanglesSimple(image, facecfg)

    height, width, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("extensions/batch-face-swap/scripts/haarcascade_frontalface_default.xml")
    all_faces = []
    stepSize = int(facecfg.multiScale)
    for size in range(facecfg.minFaceSize, min(width,height), stepSize):

        # detect faces in the range size..stepSize
        minsize = size
        maxsize = size + stepSize+1

        ratio = float(maxsize) / float(size)

        # pick a scale factor so we take about stepSize*1.2 steps to go from size to maxsize
        # scale^(stepsize*1.2) = ratio
        # (stepsize*1.2) * log(scale) = log(ratio)
        # log(scale) = log(ratio) / (stepsize * 1.2)
        scale = math.exp(math.log(ratio) / (float(stepSize)*1.1))

        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=facecfg.minNeighbors, minSize=(minsize,minsize), maxSize=(maxsize,maxsize))
        new_faces = []
        for r in faces:
            overlap = rectangleListOverlap(all_faces, r)
            if overlap is None:
                new_faces.append((r, (minsize,maxsize)))
            #else:
                #(rect,size) = overlap
                #print( "Duplicate face: " + str(r) + " in size range " + str((minsize,maxsize)) + " overlapped with " + str(rect) + " from size range " + str(size))
        all_faces.extend(new_faces)

    for i in range(0,len(all_faces)):
        r,_ = all_faces[i]
        all_faces[i] = r
    return all_faces
