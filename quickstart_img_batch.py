import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import ipdb
import cv2
from PIL import Image, ImageDraw
import time
from myutil import getShortestDistance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ipdb.set_trace(context=35)
# namely current root dir
# ROOT_DIR = os.path.abspath("/py/Mask R-CNN/Mask_RCNN-tf")
ROOT_DIR = "./"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from myutil import save_instances, cleaningBoxes

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# import coco
import samples.coco.coco as coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

input_path = "D:/tiaozhaji"
output_path = "D:/tiaozhaji-out"

cap = cv2.VideoCapture(input_path)

count = 0

for file in os.listdir(input_path):
    fullfile = os.path.join(input_path, file)

    output = file.split("_")
    name = output[len(output) -1].split(".")[0] + "-" + str(count) + ".jpg"
    print(name)

    frame = cv2.imread(fullfile)
    print("frame: ", type(frame))

    if frame is None:
        break
    count += 1
    start = time.time()
    print("count:", count)
    results = model.detect([frame], verbose=1)

    # Visualize results
    r = results[0]
    # print(r['rois'], r['class_ids'])

    # r['rois']，为矩形框的坐标：y1, x1, y2, x2 = boxes[i]
    # r['class_ids']，类别id
    # r['scores']，类别得分
    save_instances(frame, r['rois'], r['masks'], r['class_ids'],
                   class_names, os.path.join(output_path, name), r['scores'])

    print("===time===", time.time() - start)