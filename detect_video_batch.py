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
from brakegate_config.gate_config import brakeCorpDict, brakeCheckDict
from myutil import cleaningBoxes

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
from myutil import save_instances, getShortestDistance

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

video_path = "E:/BaiduNetdiskDownload/2020-04-17/"
out_path = "C:/workspace/2020-04-17-out/"

for file in os.listdir(video_path):
    input_path = os.path.join(video_path, file)

    video_name = input_path[input_path.rindex("/") + 1: -4]
    output_path = os.path.join(out_path, video_name + "-video-out.mp4")

    print(input_path)

    ip = str(video_name).split("_")[0]
    check_areas = brakeCheckDict[ip]  # 拿到该ip的摄像头的闸机开关校验区域坐标
    crops = brakeCorpDict[ip]  # 该ip下的闸机有效区域坐标

    cap = cv2.VideoCapture(input_path)
    video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

    count = 0

    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        print("frame: ", type(frame))
        count += 1
        start = time.time()
        print("count:", count)
        results = model.detect([frame], verbose=1)

        # Visualize results
        r = results[0]
        # print(r['rois'], r['class_ids'])

        content_list = [1, 15, 16]
        personList = []
        for box, class_id in zip(r['rois'], r['class_ids']):
            if class_id in content_list:
                top, left, bottom, right = box
                personList.append([top, left, bottom, right])
        #         print("personList--box: ", type(box), box)
        # print("personList: ", personList)
        afterCleanList = cleaningBoxes(personList)  # boxes清洗：area<阀值 && iou4small>0.45
        print("afterCleanList:", afterCleanList)
        taopiaoList = getShortestDistance("", count, afterCleanList, None, None)  # 计算欧式距离，判断逃票
        img = Image.fromarray(frame.astype('uint8'))  # numpy.ndarray 转PIL
        thickness = 3
        color = (0, 0, 255)  # BGR
        for box in taopiaoList:
            draw = ImageDraw.Draw(img)  # PIL中色彩是RGB
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(img.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(img.size[0], np.floor(right + 0.5).astype('int32'))
            print((left, top), (right, bottom))

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=color)
            # draw.rectangle(
            #     [tuple(text_origin), tuple(text_origin + label_size)],
            #     fill=color)
            # draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        result = np.asarray(img)
        # cv2.imshow("img", result)
        # cv2.waitKey()

        if isOutput:
            out.write(result)
        print("===time===", time.time() - start)


