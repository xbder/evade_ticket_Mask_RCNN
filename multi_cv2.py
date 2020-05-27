import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw
import time
from brakegate_config.gate_config import brakeCorpDict, brakeCheckDict
from myutil import cleaningBoxes, getShortestDistance
import configparser
from Logger import *
from detect_util import model
import traceback

evade_save_path = "D:/workspace/evade_save_path/"
cf = configparser.ConfigParser()
cf.read("./local.cfg")
# ip = cf.get("local", "ip")
isShow = True if cf.get("local", "isShow") == "True" else False
isOutput = True if cf.get("local", "isOutput") == "True" else False
# input_path = cf.get("local", "input_path") if cf.get("local", "input_path") != "0" else 0    # 从配置文件读取网络摄像头

ip = cf.get("target", "ip")
input_path = cf.get("webcam", "input_path")

ipArr = ip.split(",")
inputArr = input_path.split(",")
n = len(ipArr)

capList = []
loggerList = []
errorlog = Logger(os.path.join('./logs/error.log'), level='info')
for i in range(len(ipArr)):
    dev = inputArr[i] if inputArr[i] != "0" else 0
    capList.append(cv2.VideoCapture(dev))
    # capList.append(cv2.VideoStream(dev))
    loggerList.append(Logger(os.path.join('./logs/', ipArr[i] + ".log"), level='info'))

# for cap in capList:
#     print(cap)
#     print()

for i in range(len(capList)):
    print(capList[i])
    print(loggerList[i])

count = 0
while True:
    for i in range(len(capList)):
        try:
            cap = capList[i]
            ret, frame = cap.read()
            if frame is None:
                break
            # print("frame: ", type(frame))
            count += 1
            start = time.time()
            # print("count:", count)
            results = model.detect([frame], verbose=1)

            # Visualize results
            r = results[0]
            # print(r['rois'], r['class_ids'])

            # r['rois']，为矩形框的坐标：y1, x1, y2, x2 = boxes[i]，range of interest
            # r['class_ids']，类别id
            # r['scores']，类别得分
            # save_instances(frame, , r['masks'], r['class_ids'],
            #                class_names, os.path.join(output_path, name), r['scores'])

            content_list = [1, 15, 16]
            personList = []
            for box, class_id in zip(r['rois'], r['class_ids']):
                if class_id in content_list:
                    top, left, bottom, right = box
                    personList.append([top, left, bottom, right])
            #         print("personList--box: ", type(box), box)
            # print("personList: ", personList)
            afterCleanList = cleaningBoxes(personList)  # boxes清洗：area<阀值 && iou4small>0.45
            # print("afterCleanList:", afterCleanList)
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
                    del draw

            result = np.asarray(img)
            if isShow:
                cv2.imshow("img", result)
                cv2.waitKey(1)
            if isOutput and len(taopiaoList) > 0:  # 只有存在逃票行为时才保存图片
                save_path = os.path.join(evade_save_path, ipArr[i])
                if os.path.exists(save_path) is False:
                    os.makedirs(save_path)
                savefile = os.path.join(save_path, str(count) + "-" + str(int(time.time())) + ".jpg")
                cv2.imwrite(savefile, result)
                loggerList[i].logger.info("%s %s %s %s" % (str(count), savefile, taopiaoList, str(time.time() - start)))
            else:
                loggerList[i].logger.info("%s %s" % (str(count), str(time.time() - start)))
        except Exception as e:
            errorlog.logger.error(traceback.format_exc())

