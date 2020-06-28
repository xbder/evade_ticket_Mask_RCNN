import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw
import time
from brakegate_config.gate_config import brakeCorpDict, brakeCheckDict, brakeDirectDict
from myutil import cleaningBoxes, getShortestDistance
import configparser
from Logger import *
from detect_util import model
import traceback

evade_save_path = "D:/workspace/evade_save_path/"
normal_save_path = "D:/workspace/normal_save_path/"
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
# for i in range(len(ipArr)):
#     t1 = time.time()
#     dev = inputArr[i] if inputArr[i] != "0" else 0
#     capList.append(cv2.VideoCapture(dev))
#     # capList.append(cv2.VideoStream(dev))
#     loggerList.append(Logger(os.path.join('./logs/', ipArr[i] + ".log"), level='info'))
#     errorlog.logger.info("已连接 %s webcam, dev: %s, 耗时 %s" % (str(ipArr[i]), dev, (time.time() - t1)))

# for cap in capList:
#     print(cap)
#     print()

# for i in range(len(capList)):
#     print(capList[i])
#     print(loggerList[i])

'''
    判断点在哪个矩形框内，返回点所在的框的序号
'''
def isin(center, cropAreaList):
    where_gate = 0    # 默认处于0号闸机
    for i in range(len(cropAreaList)):
        left, top, right, bottom = cropAreaList[i]    # 左上右下
        (targetx, targety) = center

        if (targetx > left and targetx < right) and (targety > top and targety < bottom):
            where_gate = i
    return str(where_gate)

count = 0
personNumsDict = {}
final_total = 0
while True:
    time.sleep(0.5)
    for i in range(len(inputArr)):
        try:
            ip = ipArr[i]
            cropAreaList = brakeCorpDict[ip]    # 每个闸机下有效位置
            direct = brakeDirectDict[ip]    # ip对应闸机的方向

            t1 = time.time()
            # cap = capList[i]
            dev = inputArr[i] if inputArr[i] != "0" else 0
            cap = cv2.VideoCapture(dev)
            log = Logger(os.path.join('./logs/', ipArr[i] + ".log"), level='info')
            errorlog.logger.info("已连接 %s webcam, dev: %s, 耗时 %s" % (str(ipArr[i]), dev, (time.time() - t1)))

            # if cap.isOpened() is False:
            #     continue

            t2 = time.time()
            ret, frame = cap.read()
            if frame is None:
                break
            print("frame: ", type(frame))
            read_time = time.time() - t2
            count += 1
            # if count % 30 != 0:
            #     continue

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
            log.logger.info("afterCleanList: %s" % afterCleanList)
            final_total += len(afterCleanList)    # 累加人数
            log.logger.info("****** final total: %s" % (final_total))

            person_details_list = []    # 每个人详细信息的列表
            for per_person in afterCleanList:
                top, left, bottom, right = per_person
                w = right - left
                h = bottom - top
                center = (left + w / 2, top + h / 2)

                brake_num = isin(center, cropAreaList)    # 判断中心点在哪个区域内
                per_person_details = {}  # 每个人的详细信息
                per_person_details["location"] = per_person    # 每个人的位置：上左下右
                per_person_details["brake_num"] = brake_num    # 哪个闸机下
                per_person_details["direct"] = direct          # 方向

                person_details_list.append(per_person_details)    # 详细列表

                if brake_num in personNumsDict.keys():
                    tmp_nums = personNumsDict[brake_num]
                    personNumsDict[brake_num] = tmp_nums + 1
                else:
                    personNumsDict[brake_num] = 1

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
            if isOutput:    # 需要保存图片
                if len(taopiaoList) > 0:  # 存在逃票行为
                    save_path = os.path.join(evade_save_path, ipArr[i])
                    if os.path.exists(save_path) is False:
                        os.makedirs(save_path)
                    savefile = os.path.join(save_path, str(count) + "-" + str(int(time.time())) + ".jpg")
                    cv2.imwrite(savefile, result)
                else:    # 正常的图片
                    save_path = os.path.join(normal_save_path, ipArr[i])
                    if os.path.exists(save_path) is False:
                        os.makedirs(save_path)
                    savefile = os.path.join(save_path, str(count) + "-" + str(int(time.time())) + ".jpg")
                    cv2.imwrite(savefile, result)
                detailInfo = {}
                detailInfo["count"] = count
                detailInfo["savefile"] = savefile
                detailInfo["taopiaoList"] = taopiaoList
                detailInfo["read_time"] = read_time
                detailInfo["detect_time"] = str(time.time() - start)
                detailInfo["final_total"] = final_total
                detailInfo["personNumsDict"] = personNumsDict
                detailInfo["person_details_list"] = person_details_list

                log.logger.info("%s" % (str(detailInfo)))
            else:    # 不需要保存图片（只写日志）
                detailInfo = {}
                detailInfo["count"] = count
                detailInfo["read_time"] = read_time
                detailInfo["detect_time"] = str(time.time() - start)
                detailInfo["final_total"] = final_total
                detailInfo["personNumsDict"] = personNumsDict
                detailInfo["person_details_list"] = person_details_list

                log.logger.info("%s" % (str(detailInfo)))
            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            errorlog.logger.error(traceback.format_exc())

