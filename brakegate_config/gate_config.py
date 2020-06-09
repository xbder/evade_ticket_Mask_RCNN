import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

'''
    记录闸机有效位置等信息，原图：1920*1080
    images/文件命名：ip_close.jpg 或 ip_闸机编号_open.jpg
'''

brakeCorpDict = {}    # 每个灯箱下对应的闸机的有效区域（从左往右）
brakeCorpDict['10.6.8.181'] = [(0, 0, 470, 1080), (740, 0, 1220, 1080), (1390, 0, 1780, 1080)]    # 每个value，左上右下
brakeCorpDict['10.6.8.191'] = [(70, 200, 450, 900), (615, 150, 1080, 960), (1285, 140, 1640, 930), (1765, 150, 1920, 900)]
brakeCorpDict['10.6.8.192'] = [(195, 170, 570, 935), (810, 130, 1320, 960), (1495, 190, 1830, 895)]
brakeCorpDict['10.6.8.201'] = [(207, 21, 577, 673), (787, 51, 1243, 611), (1453, 145, 1793, 611)]
brakeCorpDict['10.6.8.211'] = [(145, 137, 425, 625), (637, 83, 1099, 729), (1305, 121, 1703, 709), (1809, 167, 1919, 663)]
brakeCorpDict['10.6.8.221'] = [(0, 205, 195, 995), (250, 200, 670, 890), (870, 190, 1350, 950), (1520, 200, 1760, 870)]
brakeCorpDict['10.6.8.222'] = [(365, 0, 840, 1080), (1035, 0, 1515, 1080), (1570, 0, 1920, 1080)]

# 闸机指示灯，每个通道对应的指示灯序列，(0, 0, 0, 0)为没有指示灯
gateLightDict = {}
gateLightDict['10.6.8.181'] = [(0, 0, 0, 0), (545, 509, 697, 567), (1275, 471, 1439, 535)]    # 每个value，左上右下
gateLightDict['10.6.8.191'] = [(393, 493, 543, 563), (1131, 511, 1289, 569), (1741, 539, 1827, 583), (0, 0, 0, 0)]
gateLightDict['10.6.8.192'] = [(591, 503, 765, 561), (1357, 543, 1499, 595), (1873, 561, 1920, 595)]
gateLightDict['10.6.8.201'] = [(81, 335, 165, 375), (591, 289, 743, 341), (1295, 367, 1431, 415)]
gateLightDict['10.6.8.211'] = [(453, 331, 587, 375), (1133, 323, 1275, 375), (1723, 355, 1809, 401), (0, 0, 0, 0)]
gateLightDict['10.6.8.221'] = [(0, 0, 0, 0), (113, 545, 211, 591), (685, 567, 833, 617), (1391, 583, 1517, 629)]
gateLightDict['10.6.8.222'] = [(849, 469, 1003, 521), (1543, 459, 1659, 505), (0, 0, 0, 0)]

# 闸机开关检测区域（从左到右）
brakeCheckDict = {}
brakeCheckDict['10.6.8.181'] = [(0, 540, 380, 565), (771, 431, 1201, 495), (1475, 505, 1725, 570)]    # 每个value，左上右下
brakeCheckDict['10.6.8.191'] = [(110, 445, 350, 515), (625, 570, 1065, 630), (1330, 480, 1640, 530), (1845, 590, 1920, 630)]
brakeCheckDict['10.6.8.192'] = [(245, 445, 550, 505), (830, 580, 1260, 650), (1540, 505, 1775, 555)]
brakeCheckDict['10.6.8.201'] = [(259, 371, 545, 451), (803, 265, 1211, 351), (1471, 389, 1717, 455)]
brakeCheckDict['10.6.8.211'] = [(199, 311, 419, 387), (669, 397, 1073, 453), (1323, 295, 1619, 377), (1835, 435, 1915, 473)]
brakeCheckDict['10.6.8.221'] = [(0, 560, 85, 620), (310, 480, 635, 540), (890, 580, 1295, 665), (1560, 505, 1735, 570)]
brakeCheckDict['10.6.8.222'] = [(430, 435, 795, 520), (1060, 525, 1450, 585), (1685, 425, 1865, 490)]

# 闸机通道方向：1进站，0出站
brakeDirectDict = {}
brakeDirectDict["10.6.8.181"] = 0
brakeDirectDict["10.6.8.222"] = 1

# 同一通道，同时出现的两个人距离阀值
up_distance_threshold=600
down_distance_threshold=200

# 手动nms
area_threshold=25000
iou4small_threshold=0.45

# 只检测红灯即可
red_hsv_lower = np.array([0, 43, 46])
red_hsv_upper = np.array([10, 255, 255])

# 根据以上坐标裁剪各闸机图片
def cut_brake_area():
    # 关闭状态
    image_path = "./images/"
    target_path = "./gatebrakeimage/"

    for imgfile in os.listdir(image_path):
        fullpath = os.path.join(image_path, imgfile)
        ip = imgfile.split("_")[0]

        if imgfile.__contains__("close"):    # ip_close.jpg
            boxes = brakeCheckDict[ip]
            im = Image.open(fullpath)
            for i in range(len(boxes)):
                tmp = im.crop(boxes[i])
                tmp_file = str(ip) + "_" + str(i) + "_close.jpg"
                tmp.save(os.path.join(target_path, tmp_file), quality=95)
        elif imgfile.__contains__("open"):    # ip_闸机编号_open.jpg
            boxes = brakeCheckDict[ip]
            nums = imgfile.split("_")[1]    # 该位置为闸机编号
            im = Image.open(fullpath)
            tmp = im.crop(boxes[int(nums)])
            tmp_file = str(ip) + "_" + str(nums) + "_open.jpg"
            tmp.save(os.path.join(target_path, tmp_file), quality=95)

if __name__ == '__main__':
    cut_brake_area()
