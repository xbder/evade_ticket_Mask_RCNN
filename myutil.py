import os
import sys
import random
import itertools
import colorsys

import cv2
import math
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon
import IPython.display
from brakegate_config.gate_config import up_distance_threshold, down_distance_threshold, area_threshold, iou4small_threshold, red_hsv_lower, red_hsv_upper


from mrcnn.visualize import random_colors, apply_mask

def save_instances(image, boxes, masks, class_ids, class_names, filename,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to save *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    plt.savefig(filename)
    # if auto_show:
    #     plt.show()


'''
    清洗boxes，要舍弃的：area < 阀值 || iou4small > 0.45
'''
def cleaningBoxes(boxes):
    newboxes = []
    # 1、先过滤掉面积小于阀值的
    for box in boxes:
        top, left, bottom, right = box
        w = right - left
        h = bottom - top
        if w*h > area_threshold:
            newboxes.append(box)

    finalboxes = []
    removeboxes = []
    # 2、iou4small，两两比，过滤掉较小的
    for box1 in newboxes:
        for box2 in newboxes:
            # if all(box1) == all(box2):    # 同一个框不用比较
            #     continue

            ## 这儿逻辑还是有点问题：0506、0507这两张图还需要再微调下
            iou4small, biggerbox, smallerbox = getiou4small(box1, box2)    # iou对于小者的比例，大框，小框
            print("iou4small", iou4small, biggerbox, smallerbox)
            if iou4small == 1.0:    # 过滤掉本身
                finalboxes.append(biggerbox)
                continue
            if iou4small > iou4small_threshold:    # 如果交集占小框的面积超过阀值，则只保留大框
                finalboxes.append(biggerbox)
                removeboxes.append(smallerbox)    # 已经决定要抑制掉的框，后期无论如何会抑制掉
            else:    # 否则全都保留
                finalboxes.append(box1)
                finalboxes.append(box2)
    return getDifferentSet(finalboxes, removeboxes)

'''
    求两个list的差集：list1有但list2没有的数据
'''
def getDifferentSet(list1, list2):
    # print("list1:", list1)
    # print("list2:", list2)
    finalList = []
    for arr in list1:
        if list2.__contains__(arr) is False:
            if finalList.__contains__(arr) is False:
                finalList.append(arr)
    return finalList


'''
    计算iou4small的值
    :return iou4small的比例, 较大的那个框
'''
def getiou4small(box1, box2):
    y1, x1, y2, x2 = box1
    w1 = x2 - x1
    h1 = y2 - y1
    area1 = w1 * h1
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    y3, x3, y4, x4 = box2
    w2 = x4 - x3
    h2 = y4 - y3
    area2 = w2 * h2
    x3, x4 = min(x3, x4), max(x3, x4)
    y3, y4 = min(y3, y4), max(y3, y4)

    iouarea = 0
    if (x2 <= x3 or x4 <= x1) and (y2 <= y3 or y4 <= y1):
        iouarea = 0
    else:
        lens = min(x2, x4) - max(x1, x3)
        wide = min(y2, y4) - max(y1, y3)
        iouarea = lens * wide

    if area1 > area2:
        biggerbox = box1
        smallerbox = box2
    else:
        biggerbox = box2
        smallerbox = box1
    return iouarea/min(area1, area2), biggerbox, smallerbox



'''
    获取矩形框的中心点间距
'''
def getShortestDistance(video_name, count, boxes, crop, brake_status):
    taopiaoList = []
    centerList = []

    # 这里有两种情况：
    # boxes = []，表示本来就没找到目标
    # boxes = [[]]，表示有找到的目标，但因不符合条件被抑制了
    for box in boxes:  # 对面积大小符合要求的，进行两个框中心点的距离值计算
        print("box: ", type(box), box)
        if len(box) == 4:
            top, left, bottom, right = box
            w = right - left
            h = bottom - top
            center = (left + w / 2, top + h / 2)
            centerList.append(center)
    for i in range(0, len(centerList)):  # centerList和boxes一样长
        for j in range(i, len(centerList)):
            person1x, person1y = centerList[i][0], centerList[i][1]
            person2x, person2y = centerList[j][0], centerList[j][1]
            distance = math.sqrt(((person1x - person2x) ** 2) +
                                 ((person1y - person2y) ** 2))
            print(i, j)
            print(centerList[i], centerList[j], distance)
            # fo.write("%s %s %s %s \n" % (video_name + "-" + str(count), centerList[i], centerList[j], distance))
            # fo.flush()
            if distance <= up_distance_threshold and distance > down_distance_threshold:  # 小于上阈值，说明逃票；大于下阈值，过滤掉一个人有两个框的情况
                # taopiaoList.append((box[i], box[j]))    # 返回涉嫌逃票的两个框
                taopiaoList.append(boxes[i])
                taopiaoList.append(boxes[j])
            # if distance > 0.0:
            #     taopiaoList.append(boxes[i])
            #     taopiaoList.append(boxes[j])
    return taopiaoList

'''
    检测红灯
'''
def detect_light(img):
    try:
        tmp_img = img.copy()
        tmp_img[:, :, :] = 0    # 原图拷贝涂黑，用于与处理过的图进行比较

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, red_hsv_lower, red_hsv_upper)
        _img = cv2.bitwise_and(img, img, mask=mask)    # 在原图上涂，与tmp_img比较

        if tmp_img.any() == _img.any():
            # print("正常")
            return "NORMAL"
        else:
            # print("红灯")
            return "WARNING"
    except Exception as e:
        print(e)