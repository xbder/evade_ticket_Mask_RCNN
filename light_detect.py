import os
import cv2
from brakegate_config.gate_config import gateLightDict
from myutil import detect_light
import time

'''
    闸机灯检测
'''

def prepare_data(input_dir, output_dir):
    for video_file in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_file)

        output_path = os.path.join(output_dir, video_file)
        os.makedirs(output_path)

        for img_file in os.listdir(video_path):
            ip = img_file.split("_")[0]
            img_path = os.path.join(video_path, img_file)
            img = cv2.imread(img_path)
            lightList = gateLightDict[ip]
            for i in range(len(lightList)):
                if lightList[i] == (0, 0, 0, 0):
                    continue
                left, top, right, buttom = lightList[i]
                # tmp = img.corp(lightList[i])
                tmp = img[top: buttom, left:right]
                # cv2.imshow("t", tmp)
                # cv2.waitKey()
                output_file = os.path.join(output_path, img_file + "-" + str(i) + ".jpg")
                cv2.imwrite(output_file, tmp)
            print(img_path)

def statistics_right_rate(input_path):
    all_count = 0
    right_count = 0
    start = time.time()
    fo = open("D:/staticz.txt", 'w+', encoding='utf-8')

    for colour in os.listdir(input_path):
        if colour in ["white", "green"]:
            type = "NORMAL"
            # continue
        else:
            type = "WARNING"
        colour_path = os.path.join(input_path, colour)
        for img_file in os.listdir(colour_path):
            all_count += 1
            if all_count % 1000 == 0:
                print("已处理：%d，耗时：%f" % (all_count, time.time() - start))
            img_path = os.path.join(colour_path, img_file)
            img = cv2.imread(img_path)  # CV_8UC1
            res = detect_light(img)
            fo.write(img_path + "\t" + type + "\t" + res + "\n")
            fo.flush()
            if res == type:
                right_count += 1
    fo.close()
    print("correct rate: %f" % (right_count/all_count))


if __name__ == '__main__':
    # input_dir = "D:/dataset/yolo-head-dataset/"
    # output_dir = "D:/dataset/light/"
    # prepare_data(input_dir, output_dir)
    input_path = "D:/dataset/light_classification/"
    statistics_right_rate(input_path)

