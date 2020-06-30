import os
import cv2

base_path_list = [#"E:/BaiduNetdiskDownload/2020-04-03/",
                  #"E:/BaiduNetdiskDownload/2020-04-14/",
                  "E:/BaiduNetdiskDownload/2020-04-17/"]

for base_path in base_path_list:
    for video_name in os.listdir(base_path):
        video_path = os.path.join(base_path, video_name)
        vidcap = cv2.VideoCapture(video_path)
        print(video_path)

        only_name = video_name[0: -4]
        # print(only_name)
        img_path = os.path.join("D:/mydataset/", video_name)
        os.makedirs(img_path)
        success, image = vidcap.read()
        count = 0
        success = True
        while success:
            success, image = vidcap.read()
            if success:
                cv2.imwrite(img_path + "/" + only_name + "_%06d.jpg" % count, image)  # save frame as JPEG file
                count += 1
            if count % 1000 == 0:
                print("\t已保存 %d 张" % count)
        print("  已保存 %d 张" % count)