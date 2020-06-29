import os
import shutil

'''
    复制图片文件
'''

target_path = "D:/dest/"

with open("C:/Users/vegetabledog/Desktop/log.txt", encoding='utf-8') as fo:
    for line in fo.readlines():
        arr = line.split("\t")
        filepath = arr[0]
        people_nums = arr[3]

        filename = filepath[filepath.index("\\") + 1: ]
        ip = filepath[filepath.rindex("/") + 1: filepath.index("\\")]
        # print(ip, filename)
        dest_file = ip + "+" + filename[0: -4] + "----" + str(people_nums) + ".jpg"
        print(filepath, os.path.join(target_path, dest_file))
        shutil.copyfile(filepath, os.path.join(target_path, dest_file))