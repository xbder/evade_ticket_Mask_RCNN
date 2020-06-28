import os
import json

'''
    逃票检测日志分析
'''

base_path = "C:/Users/vegetabledog/Desktop/逃票检测截至20200622日志/"

for dir_name in os.listdir(base_path):
    dir_path = os.path.join(base_path, dir_name)
    for file in os.listdir(dir_path):
        filepath = os.path.join(dir_path, file)
        with open(filepath, encoding='utf-8') as fo:
            for line in fo.readlines():
                line = line.strip("\n").strip()
                s = line[line.index("INFO: ")+6: ]
                # print(s)
                if s.__contains__("afterCleanList"):
                    pass
                elif s.__contains__("****** final total"):
                    pass
                else:
                    dict = eval(s)
                    if len(dict["person_details_list"]) >0:
                        print(dict["savefile"], len(dict["person_details_list"]), dict["person_details_list"])
