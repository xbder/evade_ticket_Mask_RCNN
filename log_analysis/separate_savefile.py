import os
import json

'''
    逃票检测日志分析
'''

base_path = "C:/Users/vegetabledog/Desktop/逃票检测截至20200622日志/"
f_dest = open("C:/Users/vegetabledog/Desktop/log.txt", 'w', encoding="utf-8")
f_error = open("C:/Users/vegetabledog/Desktop/errorlog.txt", 'w', encoding="utf-8")

for dir_name in os.listdir(base_path):
    dir_path = os.path.join(base_path, dir_name)
    for file in os.listdir(dir_path):
        if file.__contains__("error") is False:
            filepath = os.path.join(dir_path, file)
            with open(filepath, encoding='utf-8') as fo:
                for line in fo.readlines():
                    line = line.strip("\n").strip()
                    s = line[line.index("INFO: ") + 6:]
                    # print(s)
                    if s.__contains__("afterCleanList"):
                        pass
                    elif s.__contains__("****** final total"):
                        pass
                    else:
                        dict = eval(s)
                        if len(dict["person_details_list"]) > 0:
                            print(dict["savefile"], dict["read_time"], dict["detect_time"], len(dict["person_details_list"]), dict["person_details_list"])
                            f_dest.write(str(dict["savefile"]) + "\t" +
                                         str(dict["read_time"]) + "\t" +
                                         str(dict["detect_time"]) + "\t" +
                                         str(len(dict["person_details_list"])) + "\t" +
                                         str(dict["person_details_list"]) + "\n")
        else:    # 处理error.log
            filepath = os.path.join(dir_path, file)
            with open(filepath, encoding='utf-8') as fo:
                for line in fo.readlines():
                    line = line.strip("\n").strip()
                    try:
                        s = line[line.index("已连接") + 4: line.index("已连接") + 14]
                        s1 = line[line.index("耗时") + 3:]
                        print(s, s1)
                        f_error.write(s + "\t" + s1 + "\n")
                    except Exception as e:
                        pass