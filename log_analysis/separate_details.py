import os
import json

'''
    逃票检测日志分析
'''

base_path = "C:/Users/vegetabledog/Desktop/逃票检测截至20200622日志/222/"
f_dest = open("C:/Users/vegetabledog/Desktop/222.txt", 'w', encoding="utf-8")
f_error = open("C:/Users/vegetabledog/Desktop/errorlog.txt", 'w', encoding="utf-8")

for file in os.listdir(base_path):
    if file.__contains__("error") is False:
        filepath = os.path.join(base_path, file)
        with open(filepath, encoding='utf-8') as fo:
            for line in fo.readlines():
                line = line.strip("\n").strip()
                s = line[line.index("INFO: ") + 6:]
                time_str = line[0: 23]
                time_str = time_str.replace(" ", "_")
                day = time_str[0: 10]
                hours = time_str[11: 13]
                # print(time_str, day, hours)
                if s.__contains__("afterCleanList"):
                    pass
                elif s.__contains__("****** final total"):
                    pass
                else:
                    dict = eval(s)
                    if len(dict["person_details_list"]) > 0:
                        # print(str(time_str), dict["savefile"], dict["read_time"], dict["detect_time"],
                        #       len(dict["person_details_list"]), dict["person_details_list"])
                        # f_dest.write(str(time_str) + "\t" +
                        #              str(dict["savefile"]) + "\t" +
                        #              str(dict["read_time"]) + "\t" +
                        #              str(dict["detect_time"]) + "\t" +
                        #              str(len(dict["person_details_list"])) + "\t" +
                        #              str(dict["person_details_list"]) + "\n")
                        for detail in dict["person_details_list"]:
                            # detail = eval(detail)    # {'location': [492, 1436, 955, 1889], 'brake_num': '2', 'direct': 0}
                            print(type(detail))
                            if detail["brake_num"] == "None":
                                detail["brake_num"] = "0"
                            print(str(time_str), str(day), str(hours), dict["savefile"], dict["read_time"], dict["detect_time"],
                               detail["location"], detail["brake_num"], detail["direct"])
                            f_dest.write(str(time_str) + "\t" +
                                         str(day) + "\t" +
                                         str(hours) + "\t" +
                                         str(dict["savefile"]) + "\t" +
                                         str(dict["read_time"]) + "\t" +
                                         str(dict["detect_time"]) + "\t" +
                                         str(detail["location"]) + "\t" +
                                         str(detail["brake_num"]) + "\t" +
                                         str(detail["direct"]) + "\n")