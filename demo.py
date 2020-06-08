personNumsDict = {}


personNumsDict["1"] = 5

# print(personNumsDict["2"])
print(personNumsDict.keys())

def isin(center, cropAreaList):
    for i in range(len(cropAreaList)):
        left, top, right, bottom = cropAreaList[i]    # 左上右下
        (targetx, targety) = center

        if (targetx > left and targetx < right) and (targety > top and targety < bottom):
            return str(i)

cropAreaList = [(0, 0, 1, 2), (1, 1, 2, 2), (0, 0, 2, 3)]
center=(1.5, 1.5)
brake_num = isin(center, cropAreaList)
print(brake_num)

personNumsDict = {}

if brake_num in personNumsDict.keys():
    tmp_nums = personNumsDict[brake_num]
    personNumsDict[brake_num] = tmp_nums + 1
else:
    personNumsDict[brake_num] = 1

print("===", personNumsDict)