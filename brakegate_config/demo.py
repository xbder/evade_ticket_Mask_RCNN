# -*- coding=utf-8 -*-

# box1 = (577, 733, 1071, 1123)
# top, left, bottom, right = box1
# print("box1:", top, left, bottom, right, len(box1))
#
# box2 = (577, 733, 1071, 1123)
# top, left, bottom, right = box2
# print("box2:", top, left, bottom, right, len(box2))

ls = [[1, 2],
      [2, 3],
      [1, 2]]

a = [1, 2]
b = [1, 4]

print(ls.index(a))
# print(ls.index(b))
print(ls.__contains__(b))
print(ls.__contains__(a))

print(ls.remove([2, 4]))