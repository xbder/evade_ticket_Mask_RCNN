import cv2
import numpy as np

"""
    通过剥离图片为：分离图片只有2个颜色
    :param img:图片
    :param lower:最小颜色值域 HSV值
    :param upper: 最大颜色值域 HSV值
    :return: 只有两种颜色的图片
"""
def leave_color(img, lower, upper):
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        _img = cv2.bitwise_and(img, img, mask=mask)
        return _img
    except Exception as e:
        print(e)

if __name__ == '__main__':
    # 只检测红灯即可
    red_hsv_lower = np.array([0, 43, 46])
    red_hsv_upper = np.array([10, 255, 255])

    # green_hsv_lower = np.array([35, 43, 46])
    # green_hsv_upper = np.array([77, 255, 255])

    img_path = "C:/Users/vegetabledog/Desktop/hong.jpg"
    # img_path = "C:/Users/vegetabledog/Desktop/lv.jpg"
    img = cv2.imread(img_path)    # CV_8UC1
    cv2.imshow("i", img)
    cv2.waitKey()

    tmp_img = img.copy()    # 不用.copy()的话是在原图上改
    tmp_img[:, :, :] = 0
    cv2.imshow("t", tmp_img)
    cv2.waitKey()

    _img = leave_color(img, red_hsv_lower, red_hsv_upper)
    # _img = leave_color(img, green_hsv_lower, green_hsv_upper)
    cv2.imshow("_", _img)
    cv2.waitKey()
    print("img:", type(img), img.shape)
    print("tmp_img", type(tmp_img), tmp_img.shape)
    print("_img:", type(_img), img.shape)
    if tmp_img.any() == _img.any():
        print("正常")
    else:
        print("红灯")

    # 现在，能在图片上显示出来是否是红灯，但程序接不到红灯的信号
    #

    # # 自适应分割
    # dst = cv2.adaptiveThreshold(_img, 210, cv2.BORDER_REPLICATE, cv2.THRESH_BINARY_INV, 3, 10)
    # # 提取轮廓
    # contours, heridency = cv2.findContours(dst, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # # # 标记轮廓
    # # cv2.drawContours(mat_img, contours, -1, (255, 0, 255), 3)
    #
    # # 计算轮廓面积
    # area = 0
    # for i in contours:
    #     area += cv2.contourArea(i)
    #     print(cv2.contourArea(i))
    # print(area)

