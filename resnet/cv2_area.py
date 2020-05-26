import cv2

img_path = "C:/Users/vegetabledog/Desktop/hongdeng.jpg"
# img_path = "C:/Users/vegetabledog/Desktop/lvdeng.jpg"

#读取文件
mat_img = cv2.imread(img_path)
mat_img2 = cv2.imread(img_path,cv2.CV_8UC1)

#自适应分割
dst = cv2.adaptiveThreshold(mat_img2,210,cv2.BORDER_REPLICATE,cv2.THRESH_BINARY_INV,3,10)
#提取轮廓
contours,heridency = cv2.findContours(dst, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
#标记轮廓
cv2.drawContours(mat_img,contours,-1,(255,0,255),3)

#计算轮廓面积
area = 0
for i in contours:
    area += cv2.contourArea(i)
    print(cv2.contourArea(i))
print(area)

#图像show
cv2.imshow("window1", mat_img)
cv2.waitKey()