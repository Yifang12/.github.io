import cv2
import numpy as np
import matplotlib.pyplot as plt

video = cv2.VideoCapture("Panto2024.mp4")
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
ret, frame_1 = video.read()  # When the video.read() function is called in code, the next frame of the video is read
# by default.So, when we call video-.read () for the first time, it reads the first frame of the video.
frame_number = 30
# 判断帧数是否超出范围# Determine whether the number of frames is out of range
if frame_number > frame_count:
    print("Frame number exceeds video length")
    exit()

# 设置视频帧位置为第number帧（索引从0开始） # Set video frame position to number_th frame (index starts from 0)
video.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
ret, frame = video.read()
# 灰度化 # Grayscale
gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image, cmap='gray', interpolation='nearest')
plt.title("Gray Image")
plt.axis('on')
plt.show()
# 二值化 #binaryzation
ret, binary_image = cv2.threshold(gray_image, 83, 255, cv2.THRESH_BINARY)
# 取第一帧该位置为模版 # Take this position in the first frame as the template
gray_frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
_, binary_frame_1 = cv2.threshold(gray_frame_1, 83, 255, cv2.THRESH_BINARY)
template = binary_frame_1[400:571, 640:1186]
lenth, width = template.shape
# 使用Otsu阈值化方法自适应地确定Canny边缘检测的阈值
# The Otsu thresholding method is used to adaptively determine the threshold of Canny edge detection
_, threshold_img = cv2.threshold(binary_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 计算阈值的一半并转换为整数 # Calculates half of the threshold and converts it to an integer
threshold_value = threshold_img[0][0]
threshold1 = int(threshold_value * 0.5)
threshold2 = int(threshold_value)

# 进行Canny边缘检测 # Perform Canny edge detection
edge_img = cv2.Canny(binary_image, threshold1, threshold2)

plt.imshow(edge_img, cmap='gray')
plt.title("edge_img")
plt.axis('on')
plt.show()
# 寻找霍夫变换的峰值 # Find the peak of the Hough transform
peaks = cv2.HoughLinesP(edge_img, 1, np.pi / 180, threshold=10, minLineLength=300, maxLineGap=50)
# 绘制直线和矩形 # Draw lines and rectangles
plt.figure()
plt.imshow(gray_image, cmap='gray')
plt.gca().set_autoscale_on(False)

for line in peaks:
    x1, y1, x2, y2 = line[0]
    plt.plot([x1, x2], [y1, y2], 'g-', linewidth=1)
    plt.plot(x1, y1, 'yx', markersize=5)
    plt.plot(x2, y2, 'rx', markersize=5)

plt.gca().add_patch(plt.Rectangle((640, 400), width, lenth, linewidth=1, edgecolor='r', fill=False))
plt.show()