import cv2
import matplotlib.pyplot as plt

image = cv2.imread("MrBean2024.jpg")
# Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# binational
ret, binary_image = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)

# Gaussian fuzzy processing
blurred_image = cv2.GaussianBlur(binary_image , (9, 9), 0)

# The Otsu thresholding method is used to adaptively determine the threshold of Canny edge detection
_, threshold_img = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Calculates half of the threshold and converts it to an integer
threshold_value = threshold_img[0][0]
threshold1 = int(threshold_value * 0.5)
threshold2 = int(threshold_value)

# Perform Canny edge detection
edge_img = cv2.Canny(blurred_image, threshold1, threshold2)
# Draw grayscale image, Gaussian blur processing image, edge detection image
plt.figure(figsize=(20, 10))
plt.subplot(1, 3, 1)
plt.title('gray_image')
plt.imshow(gray_image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('blurred_image')
plt.imshow(blurred_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('edge_image')
plt.imshow(edge_img, cmap='gray')

plt.show()

# 霍夫变换检测圆# Hough Transform detects circles
circles = cv2.HoughCircles(edge_img, cv2.HOUGH_GRADIENT, dp=1, minDist=30, param1=50, param2=30, minRadius=30,
                           maxRadius=56)

for i in circles:
    x1, y1, r = i[0]
    cv2.circle(image, (int(x1), int(y1)), int(r), (0, 255, 0), 2)  # 参数分别为：图像、圆心坐标、半径、颜色、线宽
    cv2.line(image, (int(x1) - 5, int(y1)), (int(x1) + 5, int(y1)), (255, 0, 0), 2)  # 绘制水平线
    cv2.line(image, (int(x1), int(y1) - 5), (int(x1), int(y1) + 5), (255, 0, 0), 2)  # 绘制垂直线

# 将 BGR 格式转换为 RGB 格式
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 显示图像
plt.imshow(image_rgb)
plt.title("Output Image")
plt.axis('on')
plt.show()
