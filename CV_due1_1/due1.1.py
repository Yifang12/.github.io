import cv2
import numpy as np
import matplotlib.pyplot as plt


# 使用 OpenCV 加载图像
name = ('image3.jpg')
image = cv2.imread(name)
# 将 BGR 格式转换为 RGB 格式
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # 显示图像
# plt.imshow(image_rgb)
# plt.title("Original Image")
# plt.axis('off')
# plt.show()

# 为每个通道应用高斯滤波器如果将sigma设置为0，OpenCV会根据高斯核的大小自动计算sigma值，以确保高斯核的权重和为1。
# 因此，当sigma为0时，OpenCV会根据核的大小自动确定合适的sigma值进行高斯模糊处理。
blurred_image = cv2.GaussianBlur(image, (9, 9), 0)
# 将 BGR 格式转换为 RGB 格式
blurred_image_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
# 显示图像
# plt.imshow(blurred_image_rgb)
# plt.title("GaussianBlur Image")
# plt.axis('off')
# plt.show()

# 将彩色图像转换为灰度图像
gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray_image, cmap='gray')
# plt.title("Gray Image")
# plt.axis('off')
# plt.show()

# 使用Canny算法进行边缘检测并显示
edges = cv2.Canny(gray_image, 0.1, 200)
# plt.imshow(edges, cmap='gray')
# plt.title("Edges Image")
# plt.axis('off')
# plt.show()

# # 膨胀
# kernel = np.ones((3, 3), np.uint8)
# # 腐蚀
# erosion = cv2.erode(edges, kernel, iterations=1)
#
# dilation = cv2.dilate(erosion, kernel, iterations=1)




# 连通组件标记
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges)

# 找到最大的区域
largest_area_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
largest_area = stats[largest_area_index, cv2.CC_STAT_AREA]

# 在原图上标记最大区域
largest_area_mask = np.zeros_like(edges)
largest_area_mask[labels == largest_area_index] = 255

# # 显示结果
# plt.imshow(cv2.cvtColor(largest_area_mask, cv2.COLOR_BGR2RGB))
# plt.title("Marked Area")
# plt.axis('off')
# plt.show()

# 创建一个与原图相同大小的全零数组
result = np.zeros_like(image)

# 将最大区域在原图上标记为红色
result[labels == largest_area_index] = [0, 0, 255]

# 将其他区域保留在原图上
result = cv2.addWeighted(image, 0.5, result, 10, 0)

# # 显示结果
# plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
# plt.title("Marked Area And Other Areas")
# plt.axis('off')
# plt.show()

plt.figure(figsize=(20, 10))
plt.subplot(1, 4, 1)
plt.title('Original Image')
plt.imshow(image_rgb, cmap='jet')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Edges Image')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Marked Area')
plt.imshow(cv2.cvtColor(largest_area_mask, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Marked Area And Other Areas')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')
output_name = 'output '+name
plt.savefig(output_name)

plt.show()

# # 二值化处理
# ret, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
#
# # 查找轮廓
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # 找到最大的轮廓
# max_contour = max(contours, key=cv2.contourArea)
#
# # 填充最大的轮廓
# filled_image = image.copy()
# cv2.drawContours(filled_image, [max_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
#
# plt.imshow(filled_image)
# plt.title('Filled Image')
# plt.axis('off')
# plt.show()