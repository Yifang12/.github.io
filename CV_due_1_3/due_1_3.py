import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("MrBean2024.jpg")
# Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# binational
ret, binary_image = cv2.threshold(gray_image, 234, 255, cv2.THRESH_BINARY)

# # Gaussian fuzzy processing
# blurred_image = cv2.GaussianBlur(binary_image, (9, 9), 0)

# The Otsu thresholding method is used to adaptively determine the threshold of Canny edge detection
_, threshold_img = cv2.threshold(binary_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Calculates half of the threshold and converts it to an integer
threshold_value = threshold_img[0][0]
threshold1 = int(threshold_value * 0.5)
threshold2 = int(threshold_value)

# Perform Canny edge detection
edge_img = cv2.Canny(binary_image, threshold1, threshold2)


# Find the peak of the Hough transform
peaks = cv2.HoughLinesP(edge_img, 1, np.pi / 180, threshold=20, minLineLength=60, maxLineGap=20)
# Draw lines
plt.figure()
plt.imshow(gray_image, cmap='gray')
plt.gca().set_autoscale_on(False)

for line in peaks:
    x1, y1, x2, y2 = line[0]
    plt.plot([x1, x2], [y1, y2], 'g-', linewidth=1)
    plt.plot(x1, y1, 'yx', markersize=5)
    plt.plot(x2, y2, 'rx', markersize=5)

plt.show()

# Draw grayscale image, Gaussian blur processing image, edge detection image
plt.figure(figsize=(20, 10))
plt.subplot(1, 3, 1)
plt.title('gray_image')
plt.imshow(gray_image, cmap='gray')

# plt.subplot(1,3 , 2)
# plt.title('blurred_image')
# plt.imshow(blurred_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('edge_image')
plt.imshow(edge_img, cmap='gray')


plt.show()
