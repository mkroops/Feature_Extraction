import cv2
import numpy as np
import skimage.feature as feature
import matplotlib.pyplot as plt

image_spot = cv2.imread('onions/01_depth.png')
image_depth = cv2.imread('onions/01_depth.png')
n = cv2.cvtColor(image_depth, cv2.COLOR_BGR2GRAY)

b, g, r = cv2.split(image_spot)

distances = [1]

angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

graycom_r = feature.greycomatrix(r, distances, angles, levels=256, symmetric=True, normed=True)
graycom_g = feature.greycomatrix(g, distances, angles, levels=256, symmetric=True, normed=True)
graycom_b = feature.greycomatrix(b, distances, angles, levels=256, symmetric=True, normed=True)
graycom_n = feature.greycomatrix(n, distances, angles, levels=256, symmetric=True, normed=True)

contrast_r = feature.greycoprops(graycom_r, 'contrast')
correlation_r = feature.greycoprops(graycom_r, 'correlation')
ASM_r = feature.greycoprops(graycom_r, 'ASM')

contrast_g = feature.greycoprops(graycom_g, 'contrast')
correlation_g = feature.greycoprops(graycom_g, 'correlation')
ASM_g = feature.greycoprops(graycom_g, 'ASM')

contrast_b = feature.greycoprops(graycom_b, 'contrast')
correlation_b = feature.greycoprops(graycom_b, 'correlation')
ASM_b = feature.greycoprops(graycom_b, 'ASM')

contrast_n = feature.greycoprops(graycom_n, 'contrast')
correlation_n = feature.greycoprops(graycom_n, 'correlation')
ASM_n = feature.greycoprops(graycom_n, 'ASM')

ASM = np.mean([ASM_r, ASM_g, ASM_b, ASM_n], axis=0)
mean_ASM_r = np.mean(ASM_r)
print("mean_ASM_r", mean_ASM_r)
contrast = np.mean([contrast_r, contrast_g, contrast_b, contrast_n], axis=0)
correlation = np.mean([correlation_r, correlation_g, correlation_b, correlation_n], axis=0)

print("ASM R", ASM_r)
print("ASM G", ASM_g)
print("ASM B", ASM_b)
print("ASM N", ASM_n)
print("ASM: {}".format(ASM))
print("Contrast: {}".format(contrast))
print("Correlation: {}".format(correlation))
print("ASM type", type(ASM))
ASM_list = list(ASM)
print("ASM type", ASM_list)
labels = ['R', 'G', 'B', 'N-Infra']

x_ticks = np.arange(len(labels))

values = [np.mean(ASM_r), np.mean(ASM_g), np.mean(ASM_b), np.mean(ASM_n)]
#values = [np.mean(ASM_r), np.mean(ASM_g), np.mean(ASM_b), np.mean(ASM_n)]
print("Values", values)
plt.bar(x_ticks, values , align='center')
plt.xticks(x_ticks, labels)
plt.ylabel('Value')
plt.title('Texture Features of RGBN channels')
plt.show()
