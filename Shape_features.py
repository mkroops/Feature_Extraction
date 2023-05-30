import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('onions/10_rgb.png')
mask = cv2.imread('onions/10_truth.png')
ground_bin = cv2.imread('onions/10_truth.png', 0)

if np.count_nonzero(ground_bin) == 0:
    print("Mask is empty")
else:
    _, labels = cv2.connectedComponents(ground_bin)

print(labels)

solidities = []
non_compacts = []
circularities = []
eccentricities = []
onion_labels = []


def has_blue_object(patch):
    lower_blue = np.array([100, 0, 0])
    upper_blue = np.array([255, 10, 10])
    
   
    blue_mask = cv2.inRange(patch, lower_blue, upper_blue)
    
    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        return True
    else:
        return False
    
for label in range(1, labels.max()+1):
    patch = img.copy()
    patch_ground = mask.copy()
    patch[labels != label] = 0
    patch_ground[labels != label] = 0
    print(label)
    if(has_blue_object(patch_ground)):
        print("Onion")
        onion_labels.append(label)
    else:
        print("Weed")
    
    #cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("mask", 500, 500)
    #cv2.imshow("mask", patch_ground)
    #cv2.waitKey(1000)
    #cv2.imshow("patch",patch_ground)
    #cv2.waitKey(1000)
    print(labels[label])
    
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    moments = cv2.moments(contours[0])
    file_name7 = "E:\\UOL\\Computer Vision\\Assignment\\output\\" + "gray" + "_" +".png"
    cv2.imwrite(file_name7, patch)
    # Calculate shape features
    area = moments['m00']
    perimeter = cv2.arcLength(contours[0], True)
    solidity = area / cv2.contourArea(cv2.convexHull(contours[0]))
    non_compactness = perimeter**2 / (4*np.pi*area)
    circularity = (4*np.pi*area) / perimeter**2
    eigenvalues, _ = np.linalg.eig(np.array([[moments['mu20'], moments['mu11']], [moments['mu11'], moments['mu02']]]))


    eccentricity = np.sqrt(1 - np.min(eigenvalues) / np.max(eigenvalues))
    
    # Add shape features to lists
    solidities.append(solidity)
    non_compacts.append(non_compactness)
    circularities.append(circularity)
    eccentricities.append(eccentricity)


print("Onion labels", onion_labels)


# Plot distribution of shape features for onions and weeds
onion_solidities = []
onion_non_compacts = []
onion_circularities = []
onion_eccentricities = []


for i in onion_labels:
    onion_solidities.append(solidities[i])
    onion_non_compacts.append(non_compacts[i])
    onion_circularities.append(circularities[i])
    onion_eccentricities.append(eccentricities[i])

mean_solidities = np.mean(onion_solidities) 
mean_non_compacts = np.mean(onion_non_compacts) 
mean_circularities = np.mean(onion_circularities) 
mean_eccentricities = np.mean(onion_eccentricities) 

plot_features = [mean_solidities, mean_circularities, mean_eccentricities]
# Print means
print("Mean solidities:", mean_solidities)
print("Mean non-compacts:", mean_non_compacts)
print("Mean circularities:", mean_circularities)
print("Mean eccentricities:", mean_eccentricities)

# Define means
 
# Define labels
labels = ['Solidities', 'Circularities', 'Eccentricities']
 
# Define x-axis ticks
x_ticks = np.arange(len(labels))
 
# Create bar chart
plt.bar(x_ticks, plot_features, align='center')
plt.xticks(x_ticks, labels)
plt.ylabel('Mean Values')
plt.title('Mean Feature Values for Onion vs Weed')
# Show plot
plt.show()

'''
print("onion_solidities ", np.mean(onion_solidities))
print("weed_solidities ", np.mean(weed_solidities))
print("onion_non_compacts", np.mean(onion_non_compacts))
print("weed_non_compacts", np.mean(weed_non_compacts))
print("onion_circularities ", np.mean(onion_circularities))
print("weed_circularities ", np.mean(weed_circularities))
print("onion_eccentricities", np.mean(onion_eccentricities))
print("weed_eccentricities ", np.mean(weed_eccentricities ))

plt.hist(onion_solidities, alpha=0.5, label='Onions')
plt.hist(weed_solidities, alpha=0.5, label='Weeds')
plt.xlabel('Solidity')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.hist(np.mean(onion_solidities), alpha=0.5, label='Onions')
plt.hist(np.mean(weed_solidities), alpha=0.5, label='Weeds')
plt.xlabel('Solidity')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.hist(onion_non_compacts, alpha=0.5, label='Onions')
plt.hist(weed_non_compacts, alpha=0.5, label='Weeds')
plt.xlabel('Non-compactness')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.hist(onion_circularities, alpha=0.5, label='Onions')
plt.hist(weed_circularities, alpha=0.5, label='Weeds')
plt.xlabel('Circularity')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.hist(onion_eccentricities, alpha=0.5, label='Onions')
plt.hist(weed_eccentricities, alpha=0.5, label='Weeds')
plt.xlabel('Eccentricity')
plt.ylabel('Frequency')
plt.legend()
plt.show()
'''
