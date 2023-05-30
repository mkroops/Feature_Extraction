import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from sklearn import svm
import skimage.feature as feature
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import mahotas as mt

def extract_shape_features(patch):
    
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    moments = cv2.moments(contours[0])
    area = moments['m00']
    perimeter = cv2.arcLength(contours[0], True)
    if area != 0:
        solidity = area / cv2.contourArea(cv2.convexHull(contours[0]))
        non_compactness = perimeter**2 / (4*np.pi*area)
        circularity = (4*np.pi*area) / perimeter**2
        eigenvalues, _ = np.linalg.eig(np.array([[moments['mu20'], moments['mu11']], [moments['mu11'], moments['mu02']]]))
        eccentricity = np.sqrt(1 - np.min(eigenvalues) / np.max(eigenvalues))
    else:
        solidity = 0.0
        non_compactness = 0.0
        circularity = 0.0
        eccentricity = 0.0
    return solidity, non_compactness, circularity, eccentricity, [solidity, non_compactness, circularity, eccentricity]


def extract_texture_features(patch):
    
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    b, g, r = cv2.split(patch)

    graycom_r = feature.greycomatrix(r, distances, angles, levels=256, symmetric=True, normed=True)
    graycom_g = feature.greycomatrix(g, distances, angles, levels=256, symmetric=True, normed=True)
    graycom_b = feature.greycomatrix(b, distances, angles, levels=256, symmetric=True, normed=True)
 

    contrast_r = feature.greycoprops(graycom_r, 'contrast')
    correlation_r = feature.greycoprops(graycom_r, 'correlation')
    ASM_r = feature.greycoprops(graycom_r, 'ASM')

    contrast_g = feature.greycoprops(graycom_g, 'contrast')
    correlation_g = feature.greycoprops(graycom_g, 'correlation')
    ASM_g = feature.greycoprops(graycom_g, 'ASM')

    contrast_b = feature.greycoprops(graycom_b, 'contrast')
    correlation_b = feature.greycoprops(graycom_b, 'correlation')
    ASM_b = feature.greycoprops(graycom_b, 'ASM')

    ASM = np.mean([np.mean(ASM_r), np.mean(ASM_g), np.mean(ASM_b)], axis=0)
    contrast = np.mean([np.mean(contrast_r), np.mean(contrast_g), np.mean(contrast_b)], axis=0)
    correlation = np.mean([np.mean(correlation_r), np.mean(correlation_g), np.mean(correlation_b)], axis=0)
    print("ASM: {}".format(ASM))
    return ASM, contrast, correlation, [ASM, contrast, correlation]

train_shape_features = []
train_texture_features = []
train_labels = []
train_texture_labels = []

test_shape_features = []
test_texture_features = []
test_texture_labels = []
test_labels = []
all_train_shape_features = []


onion_labels = []
train_shape_combined = []
train_texture_combined = []
test_shape_combined = []
test_texture_combined = []
train_combined_label =[]
test_combined_label =[]

def has_blue_object(patch):
    lower_blue = np.array([100, 0, 0])
    upper_blue = np.array([255, 10, 10])
    
    blue_mask = cv2.inRange(patch, lower_blue, upper_blue)
    
    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        return True
    else:
        return False
    
for i in range(1, 21):
    
    empty_str = ''
    if i < 10:
        empty_str = '0'
        
    img = cv2.imread('onions/' + empty_str + str(i) + '_rgb.png')
    mask = cv2.imread('onions/' + empty_str + str(i) + '_truth.png')
    img_d = cv2.imread('onions/' + empty_str + str(i) + '_depth.png')
    ground_bin = cv2.imread('onions/' + empty_str + str(i) + '_truth.png', 0)
    
    if np.count_nonzero(ground_bin) == 0:
        print("Mask is empty")
    else:
        _, labels = cv2.connectedComponents(ground_bin)

    print(labels)
    
    for label in range(1, labels.max()+1):
        patch = img.copy()
        patch_ground = mask.copy()
        patch[labels != label] = 0
        patch_ground[labels != label] = 0
        print(label)

        solid, non_compact, cir, ecc, shape_combined= extract_shape_features(patch)
        ASM, Con, Correlation, texture_combined = extract_texture_features(patch)
        shape_feature = cir
        texture_feature = Correlation
        if i < 19:
            if(has_blue_object(patch_ground)):
                print("Onion")
                onion_labels.append(label)
                train_shape_features.append(shape_feature)
                train_texture_features.append(texture_feature)
                train_labels.append("Onion")
            else:
                train_shape_features.append(texture_feature)
                train_texture_features.append(ASM)
                train_labels.append("Weed")
                continue
                print("Weed")
        else:
            if(has_blue_object(patch_ground)):
                test_labels.append("Onion")
            else:
                test_labels.append("Weed")
            test_shape_features.append(shape_feature)
            test_texture_features.append(texture_feature)
        
    print(i)
    print("count train texture",len(train_texture_features))  
    print("count shape feature",len(test_shape_features))
    print("Test shape features", test_shape_features) 
    print("Test Label", test_labels)

print("Onion Label", len(test_labels))
print("count train texture",len(train_texture_features))  
print("count texture Label",len(train_texture_labels))
# Convert shape and texture features lists to numpy arrays
train_shape_features = np.array(train_shape_features)
train_texture_features = np.array(train_texture_features)
train_labels = np.array(train_labels)
test_shape_features = np.array(test_shape_features)
test_texture_features = np.array(test_texture_features)
test_labels = np.array(test_labels)

train_shape_features = train_shape_features.reshape(-1, 1)
train_labels = train_labels.reshape(-1, 1)
test_shape_features = test_shape_features.reshape(-1, 1)
train_texture_features = train_texture_features.reshape(-1, 1)
test_texture_features = test_texture_features.reshape(-1, 1)

print("train_texture_features shape:", train_texture_features.shape)
print("train_labels shape:", train_labels.shape)
print("train_shape_features shape:", train_shape_features.shape)

# Train classification models
svm_shape = svm.SVC(kernel='linear')
svm_texture = svm.SVC(kernel='linear')
svm_shape_texture = svm.SVC(kernel='linear')

# Train shape model
svm_shape.fit(train_shape_features, train_labels)

print("test_shape_features shape", test_shape_features.shape)
print("train_shape_features shape:", train_shape_features.shape)
#train_preds_shape = svm_shape.predict(train_shape_features)
print("test_shape_features shape:", test_shape_features.shape)

test_preds_shape = svm_shape.predict(test_shape_features)

#print("predicted shape", train_preds_shape)
print("test predicted shape", test_preds_shape)
# Train texture model

svm_texture.fit(train_texture_features, train_labels)
#train_preds_texture = svm_texture.predict(train_texture_features)
test_preds_texture = svm_texture.predict(test_texture_features)
#print("Train _ predict",train_preds_texture)
print("Test Predict", test_preds_texture)

# Evaluate models
print("Shape model precision and recall:")
#print(classification_report(train_labels, train_preds_shape))
print(classification_report(test_labels, test_preds_shape))
#print("Texture model precision and recall:")
#print(classification_report(train_texture_labels, train_preds_texture))
print(classification_report(test_labels, test_preds_texture))

print(classification_report(test_labels, test_preds_texture))

# Train shape + texture model
train_shape_texture_features = np.concatenate((train_shape_features, train_texture_features), axis=1)
test_shape_texture_features = np.concatenate((test_shape_features, test_texture_features), axis=1)


svm_shape_texture.fit(train_shape_texture_features, train_labels)
#train_preds_shape_texture = svm_shape_texture.predict(train_shape_texture_features)
test_preds_shape_texture = svm_shape_texture.predict(test_shape_texture_features)


print("*****************************************")
print("Shape + texture model precision and recall:")
#print(classification_report(train_combined_label, train_preds_shape_texture.ravel()))
print(classification_report(test_labels, test_preds_shape_texture.ravel()))

