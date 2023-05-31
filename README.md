# Feature_Extraction
II. FEATURE EXTRACTION
A. Literature Review for Task 2
Using shape and texture information, Automatic Age Esti-
mation from Face Image used to determine an individual age
automatically from facial photos. In the method features are
first extracted using a landmark detection algorithm then
texture features are extracted using a set of Gabor filters. To
determine the age, features are then integrated using a support
vector regression model. A method for extracting features
from skin lesion images that can be used for computer aided
diagnosis which use combination of shape and texture features,
such as perimeter, area, compactness, eccentricity and other
statistical moments of texture. The results demonstrate that
the suggested method works better when shape and texture
features are combined than when either feature and shape are
used alone.
B. Shape Features
Shape is regarded as a crucial indication for humans to
identify and recognise the physical objects in the world.
Shape helps to encode basic geometrical forms like straight
lines in various orientations. The technique of extracting
shape features involves finding and measuring the geometric
characteristics of an object or area of interest within an image
. Shape characteristics are used to evaluate the similarity
between shapes in shape based image retrieval. Shapes are fre-
quently described using basic geometric properties including
the solidity, eccentricity, circularity, non compactness etc.
C. Solidity
Compact of an object is measured by solidity. Solidity is a
useful descriptor when distinguishing between objects with
similar shapes but different levels of concavity. A non-convex
form will have a solidity value less than 1, whereas a com-
pletely convex shape will have a solidity value of 1. Solidity is
frequently utilised in shape based image retrieval and pattern
recognition applications because it is easily calculated using
straightforward mathematical methods.
Solidity = Area of object
Area of convex hull of object (3)
where Area of the object is the total number of pixels in the
object and Area of convex hull is the total number of pixels
in the smallest convex polygon.
D. Non Compactness
Non compactness describes the degree of irregularity in
an objects boundaries. It tells us how far the boundary of
an object deviates from a convex shape. The ratio of
the objects perimeter to the perimeter of convex hull must
be calculated in order to determine whether an object is non
compact or not. Convex hull is the smallest convex polygon
that can completely surround the object. If an object has
perfect convexity, Its non compactness value will be 1, which
indicates that its perimeter will be 1:1 with the perimeter of its
convex hull. On the other hand, if it is not convex, the object’s
non compactness score will be higher than 1.
N on-compactness = Object perimeter
Convex hull perimeter (4)
E. Circularity
Circle like object is measured using Circularity. Non circular
object will have a circularity value less than 1 while perfect
circular object will have value of 1. The ratio of an object area
to its square of perimeter is used to determine circularity.
Circularity can be formulated as:
Circularity = 4 ∗ π ∗ Area
Perimeter2 (5)
where pi equals to 3.14.
Circularity is used to distinguish between round or nearly
circular geometries and those with more asymmetrical shapes.
Circularity processing efficiency makes it a preferred form
descriptor in applications for shape based picture retrieval and
pattern recognition.
F. Eccentricity
Eccentricity is a form characteristic that indicates how
stretched out or extended an object is. It is described as the
ratio of the major axis length to the distance between the foci
of an ellipse suited to the shape of the object. Eccentricity
can be formulated as:
Eccentricity = Distance between foci
Main axis length (6)
Where foci represents two fixed points located within an
ellipse. Eccentricity of an object will be 0 if it has a perfectly
round shape whereas it will be close to 1 if it has a greatly
elongated shape. Eccentricity can be helpful in identifying
objects with various degrees of stretching or elongation.
Solidities Non compacts Circularities Eccentricities
0.4773 8.5847 0.1903 0.8539
TABLE III: Mean values of Shape Features of Onion
Fig. 10: Bar Graph of Shape Features
In table 3, The Solidities of onion is 0.4773 which indicates
that shape of the onion is 47 percent close to a convex shape.
Non compactness is 8.5847 which is not plotted in Fig 10,
Because Non compactness values goes higher than 1 where
other features are in between 0 to 1. If non compactness
goes higher than 1, It is not convex in shape so onion is not
convex in shape. Circularities is 0.1903 which represents the
roundness of the onion, In this case onion is less circular in
shape. Eccentricities is 0.8539 which is closer to 1, So Onion
is elongated in shape where it deviates from a perfect circle.
So eccentricities is most important feature that would be the
useful in distinguishing onions from weeds.
G. Texture Features
Texture features are the observable patterns or structures
in an image that describe the smoothness, roughness, or
regularity of the texture of the image. A popular method for
obtaining texture information is the Gray-Level Co-occurrence
Matrix approach. It analyses the spatial relationships between
pairs of pixels in the image in order to learn more about an
image texture.
GLCM technique will build a matrix to show how frequently
pairs of pixel values appear in the image at various relative
places. Then, By using this matrix, different texture features
such as Angular Second Moment, Contrast, Correlation [19].
H. Angular Second Moment
Angular Second Moment measures how consistent or homo-
geneous an image is at the local level. It can be discovered by
looking at the Gray-Level Co-occurrence Matrix which shows
the frequency with which pairs of pixel values appear in an
image at various spatial orientations.
ASM can be formulated as
ASM =
N∑
i=1
N∑
j=1
P (i, j)2 (7)
where P(i, j) is the probability of the occurrence of the pixel
pairs (i, j) in the GLCM matrix.
The squared GLCM matrix elements are added up to
determine the ASM value. A higher ASM value denotes
a more consistent texture. The resultant value depicts the
distribution of grey levels in the image. In other words,
ASM is a measurement of the image local homogeneity. It
is used in Various image analysis tasks which inculdes image
segmentation, object recognition and classification [19].
I. Contrast
Contrast is calculated from the GLCM by adding the
squared differences between each element and its mean which
is weighted by distance and it measures the local intensity
variations between neighbouring pixels in an image. In order
to differentiate between textures with varying degrees of
sharpness or detail, higher contrast values suggest greater
variances in intensity between neighbouring pixels.
Contrast =
N∑
i=1
N∑
j=1
(i − j)2 · P (i, j) (8)
where i and j are the row and column, P(i,j) is the probability
of two pixels with gray levels i and j, (i-j)2 is the product of
the squared difference between the gray levels, and probability
P(i,j) is weighted by the distance between the gray level [19].
J. Correlation
Correlation that measure the degree of linear relationship
between the grey levels of adjacent pixels in an image. It
is calculated from Gray Level Co-occurrence Matrix which
displays the frequency with which pairs of pixel values appear
in the image at various relative positions. The scale range
from from -1 to 1 where positive numbers denote a positive
correlation, negative values denote a negative correlation and
0 denotes no connection.
Correlation =
∑N
i=1
∑N
j=1(i − μi)(j − μj )P (i, j)
σiσj
(9)
Where μi and μj are the mean values of the ith row and
jth column, σi and σj are standard deviations. P (i, j) is the
probability of the occurrence of the (i, j)th pixel pair [19].
In Table 4, 5, Ch represents Channel, R for red, G for
Green, B for Blue, N for near infra-red and ASM for Angular
Second Moment. Table 4, 5 display the results of calculations
for various image features such as ASM, Contrast, Correlation
as well as colour channels such as R, G, B, and N infra-red for
two classes Onion and Weed at four distinct orientations (0°,
45°, 90°, and 180°. Higher ASM value will have more uniform
Feature Ch 0◦ 45◦ 90◦ 135◦
ASM R 0.99663 0.99656 0.99667 0.99663
ASM G 0.99663 0.99656 0.99667 0.99663
ASM B 0.99663 0.99656 0.99667 0.99663
ASM N 0.99663 0.99656 0.99667 0.99663
Contrast R 1.18384 1.52709 0.80396 1.28398
Contrast G 1.76971 2.29105 1.23787 1.95684
Contrast B 1.02187 1.36783 0.71116 1.09960
Contrast N 2.86128 4.02295 1.76478 3.27517
Correlation R 0.93760 0.91955 0.95763 0.93236
Correlation G 0.95135 0.93706 0.96598 0.94624
Correlation B 0.94633 0.92819 0.96265 0.94227
Correlation N 0.97178 0.96034 0.98260 0.96771
TABLE IV: Calculation for four orientation of Onion Clas
Feature Ch 0◦ 45◦ 90◦ 135◦
ASM R 0.99715 0.99713 0.99721 0.99713
ASM G 0.99715 0.99713 0.99721 0.99713
ASM B 0.99715 0.99713 0.99721 0.99713
ASM N 0.99715 0.99713 0.99721 0.99713
Contrast R 2.73301 3.14430 1.73082 3.08580
Contrast G 4.34181 4.91673 2.89017 5.21507
Contrast B 2.41431 2.71169 1.54666 2.89837
Contrast N 5.84840 6.62130 3.60851 7.18864
Correlation R 0.89912 0.88390 0.93606 0.88606
Correlation G 0.91463 0.90329 0.94313 0.89742
Correlation B 0.90466 0.89286 0.93887 0.88549
Correlation N 0.95034 0.94375 0.96933 0.93893
TABLE V: Calculation for four orientation of Weed Class
grey level distribution. Higher contrast value indicates greater
difference in pixel intensities which gives sharper image. The
contrast value for weed class higher that onion class. Higher
correlation value indicates greater linear dependence which
results in smoother image. In both table correlation value is
relatively high around 0.9 to 0.97.
Fig. 11: Bar Graph For ASM Texture Features of channel RGB
and Near Infra-Red
TABLE VI: Texture Analysis of ASM Feature
Channel Onion Weed
R 0.9984223934829399 0.9963961930149517
G 0.9984223931239254 0.996396176986196
B 0.9984223936578512 0.9963962190239579
N 0.9984224012930291 0.9964014323583994
Fig 11, shows the plot diagram of ASM feature of four
different channel. Table 6, shows the average outcomes of
texture analysis using Angular Second Moment feature for
samples weed and onions for different colour channels such
as R, G, B, and Near infra-red. ASM value for weed is lower
means it is heterogeneous texture whereas onion has higher
compared to weed which means it is more homogeneous
texture across different channel.
Onion image data set is trained with method called SVC
model. The main idea of this concept is to increase the margin
between several classes in a dataset. Two different features
are trained seperately and trained together to indentify which
feature is more accurate in classifying onion from weed.
Class Precision Recall F1-score Support
Onion 0.56 0.93 0.70 95
Weed 0.0 0.0 0.0 69
Overall Accuracy 0.54
TABLE VII: Metrics for Circularity Shape Model
Class Precision Recall F1-score Support
Onion 0.51 0.73 0.60 95
Weed 0.07 0.03 0.04 69
Overall Accuracy 0.43
TABLE VIII: Metrics for Correlation Texture Model
Class Precision Recall F1-score Support
Onion 0.57 0.97 0.72 95
Weed 0.00 0.00 0.00 69
Overall Accuracy 0.56
TABLE IX: Metrics for Circularity Shape and Correlation
Texture Model
It is noticed that in Table 7, Circular shape feature has
precision and recall of onion class which is 0.56 and 0.93
respectively, also F1 score has 0.70. It can identify onions of
accuracy 54 percent. Whereas while considering correlation
texture feature in table 8, which has overall accuracy of 0.43
compared to circularity shape feature but it can able to identify
43 percent of onions. While combining shape and texture
features in table 9, It has higher precision and recall and F1
score compared to circularity shape feature for onion class. It
has achieved overall accuracy of 0.56 which is higher when
combining circularity shape and correlation texture feature
together. Therefore, Combining Shape and texture feature
together will give better result compared to predicting model
only on shape and only on texture feature.
