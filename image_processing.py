import numpy as np
import imageio
import matplotlib.pyplot as plt
import skimage.color as color
import skimage.feature as feature
import skimage.filters as filters
import skimage.measure as measure
import skimage.util as util



# Task 1: Determining the size of the avengers_imdb.jpg image and producing a greyscale and a black-and-white representation of it

avengers_im=imageio.imread('image_data/avengers_imdb.jpg')
print ("The size of the avengers_imdb.jpg image is: ",avengers_im.shape)

# Viewing the original image
plt.imshow(avengers_im)
plt.show()

# Converting image to greyscale
avengers_grey=color.rgb2gray(avengers_im)

plt.imshow(avengers_grey,cmap=plt.cm.gray, interpolation='nearest')
plt.title('Avengers image in greyscale')
plt.axis("off")
plt.savefig('avengersgrey.png')
plt.show()

# Converting image to black and white using the threshold method

# Finding a suitable threshold value
threshold=filters.threshold_mean(avengers_grey)
print("Mean method threshold = ", threshold)

# Reading the image data into a 1D array
avengers_img_data=np.asarray(avengers_grey)

# Creating binary data of the image
threshold_data=np.where(avengers_img_data>threshold,255,0)
plt.imshow(threshold_data,cmap=plt.cm.gray)
plt.title("Avengers image in Black & White")
plt.axis("off")
plt.savefig('avengersb&w.png')
plt.show()




# Task 2: Adding Gaussian random noise in the bush_house_wikipedia.jpg (with variance 0.1) and filtering the perturbed image with a Gaussian mask (sigma equal to 1) and a uniform smoothing mask (size 9x9).

# Reading the Bush House image
bush_house_im=imageio.imread('image_data/bush_house_wikipedia.jpg')

# Viewing the original image
plt.imshow(bush_house_im)
plt.show()

# Adding Gaussian random noise to bush house image with variance 0.1
bush_noise=util.random_noise(bush_house_im, mode='gaussian',var=0.1)
plt.imshow(bush_noise)
plt.show()

# Filtering the noisy image with a Gaussian mask, with sigma = 1 and applying a smoothing mask.
filter_bush=filters.gaussian(bush_noise,sigma=1,mode='nearest',truncate=9)
plt.imshow(filter_bush)
plt.axis("off")
plt.title("Bush house with Gaussian noise, Gaussian mask and smoothing mask")
plt.savefig('bush_house_Gaussian_mask.png')
plt.show()





# Task 3: Dividing the forestry_commission_gov_uk.jpg into 5 segments using k-means segmentation.

# Reading the Forest Commission Gov UK image
forest_im=imageio.imread('image_data/forestry_commission_gov_uk.jpg')
plt.imshow(forest_im)
plt.show()

from sklearn import cluster

# Converting the image to greyscale
forest_grey=color.rgb2gray(forest_im)
plt.imshow(forest_grey,cmap=plt.cm.gray)
plt.show()

# Reshaping the data to a 1D array
forest_data=np.asarray(forest_grey).reshape((-1,1))

# Applying a k-means segmentation with 5 clusters
km=cluster.KMeans(n_clusters=5)
km.fit(forest_data)

# Getting cluster centre coordinates as a 1D array
labels=km.labels_
values=km.cluster_centers_.squeeze()

forest_im_segment=np.choose(labels,values)
forest_im_segment.shape=forest_grey.shape

# Getting the max and min intensity of the image
im_intensity_min=forest_grey.min()
im_intensity_max=forest_grey.max()
plt.imshow(forest_im_segment,cmap=plt.cm.gray,vmin=im_intensity_min,vmax=im_intensity_max)
plt.title("Forest image with k-Means Segmentation")
plt.axis("off")
plt.savefig('forest_k_means.png')
plt.show()





# Task 4: Performing Canny edge detection and applying the Hough transform on rolland_garros_tv5monde.jpg.

# Reading the Rolland Garros Tv5 image
rolland_im=imageio.imread('image_data/rolland_garros_tv5monde.jpg')
plt.imshow(rolland_im)
plt.show()

import skimage.transform as transform

# Converting the image to greyscale
rolland_grey=color.rgb2gray(rolland_im)

# Performing Canny Edge detection on the image
edges=feature.canny(rolland_grey)

# Applying a Hough Transform to the image
lines=transform.probabilistic_hough_line(edges,threshold=20,line_length=100, line_gap=3)
plt.figure()
for line in lines:
    p0,p1=line
    plt.plot((p0[0],p1[0]),(p0[1],p1[1]))
plt.title("Probabilistic Hough")
plt.axis("off")
plt.savefig('rolland_Hough.png')
plt.show()