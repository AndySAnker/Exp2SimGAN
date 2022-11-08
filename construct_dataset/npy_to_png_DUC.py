import numpy as np
from skimage.util import view_as_blocks
import matplotlib.pyplot as plt
import pdb

def norm_im(im):    return (im - im.min())/(im.max() - im.min()) # Normalisation

def validate_image(image):
	"""
	Function that calculates the percentage-wise scattering of the 1% most scattering pixels in the image. 
	If the value is vary high it can mean that the image is full of background.
	"""
	placeholder1 = image.flatten()
	placeholder1.sort()
	return sum(placeholder1[-int(0.01*len(placeholder1)):]) / sum(placeholder1) * 100

# npy files with data
dimer = np.load('dimer_hk_simulated.npy')
goodenough = np.load('goodenough_hk_simulated.npy')
#simulated[:][0] - are 240x400 pixel images of H-K cuts with the full resolution convolution ("hk_tobyfit")
#simulated[:][1] - are 240x400 pixel images of H-K cuts with the model evaluated only at the pixel centres - e.g. without resolution ("hk_nores")

# What data to use
data_type1 = 0 
data_type2 = 1
balanced = True

grayscale = True
colormap = 'gray' if grayscale == True else 'viridis'

# Folder the files should be saved in
save_path = "../datasets/goodenoughANDdimer_balanced/"
# How many files to use
How_many_dimer = len(dimer) 

# Shuffling
data_iter_list = list(range(How_many_dimer))
np.random.shuffle(data_iter_list)
print ("I will start to transform "+ str(How_many_dimer) +" pictures of dimer class!")
for iter, i in enumerate(data_iter_list):
	image1 = np.squeeze(dimer[i][data_type1])
	image2 = np.squeeze(dimer[i][data_type2])
	if validate_image(image1) < 99.9 and validate_image(image2) < 99.9:
		if iter < int(0.2*len(data_iter_list)):
			image1 = norm_im(image1)
			image2 = norm_im(image2)
			plt.imsave(save_path+"/testA/Dimer_%d.png" % i, image1, vmin=0, vmax=1, cmap=colormap)			
			plt.imsave(save_path+"/testB/Dimer_%d.png" % i, image2, vmin=0, vmax=1, cmap=colormap)			
		else:
			image1 = norm_im(image1)
			image2 = norm_im(image2)
			plt.imsave(save_path+"/trainA/Dimer_%d.png" % i, image1, vmin=0, vmax=1, cmap=colormap)			
			plt.imsave(save_path+"/trainB/Dimer_%d.png" % i, image2, vmin=0, vmax=1, cmap=colormap)	


How_many_goodenough = len(goodenough) 
data_iter_list = list(range(How_many_goodenough))
np.random.shuffle(data_iter_list)
data_iter_list = data_iter_list[:int(len(dimer))]
print ("I will start to transform "+ str(How_many_goodenough) +" pictures of goodenough class!")
for iter, i in enumerate(data_iter_list):
	image1 = np.squeeze(goodenough[i][data_type1])
	image2 = np.squeeze(goodenough[i][data_type2])
	if validate_image(image1) < 99.9 and validate_image(image2) < 99.9:
		if iter < int(0.2*len(data_iter_list)):
			image1 = norm_im(image1)
			image2 = norm_im(image2)
			plt.imsave(save_path+"/testA/Goodenough_%d.png" % i, image1, vmin=0, vmax=1, cmap=colormap)			
			plt.imsave(save_path+"/testB/Goodenough_%d.png" % i, image2, vmin=0, vmax=1, cmap=colormap)			
		else:
			image1 = norm_im(image1)
			image2 = norm_im(image2)
			plt.imsave(save_path+"/trainA/Goodenough_%d.png" % i, image1, vmin=0, vmax=1, cmap=colormap)			
			plt.imsave(save_path+"/trainB/Goodenough_%d.png" % i, image2, vmin=0, vmax=1, cmap=colormap)			
		
print ("I have converted all images to png")
