import numpy as np
from skimage.util import view_as_blocks
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
dimer = np.load('dimer_140meV_60x60x60.npy')
goodenough = np.load('goodenough_140meV_60x60x60.npy')

grayscale = True
colormap = 'gray' if grayscale == True else 'viridis'

# Folder the files should be saved in
save_path = "../datasets/3D_INS/"
# How many files to use
How_many_dimer = len(dimer) 

# Shuffling
data_iter_list = list(range(How_many_dimer))
np.random.shuffle(data_iter_list)
print ("I will start to transform "+ str(How_many_dimer) +" pictures of dimer class!")
for iter, i in enumerate(data_iter_list):
    image1 = np.squeeze(dimer[i][0])
    image2 = np.squeeze(dimer[i][1])
    print (validate_image(image1))
    print (validate_image(image2))
    if validate_image(image1) < 99.9 and validate_image(image2) < 99.9:
        if iter < int(0.2*len(data_iter_list)):
            image1 = norm_im(image1)
            image2 = norm_im(image2)
            np.save(save_path+"/testA/Dimer_%d.png" % i, image1)
            np.save(save_path+"/testB/Dimer_%d.png" % i, image2)
        else:
            image1 = norm_im(image1)
            image2 = norm_im(image2)
            np.save(save_path+"/trainA/Dimer_%d.png" % i, image1)
            np.save(save_path+"/trainB/Dimer_%d.png" % i, image2)

How_many_goodenough = len(goodenough) 
data_iter_list = list(range(How_many_goodenough))
np.random.shuffle(data_iter_list)
data_iter_list = data_iter_list[:int(len(dimer))]
print ("I will start to transform "+ str(How_many_goodenough) +" pictures of goodenough class!")
for iter, i in enumerate(data_iter_list):
    image1 = np.squeeze(goodenough[i][0])
    image2 = np.squeeze(goodenough[i][1])
    if validate_image(image1) < 99.9 and validate_image(image2) < 99.9:
        if iter < int(0.2*len(data_iter_list)):
            image1 = norm_im(image1)
            image2 = norm_im(image2)
            np.save(save_path+"/testA/Goodenough_%d.png" % i, image1)
            np.save(save_path+"/testB/Goodenough_%d.png" % i, image2)
        else:
            image1 = norm_im(image1)
            image2 = norm_im(image2)
            np.save(save_path+"/trainA/Goodenough_%d.png" % i, image1)
            np.save(save_path+"/trainB/Goodenough_%d.png" % i, image2)
        
print ("I have converted all images to png")
