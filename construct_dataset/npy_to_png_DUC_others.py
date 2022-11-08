import numpy as np
from skimage.util import view_as_blocks
import matplotlib.pyplot as plt
import pdb

def norm_im(im):    return (im - im.min())/(im.max() - im.min()) # Normalisation

# npy files with data
exp = np.load('exp_data.npy')
#exp_data.npy - a (2,240,400)-sized array with (0,:,:) the H-K image, and (1,:,:) the Q-E image

sim_ref_goodenough = np.load('sim_ref_data_goodenough.npy')
sim_ref_dimer = np.load('sim_ref_data_dimer.npy')
#(0,:,:) - is the H-K image with full resolution ("tobyfit")
#(1,:,:) - is the H-K image without full resolution but including the effects of integrating over the dimensions not ploted ("integrated")
#(2,:,:) - is the H-K image without resolution ("nores")
#(3,:,:) - is the Q-E "tobyfit" image 
#(4,:,:) - is the Q-E "integrated" image 
#(5,:,:) - is the Q-E "nores" image 

# What data to use
grayscale = True
colormap = 'gray' if grayscale == True else 'viridis'

# Folder the files should be saved in
save_path = "./"

print (np.shape(exp))
image = np.squeeze(exp[0])
image = norm_im(image)
plt.imsave(save_path+"/Exp.png", image, vmin=0, vmax=1, cmap=colormap)			
plt.imsave(save_path+"/Exp_MaskPercentile1.png", norm_im(np.clip(image, np.percentile(image, 1), np.percentile(image, 99))), vmin=0, vmax=1, cmap=colormap)
plt.imsave(save_path+"/Exp_MaskPercentile5.png", norm_im(np.clip(image, np.percentile(image, 5), np.percentile(image, 95))), vmin=0, vmax=1, cmap=colormap)
plt.imsave(save_path+"/Exp_MaskMedian10Upper.png", norm_im(np.clip(image, 0, np.median(image)*10)), vmin=0, vmax=1, cmap=colormap)
plt.imsave(save_path+"/Exp_MaskMedian10.png", norm_im(np.clip(image, -np.median(image)*10, np.median(image)*10)), vmin=0, vmax=1, cmap=colormap)

print (np.shape(sim_ref_goodenough[0]))
image = np.squeeze(sim_ref_goodenough[0])
image = norm_im(image)
plt.imsave(save_path+"/sim_ref_goodenough_TobyFit.png", image, vmin=0, vmax=1, cmap=colormap)

image = np.squeeze(sim_ref_goodenough[2])
image = norm_im(image)
plt.imsave(save_path+"/sim_ref_goodenough_Nore.png", image, vmin=0, vmax=1, cmap=colormap)

image = np.squeeze(sim_ref_dimer[0])
image = norm_im(image)
plt.imsave(save_path+"/sim_ref_dimer_TobyFit.png", image, vmin=0, vmax=1, cmap=colormap)

image = np.squeeze(sim_ref_dimer[2])
image = norm_im(image)
plt.imsave(save_path+"/sim_ref_dimer_Nore.png", image, vmin=0, vmax=1, cmap=colormap)
