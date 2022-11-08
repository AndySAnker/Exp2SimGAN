# Deterministic uncertainty quantification
# This notebook corresponds to the section How much can we trust the predictions? of the paper Interpretable, calibrated neural networks for analysis and understanding of neutron spectra
#To run the models you will need to download the pre-trained network weights, these are available from 10.5281/zenodo.4088240
#You can set up the required Conda Python environment by using the environment_torch.yml file included in this repository.
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

import time, glob, os, pickle, random, copy
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from models_large import CNN_DUQ as CNN_DUQ_6
import numpy as np
import matplotlib.pyplot as plt
print ("Predicting with resolution trained DUQ classifier!")
def norm_im(im):
    maxval = np.max(im)
    minval = np.min(im)
    return (im - minval)/(maxval - minval)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

model_s = CNN_DUQ_6(input_size=(240, 400, 1), num_classes=2, embedding_size=64,
               learnable_length_scale=False, length_scale=1., gamma=1.)
model_s.load_state_dict(torch.load('../../train_model/Nore/uq-discrim-newer-87.pt',map_location=torch.device('cpu')))
save_path = "save_masked_files_All_balanced/"

print ("trained_models_balanced_percentile0p2clip-model87")
percentile_Lowerpercent = 0
percentile_Upperpercent = 99.8
# Try Franken-data
npy_files = sorted(glob.glob('files_to_predict_on_All_balanced/*.npy'))
for fdat in npy_files:
    expt_f = np.load(fdat)
    print(fdat)
    expt = norm_im(expt_f)
    expt = expt.reshape(1, 240, 400, 1)
    xt = norm_im(expt)
    xt = np.clip(xt, np.percentile(xt,percentile_Lowerpercent), np.percentile(xt,percentile_Upperpercent))
    xt = norm_im(xt)
    plt.imsave(save_path+fdat.replace("files_to_predict_on_All_balanced/", ""), xt, vmin=0, vmax=1)
    xt = np.moveaxis(xt, -1, 1)
    x =  torch.from_numpy(xt).float()
    with torch.no_grad():
        output = model_s(x)
        print('Dimer', 'Goodenough')
        print((output[1].numpy()))
    print()

png_files = sorted(glob.glob('files_to_predict_on_All_balanced/*.png'))
# Try simulated brille
for fdat in png_files:
    expt = plt.imread(fdat)
    expt = rgb2gray(expt)
    print(fdat)
    expt = expt.reshape(1, 240, 400, 1)
    xt = norm_im(expt)
    xt = np.clip(xt, np.percentile(xt,percentile_Lowerpercent), np.percentile(xt,percentile_Upperpercent))
    xt = norm_im(xt)
    print (save_path+fdat.replace("files_to_predict_on/", ""))
    print (np.squeeze(xt).shape)
    plt.imsave(save_path+fdat.replace("files_to_predict_on_All_balanced/", ""), np.squeeze(xt), vmin=0, vmax=1)
    xt = np.moveaxis(xt, -1, 1)
    x =  torch.from_numpy(xt).float()
    with torch.no_grad():
        output = model_s(x)
        print('Dimer', 'Goodenough')
        print((output[1].numpy()))
    print()

print ("fakeA goodenough")
Goodenough_models = sorted(glob.glob("/mnt/beegfs/home/pearl075/section_1_INS_data/goodenoughANDdimer/results/dataset_All_balanced/test_400/images/fake_A/fakeA_Goodenough*rescaled.png"))
print ("Calculating average for simulated Goodenough that has been cleaned with GAN")
print ("Total of files: ", len(Goodenough_models))
np.random.shuffle(Goodenough_models)
all_DUQ_results = np.zeros((int(len(Goodenough_models)), 2))
accuracy = 0
for iter, fdat in enumerate(Goodenough_models):
    expt = plt.imread(fdat)
    expt = rgb2gray(expt)
    expt = expt.reshape(1, 240, 400, 1)
    xt = norm_im(expt)
    xt = np.clip(xt, np.percentile(xt,percentile_Lowerpercent), np.percentile(xt,percentile_Upperpercent))
    xt = norm_im(xt)
    plt.imsave(save_path+fdat.replace("/mnt/beegfs/home/pearl075/section_1_INS_data/goodenoughANDdimer/results/dataset_All_balanced/test_400/images/fake_A/", ""), np.squeeze(xt), vmin=0, vmax=1)
    xt = np.moveaxis(xt, -1, 1)
    x =  torch.from_numpy(xt).float()
    with torch.no_grad():
        output = model_s(x)
        all_DUQ_results[iter][0] = output[1].numpy()[0][0]
        all_DUQ_results[iter][1] = output[1].numpy()[0][1]
        if output[1].numpy()[0][0] < output[1].numpy()[0][1]:
            accuracy += 100/np.shape(all_DUQ_results)[0]
print ("Dimer mean and RMS: ", np.mean(all_DUQ_results[:,0]), np.std(all_DUQ_results[:,0]))
print ("Goodenough mean and RMS: ", np.mean(all_DUQ_results[:,1]), np.std(all_DUQ_results[:,1]))
print ("Accuracy: ", accuracy)


Goodenough_models = sorted(glob.glob("/mnt/beegfs/home/pearl075/section_1_INS_data/goodenoughANDdimer/results/dataset_All_balanced/test_400/images/fake_B/fakeB_Goodenough*rescaled.png"))
print ("Calculating average for simulated Goodenough that has been cleaned with GAN")
print ("Total of files: ", len(Goodenough_models))
np.random.shuffle(Goodenough_models)
all_DUQ_results = np.zeros((int(len(Goodenough_models)), 2))
accuracy = 0
for iter, fdat in enumerate(Goodenough_models):
    expt = plt.imread(fdat)
    expt = rgb2gray(expt)
    expt = expt.reshape(1, 240, 400, 1)
    xt = norm_im(expt)
    xt = np.clip(xt, np.percentile(xt,percentile_Lowerpercent), np.percentile(xt,percentile_Upperpercent))
    xt = norm_im(xt)
    plt.imsave(save_path+fdat.replace("/mnt/beegfs/home/pearl075/section_1_INS_data/goodenoughANDdimer/results/dataset_All_balanced/test_400/images/fake_B/", ""), np.squeeze(xt), vmin=0, vmax=1)
    xt = np.moveaxis(xt, -1, 1)
    x =  torch.from_numpy(xt).float()
    with torch.no_grad():
        output = model_s(x)
        all_DUQ_results[iter][0] = output[1].numpy()[0][0]
        all_DUQ_results[iter][1] = output[1].numpy()[0][1]
        if output[1].numpy()[0][0] < output[1].numpy()[0][1]:
            accuracy += 100/np.shape(all_DUQ_results)[0]
print ("Dimer mean and RMS: ", np.mean(all_DUQ_results[:,0]), np.std(all_DUQ_results[:,0]))
print ("Goodenough mean and RMS: ", np.mean(all_DUQ_results[:,1]), np.std(all_DUQ_results[:,1]))
print ("Accuracy: ", accuracy)

print ("FakeA dimer")
Dimer_models = sorted(glob.glob("/mnt/beegfs/home/pearl075/section_1_INS_data/goodenoughANDdimer/results/dataset_All_balanced/test_400/images/fake_A/fakeA_Dimer*rescaled.png"))
print ("Calculating average for simulated Dimer that has been cleaned with GAN")
print ("Total of files: ", len(Dimer_models))
np.random.shuffle(Dimer_models)
all_DUQ_results = np.zeros((int(len(Dimer_models)), 2))
accuracy = 0
for iter, fdat in enumerate(Dimer_models):
    expt = plt.imread(fdat)
    expt = rgb2gray(expt)
    expt = expt.reshape(1, 240, 400, 1)
    xt = norm_im(expt)
    xt = np.clip(xt, np.percentile(xt,percentile_Lowerpercent), np.percentile(xt,percentile_Upperpercent))
    xt = norm_im(xt)
    plt.imsave(save_path+fdat.replace("/mnt/beegfs/home/pearl075/section_1_INS_data/goodenoughANDdimer/results/dataset_All_balanced/test_400/images/fake_A/", ""), np.squeeze(xt), vmin=0, vmax=1)
    xt = np.moveaxis(xt, -1, 1)
    x =  torch.from_numpy(xt).float()
    with torch.no_grad():
        output = model_s(x)
        all_DUQ_results[iter][0] = output[1].numpy()[0][0]
        all_DUQ_results[iter][1] = output[1].numpy()[0][1]
        if output[1].numpy()[0][0] > output[1].numpy()[0][1]:
            accuracy += 100/np.shape(all_DUQ_results)[0]
print ("Dimer mean and RMS: ", np.mean(all_DUQ_results[:,0]), np.std(all_DUQ_results[:,0]))
print ("Goodenough mean and RMS: ", np.mean(all_DUQ_results[:,1]), np.std(all_DUQ_results[:,1]))
print ("Accuracy: ", accuracy)

Dimer_models = sorted(glob.glob("/mnt/beegfs/home/pearl075/section_1_INS_data/goodenoughANDdimer/results/dataset_All_balanced/test_400/images/fake_B/fakeB_Dimer*rescaled.png"))
print ("Calculating average for simulated Dimer that has been cleaned with GAN")
print ("Total of files: ", len(Dimer_models))
np.random.shuffle(Dimer_models)
all_DUQ_results = np.zeros((int(len(Dimer_models)), 2))
accuracy = 0
for iter, fdat in enumerate(Dimer_models):
    expt = plt.imread(fdat)
    expt = rgb2gray(expt)
    expt = expt.reshape(1, 240, 400, 1)
    xt = norm_im(expt)
    xt = np.clip(xt, np.percentile(xt,percentile_Lowerpercent), np.percentile(xt,percentile_Upperpercent))
    xt = norm_im(xt)
    plt.imsave(save_path+fdat.replace("/mnt/beegfs/home/pearl075/section_1_INS_data/goodenoughANDdimer/results/dataset_All_balanced/test_400/images/fake_B/", ""), np.squeeze(xt), vmin=0, vmax=1)
    xt = np.moveaxis(xt, -1, 1)
    x =  torch.from_numpy(xt).float()
    with torch.no_grad():
        output = model_s(x)
        all_DUQ_results[iter][0] = output[1].numpy()[0][0]
        all_DUQ_results[iter][1] = output[1].numpy()[0][1]
        if output[1].numpy()[0][0] > output[1].numpy()[0][1]:
            accuracy += 100/np.shape(all_DUQ_results)[0]
print ("Dimer mean and RMS: ", np.mean(all_DUQ_results[:,0]), np.std(all_DUQ_results[:,0]))
print ("Goodenough mean and RMS: ", np.mean(all_DUQ_results[:,1]), np.std(all_DUQ_results[:,1]))
print ("Accuracy: ", accuracy)


print ("testA goodenough")
Goodenough_models = sorted(glob.glob("/mnt/beegfs/home/pearl075/section_1_INS_data/datasets/goodenoughANDdimer_balanced/testA/Goodenough*.png"))
print ("Calculating average for simulated Goodenough that has NOT been cleaned with GAN")
print ("Total of files: ", len(Goodenough_models))
np.random.shuffle(Goodenough_models)
all_DUQ_results = np.zeros((int(len(Goodenough_models)), 2))
accuracy = 0
for iter, fdat in enumerate(Goodenough_models):
    expt = plt.imread(fdat)
    expt = rgb2gray(expt)
    expt = expt.reshape(1, 240, 400, 1)
    xt = norm_im(expt)
    xt = np.clip(xt, np.percentile(xt,percentile_Lowerpercent), np.percentile(xt,percentile_Upperpercent))
    xt = norm_im(xt)
    plt.imsave(save_path+"realA_"+fdat.replace("/mnt/beegfs/home/pearl075/section_1_INS_data/datasets/goodenoughANDdimer_balanced/testA/", ""), np.squeeze(xt), vmin=0, vmax=1)
    xt = np.moveaxis(xt, -1, 1)
    x =  torch.from_numpy(xt).float()
    with torch.no_grad():
        output = model_s(x)
        all_DUQ_results[iter][0] = output[1].numpy()[0][0]
        all_DUQ_results[iter][1] = output[1].numpy()[0][1]
        if output[1].numpy()[0][0] < output[1].numpy()[0][1]:
            accuracy += 100/np.shape(all_DUQ_results)[0]
print ("Dimer mean and RMS: ", np.mean(all_DUQ_results[:,0]), np.std(all_DUQ_results[:,0]))
print ("Goodenough mean and RMS: ", np.mean(all_DUQ_results[:,1]), np.std(all_DUQ_results[:,1]))
print ("Accuracy: ", accuracy)


Goodenough_models = sorted(glob.glob("/mnt/beegfs/home/pearl075/section_1_INS_data/datasets/goodenoughANDdimer_balanced/testB/Goodenough*.png"))
print ("Calculating average for simulated Goodenough that has NOT been cleaned with GAN")
print ("Total of files: ", len(Goodenough_models))
np.random.shuffle(Goodenough_models)
all_DUQ_results = np.zeros((int(len(Goodenough_models)), 2))
accuracy = 0
for iter, fdat in enumerate(Goodenough_models):
    expt = plt.imread(fdat)
    expt = rgb2gray(expt)
    expt = expt.reshape(1, 240, 400, 1)
    xt = norm_im(expt)
    xt = np.clip(xt, np.percentile(xt,percentile_Lowerpercent), np.percentile(xt,percentile_Upperpercent))
    xt = norm_im(xt)
    plt.imsave(save_path+"realB_"+fdat.replace("/mnt/beegfs/home/pearl075/section_1_INS_data/datasets/goodenoughANDdimer_balanced/testB/", ""), np.squeeze(xt), vmin=0, vmax=1)
    xt = np.moveaxis(xt, -1, 1)
    x =  torch.from_numpy(xt).float()
    with torch.no_grad():
        output = model_s(x)
        all_DUQ_results[iter][0] = output[1].numpy()[0][0]
        all_DUQ_results[iter][1] = output[1].numpy()[0][1]
        if output[1].numpy()[0][0] < output[1].numpy()[0][1]:
            accuracy += 100/np.shape(all_DUQ_results)[0]
print ("Dimer mean and RMS: ", np.mean(all_DUQ_results[:,0]), np.std(all_DUQ_results[:,0]))
print ("Goodenough mean and RMS: ", np.mean(all_DUQ_results[:,1]), np.std(all_DUQ_results[:,1]))
print ("Accuracy: ", accuracy)


print ("testA Dimer")
Dimer_models = sorted(glob.glob("/mnt/beegfs/home/pearl075/section_1_INS_data/datasets/goodenoughANDdimer_balanced/testA/Dimer*.png"))
print ("Calculating average for simulated Dimer that has NOT been cleaned with GAN")
print ("Total of files: ", len(Dimer_models))
np.random.shuffle(Dimer_models)
all_DUQ_results = np.zeros((int(len(Dimer_models)), 2))
accuracy = 0
for iter, fdat in enumerate(Dimer_models):
    expt = plt.imread(fdat)
    expt = rgb2gray(expt)
    expt = expt.reshape(1, 240, 400, 1)
    xt = norm_im(expt)
    xt = np.clip(xt, np.percentile(xt,percentile_Lowerpercent), np.percentile(xt,percentile_Upperpercent))
    xt = norm_im(xt)
    plt.imsave(save_path+"realA_"+fdat.replace("/mnt/beegfs/home/pearl075/section_1_INS_data/datasets/goodenoughANDdimer_balanced/testA/", ""), np.squeeze(xt), vmin=0, vmax=1)
    xt = np.moveaxis(xt, -1, 1)
    x =  torch.from_numpy(xt).float()
    with torch.no_grad():
        output = model_s(x)
        all_DUQ_results[iter][0] = output[1].numpy()[0][0]
        all_DUQ_results[iter][1] = output[1].numpy()[0][1]
        if output[1].numpy()[0][0] > output[1].numpy()[0][1]:
            accuracy += 100/np.shape(all_DUQ_results)[0]
print ("Dimer mean and RMS: ", np.mean(all_DUQ_results[:,0]), np.std(all_DUQ_results[:,0]))
print ("Goodenough mean and RMS: ", np.mean(all_DUQ_results[:,1]), np.std(all_DUQ_results[:,1]))
print ("Accuracy: ", accuracy)


Dimer_models = sorted(glob.glob("/mnt/beegfs/home/pearl075/section_1_INS_data/datasets/goodenoughANDdimer_balanced/testB/Dimer*.png"))
print ("Calculating average for simulated Dimer that has NOT been cleaned with GAN")
print ("Total of files: ", len(Dimer_models))
np.random.shuffle(Dimer_models)
all_DUQ_results = np.zeros((int(len(Dimer_models)), 2))
accuracy = 0
for iter, fdat in enumerate(Dimer_models):
    expt = plt.imread(fdat)
    expt = rgb2gray(expt)
    expt = expt.reshape(1, 240, 400, 1)
    xt = norm_im(expt)
    xt = np.clip(xt, np.percentile(xt,percentile_Lowerpercent), np.percentile(xt,percentile_Upperpercent))
    xt = norm_im(xt)
    plt.imsave(save_path+"realB_"+fdat.replace("/mnt/beegfs/home/pearl075/section_1_INS_data/datasets/goodenoughANDdimer_balanced/testB/", ""), np.squeeze(xt), vmin=0, vmax=1)
    xt = np.moveaxis(xt, -1, 1)
    x =  torch.from_numpy(xt).float()
    with torch.no_grad():
        output = model_s(x)
        all_DUQ_results[iter][0] = output[1].numpy()[0][0]
        all_DUQ_results[iter][1] = output[1].numpy()[0][1]
        if output[1].numpy()[0][0] > output[1].numpy()[0][1]:
            accuracy += 100/np.shape(all_DUQ_results)[0]
print ("Dimer mean and RMS: ", np.mean(all_DUQ_results[:,0]), np.std(all_DUQ_results[:,0]))
print ("Goodenough mean and RMS: ", np.mean(all_DUQ_results[:,1]), np.std(all_DUQ_results[:,1]))
print ("Accuracy: ", accuracy)


