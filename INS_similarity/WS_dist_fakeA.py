import numpy as np
import pdb, sys, torch, time, random
import matplotlib.pyplot as plt
import torch.nn as nn
from math import log2
from layers import SinkhornDistance

savepath = "WS_list_single_dir_fakeA/"
# Load data
X_train = np.load("latentspace_locations_fakeA/trainingset_fake_A_latentspace.npy")
X_testA = np.load("latentspace_locations_fakeA/testset_fake_A_latentspace.npy")
X_testB = np.load("latentspace_locations_fakeA/testset_fake_B_latentspace.npy")
X_animals = np.load("latentspace_locations_fakeA/animals_fake_A_latentspace.npy")
X_Exp = np.load("latentspace_locations_fakeA/Exp_fake_A_latentspace.npy")
X_franken = np.load("latentspace_locations_fakeA/franken_fake_A_latentspace.npy")
X_Rb2MnF4 = np.load("latentspace_locations_fakeA/Rb2MnF4_fake_A_latentspace.npy")
X_Nore = np.load("latentspace_locations_fakeA/sim_ref_Nore_fake_A_latentspace.npy")
X_TobyFit = np.load("latentspace_locations_fakeA/sim_ref_TobyFit_fake_A_latentspace.npy")
X_Digits = np.load("latentspace_locations_fakeA/Digits_fake_A_latentspace.npy")
samples = 20 # Max samples drawn from a dist.
print ("samples: ", samples)

# Inspired by blog post: https://github.com/dfdazac/wassdistance
WS = SinkhornDistance(eps=0.1, max_iter=100)

#  flatten data
X_train = X_train.reshape(np.shape(X_train)[0], -1)
X_testA = X_testA.reshape(np.shape(X_testA)[0], -1)
X_testB = X_testB.reshape(np.shape(X_testB)[0], -1)
X_animals = X_animals.reshape(np.shape(X_animals)[0], -1)
X_Exp = X_Exp.reshape(np.shape(X_Exp)[0], -1)
X_franken = X_franken.reshape(np.shape(X_franken)[0], -1)
X_Rb2MnF4 = X_Rb2MnF4.reshape(np.shape(X_Rb2MnF4)[0], -1)
X_Nore = X_Nore.reshape(np.shape(X_Nore)[0], -1)
X_TobyFit = X_TobyFit.reshape(np.shape(X_TobyFit)[0], -1)
X_Digits = X_Digits.reshape(np.shape(X_Digits)[0], -1)

# Here we just take random samples from latent space dist.
WS_train_list = []
WS_testA_list = []
WS_testB_list = []
WS_animals_list = []
WS_Exp_list = []
WS_franken_list = []
WS_Rb2MnF4_list = []
WS_Nore_list = []
WS_TobyFit_list = []
WS_Digits_list = []
start_time = time.time()
for i in range(1000):
    print (i)
    #X_train = random.sample(list(X_train), samples)
    #X_test = random.sample(list(X_test), samples)
    WS_train, P, C = WS(torch.tensor(random.sample(list(X_train), samples)), torch.tensor(random.sample(list(X_train), samples)))
    WS_testA, P, C = WS(torch.tensor(random.sample(list(X_testA), samples)), torch.tensor(random.sample(list(X_train), samples)))
    WS_testB, P, C = WS(torch.tensor(random.sample(list(X_testB), samples)), torch.tensor(random.sample(list(X_train), samples)))
    WS_animals, P, C = WS(torch.tensor(X_animals), torch.tensor(random.sample(list(X_train), samples)))
    WS_Exp, P, C = WS(torch.tensor(X_Exp), torch.tensor(random.sample(list(X_train), samples)))
    WS_franken, P, C = WS(torch.tensor(X_franken), torch.tensor(random.sample(list(X_train), samples)))
    WS_Rb2MnF4, P, C = WS(torch.tensor(X_Rb2MnF4), torch.tensor(random.sample(list(X_train), samples)))
    WS_Nore, P, C = WS(torch.tensor(X_Nore), torch.tensor(random.sample(list(X_train), samples)))
    WS_TobyFit, P, C = WS(torch.tensor(X_TobyFit), torch.tensor(random.sample(list(X_train), samples)))
    WS_Digits, P, C = WS(torch.tensor(X_Digits), torch.tensor(random.sample(list(X_train), samples)))
    #WS_train, P, C = WS(torch.tensor(X_train), torch.tensor(X_train[np.random.randint(len(X_train))]).repeat(2,1))
    #WS_test, P, C = WS(torch.tensor(X_test), torch.tensor(X_train[np.random.randint(len(X_train))]).repeat(2,1))
    #WS_animals, P, C = WS(torch.tensor(X_animals), torch.tensor(X_train[np.random.randint(len(X_train))]).repeat(2,1))
    #WS_Exp, P, C = WS(torch.tensor(X_Exp).repeat(2,1), torch.tensor(X_train[np.random.randint(len(X_train))]).repeat(2,1))
    #WS_franken, P, C = WS(torch.tensor(X_franken).repeat(2,1), torch.tensor(X_train[np.random.randint(len(X_train))]).repeat(2,1))
    #WS_Rb2MnF4, P, C = WS(torch.tensor(X_Rb2MnF4), torch.tensor(X_train[np.random.randint(len(X_train))]).repeat(2,1))
    #WS_Nore, P, C = WS(torch.tensor(X_Nore).repeat(2,1), torch.tensor(X_train[np.random.randint(len(X_train))]).repeat(2,1))
    #WS_TobyFit, P, C = WS(torch.tensor(X_TobyFit).repeat(2,1), torch.tensor(X_train[np.random.randint(len(X_train))]).repeat(2,1))
    #WS_Digits, P, C = WS(torch.tensor(X_Digits), torch.tensor(X_train[np.random.randint(len(X_train))]).repeat(2,1))

    WS_train_list.append(WS_train)
    WS_testA_list.append(WS_testA)
    WS_testB_list.append(WS_testB)
    WS_animals_list.append(WS_animals)
    WS_Exp_list.append(WS_Exp)
    WS_franken_list.append(WS_franken)
    WS_Rb2MnF4_list.append(WS_Rb2MnF4)
    WS_Nore_list.append(WS_Nore)
    WS_TobyFit_list.append(WS_TobyFit)
    WS_Digits_list.append(WS_Digits)
np.save(savepath+"train_WS_list_single.npy", WS_train_list)
np.save(savepath+"testA_WS_list_single.npy", WS_testA_list)
np.save(savepath+"testB_WS_list_single.npy", WS_testB_list)
np.save(savepath+"animals_WS_list_single.npy", WS_animals_list)
np.save(savepath+"Exp_WS_list_single.npy", WS_Exp_list)
np.save(savepath+"franken_WS_list_single.npy", WS_franken_list)
np.save(savepath+"Rb2MnF4_WS_list_single.npy", WS_Rb2MnF4_list)
np.save(savepath+"Nore_WS_list_single.npy", WS_Nore_list)
np.save(savepath+"TobyFit_WS_list_single.npy", WS_TobyFit_list)
np.save(savepath+"Digits_WS_list_single.npy", WS_Digits_list)
print ("WS train: ", np.mean(WS_train_list), " +/- ", np.std(WS_train_list))
print ("WS testA: ", np.mean(WS_testA_list), " +/- ", np.std(WS_testA_list))
print ("WS testB: ", np.mean(WS_testB_list), " +/- ", np.std(WS_testB_list))
print ("WS animals: ", np.mean(WS_animals_list), " +/- ", np.std(WS_animals_list))
print ("WS Exp: ", np.mean(WS_Exp_list), " +/- ", np.std(WS_Exp_list))
print ("WS franken: ", np.mean(WS_franken_list), " +/- ", np.std(WS_franken_list))
print ("WS Rb2MnF4: ", np.mean(WS_Rb2MnF4_list), " +/- ", np.std(WS_Rb2MnF4_list))
print ("WS Nore: ", np.mean(WS_Nore_list), " +/- ", np.std(WS_Nore_list))
print ("WS TobyFit: ", np.mean(WS_TobyFit_list), " +/- ", np.std(WS_TobyFit_list))
print ("WS Digits: ", np.mean(WS_Digits_list), " +/- ", np.std(WS_Digits_list))

