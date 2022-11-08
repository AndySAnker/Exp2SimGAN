import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from ignite.handlers import EarlyStopping
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss
import time
import os  
import pickle
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from models_large import CNN_DUQ
import numpy as np
import copy, glob
import matplotlib.pyplot as plt
plt.style.use('sciml-style')
from tqdm import tqdm
torch.backends.cudnn.enabled = True

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def norm_im(im):
    maxval = np.max(im)
    minval = np.min(im)
    return (im - minval)/(maxval - minval)

#def norm_im(im):
#    return im / np.max(im)

#data_path = 'mnt/beegfs/home/pearl075/interpretable-ml-neutron-spectroscopy/models/section_4-2/train-model/resolution/'
data_path = '../../../../../construct_dataset_new/'
ge_data_file =  'goodenough_hk_simulated.npy' #'simulated_resolution_goodenough.npy'
di_data_file =  'dimer_hk_simulated.npy' #simulated_resolution_dimer.npy'

Goodenough_images = sorted(glob.glob("../../../../../datasets/goodenoughANDdimer/trainA/Goodenough*"))
Dimer_images = sorted(glob.glob("../../../../../datasets/goodenoughANDdimer/trainA/Dimer*"))
Goodenough_images = sorted(glob.glob("/mnt/beegfs/home/pearl075/section_1_INS_data/goodenoughANDdimer/results/dataset_All_balanced/test_400/images/fake_A/fakeA_Goodenough*rescaled.png"))
Dimer_images = sorted(glob.glob("/mnt/beegfs/home/pearl075/section_1_INS_data/goodenoughANDdimer/results/dataset_All_balanced/test_400/images/fake_A/fakeA_Dimer*rescaled.png"))
ge_data = torch.zeros((int(len(Goodenough_images)), 240, 400))
di_data = torch.zeros((int(len(Dimer_images)), 240, 400))
percentile_Lowerpercent = 0
percentile_Upperpercent = 99.8
print ("percentile_Lowerpercent", percentile_Lowerpercent)
print ("percentile_Upperpercent", percentile_Upperpercent)
for iter, fdat in enumerate(Goodenough_images):
    expt = plt.imread(fdat)
    expt = rgb2gray(expt)
    expt = expt.reshape(240, 400)
    xt = norm_im(expt)
    xt = np.clip(xt, np.percentile(xt,percentile_Lowerpercent), np.percentile(xt,percentile_Upperpercent))
    xt = norm_im(xt)
    x = torch.from_numpy(xt).float()
    ge_data[iter] = x
for iter, fdat in enumerate(Dimer_images):
    expt = plt.imread(fdat)
    expt = rgb2gray(expt)
    expt = expt.reshape(240, 400)
    xt = norm_im(expt)
    xt = np.clip(xt, np.percentile(xt,percentile_Lowerpercent), np.percentile(xt,percentile_Upperpercent))
    xt = norm_im(xt)
    x = torch.from_numpy(xt).float()
    di_data[iter] = x
#train_data_file = 'training_data.pickle'
print (ge_data_file)
print (ge_data.shape, di_data.shape)

#ge_data = np.squeeze(np.load(data_path+ge_data_file).squeeze())
#di_data = np.squeeze(np.load(data_path+di_data_file))
#ge_data = ge_data[:3000,1,:,:]
#di_data = di_data[:3000,1,:,:]
ge_data = ge_data[:2622]
di_data = di_data[:2622]

print (len(ge_data), len(di_data))
labels = np.zeros((len(ge_data) + len(di_data), 1))
labels[:len(ge_data)] = 1.

X, y = shuffle(np.concatenate((ge_data, di_data)), labels)
y = np.array([int(b[0]) for b in y])
#X = np.concatenate((ge_data, di_data))
#y = labels
X = np.expand_dims(X, axis=3)
X = np.moveaxis(X, -1,1)
#X = np.clip(X, 0, 120)
#X = np.clip(X, 0, 10*np.median(X))
d = [norm_im(i) for i in X]
X = np.array(d)
np.nan_to_num(X, copy = False, nan=0)

#
if torch.cuda.is_available():  
  print("GPU found")
  device = "cuda:0" 
else:  
  print("GPU not found")
  device = "cpu"

batch_size = 64
X_train = X[:int(0.8*len(X))]
y_train = y[:int(0.8*len(X))]
X_test = X[int(0.8*len(X)):]
y_test = y[int(0.8*len(X)):]
print(X.shape, y.shape)

ds_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), F.one_hot(torch.from_numpy(y_train)).float())
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)

ds_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), F.one_hot(torch.from_numpy(y_test)).float())
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)

np.random.seed(0)
torch.manual_seed(0)

l_gradient_penalty = 1.0

model = CNN_DUQ(input_size=(240, 400, 1), num_classes=2, embedding_size=64,
               learnable_length_scale=False, length_scale=1., gamma=1.)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
#optimizer = torch.optim.Adam(model.parameters())

def calc_gradient_penalty(x, y_pred):
    gradients = torch.autograd.grad(
            outputs=y_pred,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
        )[0]

    gradients = gradients.flatten(start_dim=1)
    # L2 norm
    grad_norm = gradients.norm(2, dim=1)
    # Two sided penalty
    gradient_penalty = ((grad_norm - 1) ** 2).mean()
# One sided penalty - down
#     gradient_penalty = F.relu(grad_norm - 1).mean()
    return gradient_penalty


def output_transform_acc(output):
    y_pred, y, x, z = output
    #print (y_pred[:10])
    #print (y[:10])

    y = torch.argmax(y, dim=1)
    #print (torch.argmax(y_pred, dim=1)[:10])
    #print (y[:10])
    return y_pred, y

def output_transform_bce(output):
    y_pred, y, x, z = output

    return y_pred, y

def output_transform_gp(output):
    y_pred, y, x, z = output

    return x, y_pred


def step(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = batch
    x, y = x.cuda(), y.cuda()
    x.requires_grad_(True)
    z, y_pred = model(x)
    loss1 =  F.binary_cross_entropy(y_pred, y)
    loss2 = l_gradient_penalty * calc_gradient_penalty(x, y_pred)
    loss = loss1 + loss2
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.update_embeddings(x, y)

    return loss.item()

def eval_step(engine, batch):
    model.eval()
    x, y = batch
    x, y = x.cuda(), y.cuda()
    x.requires_grad_(True)
    z, y_pred = model(x)
    return y_pred, y, x, z

trainer = Engine(step)
evaluator = Engine(eval_step)

#handler = EarlyStopping(patience=10, score_function=output_transform_acc, trainer=trainer)
# Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
#evaluator.add_event_handler(Events.COMPLETED, handler)

metric = Accuracy(output_transform=output_transform_acc)
metric.attach(evaluator, "accuracy")

metric = Loss(F.binary_cross_entropy, output_transform=output_transform_bce)
metric.attach(evaluator, "bce")

metric = Loss(calc_gradient_penalty, output_transform=output_transform_gp)
metric.attach(evaluator, "gp")


ds_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), F.one_hot(torch.from_numpy(y_train)).float())
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)

ds_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), F.one_hot(torch.from_numpy(y_test)).float())
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=200, shuffle=False)

model = model.cuda()


t_start = time.time()
@trainer.on(Events.EPOCH_COMPLETED)
def log_results(trainer):
    evaluator.run(dl_train)
    metrics_train = evaluator.state.metrics
    torch.save(model.state_dict(), 'trained_models_balanced_percentile0p2clip/uq-discrim-newer-'+str(trainer.state.epoch)+'.pt') 
    
    print("Test Results Train - Epoch: {} Acc: {:.4f} BCE: {:.2f} GP {:.2f} Tot Time: {:.2f} s"
          .format(trainer.state.epoch, metrics_train['accuracy'], metrics_train['bce'], metrics_train['gp'],
                 time.time() - t_start))

    evaluator.run(dl_test)
    metrics_validate = evaluator.state.metrics
    print("Test Results Validate - Epoch: {} Acc: {:.4f} BCE: {:.2f} GP {:.2f} Tot Time: {:.2f} s"
          .format(trainer.state.epoch, metrics_validate['accuracy'], metrics_validate['bce'], metrics_validate['gp'],
                 time.time() - t_start))

trainer.run(dl_train, max_epochs=100)
torch.save(model.state_dict(), './uq-discrim-newer.pt')
