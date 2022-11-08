[ChemRxiv] XXX  |  [Paper] XXX

# Exp2SimGAN
Welcome to Exp2SimGAN, a machine learning algorithm that learns to translate between simulated- and experimental data! 

1. [Exp2SimGAN](#Exp2SimGAN)
2. [Getting started (with Colab)](#getting-started-with-colab)
3. [Getting started (own computer)](#getting-started-own-computer)
    1. [Install requirements](#install-requirements)
    2. [Simulate data](#simulate-data)
    3. [Train model](#train-model)
    4. [Predict](#predict)
4. [Author](#author)
5. [Cite](#cite)
6. [Acknowledgments](#Acknowledgments)
7. [License](#license)  

We here apply DeepStruc for the structural analysis of a model system of mono-metallic nanoparticle (MMNPs) with seven
different structure types and demonstrate the method for both simulated and experimental PDFs. DeepStruc can reconstruct
simulated data with an average mean absolute error (MAE) of the atom xyz-coordinates on 0.093 ± 0.058 Å after fitting a
contraction/extraction factor, an ADP and a scale parameter.
We demonstrate the generative capability of DeepStruc on a dataset of face-centered cubic (fcc), hexagonal closed packed
(hcp) and stacking faulted structures, where DeepStruc can recognize the stacking faulted structures as an interpolation
between fcc and hcp and construct new structural models based on a PDF. The MAE is in this example 0.030 ± 0.019 Å.

The MMNPs are provided as a graph-based input to the encoder of DeepStruc. We compare DeepStruc with a similar [DGM](https://github.com/AndyNano/CVAE.git)
without the graph-based encoder. DeepStruc is able to reconstruct the structures using a smaller dimension of the latent
space thus having a better generative capabillity. We also compare DeepStruc with a [brute-force modelling](https://github.com/AndyNano/Brute-force-PDF-modelling.git) approach and a [tree-based classification algorithm](https://github.com/AndyNano/MetalFinder.git). The ML models are significantly faster than the brute-force approach, but DeepStruc can furthermore create a latent space from where synthetic structures can be sampled which the tree-based method cannot!
The baseline models can be found in other repositories: [brute-force](https://github.com/AndyNano/Brute-force-PDF-modelling.git),
[MetalFinder](https://github.com/AndyNano/MetalFinder.git) and [CVAE](https://github.com/AndyNano/CVAE.git).
![alt text](img/DeepStruc.png "DeepStruc")


# Getting started (with Colab)
Using DeepStruc on your own PDFs is straightforward and does not require anything installed or downloaded to your computer.
Follow the instructions in our [Colab notebook](https://colab.research.google.com/github/EmilSkaaning/DeepStruc/blob/main/DeepStruc.ipynb)
and try to play around. 

# Getting started (own computer)
Follow these step if you want to train DeepStruc and predict with DeepStruc locally on your own computer.

## Install requirements
See the [install](/install) folder. 

## Simulate data
See the [data](/data) folder. 

## Train model
To train your own DeepStruc model simply run:
```
python train.py
```
A list of possible arguments or run the '--help' argument for additional information.  
If you are intersted in changing the architecture of the model go to __train.py__ and change the _model_arch_ dictionary. 
 
| Arg | Description | Example |  
| --- | --- |  --- |  
| `-h` or `--help` | Prints help message. |    
| `-d` or `--data_dir` | Directory containing graph training, validation and test data. __str__| `-d ./data/graphs`  |  
| `-s` or `--save_dir` | Directory where models will be saved. This is also used for loading a learner. __str__ | `-s bst_model`  |   
| `-r` or `--resume_model` | If 'True' the save_dir model is loaded and training is continued. __bool__| `-r True`  |
| `-e` or `--epochs` | Number of maximum epochs. __int__| `-e 100`  |  
| `-b` or `--batch_size` | Number of graphs in each batch. __int__| `-b 20`  |  
| `-l` or `--learning_rate` | Learning rate. __float__| `-l 1e-4`  |  
| `-B` or `--beta` | Initial beta value for scaling KLD. __float__| `-B 0.1`  |  
| `-i` or `--beta_increase` | Increments of beta when the threshold is met. __float__| `-i 0.1`  |  
| `-x` or `--beta_max` | Highst value beta can increase to. __float__| `-x 5`  |  
| `-t` or `--reconstruction_th` | Reconstruction threshold required before beta is increased. __float__| `-t 0.001`  |  
| `-n` or `--num_files` | Total number of files loaded. Files will be split 60/20/20. If 'None' then all files are loaded. __int__| `-n 500`  |  
| `-c` or `--compute` | Train model on CPU or GPU. Choices: 'cpu', 'gpu16', 'gpu32' and 'gpu64'. __str__| `-c gpu32`  |  
| `-L` or `--latent_dim` | Number of latent space dimensions. __int__| `-L 3`  |  

## Predict
To predict a MMNP using DeepStruc or your own model on a PDF:
```
python predict.py
```
A list of possible arguments or run the '--help' argument for additional information.  
 
| Arg | Description | Example |  
| --- | --- |  --- |  
| `-h` or `--help` | Prints help message. |    
| `-d` or `--data` | Path to data or data directory. If pointing to data directory all datasets must have same format. __str__| `-d data/experimental_PDFs/JQ_S1.gr`  |  
| `-m` or `--model` | Path to model. If 'None' GUI will open. __str__ | `-m ./models/DeepStruc`  |   
| `-n` or `--num_samples` | Number of samples/structures generated for each unique PDF. __int__| `-n 10`  |
| `-s` or `--sigma` | Sample to '-s' sigma in the normal distribution. __float__| `-s 7`  |  
| `-p` or `--plot_sampling` | Plots sampled structures on top of DeepStruc training data. Model must be DeepStruc. __bool__| `-p True`  |  
| `-g` or `--save_path` | Path to directory where predictions will be saved. __bool__| `-g ./best_preds`  |  
| `-i` or `--index_plot` | Highlights specific reconstruction in the latent space. --data must be specific file and not directory and  '--plot True'. __int__| `-i 4`  |  
| `-P` or `--plot_data` | If True then the first loaded PDF is plotted and shown after normalization. __bool__| `-P ./best_preds`  |  
  

# Authors
__Andy S. Anker__<sup>1</sup>   
__Keith T. Butler__<sup>2,4</sup>  
__Manh Duc Le__<sup>3</sup>  
__Toby G. Perring__<sup>3</sup>     
__Jeyan Thiyagalingam__<sup>2</sup>    
 
<sup>1</sup> Department of Chemistry and Nano-Science Center, University of Copenhagen, Denmark.   
<sup>2</sup> Scientiﬁc Computing Department, Rutherford Appleton Laboratory, England.   
<sup>3</sup> ISIS Neutron and Muon Source, Rutherford Appleton Laboratory, England.   
<sup>4</sup> Current affiliation: School of Engineering and Materials Science, Queen Mary University of London, England.   
Should there be any question, desired improvement or bugs please contact us on GitHub or 
through email: __andy@chem.ku.dk__.

# Cite
If you use our code or our results, please consider citing our papers. Thanks in advance!
```
@article{anker2022Exp2SimGAN,
  title={Using generative adversarial networks to match experimental and simulated inelastic neutron scattering data},
  author={Andy S. Anker, Keith T. Butler, Manh D. Le, Toby G. Perring, Jeyan Thiyagalingam},
  year={2022}}
```

# Acknowledgments
Our code is developed based on the the following publication:
```
@inproceedings{han2021dual,
  title={Dual contrastive learning for unsupervised image-to-image translation},
  author={Han, Junlin and Shoeiby, Mehrdad and Petersson, Lars and Armin, Mohammad Ali},
  year={2021}
}
```
```
@inproceedings{park2020cut,
  title={Contrastive Learning for Unpaired Image-to-Image Translation},
  author={Taesung Park and Alexei A. Efros and Richard Zhang and Jun-Yan Zhu},
  year={2020}
}
```

# License
This project is licensed under the Apache License Version 2.0, January 2004 - see the [LICENSE](LICENSE) file for details.
