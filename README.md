[ChemRxiv] XXX  |  [Paper] XXX

# Exp2SimGAN
Welcome to Exp2SimGAN, a machine learning algorithm that learns to translate between simulated- and experimental data! 

1. [Exp2SimGAN](#Exp2SimGAN)
2. [Getting started](#getting-started)
    1. [Install requirements](#install-requirements)
    2. [Simulate data](#simulate-data)
    3. [Train model](#train-model)
    4. [Predict](#predict)
3. [Author](#author)
4. [Cite](#cite)
5. [Acknowledgments](#Acknowledgments)
6. [License](#license)  

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


# Getting started
Follow these step if you want to train Exp2SimGAN and predict with Exp2SimGAN locally on your own computer.

## Install requirements
See the [install](/install) folder. 

## Train model
To train your own DeepStruc model simply run:
```
python train.py
```
A list of possible arguments can be found in the following files:
- [base options](/options/base_options.py)
- [train options](/options/train_options.py)

## Predict
To predict a MMNP using DeepStruc or your own model on a PDF:
```
python predict.py
```
A list of possible arguments can be found in the following files:
- [base options](/options/base_options.py)
- [test options](/options/test_options.py)

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
