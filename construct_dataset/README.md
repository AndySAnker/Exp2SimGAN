[ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/63a15e21a53ea6c3c751564f)  |  [Paper] XXX

# Exp2SimGAN - Construct data

The data to train the algorithm is deposited at Zenodo - https://zenodo.org/record/7308423#.Y2zgoOzML0o:
- dimer_hk_labels.npy
- dimer_hk_simulated.npy
- goodenough_hk_labels.npy
- goodenough_hk_simulated.npy

and experimental data:
- exp_data.npy

and simulated reference data:
- sim_ref_data_dimer.npy
- sim_ref_data_goodenough.npy

Use the python files in this folder to open the datasets and convert them to the right trainingset format of Exp2SimGAN.

The data generation is described further in the publication [_Interpretable, calibrated neural networks for analysis and understanding of neutron spectra_](https://iopscience.iop.org/article/10.1088/1361-648X/abea1c) that has another dedicated [GitHub repository.](https://github.com/keeeto/interpretable-ml-neutron-spectroscopy)
