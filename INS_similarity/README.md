[ChemRxiv] XXX  |  [Paper] XXX

# Exp2SimGAN - Trusting the Machine

We have  implemented an quantification for how much you can trust the results of Exp2SimGAN on your data inspired by a [FID score]("https://arxiv.org/abs/1706.08500"). Here we calculate the Wasserstein distance between the dataset and the trainingset in the featurespace. A small Wasserstein distance represents data with high similarity to the trainingset and the model can confidently be applied on this dataset. However, a large Wasserstein distance represents data with low similarity to the trainingset and the user has to be cautious to use the model on this dataset.

<p align="center">
  <img width="400" src="../imgs/TrustingTheMachine.png">
</p>

In order to calculate and visualize the Wasserstein distances form a dataset to the trainingset follow the procedure:
1. Run test.py on a dataset and save latent space in the last line of test.py
2. Run WS_dist_fakeA.py or WS_dist_fakeB.py to calculate the Wasserstein distances between the latent spaces
3. Run Ridgeline_plot_fakeA.py or Ridgeline_plot_fakeB.py to make visualisations of latent spaces
