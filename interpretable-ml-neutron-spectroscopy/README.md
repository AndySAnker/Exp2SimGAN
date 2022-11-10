# DUQ Classifier
This is a truncated and altered version of the repository of code associated with the publication [_Interpretable, calibrated neural networks for analysis and understanding of neutron spectra_](https://arxiv.org/abs/2011.04584)

Here we only use the DUQ classifier.

## Running the codes

There are different `conda` environments associated with the different codes:

* To run the uncertainty quntification netowrks you will need to load the `pytorch` environment in `environment-torch.yml`

## Using Docker

You can also use Docker container to run the code.

```
docker pull mducle/ml_ins_duq
docker run -ti -p8888:8888 ml_ins_duq 
```

This will start the notebook.
You should then navigate to `http://localhost:8888/notebooks/models/duq/notebook/duq-publication.ipynb` to load the notebook.
The password is `pcsmo`.
