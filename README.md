# Monograph
Unsupervised Machine Learning of the Ising and XY model.

A brief explanation:
1 - The metropolis files can be used to generate samples for both models. Periodic boundaries are assumed.

2 - The analysis files can be used to perform PCA, KPCA, AE and VAE on the samples, as well as a probabilistic analysis of the latent variables obtained with these methods.

3 - The previous files make use of the VAE script. The file must be on the same folder, otherwise the import command must be modified accordingly.

4 - The analysis files make use of pytorch (v 1.10).

5 - The density of states calculator provides an estimate for the Ising model.
