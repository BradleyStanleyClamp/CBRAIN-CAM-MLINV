# CBRAIN-CAM - a neural network climate model parameterization
# Physically-constrained & Physically-informed fork 

Second Fork Author: Bradley Stanley-Clamp - <bradleysc@robots.ox.ac.uk>
Fork Author: Tom Beucler - <tom.beucler@gmail.com> - https://wp.unil.ch/dawn
Main Repository Author: Stephan Rasp - <raspstephan@gmail.com> - https://raspstephan.github.io

This is a fork of the work by Beucler .et al Climate-invariant Machine learning, looking at how we can learn invariant transformations in a data driven approach. 

Thank you for checking out this fork of the CBRAIN repository (https://github.com/raspstephan/CBRAIN-CAM), dedicated to building *physically-constrained and physically-informed* climate model parameterizations. This is a working fork in a working repository, which means that recent commits may not always be functional or documented. 

If you are looking for the version of the code that corresponds to the climate-invariant paper, check out this release: 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5775489.svg)](https://doi.org/10.5281/zenodo.5775489)

If you are looking for the version of the code that corresponds to the PNAS paper, check out this release: https://github.com/raspstephan/CBRAIN-CAM/releases/tag/PNAS_final

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1402384.svg)](https://doi.org/10.5281/zenodo.1402384)

The modified SPCAM3 climate model code is available at https://gitlab.com/mspritch/spcam3.0-neural-net (branch: `nn_fbp_engy_ess`)


### Papers using this fork

> (Submitted) Beucler, T., Pritchard, M., Yuval, J., Gupta, A., Peng, L., Rasp, S., 
> Ahmed, F., O'Gorman, P.A., Neelin, J.D., Lutsko, N.J. and Gentine, P.: 
> Climate-Invariant Machine Learning. 
> arXiv preprint arXiv:2112.08440.
> https://arxiv.org/abs/2112.08440

> Beucler, T., Pritchard, M., Rasp, S., Ott, J., Baldi, P., & Gentine, P.: 
> Enforcing Analytic Constraints in Neural-Networks Emulating Physical Systems. 
> Physical Review Letters, 126.9: 098302. Editors’ Suggestion. 
> [arXiv pdf](https://arxiv.org/abs/1909.00912)
> https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.126.098302

> Brenowitz, N., T. Beucler, M. Pritchard & C. Bretherton: 
> Interpreting and Stabilizing Machine-Learning Parametrizations of Convection. 
> Journal of the Atmospheric Sciences, 77, 4357-4375.
> https://journals.ametsoc.org/view/journals/atsc/77/12/jas-d-20-0082.1.xml

> (Workshop) Beucler, T., Pritchard, M., Gentine, P., & Rasp, S.: 
> Towards Physically-Consistent, Data-Driven Models of Convection. 
> IEEE International Geoscience and Remote Sensing Symposium 2020. 
> [arXiv pdf](https://arxiv.org/abs/2002.08525
> https://ieeexplore.ieee.org/document/9324569

> (Workshop) Beucler, T., Rasp, S., Pritchard, M., & Gentine, P.: 
> Achieving Conservation of Energy in Neural Network Emulators for Climate Modeling. 
> 2019 International Conference on Machine Learning.
> https://arxiv.org/abs/1906.06622

### Papers using the main repository

> S. Rasp, M. Pritchard and P. Gentine, 2018.
> Deep learning to represent sub-grid processes in climate models
> https://arxiv.org/abs/1806.04731
 
> P. Gentine, M. Pritchard, S. Rasp, G. Reinaudi and G. Yacalis, 2018. 
> Could machine learning break the convection parameterization deadlock? 
> Geophysical Research Letters. http://doi.wiley.com/10.1029/2018GL078202


## Repository description

The main components of the repository are:

- `cbrain`: Contains the cbrain module with all code to preprocess the raw data, run the neural network experiments and analyze the data.
- `pp_config`: Contains configuration files and shell scripts to preprocess the climate model data to be used as neural network inputs
- `nn_config`: Contains neural network configuration files to be used with `run_experiment.py`.
- `notebooks`: Contains Jupyter notebooks used to analyze data. All plotting and data analysis for the papers is done in the subfolder `presentation`. `dev` contains development notebooks.
- `wkspectra`: Contains code to compute Wheeler-Kiladis figures. These were created by [Mike S. Pritchard](http://sites.uci.edu/pritchard/)
- `save_weights.py`: Saves the weights, biases and normalization vectors in text files. These are then used as input for the climate model.

