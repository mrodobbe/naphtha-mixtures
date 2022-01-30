# Property Prediction of Light Oil Fractions
Machine learning method for predicting physicochemical properties of light oil fractions

This repository contains the GauL-HDAD-derived algorithm for representing chemical mixtures as a numeric vector and neural networks for mixture property prediction.

## Installation
This package is written in python and can be run from command prompt. This requires the installation of several python packages, which are listed below. It is recommended to use Anaconda for managing packages (https://docs.anaconda.com/anaconda/install/).
* Python 3.8 or higher
* NumPy         `conda install numpy`
* RDKit         `conda install -c conda-forge rdkit`
* Scikit-learn  `conda install -c conda-forge scikit-learn`
* joblib        `conda install -c anaconda joblib`
* TensorFlow 2  `pip install tensorflow`
* matplotlib    `conda install -c conda-forge matplotlib`

The package can then be installed by cloning this repository.

## Data
The data that is used in the original work is available in the folder `Data`. Five files are included:
* `labeled_library.pickle`  A library with all C<sub>4</sub> to C<sub>12</sub> PIONA molecules and their experimental data (if available)
* `mei_input.pickle`        The input values (lumped naphtha samples) from Mei _et al._
* `mei_output.pickle`        The output values (boiling points and mixture properties) from Mei _et al._
* `pyl_input.pickle`        The input values (lumped naphtha samples) from Pyl _et al._
* `pyl_output.pickle`        The output values (boiling points and mixture properties) from Pyl _et al._

Changing the input and output sources is possible in the file `input.py`.

## Training
### Pretrained Model
This repository contains a folder named `pretrained`. In that folder, pretrained pure compound property models are available and the gaussian mixture models for the molecular representations are premade using all molecules in `Data/labeled_library.pickle`.
Using the pretrained model speeds up training, since only the boiling points and the desired mixture property has to be trained.
Training mixture properties using the pretrained model is possible via following command:
`python train.py pretrained <property>`
Replace <property> with the mixture property that you want to predict.

Currently available properties:
* _Using data from Pyl et al._:
    Specific Gravity: sg   >>> `python train.py pretrained sg`
* _Using data from Mei et al._:
    Liquid Density: d20 or density   >>> `python train.py pretrained d20` or `python train.py pretrained density`
    Dynamic Viscosity: mu or viscosity   >>> `python train.py pretrained mu` or `python train.py pretrained viscosity`
    Surface Tension: st or surface tension   >>> `python train.py pretrained st` or `python train.py pretrained "surface tension"`

All results will be found in the `pretrained` folder.

### New Model
It is also possible to train all models yourself. Due to the large number of hydrocarbons, the creation of Gaussian mixture models will take several hours. 
You can train the models simply using `python train.py <folder> <property>`
Replace <folder> with the name of the folder where you want to store your results. 
Replace <property> with the desired property, as statedabove.

## How to cite?
When using this prediction model for your own publication, please cite the original papers:

*Learning Molecular Representations for Thermochemistry Prediction of Cyclic Hydrocarbons and Oxygenates <br>
Dobbelaere, M.R.; Plehiers, P.P.; Van de Vijver, R.;  Stevens, C.V.; Van Geem, K.M. <br>
J. Phys. Chem. A 2021, 125, 23, 5166–5179*

*Machine Learning for Physicochemical Property Prediction of Complex Hydrocarbon Mixtures <br>
Dobbelaere, M.R.; Ureel, Y.; Vermeire, F.H.; Stevens, C.V.; Van Geem, K.M. <br>
Submitted to Industrial and Engineering Chemistry Research, 2022*

