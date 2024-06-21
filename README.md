# Thesis ***Automatic Learning For Estimating the T2 Spectrum and Myelin Water Fraction Mapping from MR Data*** - Source code
This is the repository that replicates results obtained in the Master thesis entitled "Automatic Learning For Estimating the $T_{2}$ Spectrum and Myelin Water Fraction Mapping from MR Data" presented by Daniel Vallejo Aldana - Research Center in Mathematics (CIMAT) Guanajuato Mexico.

- *@Author*: Daniel Vallejo Aldana
- *@Contact*: daniel.vallejo@cimat.mx

* To run all codes from this repository install the required packages described in the requirements file.

This repository is structured in three main parts. 
* The first corresponds to the processing stage, where the images are processed to correct motion and remove noise.
* The second part corresponds to the supervised learning approach, where the synthetic data is generated from parameters given by the user, and the Multi-Layer Perceptron model is trained using this synthetic database.
* The third part corresponds to the self-supervised learning approach. Like the supervised learning approach, we generate a synthetic database and use it to train the self-supervised model 
