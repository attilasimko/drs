# Code repository for the manuscript titled "Towards MR contrast independent synthetic CT generation."

Submitted to the special edition in Artificial Intelligence of Zeitschrift fuer Medizinische Physik.

### Data generation

The data used for training is based on a privately acquied dataset, and the pre-processing steps are collected in "datageneration.py". This code contains decomposing the MR signal into the synthetic quantitative maps (PD, T1 and T2) used to train our proposed robust model. The outputs of this script are the files used for training.

### Training code

The script "training.py" uses a DataGenerator to load in the dataset created previously, build and compile the SRResNet architecture and train the model for sCT generation. The argument "--case" determines if the model will be trained on the direct MR images ("I") or on the synthetic quantitative maps ("II"). Other arguments are "--lr" for the learning rate, "--dropout" for the dropout-rate, "--gpu" for selecting between multiple available GPUs.

### Evaluating code

A figure similar to that presented in the paper can be created using your own images with the script "evaluating_range.py" which transfers the contrast of the images to a wide range, and evaluates the sCT models on each contrast. The script loads in both trained models, steps through the ranges also presented in the paper and evaluates a subset of our validation dataset on the contrasts.

Two example images are also presented:
- "I_contrast_map.png" shows the results for the first model (sCT_MR)
- "II_contrast_map.png" show the results for the second model (sCT_sQM)

Note the different windowing of the images!

### Trained-weights

The trained weights can be downloaded from google drive:
- https://drive.google.com/file/d/1UYDE7LHmp3CAUFRfDfm-NxtsXrc6ZRC-/view?usp=share_link
The weights for the sCT_MR model evaluated in the paper.
- https://drive.google.com/file/d/1p6gAzaz6eyHaocwM-gOp1COzeYM8Xos5/view?usp=share_link 
The weights for the sCT_sQM model evaluated in the paper.
- https://drive.google.com/file/d/1hHXXddShhhsH5u8ob564cmHAMWzAedsn/view?usp=share_link
The contrast transfer model presented in "Changing the Contrast of Magnetic Resonance Imaging Signals using Deep Learning" (https://proceedings.mlr.press/v143/simko21a.html)

An evaluation method using all three models is presented in "evaluating_range.py".