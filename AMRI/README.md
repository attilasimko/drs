# Code repository for the manuscript 'Improving MR image quality with a multi-task, multi-domain CNN using convolutional losses.'

Submitted to IEEE Transactions on Medical Imaging.
The code uses the artefact augmentations from the repository: https://github.com/SimoneRuiter/artefacts

### Data generation


The fastMRI dataset is processed in 'data generation/fastMRI.py' while the in-house Pelvic dataset is processed for training and evaluation in 'data generation/Pelvic.py'. This code contains all the pre-processing of the images required to train the model.

### Training code

The script "training.py" uses a DataGenerator to load in the dataset created previously, build and compile the KIKI-Net architecture and train the model to correct the MRI artefacts.

### Evaluating code

The evaluation of correcting for the individual artefacts can be found in the 'Evaluations/' folder for subsampling, motion, noise and bias in 'subsampling.py', 'motion.py', 'noise.py' and 'bias.py', respectively. Similarly, the evaluation of the multi-task learning method can be found in 'Evaluations/multitask.py', while the individual model components are evaluated in 'Evaluations/components.py'.

### Trained-weights

The trained weights of the proposed model and the models trained to correct individual artefacts can be found on Google Drive:
- https://drive.google.com/drive/folders/18YN0A0whSXIvaizG1nZ58IaI7Wuev6WK?usp=share_link