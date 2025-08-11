## Variational Inverting Network

## Requirements and Dependencies

* Ubuntu 16.04, cuda 10.0
* Python 3.6, Pytorch 1.1.0

## How to Generate Results for Helmholtz Example

1. generating learning examples: genelearningExamples.py
2. prepare the training and testing data: /data_train_test/prepare_training_datasets.py
3. training VINet: train_inv_simu.py
4. model error plot: DrawModelError.py
5. estimated results: DrawInvertU.py

