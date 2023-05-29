# Food101 Image Classification with Various Models

This repository contains the implementations of several models to tackle the Food-101 classification task. I have utilized various deep learning architectures including Basic CNN, All Convolutional Net, Regularization, and Transfer Learning methods.

# Table of Contents
- [Food101 Image Classification with Various Models](#food101-image-classification-with-various-models)
- [Table of Contents](#table-of-contents)
- [Basic CNN](#basic-cnn)
  - [Results](#results)
- [All Convolutional Net](#all-convolutional-net)
  - [Results](#results-1)
- [Regularization](#regularization)
  - [Results](#results-2)
- [Transfer Learning](#transfer-learning)
  - [Results](#results-3)
- [Copyright](#copyright)
  
# Basic CNN
The Food101 model is a convolutional neural network (CNN) architecture designed to classify images of food into 101 different categories. The model uses a basic architecture with three convolutional layers and two fully connected layers. For more details, please refer to the [Basic CNN](basic_cnn) folder.

## Results
* **Validation loss:** 3.81039
* **Training loss:** 1.262546777
* **Test Accuracy:** 39.8380%

# All Convolutional Net
This model utilizes an All Convolutional Net architecture, using four 2D convolutional layers, each followed by a ReLU activation function. Following the convolutional layers, the output passes through an adaptive average pooling layer and a flattening layer. For further details, see the associated code files.

## Results
* **Validation loss:** 4.44300222
* **Training loss:** 4.41617
* **Test Accuracy:** 17.5942%

For more details, please refer to the [All Convolutional Net](all_convolutional) folder.

# Regularization
This model introduces regularization techniques, such as dropout and data augmentation, to enhance the performance of the CNN. For more details, please refer to the [Regularization](regularization) folder.

## Results
* **Validation loss:** 3.059
* **Training loss:** 3.150
* **Test Accuracy:** 24.546%

# Transfer Learning
This model utilizes transfer learning to improve the performance of the CNN. The model uses a pre-trained ResNet-50 model, with the final fully connected layer replaced with a new fully connected layer with 101 output units. For more details, please refer to the [Transfer Learning](transfer_learning) folder.

## Results
* **Final Model Accuracy:** 58.56%

# Copyright
This software is provided for educational purposes only. It is prohibited to use this code, for any college assignment or personal use. Unauthorized distribution, modification or commercial usage is strictly forbidden. Please respect the rights of the author.
