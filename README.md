# CIFAR-10 Image Classification using CNN

This project demonstrates the implementation of a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Requirements](#requirements)
- [Contributing](#contributing)




## Introduction

This project uses a CNN model to classify images in the CIFAR-10 dataset. The model includes several convolutional layers followed by max pooling, dropout, and fully connected layers. The training process is carried out on a GPU for faster computation.

## Dataset

The CIFAR-10 dataset is a widely used benchmark dataset in machine learning and computer vision. It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Model Architecture

The CNN model is built using TensorFlow and Keras. The architecture includes:

- Convolutional layers with ReLU activation and L2 regularization
- Max pooling layers to reduce spatial dimensions
- Dropout layers to prevent overfitting
- Global average pooling instead of flattening
- Fully connected (dense) layers with ReLU activation
- Output layer with softmax activation for classification

## Training

The model is compiled with the Adam optimizer and categorical crossentropy loss function. The training is performed for 100 epochs with validation on the test set.

## Evaluation

The model's performance is evaluated using the test set, and the accuracy is reported. The training and validation accuracy are plotted to visualize the training process.

## Results

The results show the accuracy of the model on the test set. The training and validation accuracy plots provide insights into the model's learning process.

## Usage

To run this project, follow these steps:

1. Open the Google Colab notebook:
    [Open Colab Notebook](https://colab.research.google.com/drive/1HUJAVwk7BIULO0EJN3cOJKjQWvin9JxC#scrollTo=p2wvL1Ynanpb)

2. Ensure the runtime type is set to GPU:
    - Click on `Runtime` in the menu
    - Select `Change runtime type`
    - Choose `T4 GPU` from the `Hardware accelerator` dropdown

3. Run all cells in the notebook to execute the code.

## Requirements

- TensorFlow
- Keras
- Matplotlib
- NumPy

These packages are typically pre-installed in the Google Colab environment.
## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

