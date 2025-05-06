# Happy/Sad Image Classification Using Deep Learning

This project demonstrates the use of deep learning techniques to classify images as happy or sad, leveraging a Convolutional Neural Network (CNN) implemented with the Keras library. The model achieves an impressive accuracy rate of nearly 100%, showcasing the effectiveness of modern machine learning approaches in analyzing and interpreting visual data.

## Overview

The core of this project involves training a CNN to differentiate between happy and sad expressions in images. The architecture includes three convolutional layers, each followed by a max-pooling layer, and concludes with dense layers for classification. This setup is optimized using the Adam optimizer and Binary Cross Entropy as the loss function, resulting in a highly efficient and accurate model.

## Architecture

- **Convolutional Layers**: Utilizes 3x3 kernels with a stride of 1 and ReLU activation functions to extract features from the input images.
- **Max Pooling Layers**: Applies 2x2 kernels with a stride of 2 to downsample the feature maps, reducing computational complexity.
- **Flatten Layer**: Converts the 3D feature maps into a 1D feature vector for processing by subsequent layers.
- **Dense Layers**: Consists of two layers with ReLU and Sigmoid activation functions, producing a binary classification output indicating whether the image depicts a happy or sad expression.

## Input Shape

The model expects input images to be resized to a shape compatible with its architecture. Preprocessing steps include resizing images to a standard size and normalizing pixel values.
