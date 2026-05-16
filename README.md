# Handwritten Digit Recognition using PyTorch

## Overview
This project implements a handwritten digit recognition model using the MNIST dataset and PyTorch.

The model is trained to classify handwritten digits (0–9) using a neural network.

## Features
- Data preprocessing using torchvision transforms
- MNIST dataset loading
- Neural network implementation
- Training and evaluation pipeline
- GPU support with CUDA
- Accuracy monitoring during training

## Technologies Used
- Python
- PyTorch
- Torchvision
- NumPy

## Model Architecture
Input: 784 features (28×28 images)

Layers:
- Linear(784 → 128)
- ReLU
- Linear(128 → 10)

Loss Function:
- CrossEntropyLoss

Optimizer:
- Adam

## Run

Install dependencies:

```bash
pip install -r requirements.txt
