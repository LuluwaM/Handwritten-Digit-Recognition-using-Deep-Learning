# Handwritten-Digit-Recognition-using-Deep-Learning

## 📌 Project Idea
This project is part of the Machine Learning Fundamentals Nanodegree program from Udacity.

This project focuses on building a deep learning model to recognize handwritten digits using the MNIST dataset.  As a prototype for an OCR system, the goal is to preprocess the data, build and train a neural network using PyTorch, and achieve at least 90% accuracy on the test set.

## 🎯 Objectives
- Preprocess and prepare the MNIST dataset for training.  
- Build and train a neural network classifier with PyTorch.  
- Achieve at least 90% accuracy.
- Save the trained model.


## 📋 Project Steps
1. **Load and Preprocess Data**  
   - Load the MNIST dataset using `torchvision.datasets`.  
   - Use transforms to convert images to tensors, normalize, and flatten them.  
   - Create DataLoaders for training and testing sets.

2. **Visualize Dataset**  
   - Explore and visualize samples before and after preprocessing.  

3. **Build Neural Network**  
   - Design a neural network using PyTorch to classify handwritten digits.  

4. **Train and Tune Model**  
   - Train the model using the training data.  
   - Tune hyperparameters and network architecture to reach at least 90% test accuracy.

5. **Save Trained Model**  
   - Save the trained model using `torch.save`.
  
## 🛠️ Tools Used
- Python  
- PyTorch  
- Torchvision  
- Jupyter Notebook  
