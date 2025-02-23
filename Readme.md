Potato Disease Detection Using CNN
## Overview
This project aims to detect potato diseases using a Convolutional Neural Network (CNN) model. The system classifies images of potato leaves into different categories based on their health status, helping farmers and agricultural experts identify and manage diseases effectively.

## Dataset
The dataset consists of images of potato leaves categorized into three classes:
Healthy
Early Blight
Late Blight

## Source
The dataset can be obtained from PlantVillage or other publicly available agricultural image datasets.
## Model Architecture
The CNN model consists of multiple layers:
Convolutional Layers for feature extraction
Pooling Layers for dimensionality reduction
Fully Connected Layers for classification
Softmax Activation for multi-class classification

## Requirements
Install the necessary dependencies before running the project:
pip install tensorflow keras numpy pandas matplotlib opencv-python

## Implementation Steps
1)Data Preprocessing
Image resizing and normalization
Splitting the dataset into training and validation sets
2)Model Training
Build the CNN architecture using TensorFlow/Keras
Train the model using the dataset
Monitor accuracy and loss metrics
3)Model Evaluation
Evaluate performance using test images
Calculate accuracy, precision, recall, and F1-score
4)Prediction & Deployment
Test the model on new images
Deploy using Flask or Streamlit for a user-friendly interface

## Usage
Run the training script:
python train.py
To test the model:
python predict.py --image test_image.jpg

## Results
The trained model achieves high accuracy in classifying potato diseases. The results can be visualized using Matplotlib for loss and accuracy trends.

## Future Improvements
Expand the dataset for better generalization
Implement transfer learning for improved accuracy
Deploy the model as a mobile application for easy field access

## Contributors
V Surendra
surendravendra87@gmail.com
