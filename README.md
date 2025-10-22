# Traffic Sign Classification Using Deep Learning
University project Sorbonne Data Analytics

This project focuses on the recognition and classification of traffic signs using Convolutional Neural Networks (CNNs). Accurate traffic sign recognition is a key component of autonomous driving systems, enabling vehicles to perceive their environment and make informed decisions.  
 
Project Overview  
The goal of this project is to develop a robust and accurate model capable of recognizing and classifying traffic signs from images.
We used the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains over 50,000 labeled images across 43 classes.

Dataset  
Source: GTSRB Dataset on Kaggle
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign 
Classes: 43 types of traffic signs
Images: ≈ 50,000
Preprocessing:
Resized to 32×32
Normalized pixel values
Data augmentation (rotation, shift, shear, zoom, and horizontal flip)

Model Architecture
We implemented and optimized several CNN architectures using TensorFlow and Keras, including:
- A baseline CNN with ReLU activation
- An optimized CNN with batch normalization, dropout, and global average pooling
- A Swish-activated CNN achieving superior performance
We also extended the project with a binary classification model to first detect whether an image contains a traffic sign or not, before passing it to the multi-class classifier.

Training & Optimization
- Optimizer: Adam (learning rate = 0.001)
- Batch size: 256
- Epochs: 20–40
- Early stopping: Enabled to prevent overfitting
- Validation strategy: Stratified K-Fold Cross Validation (3 folds)

Model performances:
- Baseline CNN (ReLU): 96.4% accuracy
- Optimized CNN (ReLU): 99.8% accuracy
- Swish CNN: 99.87% accuracy, loss = 0.0057
- Binary classifier (Sigmoid): 99.75% validation accuracy

Key Insights
- Swish activation improved gradient flow and generalization.
- Data augmentation significantly reduced overfitting.
- The binary+multiclass hybrid model enhanced classification reliability.

Future Work
- Enhance robustness under adverse weather conditions (rain, fog, snow).
- Test model transferability to other countries’ traffic signs.
- Combine vision-based detection with LIDAR and radar data for real-world safety.

Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy / Pandas
- Matplotlib / Seaborn
- Scikit-learn

Authors
Nadia Medjdoub
Karim Ameur
