# Blood Cell Image Classification

**Project Overview**

This project focuses on classifying blood cell images into eight distinct classes using various machine learning models. With the rising importance of integrating artificial intelligence into the modern healthcare industry, this project aims to automate and enhance the process of identifying blood cell types, offering significant potential benefits to the medical community. The project involves image recognition, pattern analysis, and the acceleration of various medical diagnoses, ultimately improving the efficiency and accuracy of hematology disease detection. By harnessing the power of machine learning, the project not only strives for high accuracy but also considers additional factors such as runtime to ensure efficient decision-making.

**Technologies Used**

- **Programming Languages**: Python
- **Libraries**: Keras, scikit-learn, numpy, pandas, matplotlib
- **Tools**: Jupyter Notebook, Google Colaboratory

**Project Achievements**

- **Improved Classification Accuracy**:
    - Achieved an overall classification accuracy of 94% using the Convolutional Neural Network (CNN) model, demonstrating the effectiveness of deep learning techniques in image classification tasks.
- **Efficient Runtime Performance**:
    - Reduced training and inference times significantly through effective preprocessing techniques and optimized hyperparameter tuning, ensuring models are both accurate and efficient.
- **Comprehensive Model Comparison**:
    - Provided an in-depth comparison of four different machine learning models, highlighting the strengths and weaknesses of each approach in terms of accuracy, runtime, and practical applicability.

**Period**

- 2023.7 ~ 2023.11

**GitHub Repository**

- https://github.com/TommyYoungjunCho/Blood-Cell-Image-Classification

# Project Details

1. **Data Description**:
    - **Dataset**: BloodMNIST dataset
    - **Images**: 17,092 total images (13,673 training images and 3,419 test images)
    - **Classes**: 8 blood cell types
    - **Image Size**: 28x28 pixels with 3 color channels (RGB)
    - **Distribution**:
        - Basophils: 7.12%
        - Eosinophils: 18.24%
        - Erythroblasts: 9.08%
        - Immature Granulocytes: 16.94%
        - Lymphocytes: 7.10%
        - Monocytes: 8.31%
        - Neutrophils: 19.48%
        - Platelets: 13.74%
    
2. **Data Exploration and Preprocessing**:
    - **Normalization**: Adjusted and scaled data to a standard range between 0 and 1.
    - **Grayscale Conversion**: Transformed color images into grayscale to reduce complexity and enhance important features.
    - **Feature Extraction**:
        - Pixel Intensity: Analyzed the brightness levels in grayscale images.
        - GLCM (Gray-Level Co-occurrence Matrix): Extracted texture features such as correlation, homogeneity, contrast, energy, and dissimilarity.
    
3. **Machine Learning Models**:
    - **Model 1: Fully Connected Neural Network (FCNN)**:
        - **Architecture**:
            - Input Layer: 784 neurons (28x28 pixels)
            - Hidden Layers: 2 layers with 500 neurons each
            - Output Layer: 8 neurons (one for each class)
        - **Activation Functions**: ReLU for hidden layers, Softmax for output layer
        - **Optimization**: Adam optimizer
        - **Loss Function**: Sparse Categorical Crossentropy
    - **Model 2: Convolutional Neural Network (CNN)**:
        - **Architecture**:
            - Convolutional Layers: Multiple layers with filters for feature extraction
            - Pooling Layers: Max pooling for dimensionality reduction
            - Fully Connected Layers: Similar to FCNN structure in the later stages
        - **Activation Functions**: ReLU for convolutional and fully connected layers, Softmax for output layer
        - **Optimization**: Adam optimizer
        - **Loss Function**: Sparse Categorical Crossentropy
    - **Other Models**: Implemented and compared additional algorithms covered in the course, including at least one ensemble method.
    
4. **Hyperparameter Tuning**:
    - Conducted a hyperparameter search for each model to optimize performance.
    - Tuned parameters such as batch size, number of epochs, number of neurons in hidden layers, learning rate, and more.
    - Recorded results and runtimes for different hyperparameter combinations.

## Notion Portfolio Page
- [[Notion Portfolio Page Link](https://magic-taleggio-e52.notion.site/Portfolio-705d90d52e4e451488fb20e3d6653d3b)](#) 
