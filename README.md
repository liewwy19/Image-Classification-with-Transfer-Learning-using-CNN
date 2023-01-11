
# Image Classification with Transfer Learning using CNN


![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)



## Summary
This project implement a Convolutional Neural Network (CNN) architecture model (i.e. VGG-16) for Image Classification via Transfer Learning using Tensorflow Keras library in Python.
## Abstract
There are several types of common concrete cracks namely hairline cracks which
usually develop in concrete foundation as the concrete cures, shrinkage cracks which
occur while the concrete is curing, settlement cracks which happen when part of
concrete sinks or when the ground underneath the slab isn't compacted properly as
well as structural cracks which form due to incorrect design.

Concrete cracks may endanger the safety and durability of a building if not being
identified quickly and left untreated. With the help of well establish Convolutional Neural Network (CNN) architecture, this model is tasked to
perform image classification to classify concretes with or without cracks. This model is impactful and may save thousands of lives.
## Data Set
Datasets used for developing this model can be obtain from [Concrete Crack Images for Classification](https://data.mendeley.com/datasets/5y9wdsg2zt/2) (Contributed by: Özgenel, Çağlar Fırat).

This dataset is divided into two as negative and positive crack images for image classification purpose. Each class has 20,000 images with a total of 40,000 images of 277x277 pixels in RGB channels.

## Run Locally

Clone the project

```bash
  git clone https://github.com/liewwy19/Image-Classification-with-Transfer-Learning-using-CNN.git
```

Download and extract the dataset into a sub-folder named "datasets" within the project's folder. Your project folder structure should look like the following:
```bash
Project Folder
  |
  |--- datasets
  |       |--- Negative      # sub-folder contains 20,000 images with no cracks   
  |       |--- Positive      # sub-folder contains 20,000 images with positive cracks    
  |
  |--- saved_models          # folder with all the related saved models
  
```


## The Methodology
+ Import packages and define contant variable
+ Data Loading
    + split and load the data into tensorflow datasets using keras.utils library
    + display some images from train dataset for quick visual (optional)
+ Data Preprocessing
    + further split the validation dataset into validation-test split
    + convert the BatchDataset into PrefetchDataset
    + create a small pipeline for data augmentation
    + test the data augmentation (optional)
+ Model Development
    + instantiate the object to prepare the input layer 
    + instantiate the base model (feature extractor)
    + disable the training for the feature extractor by freezing the layers
    + instantiate the classification layers
    + using functional API to link all of the modules together
+ Model Compilation
+ Model training
+ Apply the Fine-tuning Transfer Learning
    + freeze only the earlier layer within the base model
    + compile the updated model
    + continue the model training with new configuration
+ Model Evaluation
+ Model Deployment
    + Model Analysis with test dataset prediction
    + Model Saving

## The Model

### Base Model (CNN-VGG16) Summary
![](https://github.com/liewwy19/Image-Classification-with-Transfer-Learning-using-CNN/blob/main/assets/1_CrjJwSX9S7f759dK2EtGJQ.jpg?raw=True)

(Image source: https://neurohive.io/en/popular-networks/vgg16/)

Note: Since the dataset used in this project is in 227x227 pixels, I did not resize it to 224x224 like what it listed in the diagram of VGG-16 Architecture above.

![](https://github.com/liewwy19/Image-Classification-with-Transfer-Learning-using-CNN/blob/main/feature_extractor_model_summary.png?raw=True)


### Final Model Summary
![](https://github.com/liewwy19/Image-Classification-with-Transfer-Learning-using-CNN/blob/main/final_model_summary.png?raw=True)
## The Process (Transfer Learning)

#### Trainable params (Using Pre-trained model as Feature Extrators)
![](https://github.com/liewwy19/Image-Classification-with-Transfer-Learning-using-CNN/blob/main/assets/params.png?raw=True)

![](https://github.com/liewwy19/Image-Classification-with-Transfer-Learning-using-CNN/blob/main/assets/training.png?raw=True)

#### Trainable params (Fine Tuning Pre-trained Models for Domain Adaptation)
![](https://github.com/liewwy19/Image-Classification-with-Transfer-Learning-using-CNN/blob/main/assets/params_fine_tune.png?raw=True)

![](https://github.com/liewwy19/Image-Classification-with-Transfer-Learning-using-CNN/blob/main/assets/training_fine_tune.png?raw=True)
## The Analysis
The model able to achieve an excellent accuracy and f1-score for both the image classes. Using the award winning VGG-16 model architeture as base model definately part of the reason for this astoundingly results. 

This results also due to the adequate amount of training dataset for both image classes. Matter a fact, I do notice that the differences between the 2 classes being identified are too obvious hence the perfect accuracy. 

![](https://github.com/liewwy19/Image-Classification-with-Transfer-Learning-using-CNN/blob/main/confusion_matrix.png?raw=True)


Referring to the accuracy chart below, we can conclude that we have Higher Start, Higher Slope, and Higher asymptote, which are exactly the 3 benefits we get when applying transfer learning corectly and appropriately.


![](https://github.com/liewwy19/Image-Classification-with-Transfer-Learning-using-CNN/blob/main/chart_tensorboard_acc.png?raw=True)

![](https://github.com/liewwy19/Image-Classification-with-Transfer-Learning-using-CNN/blob/main/chart_tensorboard_loss.png?raw=True)
## The Results

#### Model evaluation using the base model directly without any training
![](https://github.com/liewwy19/Image-Classification-with-Transfer-Learning-using-CNN/blob/main/assets/evaluation_before.png?raw=True)

#### Model evaluation after training the base model and then fine-tuning it and further training
![](https://github.com/liewwy19/Image-Classification-with-Transfer-Learning-using-CNN/blob/main/assets/evaluation_after.png?raw=True)


## Contributing

This project welcomes contributions and suggestions. 

    1. Open issues to discuss proposed changes 
    2. Fork the repo and test local changes
    3. Create pull request against staging branch


## Acknowledgements
 - Özgenel, Çağlar Fırat (2019), “Concrete Crack Images for Classification”, Mendeley Data, V2, doi: 10.17632/5y9wdsg2zt.2
 - [VGG16 – Convolutional Network for Classification and Detection](https://neurohive.io/en/popular-networks/vgg16/)
 - [Transfer learning and fine-tuning - TensorFlow Tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning)
 - [Selangor Human Resource Development Centre (SHRDC)](https://www.shrdc.org.my/)

