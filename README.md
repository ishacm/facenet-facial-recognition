# Facial Recognition with Facenet

## Overview
This project develops a facial recognition system using TensorFlow & other supporting tools. The pre-processing of images is done using aalignment, generating facial embeddings & training SVM classifier.

## Table of Contents
- [Introduction to Facial Recognition](#introduction-to-facial-recognition)
- [Preprocessing Images using Facial Detection and Alignment](#preprocessing-images-using-facial-detection-and-alignment)
- [Generating Facial Embeddings in TensorFlow](#generating-facial-embeddings-in-tensorflow)
- [Training an SVM Classifier](#training-an-svm-classifier)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Preparing the Data](#preparing-the-data)
- [Preprocessing](#preprocessing)
- [Creating Embeddings in TensorFlow](#creating-embeddings-in-tensorflow)
- [Training a Classifier](#training-a-classifier)
- [Evaluating the Results](#evaluating-the-results)
- [Conclusion](#conclusion)

## Introduction to Facial Recognition

Facial recognition is a biometric solution that measures unique characteristics about one's face. Applications available today include flight check-in, tagging friends and family members in photos, and "tailored" advertising. To perform facial recognition, you need a way to uniquely represent a face.

### FaceNet

FaceNet uses a convolutional neural network architecture with a "triplet loss" function. This loss function minimizes the distance from positive examples while maximizing the distance from negative examples.

### Vector Embeddings

An embedding maps input features to vectors, and in a facial recognition system, these inputs are images containing a subject's face, mapped to a numerical vector representation.

Since these vector embeddings are in a shared vector space, vector distance can be used to calculate similarity between two faces. 

## Environment Setup

Use of Docker to install TensorFlow, OpenCV, and Dlib. Dlib provides a library for facial detection and alignment.
Create a requirements.txt for Python dependencies and a Dockerfile to create your Docker environment.
To build this image, run:
`docker build -t colemurray/medium-facenet-tutorial -f Dockerfile`

## Preparing the Data
The LFW (Labeled Faces in the Wild) dataset as training data from: http://vis-www.cs.umass.edu/lfw/lfw.tgz used

## Preprocessing
Preprocess the images to solve problems such as lighting differences, occlusion, alignment, and segmentation. You'll find the largest face in an image and align it based on the location of the eyes and bottom lip.
Download Dlib's face landmark predictor:
`curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2`
`bzip2 -d shape_predictor_68_face_landmarks.dat.bz2`
Run the preprocessing script in the Docker environment:

## Training a Classifier
Load the images from the queue you created, apply preprocessing, and feed them into the model to generate 128-dimensional embeddings. Use these embeddings as feature inputs into a scikit-learn SVM classifier.

## Evaluating the Results
Evaluate your classifier by feeding new images that it has not trained on

## License

Free to use, subject to License mentioned.


