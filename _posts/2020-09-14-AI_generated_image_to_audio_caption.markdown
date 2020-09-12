---
layout: post
title: AI Generated Image to Audio Captions
date: 2020-09-11 18:32:20 +0300
description: Creating audio image captions using artificial intelligence.
img: image_captions/image_captions_cover_photo2.jpg # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [Convolutional Neural Network, Deep Learning, Natural Language Processing, CNN, RNN, LSTM, NLP]
---
# Introduction
deep learning is becoming a key instrument in artificial intelligence applications.

You can see my code on my [GitHub](https://github.com/maryam4420/Predicting-Startup-Success) repo. 

![Overview](../assets/img/image_captions/Itro_Pic.jpg){: .postImage}
# Data 

In order to generate meaningful captions, we need to train a model for both images and descriptions at the same time. For this project, I used the open source Flicker 8K from [Kaggle](https://www.kaggle.com/shadabhussain/flickr8k), which includes 8000 images and each image has five captions. I used 6000 images their descriptions for training and the rest for testing. Below are two sample image and their captions, used in the the training data. As you can see, the captions are describing the image slightly differently but are very similar. 

![Sample Image and Captions](../assets/img/image_captions/Sample_Img&Caption1.jpg){: .postImage}
**Captions:**
- A child in a pink dress is climbing up a set of stairs in an entry way.
- A girl going into a wooden building.
- A little girl climbing into a wooden playhouse.
- A little girl climbing the stairs to her playhouse.
- A little girl in a pink dress going into a wooden cabin.

![Sample Image and Captions](../assets/img/image_captions/Sample_Img&Caption2.jpg){: .postImage}
**Captions:**
- A black dog and a spotted dog are fighting. 
- A black dog and a tri-colored dog playing with each other on the road. 
- A black dog and a white dog with brown spots are staring at each other in the street. 
- Two dogs of different breeds looking at each other on the road. 
- Two dogs on pavement moving toward each other.

# Processing Image data
I used transfer learning to interpret the content of the images. Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task. It is popular approach in deep learning where pre-trained models are used as a starting point and in fact, so much of the progress in deep learning over the past few years is attributable to availability of such pre-trained models. There are many of Convolutional Neural Network (CNN) pre-trained models available to choose from such as VGG16, ResNet50, Xception. For tis project, I used the InceptionV3, which was created by Google Research and trained on ImageNet dataset (1.4M images) that includes 1000 different image classes.

I converted all the images to size 299x299 as required by InceptionV3 and passed them to the model to extract the 2048 lentgh feature vectors also known as the "bottleneck features". To do this I froze the Base layers of the model to .... and avoid relearning the features. The below image shows the model, inputs (images) and the vector. I then used this vector as an input to merge with the processed text data (descriptions), but more on this later!

![InceptionV3](../assets/img/image_captions/InceptionV3.jpg){: .postImage}

# Processing Text data (Captions)


# Final Neural Network Model

# Predicting the Captions

# Adding Speech

# Results

# Future Work




