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

Artificial intelligence has become a hot topic nowadays and it's being used by companies in all industries. There are many examples of great AI technologies that we use in our everyday lives, such as:

- Ride-sharing apps, like Uber and Lyft
- Food and groceries apps, like Grobhub and Instacart
- Product or music Recommender engines use by many companies, like Amazon, Spotify, and Youtube
- Maps and direction apps, like Google Maps
- Smart reply in Gmail 
- Chatbots, implemented in many websites
- Fraud detection applications, used in financial institutions 

Of course, it's not all unicorns and rainbows! Like any other technology, AI has it's downsides and can be misused, but let's focus on the positive here! On a personal level, AI has made our lives more efficient in many ways and will continue to do so. It has also significantly impacted and changed many industries while creating new opportunities, and will continue to do so!  

For this project I decided to use deep learning, which is a key instrument in AI applications, and generate captions for images using AI! Wait, can we really do that?? Yes! Doing something like this was inconceivable a few years ago, but the recent advancements in deep learning has made it possible. Generating captions or in other words, the ability to understand and describe an image, is definitely cool and exciting, but there are also many real world applications for it. For instance, for self-driving cars, where we need to understand what is on the road and around the car. Another example would be for CCTV cameras, where it would be great to understand what scenes are alarming to prevent crimes and protect people's lives. But my main motivation for picking up this project was to help the blinds and people with the visual impairments by creating a technology that converts images to captions, and then to speech. I will showcase that here, but to create a great tool, we will need to train a huge amount of data. 

This project is rather complex and has many steps which I will in this post. The blow figure shows the overview, just to give you the big picture:

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

- A person and their tan dog go for a ride in a rowboat.
- A woman and a dog rowing down a wide river.
- A woman wearing an orange jacket sitting in a canoe on a lake with a dog behind her.
- A woman with a dog canoe down a river.
- Woman and dog in rowboat on the water.

# Processing Image data
I used transferred learning to interpret the content of the images. Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task. It is popular approach in deep learning where pre-trained models are used as a starting point and in fact, so much of the progress in deep learning over the past few years is attributable to availability of such pre-trained models. There are many of Convolutional Neural Network (CNN) pre-trained models available to choose from such as VGG16, ResNet50, Xception. For tis project, I used the InceptionV3, which was created by Google Research and trained on ImageNet dataset (1.4M images) that includes 1000 different image classes.

I converted all the images to size 299x299 as required by InceptionV3 and passed them to the model to extract the 2048 lentgh feature vectors also known as the "bottleneck features". For a classification tax a Sofmax function would be applied after this steps perform the classification task To do this I froze the Base layers of the model to .... and avoid relearning the features. The below image illustrates the model, inputs (images) and the vector. I then used this vector as an input to merge with the processed text data (descriptions), but more on this later!

Add a quick description of how the model words, freezing base and extracting before the Softmax is applied: IncpetionV3 is a complex model, but in simple words ...

![InceptionV3](../assets/img/image_captions/InceptionV3.jpg){: .postImage}

# Processing Text data (Captions)
To be able to analyse the descriptions ...

## 1. Cleaning the Text
I first cleaned the descriptions to reduce the size of the vocabulary of words, by the following ways: This is important because with having fewer words, we can create a smaller model which can be trained faster.
- Converted all the words to lower case
- Removed punctuations (e.g. "!", "-")
- Removed all numbers 
- Removed all words with less than two characters (e.g. "a")  **?

## 2. Defining a fixed sequence length and starting/ending points
The input sequences for all neural networks should have the same length (because we need to pack them all in a single tensor). So for example, when working with reviews text, they are usually truncated to a certain size. For the case of the captions, since they are not too long, I looked at the maximum caption length in the data , which was 40, and used that as the fixed sequence length, which allows us to preserve all the information. In order to have the same length for all the captions, I padded the shorter ones with zeros. 

The way that the final model works is that it will generate a caption one word a time. So we need to define a starting point to kick off the generation process. We also need an ending point to signal the end of the caption. The model will stop the generating new words if it reaches this stopping word or the maximum length. I did this with adding "startseq" and "endseq" to all the captions.

To make this more concrete, below is an example for how the captions of the first image above were modified based on step 1 step 2:

- **startseq** a child in a pink dress is climbing up a set of stairs in an entry way **endseq** 
- **startseq** a girl going into a wooden building **endseq** 
- **startseq** a little girl climbing into a wooden playhouse **endseq** 
- **startseq** a little girl climbing the stairs to her playhouse **endseq** 
- **startseq** a little girl in a pink dress going into a wooden cabin **endseq** 

## 3. Removing the outliers

Next I removed the outlier words by removing the words that were repeated in the entire vocabulary of data less than 10 times. This is not a mandatory step, but removing the outliers, not only results in a better prediction, but it helps with saving the memory: The data has 6000 images, and each image has 5 captions, and if each caption has 10 word. and after adding embedding, this will make for 6,000*5*10*300=90,000,000. that is only the text data!

This step reduced the number of unique words in the vocabulary from xx to xx.

## 4. Tokenizing
Next we need to tokenize the words and convert them to integers before feeding them into the model.  I broke down the sentences to words. After data cleaning and removing the outliers there ever 1600 unique words in the dataset, then I assigend a tokenized the words. In other words I assigned an integer to each unique word.

## 5. Word Embeddings 

I used transferred learning again to do word embedding, while leveraging a model that was trained on a much larger text dataset than my data here, and extracted (semantically-meaningful) feature vectors from the captions. For this project I used Global Vectors for Word Representation (GloVe) with 200 word dimension. GloVe which is an unsupervised learning algorithm to obtain vector representation for words. In simple words, GloVe allows us to take a corpus of text and transform each word into a position in a high-dimensional space. In other words, using the precomputed word embeddings that is available in GloVe, I created an embedding matrix for all the 1600 unique words in the vocabulary. This embedding matrix will later be loaded into the final model before training. Note that if a word is in the our vocabulary but is not in GloVe, the values of the vectors for that word will be zeros.

To make this more clear, here's a toy example of how a sample captions will look like when being fed into the model:
![Text tensor input example](../assets/img/image_captions/text_tensor_example.jpg){: .postImage}

This was just an example and each caption will have a similar triangle Note that the numbers shown above are made up but they will numbers between 0 and 1 because they are probability values. Also, as discussed above, each captain will have a length of 40. 
 
That was a lot of steps for processing the text data! To summarize, I cleaned the text, defined a fixed length and defined a starting point and stopping point for the model, removed the outliers, tokenized the words, and did the word embedding using a pre-trained GLoVE mode. 

Now that we have processed the images and text data, we are ready to feed them into the final model.

# Final Neural Network Model Architecture 

##1. Setting UP The Model
Now that have processed both images and captions, we can feed them into the final neural network model. Since we have two inputs we have to use the Function API model (instead of the sequential model) which can take more than one input. The below figure illustrates the model and it's inout and output. 

![Final Model](../assets/img/image_captions/Final_model1.jpg){: .postImage}

But let me explain the steps in a bit more detail:

**Input 1:** The first input of the model will be images that were converted into features vectors of 2048 length

**Input 2:** The second input of the model will be the text sequences which have a length of 40 and embedding dimension of 200. But we first need to feed this into an LSTM layer. LSTM (Long Short-Term Memory) is just a special recurrent network layer that can process sequences and understand the order of the words.
**Merging the inputs:** Next we need to merge both data inputs and convert them to a single tensor and now we can feed this tensor to the functional API mode. Note that, to do the merge, we need to do make the output of the previous steps the same length. In this case, I converted those outputs 256-length.
**Model:** So the model takes the tensor input of image and text data, and then I built two dense layers and a Softmax on top of the final layer (to convert the data in the previous layers into probabilities). I then fitted the model using an "adam" optimizer and used "categorical_crossentropy" to measure the loss of the model. 
**Output** The output of the model is a single vector, including probabilities for each of the 1600 unique words (which are conditioned on images). Then we can use this output to predict for the captions for images in the test data. I will explain how to that further down.

The below image was outputted by the model, it shows the input and output of each layer and the random drop outs that I used to avoid outfitting. 

![Final Model](../assets/img/image_captions/Final_model2.jpg){: .postImage}

##2. Generate Captions

Now we can generate the captions using a for loop and let me explain this via an example which is illustrated in the figure below. The first step is starting caption (a string) including only "startseq" as it's first word. Then we can predict the next words of the caption using a for loop as follows:
**Iteration 1:** Then we the model receives the image + "startseq" as input and makes a prediction for the next word, "little", using the final image and text vector discussed above.
**Iteration 2:** Then we the model receives the image + "startseq little" as input and makes a prediction for the next word, "girl".
    .
    .
**Iteration 7:** Then we the model receives the image + "startseq little girl climbing into wooden playhouse" as input and makes a prediction for the next word, which is "endseq" for this example and this iteration. This gives the model the signal to stop predicting. As mentioned before, if the model would reach the maximum length of 40 before before reaching the word, "endseq", it would stop predicting at that point.

![Caption Prediction Example](../assets/img/image_captions/Caption_pred_example.jpg){: .postImage}
 
 Note that this example shows words under "Partial Captions" but as discussed before, they are converted to numbers since that's what the model understands. 

# Adding Speech

To be updated soon!

# Results

Blow are a few examples of the captions generated by the model

![Little_boy_in_red](../assets/img/image_captions/Little_boy_in_red.jpg){: .postImage}

**Generated caption: ** xxxx

![People_image](../assets/img/image_captions/People_image.jpg){: .postImage}

**Generated caption:** A group of people are gathered around a building

![Boy_on_swing](../assets/img/image_captions/Boy_on_swing.jpg){: .postImage}

**Generated caption:** A boy is swinging on a swing set

![Dog](../assets/img/image_captions/Dog.jpg){: .postImage}

**Generated caption:** A dog is chasing a ball in the grass



# Conclusions 
...
We were able to build a decent model to generate captions with training a neural network model on only 6000 images and captions. The model was strengthened by the power of the transferred learning (the InceptionV3 model for CNN and the GLoVE model for word-embedding), where the models were previously trained on very large image and text datasets. It should be noted that the testing images should be semantically related to the training images. For example, if we only train the model on cats and dogs, the model can only predict cats or dogs. Regardless of the model that we use, we cannot expect it to recognise fruits or flowers! On that note, for future work I would like to work with a larger dataset containing more types of images so that the model can make predictions for a wider type of images. I would also like to use a gp3 model instead of an LSTM model. However, for captions, which mostly contain factual and straight forward text as opposed to poetry or legal language, I don't expect a huge improvement. 

If we are able to train a huge number of images and captions, and turn videos or high frequency images to speech. Using this, I hope that we can create an accessible technology that can easily translate images / videos to audio. Imagine you can't see! Now imagine your surrounding can be described to you!

You can see my code on my [GitHub](https://github.com/maryam4420/Predicting-Startup-Success) repo. Note that, if you use my code or if I run train the model again, the resulted captions will be slightly different, due to the stochastic nature of the model.




