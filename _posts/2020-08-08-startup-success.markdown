---
layout: post
title: Startup Success Prediction
date: 2020-08-08 13:32:20 +0300
description: Startup success prediction using classification.
img: startup-success/Startup-Cover.jpg # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [Classification, Startup]
---
## Overview
Can we use historical data to predict a startup's success or failure? I used the power of data, supervised learning, and classification algorithms to explore this million dollar question and better understand the key factors contributing to success of a company! 

Below is a summary of the data, methodology, results, and the app I created to predict the chance of success. See my [GitHub](https://github.com/maryam4420/Predicting-Startup-Success) for my code and more details.  

![Overview](../assets/img/startup-success/Startup-Overview.jpg){: .postImage}

# Data

I used a 2015 snapshot of CrunchBase data including historical data for more than 60,000 companies.
The majority of the companies were founded in the last 10 to 15 years prior to 2015 and the data included 
information such as:

**Status** (Operating, Acquisition, IPO, or Closed)

**Funding Information** 
- Amount (US$)
- Date of the first and last funding
- Number of rounds

**Location** 
- Country 
- State

**Industry**

# Project Design

To narrow down the questions, I approached this from a venture capital (“VC”) point of view and focused on companies that have received at least one round of funding. The idea is that the VCs can use this information to decide if they should invest in a company or not. The results however, should be coupled with intuition and industry knowledge as historical data is not always a good indication of what might happen in the future! 

I used the following criteria to determine success and failure and labeled the data accordingly:

**Success**

- Acquisition
- IPO

**Failure**
- Closed
- Operating for less than six years and no funding in the last three years.

It is hard to label the operating companies, however, the majority of the companies in my data were in operating status and in order to use a portion of those companies in my analysis, I tried to identify the companies that haven't been very successful to attract funding to label them as "failed".
 
 In general, the average time between funding rounds for most companies is 12 to 18 months. Therefore, I labeled the companies that have been operating only for five years, received funding once, but have not attracted funding in the last three years as failed. 

# Algorithms / Results

In total, I built, trained, evaluated, and tested eight algorithms. Logistic Regression performed the best which is great because it's more interpretable than other models. I further tuned and tested the Logistic Regression model using a Grid Search Cross-validation. The final F2 score is 0.8040, achieved by a probability threshold of 0.20.

I used F2 as my main metric to place extra emphasis on recall to catch any potential "unicorns" even at the expense of investing in a few "duds". This is of course subjective and a VC firm with lower risk tolerance may choose to place more emphasis on precision than recall. 

As a second metric, I also looked at the ROC AUC scores. Logistic Regression performed the best using this metric as well. See the below table and figure for performance of all the models tested:

![Model Comparsion](../assets/img/startup-success/Startup-Models.jpg){: .postImage}

![ROC Curves](../assets/img/startup-success/Startup-ROC%20Curves.jpg){: .postImage}

# Feature Importance

The figure below shows the coefficients and importance of each feature. The following can be inferred by comparing the coefficients:

- Funding amount and time to receive the first funding are crucial in a startup's success.
- Companies in the U.S. have a better chance to succeed than other countries
- Companies in California have a better chance to succeed than other U.S. states.

![Feature Importance](../assets/img/startup-success/Startup-Feature%20Importance.jpg){: .postImage}

# Flask App 

Finally, I created an app using Flask, to predict the probability of success. Here’s a video of how it works:

[![Demo Startup Success App](https://j.gifs.com/r84WNK.gif){: .postImage}](https://www.youtube.com/watch?v=OIZRC9J9Voc)