---
layout: post
title: Startup Success Prediction
date: 2020-08-08 13:32:20 +0300
description: Startup success prediction using classification.
img: Startup-Cover.jpg # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [Classification, Startup]
---
## Overview
Can we use historical data to predict a startup success or failure? I tried to use the power of data, supervised learning, and classification algorithms to answer this million dollar question and understand the key factors contributing to success of a company! 

Below is a summary of the steps performed, results, and the app I created to predict the chance of success. See my github for my code as well as more details.

![Overview](../assets/img/startup-success/Startup-Overview.jpg)

# Data

I used a 2015 snapshot of CrunchBase data including historical data for more than 60,000 companies.
The majority of the companies were founded in the last 10 to 15 years prior to 2015 and the data included 
information such as:

**Status** (Operating, Acquisition, IPO, or Closed)

**Funding Info** 
- Amount
- Date of the first and last funding
- Number of rounds

**Location** (country and state)

**Industry**

# Project Design

To narrow the questions, I approached this from a venture capital (“VC”) point of view and focused on companies that have received at least one round of funding. The idea is that the VCs can use this information to decide if they should invest in a company. I used the following criterias for success and failure:

**Success**

- Acquisition
- IPO

**Failure**
- Closed
- Operating for less than six years and no funding in the last three years.

It is hard to label the operating companies, however, the majority of the companies in my data were in operating status and in order to use a portion of those companies in my analysis, I tried to identify the companies that haven't been very successful to attract funding. 

The average time between fundings for most companies is 12 to 18 months, so I labeled the companies that have been operating only for five years, received funding once, but have not attracted funding in the last three years as failed. 

# Algorithms / Results

In total, I built, trained, evaluated, and tested eight algorithms. Logistic Regression performed the best and is the best model in terms of interpretability. So, I further tuned and tested the Logistic Regression model using a Grid Search Cross-validation. The final F2 score is 0.8040 achieved by a probability threshold of 0.20. 

I used F2 as my main metric to place extra emphasis on recall to catch any potential "unicorns" even at the expense of investing in a few "duds". This is of course subjective and a VC firm with lower risk tolerance may choose to place more emphasis on precision than recall. 

As a second metric, I also looked at the ROC AUC scores, Logistic Regression also performed the best using that metric. Below are a table and figure detailing the performance of the models tested:

![Model Comparsion](../assets/img/startup-success/Startup-Models.jpg)

![ROC Curves](../assets/img/startup-success/Startup-ROC%20Curves.png)

## Feature Importance

The figure below shows how important the features are. The following can be inferred by comparing the coefficients:

Funding amount and time to receive the first funding are crucial for success.
Companies in the USA have a better chance in success than other countries
Companies in California have a better chance in success than other U.S. states.

![Feature Importance](../assets/img/startup-success/Startup-Feature%20Importance.png) <!-- .element width="100%" -->

# Flask App 

Finally, I created an app using Flask, to predict the probability of success. Here’s a video of how it works:

[![Demo Startup Success App](https://j.gifs.com/1WAkEm.gif)](https://www.youtube.com/watch?v=OIZRC9J9Voc)