---
layout: post
title: Analyzing Amazon Reviews
date: 2020-08-24 18:32:20 +0300
description: Analyzing amazon reviews using NLP and recommendation system.
img: amazon_reviews/Amazon_Reviews_Cover.jpg # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [Natural Language Processing, NLP, Topic Modeling, Recommender System, Unsupervised Learning]
---
# Introduction
There is no doubt that the internet has revolutionized the way we live in so many ways. People like myself, are increasingly shopping online, writing reviews, and using other customers’ reviews to make better shopping decisions. This creates a great amount of valuable information available, and many companies use such information to improve their products, keep their customers happy, and increase sales. So it is important to be able to drive insights from the reviews to help with business decisions.

On the other hand, while shopping has become easier in so many ways, one problem that consumers are facing is that online stores have millions of products available and finding the right product can become overwhelmingly difficult due to the “information overload”. Therefore, recommending the right products to the customers is important and leads to both customer satisfaction and higher sales and profits.

For this project, I decided to use Natural Language Processing to analyze Amazon reviews for headphone products to understand what customers like or dislike about the products, and see what we can learn from them. Additionally, I created a content-based recommender system to make finding the right products easier. Below is a summary of the data, process, and results. You can see my code on my [GitHub](https://github.com/maryam4420/Predicting-Startup-Success) repo. 

# Data and Preprocessing

I downloaded the electronics metadata and electronics’ reviews from [here](http://jmcauley.ucsd.edu/data/amazon/). Then merged them together to get the products information and their reviews in one dataset. Then I extracted the headphone data from it. After removing missing values and duplicates, the final headphones dataset included  ~ 47,000 records, including ~ 1,200 unique products and ~ 23,000 unique users/reviewers.

To make the reviews ready for analysis and remove “noise”, I performed the following preprocessing steps using NLTK: 

- Converted and standardized accented characters into ASCII characters
- Removed special characters (e.g. special symbols and punctuation)
- Removed stopwords (e.g. ”a”, “and”, “but”, and “how” )
- Lemmatized the reviews (e.g. converted ran, run, and running to run)
- Normalized the reviews (e.g. converted to lowercase, removed apostrophes, etc.)

Here’s an example of a preprocessed review:

**Original review:** “The &#34;OK&#34; button broke after owning it less than a month.  Had to return to for a refund.  Buying a new one directly from Roku this time even though it costs a little extra.Broke in less than a month”

**Preprocessed review:** “ok button break own less month return refund buy new one directly roku time even though cost little extrabroke less month“

# Exploratory Data Analysis

Let’s first take a look at a few graphs to get a general understanding of the data. 

Below are the number of reviews per year and the number of unique products per year. The data includes headphone product/reviews from the year 2000 through mid 2014. As expected we, we can see that the number of reviews per year has increased significantly year over year. But also the number of unique headphones sold through Amazon has increased significantly in the recent years.
 
 
![Reviews Per Year](../assets/img/amazon_reviews/Reviews_per_year.jpg){: .postImage}
![Unique Product Per Year](../assets/img/amazon_reviews/Unique_product_per_year.jpg){: .postImage}

 Each product has a rating of one through five and as shown below, more than 50% of the products had a rating of five and only 6% had a rating of one! 
 
![Reviews Per Rating](../assets/img/amazon_reviews/Reviews_per_rating.jpg){: .postImage}
![Reviews Percentage](../assets/img/amazon_reviews/Reviews_pie_chart.jpg){: .postImage}
 
 I was also interested to see the length of the reviews. The graph below shows the review length for each rating after removing the outliers. The lengths are relatively consistent across the ratings. 

![Reviews length](../assets/img/amazon_reviews/Box_plot_reviews_length.jpg){: .postImage}

Finally, the graph below shows the distribution of the review lengths. The majority of people have written reviews with less than 1,000 characters, which is about ⅓ of a page. 

![Distribution of length](../assets/img/amazon_reviews/Distribution_of_review_length.jpg){: .postImage}
 
 After exploring the data, I assigned a rating class of one to reviews with ratings of four and five (positive reviews), and a rating class of zero to reviews with ratings of one and two (negative reviews). Additionally, for computational purposes, I only focused on the more recent reviews (after 2011). The final dataset included ~ 24,000 reviews which was later converted to ~ 1.2 M bigrams and trigrams!

# Topic Modeling (Unsupervised Learning)


I divided the topics into positive and negative to be able to look at them separately and compare them. I used both Countectorizer and TF-IDF for feature extraction and Non-negative Matrix Factorization (NMF) and Latent Semantic Analysis (LDA) to reduce dimensionality and divide the “documents” into the main topics. TF-IDF with NMF performed the best and below are the topics I found:

**Positive Topics:** 

![Positive Topics](../assets/img/amazon_reviews/Positive topics.jpg){: .postImage}

**Negative Topics:**

![Negative Topics](../assets/img/amazon_reviews/Negative topics.jpg){: .postImage}

To summarize, people have talked about 1) the sound and quality of the headphones, 2) comfort and fit, and 3) if they recommend the product. Interestingly enough, there was no mention of the price in the negative comments, which suggests these lower rating reviews were for cheaper products with lower quality. 

Additionally, I noted that there was no mention of noise cancelling in the positive topics, so it looks like the noise cancelling did not quite meet people’s expectations and resulted in negative reviews. Note that these reviews were written before 2015 and noise cancelling technology has most likely improved in the recent years.

# WordCloud (Supervised Learning)

The results of the unsupervised topic modeling was interesting and inline with my expectations. But I wanted to take another approach and zoom in a bit more! I decided to compare two brands and find out the most frequent things people have said about them. Below is a graph of the top 10 brands in the data. I decided to focus on Sony and Panasonic as they are rivals and headquartered in the same country. 

![Top 10 Brands](../assets/img/amazon_reviews/Top_10_brands.jpg){: .postImage}

I converted the reviews into words and bigrams using a Countvectorizer and assigned a frequency and average rating to them based on ratings. Then, I sorted them based on rating to get the reviews with the highest sentiment.  Below are the WordClouds for both Sony and Panasonic. 

**Sony:**

![Sony](../assets/img/amazon_reviews/Sony.jpg){: .postImage}

**Panasonic:**

![Panasonic](../assets/img/amazon_reviews/Panasonic.jpg){: .postImage}

It turns out, we can learn a lot by comparing the most frequent comments; 

- For both brands, most people have agreed with great price, quality, and fit. 
- For Sony, many people have talked about the overall sound quality, but for Panasonic, many people have specifically mentioned that they like the bass! 
- Many people mentioned that they have purchased a second pair of Sony headphones, but that was not the case for Panasonic!

# Recommender System 

This section will be coming soon!