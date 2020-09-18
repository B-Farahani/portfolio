---
layout: post
title: Analyzing Amazon Reviews
date: 2020-08-24 18:32:20 +0300
description: Analyzed Amazon reviews using NLP and built a recommendation system.
img: amazon_reviews/Amazon_Reviews_Cover.jpg # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [Natural Language Processing, NLP, Topic Modeling, Recommender System, Unsupervised Learning]
---
# Introduction
There is no doubt that the internet has revolutionized the way we live today and how businesses operate. People, including myself, are increasingly shopping online, writing reviews, and using other customers’ reviews to make better shopping decisions. These reviews create a significant amount of valuable information, and many companies analyze such information to improve their products, keep their customers happy, and increase sales. So in today's competitive business environment, it is crucial to be able to drive insights from the reviews to make smarter business decisions.

On the other hand, while shopping has become easier in so many ways, one problem that consumers are increasingly facing is that online stores have millions of products available, and finding the desired product can become overwhelmingly difficult due to the “information overload”. Therefore, recommending the right products to the customers is important and leads to both customer satisfaction and higher sales and profits.

For this project, I decided to use Natural Language Processing to analyze Amazon reviews for headphone products to understand what customers like or dislike about the products and see what we can learn from them. Additionally, I created a content-based recommender system to make finding the right products easier for the customers. Below is a summary of the data, process, and results. You can see my code on my github [rep](https://maryam4420.github.io/portfolio/amazon_reviews/).

![Overview](../assets/img/amazon_reviews/Itro_Pic.jpg){: .postImage}
# Data and Preprocessing

I downloaded the electronics metadata and electronics reviews from [here](http://jmcauley.ucsd.edu/data/amazon/). Then I merged them together to get the products' information and their reviews in one dataset. Then I extracted the headphone data from it. After removing missing values and duplicates, the final headphones dataset included  ~ 47,000 records, including ~ 1,200 unique products and ~ 23,000 unique users/reviewers.

To make the reviews ready for analysis and remove “noise”, I performed the following preprocessing steps using NLTK: 

- Converted and standardized accented characters into ASCII characters
- Removed special characters (e.g. special symbols and punctuation)
- Removed stopwords (e.g. ”a”, “and”, “but”, and “how” )
- Lemmatized the reviews (e.g. converted ran, run, and running to run)
- Normalized the reviews (e.g. converted to lowercase, removed apostrophes, etc.)

Here’s an example of a preprocessed review:

**Original review:** “Bought these to replace some bluetooth headphones that finally gave out.  I've paid a lot more for headphones that didn't sound as good as these do.  At $12 I doubt you'll find a better deal.Great sound, great price”

**Preprocessed review:** “buy replace bluetooth headphones finally give ive pay lot headphones didnt sound good doubt youll find better dealgreat sound great price“

# Exploratory Data Analysis

Let’s first take a look at a few graphs to get a general understanding of the data. 

Below are the number of reviews per year and the number of unique products per year. The data includes headphone products/reviews from the year 2000 through mid-2014. As expected, we can see that the number of reviews per year has increased significantly year over year. But at the same time, the number of unique headphones sold through Amazon has risen remarkably in recent years.
 
 
![Reviews Per Year](../assets/img/amazon_reviews/Reviews_per_year.jpg){: .postImage}
![Unique Product Per Year](../assets/img/amazon_reviews/Unique_product_per_year.jpg){: .postImage}

Each product has a rating of one through five, and as shown below, more than 50% of the products had a rating of five, while only 6% had a rating of one!  
 
![Reviews Per Rating](../assets/img/amazon_reviews/Reviews_per_rating.jpg){: .postImage}
![Reviews Percentage](../assets/img/amazon_reviews/Reviews_pie_chart.jpg){: .postImage}
 
I was also interested to see the length of the reviews. The graph below shows the review length for each rating after removing the outliers. The lengths are relatively consistent across the ratings. 

![Reviews length](../assets/img/amazon_reviews/Box_plot_reviews_length.jpg){: .postImage}

Finally, the graph below shows the distribution of the length of the reviews. The majority of people have written reviews with less than 1,000 characters, which is about ⅓ of a page. 

![Distribution of length](../assets/img/amazon_reviews/Distribution_of_review_length.jpg){: .postImage}
 
 After exploring the data, I assigned a rating class of one to reviews with ratings of four and five (positive reviews), and a rating class of zero to reviews with ratings of one and two (negative reviews). Additionally, for computational purposes, I only focused on the more recent reviews (after 2011). The final dataset included ~ 24,000 reviews which was later converted to ~ 1.2 M bigrams and trigrams!

# Sentiment Analysis
Sentiment analysis is a text analysis method that detects polarity (e.g. positive or negative opinion), and it aims to measure the attitude, sentiment, intensity, and emotions of the speaker. For example, words like 'love', 'enjoy', and 'amazing' convey a positive sentiment, while words like 'hate', 'dislike', and 'bad' convey a negative sentiment. To better understand how happy or upset the customers are, I analyzed the reviews, separately for each rating, using VADER (Valence Aware Dictionary for Sentiment Reasoning). The figures below show the results, and we can see that the customers' tone is more positive for higher ratings. In other words, by looking at the reviews, we can confirm that the customers are more satisfied with the products that have a higher rating.

![Average Sentiment Per Rating](../assets/img/amazon_reviews/Average_Sentiment_Per_Rating.jpg){: .postImage}

![Ratings' Average Sentiment Per Year](../assets/img/amazon_reviews/Ratings'_Average_Sentiment_Per_Year.jpg){: .postImage} 

I also looked at the average sentiment for all the reviews, combined, for the last decade. As shown below, overall, sentiment has improved through the years, which makes sense, given Amazon's success in recent years.

![Average Sentiment Per Year](../assets/img/amazon_reviews/Average_Sentiment_Per_Year.jpg){: .postImage}

# Topic Modeling (Unsupervised Learning)


I divided the topics into positive and negative to be able to look at them separately and compare them. I used both Countvectorizer and TF-IDF for feature extraction and used Non-negative Matrix Factorization (NMF) and Latent Semantic Analysis (LDA) to reduce dimensionality and find the main topics in the corpus (separately for the positive reviews and for the negative reviews). TF-IDF with NMF performed the best and below are the topics I found:

**Positive Topics:** 

![Positive Topics](../assets/img/amazon_reviews/Positive topics.jpg){: .postImage}

**Negative Topics:**

![Negative Topics](../assets/img/amazon_reviews/Negative topics.jpg){: .postImage}

To summarize, people have talked about 1) the sound and quality of the headphones, 2) comfort and fit, and 3) if they recommend the product. Interestingly enough, there was no mention of the price in the negative comments, which suggests these lower rating reviews were for cheaper products with lower quality. 

Additionally, I noted that there was no mention of noise-cancelling in the positive topics, so it looks like the noise-cancelling did not quite meet people’s expectations and resulted in negative reviews. Note that these reviews were all written before 2015, and noise-cancelling technology has most likely improved in recent years.

# WordCloud (Supervised Learning)

The results of the unsupervised topic modeling were interesting and in line with my expectations. But I wanted to take another approach and zoom in a bit more! I decided to compare two brands and find out the most frequent things people have said about them. I looked at the top 10 brands in the data, that are shown below, and decided to focus on Sony and Panasonic as they are rivals and headquartered in the same country. 

![Top 10 Brands](../assets/img/amazon_reviews/Top_10_brands.jpg){: .postImage}

I converted the reviews into words and bigrams using a Countvectorizer and assigned a frequency and average rating to them based on ratings. Then, I sorted them based on rating to get the reviews with the highest sentiment.  Below are the WordClouds for both Sony and Panasonic. 

## Sony:

![Sony](../assets/img/amazon_reviews/Sony.jpg){: .postImage}

## Panasonic:

![Panasonic](../assets/img/amazon_reviews/Panasonic.jpg){: .postImage}

It turns out that we can learn a lot by comparing the most frequent comments; 

- For both brands, most people have agreed with the great price, quality, and fit. 
- For Sony, many people have talked about the overall sound quality, but for Panasonic, many people have specifically mentioned that they like the bass! 
- Many people specified that they purchased a second pair of Sony headphones, but that was not the case for Panasonic!

# Recommender System 
There are a few kinds of recommender systems that are commonly used by companies: content-based filtering and collaborative filtering (item-based and user-based). Companies use one or a variety of these methods to recommend the best products to the customers and consequently increase sales. The appropriate recommender engine and the recommended products are chosen based on a variety of factors such as customers' demographics and the nature of the product. For example, movies, electronic products, and news articles could require their unique and different recommendation strategies since they are very different in nature. The best method and recommendations also significantly depend on business needs and direction. For example, other than what the customer would find desirable, a company may want to promote certain products more than other products or perhaps recommend products that are on sale.

For this project, I decided to create a content-based filtering using reviews to help customers find the desired product based on other customers' reviews. Generally, in content-based filtering, the similarity between products is calculated based on the attributes of the products (e.g. text/description, brand, price). This method is often used to solve the cold start problem, where user information such as ratings or users' purchase history is not available. 

In the case of Amazon's headphones, and this dataset, there are many reviews available, so I created a content-based filtering recommender system using reviews to help the customer get a recommendation, based on the reviews that have the closest distance (cosine similarity) with what the user is looking for. To showcase this, I have included below an example of a user's search/text input and the top three recommended products: 

**Customers' text:** "Comfortable headphone, with great sound quality and good looking design!"

![Recommendation](../assets/img/amazon_reviews/Recommendation.jpg){: .postImage}

This recommendation system can, of course, be further tuned to allow the customers to filter for features such as price or brand. Also, as mentioned above, the products' descriptions can easily be used instead of the reviews as an alternative strategy and to also capture the new products that do not have reviews yet.

# Conclusions

It's crucial for all businesses to understand what the customers are thinking about the company and their products, and how satisfied they are. Natural language processing allows us to achieve this without a human reading all the comments one by one. In this project, I used NLP to find the main topics discussed by customers for headphones and performed sentiment analysis, using unsupervised learning techniques. I compared the most frequent things customers have said about the products of two rival brands (Sony and Panasonic) and found some interesting insights. And finally, I built a recommender engine using content-based filtering to help customers find their desired products based on what other customers have said about the products. 
