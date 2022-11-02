# Sentiment Analysis of the Indian Financial Market

Sentiment Analysis is a natural language processing (NLP) technique which is used to interpret the emotional tonality of a text. 

## Obejctive: The purpose of the work was to analyze how a negative or positive statement could influence the position of a company in the financial market. For the above reason, a dataset of around 25000 tweets was scrapped using the Tweepy library following by data cleaning using Regex. The data was then allotted a positive or negative sentiment using the flair package. Finally, using the tensor flow package, the data was tokenized and applied to the proposed GRU model. Different metrics such as accuracy, recall, precision, f1 have been used to determine the outcomes and results of the conducted research. 

In contribution, a unique twitter dataset was created that is based solely on Indian financial market and can be further used for financial research and analysis.

A sentiment analysis-based machine learning strategy for financial market prediction from Twitter releases was presented in this research. An advanced Sentiment Analysis model was developed to obtain predictions on whether a live tweet has a positive or negative sentiment attached to it. The sentiment was used to analyze the impact of the statement. A GRU model was implemented on a dataset that was created using Twitter API to obtain live tweets on the Indian financial market. The dataset was cleaned and sentiment analysis was performed before fitting our model onto the testing dataset. Different cases were tried by changing the hyperparameters of 
the model, optimum performance measures were obtained through this- an f1 score of 84.11%, recall of 84.91%, and precision of 83.30% and an accuracy score of 86.20% on the limited dataset. 
