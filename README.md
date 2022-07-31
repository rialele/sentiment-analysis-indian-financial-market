# Sentiment Analysis of the Indian Financial Market

Sentiment Analysis is a natural language processing (NLP) technique which is used to interpret the emotional tonality of a text. 

The purpose of the work was to analyze how a negative or positive statement could influence the position of a company in the financial market. For the above reason, a dataset of around 25000 tweets was scrapped using the Tweepy library following by data cleaning using Regex. The data was then allotted a positive or negative sentiment using the flair package. Finally, using the tensor flow package, the data was tokenized and applied to the proposed GRU model. Different metrics such as accuracy, recall, precision, f1 have been used to determine the outcomes and results of the conducted research. 

In contribution, a unique twitter dataset was created that is based solely on Indian financial market and can be further used for financial research and analysis
