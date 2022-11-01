#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis of Financial Markets

# #### importing necessary packages

# In[5]:


import os
import tweepy as tw
import pandas as pd
import re


# #### accessing twitter api

# In[7]:


auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)


# #### scraping tweets from twitter into tweet_head

# In[ ]:


search_words = "stock markets"
new_search = search_words + "-filter:retweets"
new_search


tweets = tw.Cursor(api.search_tweets,q=search_words,lang="en").items(5000)

tweet_text= [[tweet.text] for tweet in tweets]

tweet_data = pd.DataFrame(data=tweet_text, columns=["Text"])


# In[ ]:


tweet_data.head()


# #### importing the flair package

# In[8]:


import flair
sentiment_model = flair.models.TextClassifier.load('en-sentiment')


# #### testing flair on demo sentences

# In[9]:


sentence = flair.data.Sentence("Hello sucker")
sentiment_model.predict(sentence)


# In[10]:


sentence


# In[11]:


sentence1 = flair.data.Sentence("you look happy")
sentiment_model.predict(sentence1)


# In[12]:


sentence1


# #### code for assigning '1' to sentences with a positive sentiment, and '0' to negative

# In[ ]:


sentiments = []
for tweet in tweet_data['Text']: 
    sentence = flair.data.Sentence(tweet)
    sentiment_model.predict(sentence)  
    sentiments.append(sentence.labels[0].value)
    
tweet_data['Sentiment'] = sentiments


# In[ ]:


sentiment_dict={"POSITIVE": 1, "NEGATIVE": -1 }


# In[ ]:


import numpy as np
tweet_data["Sentiment"]=tweet_data["Sentiment"].map(sentiment_dict)


# In[ ]:


tweet_data


# In[ ]:


tweet_data.to_csv(r'C:\Users\Diyora\Python\Mini Project\Our_tweets.csv')


# #### additional imports

# In[14]:


# Data imports
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ML imports
from sklearn.model_selection import train_test_split,GridSearchCV
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score,recall_score


# #### data cleaning on primary data

# In[15]:


tweet_data=pd.read_csv("5k_tweets.csv")
tweet_data['Text'] = tweet_data['Text'].apply(lambda x: x.lower())  # transform text to lowercase
tweet_data['Text'] = tweet_data['Text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x)) #accepting everything that is a letter or digit
print(tweet_data.shape)
tweet_data=tweet_data.drop("Unnamed: 0", axis= 1)
tweet_data.head(5)


# #### appending secondary data

# In[16]:


df = pd.read_csv("stock_data.csv")
df = df.sample(frac=1).reset_index(drop=True)
df.head()
# clean tweet text
df['Text'] = df['Text'].apply(lambda x: x.lower())  # transform text to lowercase
df['Text'] = df['Text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x)) #accepting everything that is a letter or digit
print(df.shape)
df.head(20)


# In[17]:


a=tweet_data.append(df)


# In[18]:


a.shape


# #### visualizing the data

# In[19]:


a['Sentiment'].value_counts().sort_index().plot.bar()


# #### tokenization

# In[21]:


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet_data['Text'].values)
X = tokenizer.texts_to_sequences(tweet_data['Text'].values)
X = pad_sequences(X)
print("X tokenized data = ", X[:5])


# In[22]:


# Y as buckets of Sentiment column
y = pd.get_dummies(tweet_data['Sentiment']).values
y


# #### GRU Model

# In[23]:


model = Sequential()
model.add(Embedding(10000, 512, input_length=X.shape[1]))
model.add(Dropout(0.2))
model.add(GRU(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.15))
model.add(GRU(512, dropout=0.2, recurrent_dropout=0.15))
model.add(Dense(2, activation='softmax'))


# In[24]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# #### train-test split

# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# #### fitting the model, view training accuracy (changed everytime we ran the model)

# In[26]:


batch_size = 32
epochs = 8

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)


# #### model predications

# In[27]:


Predictions = model.predict(X_test)


# In[28]:


pos_count, neg_count = 0, 0
real_pos, real_neg = 0, 0
for i, prediction in enumerate(Predictions):
    if np.argmax(prediction)==1:
        pos_count += 1
    else:
        neg_count += 1
    if np.argmax(y_test[i])==1:    
        real_pos += 1
    else:
        real_neg +=1

print('Positive predictions:', pos_count)
print('Negative predictions:', neg_count)

print('Real neutral:', real_pos)
print('Real negative:', real_neg)


# In[29]:


import matplotlib.pyplot as plt


# In[30]:


print(history.history['loss'], )
predictions = [pos_count, neg_count]
real = [real_pos, real_neg]
labels = ['Positive', 'Negative']


# #### displaying model outcomes

# In[31]:


x = np.arange(len(labels))
width = 0.35 

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, real, width, label='Real')
rects2 = ax.bar(x + width/2, predictions, width, label='Predictions')

ax.set_ylabel('Scores')
ax.set_title('Count of Classifications')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


fig.tight_layout()

plt.show()


# In[32]:


fig, ax = plt.subplots()
loss = history.history['loss']
epoch = [item for item in range(1,9)]
accuracy = history.history['accuracy']

ax.plot(epoch, loss, label = "Loss")
ax.plot(epoch, accuracy, label = "Accuracy")

ax.set_xlabel('Epoch')
ax.set_title('Accuracy and Loss per epoch')
plt.legend()
plt.show()


# #### final results

# In[33]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
y_test_arg=np.argmax(y_test,axis=1)
Y_pred = np.argmax(model.predict(X_test),axis=1)

print("The precision score is",precision_score(y_test_arg, Y_pred))
print("The recall score is", recall_score(y_test_arg, Y_pred))
print("The F1-score is", f1_score(y_test_arg, Y_pred))
print("The acc-score is", accuracy_score(y_test_arg, Y_pred))

