
Conversation opened. 1 unread message.

Skip to content
Using Gmail with screen readers
duk 
Enable desktop notifications for Gmail.
   OK  No thanks
1 of many
Fwd:
Inbox

SHAMIL AHAMMED E A . <shamil.ds22@duk.ac.in>
11:05 PM (11 minutes ago)
to me


---------- Forwarded message ---------
From: Nabeel P <nabeelp029@gmail.com>
Date: Tue, May 2, 2023, 9:12 PM
Subject: Fwd:
To: <shamil.ds22@duk.ac.in>




---------- Forwarded message ---------
From: Shahil T <shahilt412@gmail.com>
Date: Tue, May 2, 2023 at 7:41 PM
Subject: Fwd:
To: Nabeel P <nabeelp029@gmail.com>



---------- Forwarded message ---------
From: Roshan T . <roshan.ds22@duk.ac.in>
Date: Tue, 2 May 2023, 2:05 pm
Subject:
To: shahilt412@gmail.com <shahilt412@gmail.com>


import pandas as pd
import streamlit as st
import pickle
bbc_text = pd.read_csv("bbc-text.txt")
bbc_text = bbc_text.rename(columns = {"text":"News_Headline"}, inplace = False)
bbc_text.category = bbc_text.category.map({"tech":0,"business":1,"sport":2,"entertainment":3,"politics":4})
from sklearn.model_selection import train_test_split
X = bbc_text.News_Headline
y = bbc_text.category
#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.6, random_state = 1)
from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer(stop_words = "english", lowercase = False)
#fit the vectorizer on the training data
vector.fit(X_train)
X_transformed = vector.transform(X_train)
X_transformed.toarray()
# for text data
X_test_transformed = vector.transform(X_test)
from sklearn.naive_bayes import MultinomialNB
naivebayes = MultinomialNB()
naivebayes.fit(X_transformed, y_train)
from sklearn.metrics import classification_report
print(classification_report(naivebayes.predict(X_test_transformed), y_test))
import pickle
saved_model = pickle.dumps(naivebayes)
s = pickle.loads(saved_model)

st.header('Text Classifier')
input = st.text_area("Please enter the text", value="")
vec = vector.transform([input]).toarray()
if st.button("Predict"):
    
    st.write(str(list(naivebayes.predict(vec))[0]).replace('0', 'TECH').replace('1', 'BUSINESS').replace('2', 'SPORTS').replace('3','ENTERTAINMENT').replace('4','POLITICS'))

