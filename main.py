import string

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

## nltk.download('stopwords')

df = pd.read_csv('spam_ham_dataset.csv')
df['text'] = df["text"].apply(lambda x: x.replace('\r\n', ''))

stemmer = PorterStemmer()
stemmer.stem('sophistication')
corpus = []

stopwords_set = set(stopwords.words('english'))

for i in range(len(df)):
    text = df['text'].iloc[i].lower()
    text = text.translate(str.maketrans('','', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    corpus.append(text)


vectorizer = CountVectorizer()

X = vectorizer.fit_transform(corpus).toarray()
y = df.label_num

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


clf = RandomForestClassifier(n_jobs=-1)

clf.fit(X_train, y_train)
clf.score(X_test, y_test)

# Calculation of the score : Model Accuracy Score: 0.9681159420289855 (as of latest train/test)
score = clf.score(X_test, y_test)
print("Model Accuracy Score:", score)

email_to_clasify = df.text.values[21]

# Preprocess the email text
email_text = email_to_clasify.lower().translate(str.maketrans('','', string.punctuation)).split()
email_text = [stemmer.stem(word) for word in text if word not in stopwords_set]
email_text = ' '.join(email_text)

email_corpus = [email_text]
X_email = vectorizer.transform(email_corpus)

# Make the prediction
predicted_label = clf.predict(X_email)

# Get the actual label from the DataFrame
actual_label = df.label_num.iloc[21]
 
# Print predicted and actual labels
print("Predicted Label:", predicted_label[0])  # predicted_label is an array
print("Actual Label:", actual_label)