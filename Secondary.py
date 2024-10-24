import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer  # Use TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# nltk.download('stopwords')

# Load dataset
df = pd.read_csv('spam_ham_dataset.csv')
df['text'] = df["text"].apply(lambda x: x.replace('\r\n', ''))

# Initialize the stemmer
stemmer = PorterStemmer()
corpus = []
stopwords_set = set(stopwords.words('english'))

# Preprocess the text
for i in range(len(df)):
    text = df['text'].iloc[i].lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    corpus.append(text)

# Vectorize the corpus with TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
y = df.label_num

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the Random Forest Classifier
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(X_train, y_train)

# Calculate and print the model score
score = clf.score(X_test, y_test)
print("Model Accuracy Score:", score)

# Email to classify
email_index = 21  # Change this index to classify a different email
email_to_classify = df.text.values[email_index]

# Preprocess the email text
email_text = email_to_classify.lower().translate(str.maketrans('', '', string.punctuation)).split()
email_text = [stemmer.stem(word) for word in email_text if word not in stopwords_set]  # Use email_text here
email_text = ' '.join(email_text)

# Transform the preprocessed email for prediction
email_corpus = [email_text]
X_email = vectorizer.transform(email_corpus)

# Make the prediction
predicted_label = clf.predict(X_email)

# Get the actual label from the DataFrame
actual_label = df.label_num.iloc[email_index]

# Print predicted and actual labels
print("Predicted Label:", predicted_label[0])  # predicted_label is an array
print("Actual Label:", actual_label)
