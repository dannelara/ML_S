from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from get_training_data import get_training_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
import sys

data_size = 100_000
hash_size = 100_000

print("Data size set to:", data_size)
print("Hash size set to:", hash_size)

print("Initializing...")
nb = MultinomialNB()

# Fetch data
print("Loading data...")
token_list = get_training_data("names_lemmatized")[:data_size]
class_list = get_training_data("sni_full")[:data_size]

# Vectorize
print("Vectorizing...")
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform([' '.join(tokens) for tokens in token_list]).toarray()
del(token_list)
del(vectorizer)

# Scale the feature matrix
print("Scaling...")
scaler = MinMaxScaler(feature_range=(0, sys.maxsize))
X_train_scaled = scaler.fit_transform(X_train)
del(X_train)
del(scaler)

# Train the classifier
print("Training...")
start_time = time.time()
nb.fit(X_train_scaled, class_list)
fit_time = (time.time() - start_time)

# Predict on test data
print("Predicting...")
accuracy = accuracy_score(class_list, nb.predict(X_train_scaled))
print("Accuracy for %s: %0.1f%% " % ("MultinomialNB", accuracy * 100))
print("Total time spend training: %.2fs" % fit_time)
