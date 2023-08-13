import time
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from get_training_data import get_training_data

data_size = 100_0
hash_size = 100_0

# Fetch data
print("Loading data...")
token_list = get_training_data("names_lemmatized")[:data_size]
class_list = get_training_data("sni_full")[:data_size]

# Vectorize
print("Vectorizing...")
vectorizer = TfidfVectorizer(max_features=hash_size)
X_train = vectorizer.fit_transform([' '.join(tokens) for tokens in token_list])
del(token_list)

# Scale
print("Scaling...")
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
del(X_train)

# Train
print("Training...")
clf = svm.SVC(kernel='rbf', C=10, gamma=0.1)
start_time = time.time()
clf.fit(X_train_scaled, class_list)
fit_time = (time.time() - start_time)
print("Total time spent training: %.2fs" % fit_time)

# Predict on training data
print("Predicting...")
start_time = time.time()
predictions = clf.predict(X_train_scaled)
predict_time = (time.time() - start_time)
accuracy = accuracy_score(class_list, predictions)
print("Accuracy for %s: %0.1f%%" % ("SVM", accuracy * 100))
print("Total time spent predicting: %.2fs" % predict_time)
