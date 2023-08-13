from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from get_training_data import get_training_data
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

data_size = 100_000
hash_size = 100_000

print("Data size set to:", data_size)
print("Hash size set to:", hash_size)

print("Initializing...")
log_reg = LogisticRegression(solver='liblinear', max_iter=1000)

# Fetch data
print("Loading data...")
token_list = get_training_data("names_stemmed")[:data_size]
class_list = get_training_data("sni_full")[:data_size]

# Vectorize
print("Vectorizing...")
vectorizer = HashingVectorizer(n_features=hash_size, dtype=np.float64)
X_train = vectorizer.fit_transform([' '.join(tokens) for tokens in token_list])
del(token_list)
del(vectorizer)

# Scale the feature matrix
print("Scaling...")
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
del(X_train)
del(scaler)

# Train the classifier
print("Training...")
start_time = time.time()
log_reg.fit(X_train_scaled, class_list)
fit_time = (time.time() - start_time)

# Predict on test data
print("Predicting...")
accuracy = accuracy_score(class_list, log_reg.predict(X_train_scaled))
print("Accuracy for %s: %0.1f%% " % ("Logistic Regression", accuracy * 100))
print("Total time spend training: %.2fs" % fit_time)
