# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (Replace 'spam.csv' with your dataset path)
# Dataset should have two columns: 'label' (spam/ham) and 'message' (text)
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])

df = df.rename(columns={"v1": "label", "v2": "message"})
df = df[['label', 'message']]  # Keeping only the necessary columns

# Encode the labels: spam -> 1, ham -> 0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Splitting data into training and testing sets
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text vectorization using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Training the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Predicting on test data
y_pred = model.predict(X_test_vectorized)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Testing with new messages
new_messages = ["Congratulations! You've won a $1000 gift card.",
                "Hi, can we schedule a meeting tomorrow?"]
new_messages_vectorized = vectorizer.transform(new_messages)
predictions = model.predict(new_messages_vectorized)

for msg, pred in zip(new_messages, predictions):
    print(f"Message: {msg} -> {'Spam' if pred == 1 else 'Ham'}")
