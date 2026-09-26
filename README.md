SMS Spam Detection

A machine learning–based SMS Spam Detection application that classifies text messages as Spam or Not Spam (Ham) using Natural Language Processing (NLP) and a Multinomial Naive Bayes classifier.

The project includes a Streamlit web application where users can enter an SMS message and receive an instant prediction along with the model's confidence and probability breakdown.

Project Overview

Spam messages are unwanted messages that may contain misleading offers, suspicious links, fraudulent requests, or attempts to obtain personal information.

This project demonstrates how machine learning can be used to automatically identify spam messages by learning patterns from labeled SMS data.

The workflow includes:

SMS Dataset → Text Preprocessing → Feature Extraction → Model Training → Prediction → Spam/Ham Classification

Features
Classifies SMS messages as SPAM or NOT SPAM
Uses Multinomial Naive Bayes
Converts text into numerical features using CountVectorizer
Uses English stop-word removal during feature extraction
Calculates model accuracy on a test dataset
Displays prediction confidence
Shows spam and not-spam probability
Interactive Streamlit web interface
Fast real-time predictions
Includes model evaluation using a classification report
Machine Learning Approach
1. Dataset

The project uses the SMS Spam Collection dataset.

Each message is labeled as:

Label	Meaning
spam	Unwanted or potentially malicious message
ham	Legitimate message

The dataset is loaded from the SMSSpamCollection file included in the repository.

2. Data Preparation

The dataset is loaded using Pandas and converted into two main columns:

label
message

The labels are encoded as:

spam → 1
ham  → 0

The data is then divided into training and testing sets using an 80/20 split with a fixed random state of 42.

3. Feature Extraction

Machine learning models cannot directly process raw text.

The project uses CountVectorizer from Scikit-learn to convert SMS messages into numerical feature vectors.

English stop words are removed during vectorization.

CountVectorizer(stop_words='english')
4. Model

The classification algorithm used is:

Multinomial Naive Bayes

This algorithm is commonly used for text classification because it works effectively with discrete word-count features.

The model learns patterns from the training messages and predicts whether unseen messages are spam or legitimate.

Streamlit Web Application

The project includes an interactive Streamlit interface.

Users can:

Enter an SMS message.
Click Check if Spam.
Receive the predicted classification.
View the prediction confidence.
View the probability breakdown between Spam and Not Spam.

For example:

Congratulations! You've won a $1000 gift card.

The application analyzes the message and returns the corresponding classification.

The Streamlit application also provides information about the model and dataset through the sidebar.

Model Evaluation

The standalone spam.py script evaluates the trained model using:

Accuracy
Classification Report

The project also demonstrates predictions on sample messages.

Example:

Congratulations! You've won a $1000 gift card.
→ Spam

Hi, can we schedule a meeting tomorrow?
→ Ham
Tech Stack
Technology	Purpose
Python	Programming language
Pandas	Data loading and manipulation
Scikit-learn	Machine learning
CountVectorizer	Text feature extraction
Multinomial Naive Bayes	Spam classification
Streamlit	Web application
SMS Spam Collection	Training dataset
Project Structure
Spam_Detection/
│
├── SMSSpamCollection       # SMS spam/ham dataset
├── app.py                  # Streamlit web application
├── spam.py                 # Model training and evaluation
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
Installation
1. Clone the repository
git clone https://github.com/BoyidiBhuvaneswari/Spam_Detection.git
2. Navigate to the project
cd Spam_Detection
3. Create a virtual environment
python -m venv venv
4. Activate the virtual environment

Windows:

venv\Scripts\activate

macOS/Linux:

source venv/bin/activate
5. Install dependencies
pip install -r requirements.txt

The project requires:

streamlit
pandas
scikit-learn
Run the Streamlit Application

Start the application using:

streamlit run app.py

Streamlit will provide a local URL, typically:

http://localhost:8501

Open the URL in your browser and enter an SMS message to test the classifier.

Run the Machine Learning Script

You can also run the standalone machine learning implementation:

python spam.py

This script:

Loads the dataset
Encodes the labels
Splits the data
Performs text vectorization
Trains the Naive Bayes classifier
Generates predictions
Calculates accuracy
Displays the classification report
Tests the model with sample messages
Example
Input
Congratulations! You've won a $1000 gift card.
Prediction
SPAM
Input
Hi, can we schedule a meeting tomorrow?
Prediction
NOT SPAM
Important Note

The confidence displayed by the application represents the model's estimated probability for the predicted class. It should not be interpreted as a guarantee that a message is actually malicious or legitimate.

This project is intended for educational and demonstration purposes.
