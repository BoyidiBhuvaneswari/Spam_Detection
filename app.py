# SMS Spam Classifier - Streamlit Web App
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import pickle

# Page config
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="",
    layout="centered"
)

# Title and description
st.title("SMS Spam Classifier")
st.markdown("---")
st.write("**Paste any SMS message below and I'll predict if it's SPAM or NOT SPAM!**")

# Load and train the model (cached for performance)
@st.cache_resource
def train_model():
    # Load the dataset
    df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
    df = df.rename(columns={"v1": "label", "v2": "message"})
    df = df[['label', 'message']]
    
    # Encode labels
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    
    # Split data
    X = df['message']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Vectorize
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    # Calculate accuracy
    accuracy = model.score(X_test_vec, y_test)
    
    return model, vectorizer, accuracy

# Train model and show accuracy
with st.spinner("Loading spam detection model..."):
    model, vectorizer, accuracy = train_model()

# Display model info
st.success(f"Model loaded successfully! (Accuracy: {accuracy:.2%})")

# Sidebar with info
st.sidebar.title("About")
st.sidebar.write("""
This AI-powered tool uses **Machine Learning** to detect spam messages.

**Model:** Naive Bayes Classifier  
**Accuracy:** {:.2%}  
**Dataset:** SMS Spam Collection
""".format(accuracy))

st.sidebar.markdown("---")
st.sidebar.write("Built by Bhuvaneswari")

# Main input area
st.markdown("Test a Message")
user_input = st.text_area(
    "Enter SMS message:",
    height=100,
    placeholder="Example: Congratulations! You've won a $1000 gift card..."
)

# Prediction button
if st.button("Check if Spam", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter a message first!")
    else:
        # Predict
        message_vec = vectorizer.transform([user_input])
        prediction = model.predict(message_vec)[0]
        probability = model.predict_proba(message_vec)[0]
        
        # Display result
        st.markdown("### Result")
        
        if prediction == 1:  # Spam
            st.error("### SPAM")
            st.write(f"**Confidence:** {probability[1]:.2%}")
            st.write("""
            **Why this might be spam:**
            - Contains suspicious links or requests
            - Uses urgent or too-good-to-be-true language
            - Asks for personal information
            """)
        else:  # Ham (Not Spam)
            st.success("### NOT SPAM")
            st.write(f"**Confidence:** {probability[0]:.2%}")
            st.write("This message appears to be legitimate.")
        
        # Show probabilities
        st.markdown("#### Probability Breakdown")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Spam Probability", f"{probability[1]:.2%}")
        with col2:
            st.metric("Not Spam Probability", f"{probability[0]:.2%}")



# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>Machine Learning with Scikit-learn</p>
    <p>Bhuvaneswari Boyidi</p>
</div>
""", unsafe_allow_html=True)