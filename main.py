import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


DATA_JSON_FILE = 'email-text-data.json'
data = pd.read_json(DATA_JSON_FILE)


vectorizer = CountVectorizer(stop_words='english')

all_features = vectorizer.fit_transform(data.MESSAGE)

X_train, X_test, y_train, y_test = train_test_split(all_features, data.CATEGORY, 
                                                   test_size=0.3, random_state=88)
classifier = MultinomialNB()        
classifier.fit(X_train, y_train)


def email_prediction(msg):
    matrix = vectorizer.transform([msg])

    return classifier.predict(matrix)[0]


def main():
    
    st.title("SPAM e-MAIL CLASSIFICATION")
    st.subheader('Built with Python and Streamlit')

    msg = st.text_input("Enter Your Message Below... eg. Let's catch up tonight, You have a meeting scheduled for tomorrow, To join our premium memebership please enter code ALLYSON30 etc.")
    if st.button('Predict'):
        
        result = email_prediction(msg)

        if result:
            st.error("SPAM")
        else:
            st.success("HAM")
   

if __name__ == '__main__':
	main()