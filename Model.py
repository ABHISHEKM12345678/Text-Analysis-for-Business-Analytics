import joblib

# Load the trained model and vectorizer
model = joblib.load('models/text_analysis_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Example usage for making a prediction on new text data
new_text = ["I am really satisfied with this product!"]
new_text_vect = vectorizer.transform(new_text)  # Use the same vectorizer to transform new data

# Predict sentiment (positive or negative)
prediction = model.predict(new_text_vect)
print("Predicted sentiment:", prediction[0])
