# Text-Analysis-for-Business-Analytics
Text Analysis for Business Analytics
# Import necessary libraries
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Collection
# Load the dataset
data = pd.read_csv('customer_reviews.csv')  # Ensure this file exists in your directory
print("Sample Data:")
print(data.head())  # Display the first few rows of the dataset

# Expected Output: Display the first few rows of your dataset
# Sample Output:
#    reviews                         sentiment
# 0  "Great product, highly recommend!"     positive
# 1  "Worst purchase I've ever made."       negative
# 2  "Okay, not the best."                   neutral
# 3  "Love it! Will buy again!"              positive
# 4  "Did not meet my expectations."         negative

# Step 2: Data Preprocessing
def preprocess_text(text):
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    return text

# Apply preprocessing to the 'reviews' column
data['cleaned_reviews'] = data['reviews'].apply(preprocess_text)

# Display cleaned data
print("\nCleaned Data:")
print(data[['reviews', 'cleaned_reviews']].head())  # Show original and cleaned reviews

# Expected Output: Cleaned reviews without special characters
# Sample Output:
#                                            reviews              cleaned_reviews
# 0                     "Great product, highly recommend!"                     great product highly recommend
# 1                               "Worst purchase I've ever made."                              worst purchase ive ever made
# 2                                            "Okay, not the best."                                           okay not the best
# 3                                    "Love it! Will buy again!"                                 love it will buy again
# 4                              "Did not meet my expectations."                               did not meet my expectations

# Step 3: Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['cleaned_reviews'])  # Transform cleaned text into TF-IDF features

# Step 4: Model Selection
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['sentiment'], test_size=0.2, random_state=42)

# Train the model using Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Model Evaluation
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))  # Print evaluation metrics

# Expected Output: Classification report showing precision, recall, f1-score for each class
# Sample Output:
#               precision    recall  f1-score   support
#
#     negative       0.80      0.75      0.77        12
#      neutral       0.50      0.25      0.33         8
#     positive       0.75      0.90      0.82        20
#
#    accuracy                           0.73        40
#   macro avg       0.68      0.63      0.64        40
# weighted avg       0.73      0.73      0.72        40

# Step 6: Results and Insights
# Visualizing sentiment distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='sentiment', data=data)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# This plot will show the distribution of sentiments across your dataset.
