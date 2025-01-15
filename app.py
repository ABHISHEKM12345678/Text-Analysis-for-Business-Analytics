# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load Dataset
# Replace 'your_dataset.csv' with your dataset file
data = pd.read_csv('sample_dataset.csv')

# Display first few rows of the dataset
print(data.head())

# Step 2: Data Preprocessing
# Check for missing values
print(data.isnull().sum())

# Drop missing values
data.dropna(subset=['text_column', 'label_column'], inplace=True)

# Convert text to lowercase
data['text_column'] = data['text_column'].str.lower()

# Remove special characters
data['text_column'] = data['text_column'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

# Tokenization and stopword removal can be done during vectorization.

# Step 3: Exploratory Data Analysis (EDA)
# Visualize the distribution of labels
sns.countplot(data['label_column'])
plt.title("Label Distribution")
plt.show()

# Word cloud for most common words (optional)
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data['text_column']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Step 4: Feature Engineering
# Split data into training and testing sets
X = data['text_column']
y = data['label_column']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Model Building
# Using RandomForestClassifier as an example
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Step 6: Evaluation
y_pred = model.predict(X_test_tfidf)

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 7: Interpretation
# Extract feature importance (for tree-based models)
feature_importances = model.feature_importances_
important_features = pd.DataFrame({'Feature': vectorizer.get_feature_names_out(), 'Importance': feature_importances})
important_features = important_features.sort_values(by='Importance', ascending=False).head(10)
print("Top 10 Important Features:\n", important_features)

# Save the model (optional)
import joblib
joblib.dump(model, 'text_analysis_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
