import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
data = pd.read_csv("movies.csv")

# Select important columns
data = data[['Series_Title', 'Genre', 'Overview']]

# Rename columns
data.columns = ['title', 'genre', 'overview']

# Handle missing values
data['genre'] = data['genre'].fillna("")
data['overview'] = data['overview'].fillna("")

# Combine text columns
data['content'] = data['genre'] + " " + data['overview']

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

data['content'] = data['content'].apply(clean_text)

# Convert text to numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['content'])

print("✅ TF-IDF matrix created successfully!")
print("Shape of matrix:", tfidf_matrix.shape)

# Save cleaned data
data.to_csv("clean_movies.csv", index=False)
