import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load cleaned dataset
movies_df = pd.read_csv("clean_movies.csv")

# Create TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies_df['content'])

# Calculate Cosine Similarity matrix
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

print("✅ Cosine Similarity Matrix Created!")

# Recommendation function
def get_recommendations(movie_title, cosine_sim_matrix, movies_df, top_n=10):
    
    # Check if movie exists
    if movie_title not in movies_df['title'].values:
        return ["❌ Movie not found in dataset"]

    # Get index of movie
    idx = movies_df[movies_df['title'] == movie_title].index[0]

    # Get similarity scores
    similarity_scores = list(enumerate(cosine_sim_matrix[idx]))

    # Sort movies based on similarity score
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get top N similar movies (excluding itself)
    top_movies = similarity_scores[1:top_n+1]

    # Get movie titles
    recommended_movies = []
    for i in top_movies:
        recommended_movies.append(movies_df.iloc[i[0]]['title'])

    return recommended_movies


# ---------------- TESTING ---------------- #

movie_name = input("Enter a movie name: ")
recommendations = get_recommendations(movie_name, cosine_sim_matrix, movies_df)

print("\n🎬 Recommended Movies:")
for movie in recommendations:
    print("👉", movie)
