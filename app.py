import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies_df = pd.read_csv("clean_movies.csv")

# Vectorize content
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies_df['content'])

# Compute similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# Recommendation function
def get_recommendations(movie_title, top_n=10):
    if movie_title not in movies_df['title'].values:
        return ["❌ Movie not found in dataset"]

    idx = movies_df[movies_df['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_movies = similarity_scores[1:top_n+1]

    return [movies_df.iloc[i[0]]['title'] for i in top_movies]

# ---------------- UI ---------------- #

st.set_page_config(page_title="🎬 Smart Movie Recommender")

st.title("🎬 Smart Movie Recommender")
st.write("Enter a movie name and get similar movie recommendations.")

movie_name = st.text_input("Enter Movie Title:")

if st.button("Recommend"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        results = get_recommendations(movie_name)
        st.subheader("Recommended Movies:")
        for movie in results:
            st.write("👉", movie)
