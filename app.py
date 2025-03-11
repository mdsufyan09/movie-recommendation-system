from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Mapping for full language names
LANGUAGE_MAP = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German", "it": "Italian",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "hi": "Hindi", "ru": "Russian",
    "pt": "Portuguese", "ar": "Arabic", "tr": "Turkish"
}

# Load new dataset
movies = pd.read_csv('movies.csv')

# Convert release_date to (DD/MM/YYYY) format
movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.strftime('%d/%m/%Y')

# Map language codes to full names
movies['original_language'] = movies['original_language'].apply(lambda x: LANGUAGE_MAP.get(x, x))

# Precompute similarity once at startup
movies['content'] = (
    movies['title'].astype(str) + " " +
    movies['original_language'].astype(str) + " " +
    movies['overview'].astype(str)
)

vectorizer = TfidfVectorizer(stop_words='english')
movie_matrix = vectorizer.fit_transform(movies['content'])
similarity = cosine_similarity(movie_matrix)

# Normalize movie titles for faster lookup
movies['cleaned_title'] = movies['title'].astype(str).str.replace(r'[^a-zA-Z0-9 ]', '', regex=True).str.lower().str.strip()

# Function to recommend movies
def recommend_movies(movie_title, num_recommendations=8):  # Now returns 8 recommendations
    cleaned_input = re.sub(r'[^a-zA-Z0-9 ]', '', movie_title.lower().strip())

    # Find the closest match using a fast lookup
    matches = movies[movies['cleaned_title'].str.contains(cleaned_input, regex=False, na=False)]

    if matches.empty:
        return []

    movie_index = matches.index[0]
    similar_movies = sorted(list(enumerate(similarity[movie_index])), key=lambda x: x[1], reverse=True)[1:num_recommendations+1]

    return [
        {
            "id": i[0],
            "title": movies.iloc[i[0]]['title'],
            "original_language": movies.iloc[i[0]]['original_language']
        }
        for i in similar_movies
    ]

# Flask App
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        recommendations = recommend_movies(movie_name)
    return render_template('index.html', recommendations=recommendations)

@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    movie = movies.iloc[movie_id]
    return jsonify({
        "title": movie["title"],
        "release_date": movie["release_date"],
        "original_language": movie["original_language"],
        "overview": movie["overview"]
    })

if __name__ == '__main__':
    app.run(debug=True)
