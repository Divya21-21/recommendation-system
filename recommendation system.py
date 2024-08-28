import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Sample dataset: Users' ratings of movies
data = {
    'User': ['Alice', 'Alice', 'Alice', 'Bob', 'Bob', 'Bob', 'Carol', 'Carol', 'Carol', 'David', 'David', 'David'],
    'Movie': ['Inception', 'Titanic', 'Avatar', 'Inception', 'Avatar', 'Frozen', 'Titanic', 'Avatar', 'Frozen', 'Inception', 'Titanic', 'Frozen'],
    'Rating': [5, 4, 5, 5, 4, 2, 4, 5, 3, 5, 3, 4]
}

df = pd.DataFrame(data)

# Create a pivot table with users as rows, movies as columns, and ratings as values
user_movie_matrix = df.pivot_table(index='User', columns='Movie', values='Rating')

# Fill NaN with 0 for similarity computation
user_movie_matrix_filled = user_movie_matrix.fillna(0)

# Compute cosine similarity between users
user_similarity = pd.DataFrame(cosine_similarity(user_movie_matrix_filled),
                               index=user_movie_matrix.index, columns=user_movie_matrix.index)

def recommend_movies(user, user_similarity, user_movie_matrix, n_recommendations=2):
    similar_users = user_similarity[user].sort_values(ascending=False).index[1:]
    
    recommendations = set()
    for similar_user in similar_users:
        similar_user_ratings = user_movie_matrix.loc[similar_user]
        unrated_movies = similar_user_ratings[similar_user_ratings.isna()]
        recommendations.update(unrated_movies.index)
        
        if len(recommendations) >= n_recommendations:
            break
    
    return list(recommendations)[:n_recommendations]

# Example recommendation for Alice
recommended_movies = recommend_movies('Alice', user_similarity, user_movie_matrix)
print("Collaborative Filtering Recommended movies for Alice:", recommended_movies)

# Movies and genres dataset
movies = pd.DataFrame({
    'Movie': ['Inception', 'Titanic', 'Avatar', 'Frozen'],
    'Genre': ['Sci-Fi', 'Romance', 'Sci-Fi', 'Animation']
})

# Vectorize the genres
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies['Genre'])

# Create a genre similarity matrix using cosine similarity
movie_similarity = pd.DataFrame(cosine_similarity(genre_matrix),
                                index=movies['Movie'], columns=movies['Movie'])

# User ratings matrix
user_ratings = pd.DataFrame({
    'User': ['Alice', 'Bob', 'Carol'],
    'Inception': [5, 5, None],
    'Titanic': [4, None, 4],
    'Avatar': [5, 4, 5],
    'Frozen': [None, 2, 3]
}).set_index('User')

def recommend_movies_content(user, user_ratings, movie_similarity, n_recommendations=2):
    user_ratings = user_ratings.loc[user].dropna()
    
    similar_movies = pd.Series(dtype='float64')
    for movie, rating in user_ratings.items():
        similar_movies = similar_movies.add(movie_similarity[movie].multiply(rating), fill_value=0)
    
    similar_movies = similar_movies.groupby(similar_movies.index).sum()
    recommended_movies = similar_movies.drop(user_ratings.index).sort_values(ascending=False)
    
    return recommended_movies.index[:n_recommendations]

# Example recommendation for Alice
recommended_movies_content = recommend_movies_content('Alice', user_ratings, movie_similarity)
print("Content-based Recommended movies for Alice:", recommended_movies_content)