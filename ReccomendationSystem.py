import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
from math import sqrt

# Load the dataset
# You can adjust the path based on where you saved the dataset
ratings_df = pd.read_csv('C:\\Users\\jyots\\Desktop\\CodTech\\Task-4_RECOMMENDATION SYSTEM\\ml-100k\\u.data', delimiter='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Load movie data to get movie titles (if available)
movies_df = pd.read_csv('C:\\Users\\jyots\\Desktop\\CodTech\\Task-4_RECOMMENDATION SYSTEM\\ml-100k\\u.item', delimiter='|', encoding='latin-1', header=None, names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'])

# Preprocessing: Merging datasets and removing unnecessary columns
ratings_df = pd.merge(ratings_df, movies_df[['movie_id', 'movie_title']], left_on='item_id', right_on='movie_id')

# Drop unnecessary columns
ratings_df = ratings_df[['user_id', 'movie_title', 'rating']]

# Display first few rows to check
ratings_df.head()

# Create the user-item matrix
user_item_matrix = ratings_df.pivot_table(index='user_id', columns='movie_title', values='rating')

# Fill NaN values with 0s (unrated movies will be treated as 0)
user_item_matrix = user_item_matrix.fillna(0)

# Step 1: SVD Matrix Factorization
#U, sigma, Vt = svds(user_item_matrix.values, k=50)  # k is the number of latent factors
num_users, num_movies = user_item_matrix.shape
k = min(num_users, num_movies) - 1  # or any smaller value, e.g., 20
U, sigma, Vt = svds(user_item_matrix.values, k=k)

# Reconstruct the matrix approximation
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)

# Create a DataFrame with the predicted ratings
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)

# Show predicted ratings for the first user
predicted_ratings_df.iloc[0].head()

# Step 2: Train-Test Split (80% train, 20% test)
train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Create user-item matrices for the training and testing data
train_matrix = train_data.pivot_table(index='user_id', columns='movie_title', values='rating').fillna(0)
test_matrix = test_data.pivot_table(index='user_id', columns='movie_title', values='rating').fillna(0)

# Rebuild the SVD model on the train data
# U, sigma, Vt = svds(train_matrix.values, k=50)
num_users_train, num_movies_train = train_matrix.shape
k_train = min(num_users_train, num_movies_train) - 1  # or any smaller value
U, sigma, Vt = svds(train_matrix.values, k=k_train)
sigma = np.diag(sigma)
train_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

# Convert train and test data to arrays
train_data_values = train_matrix.values
test_data_values = test_matrix.values

# Calculate RMSE on test data
def calculate_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

rmse = calculate_rmse(test_data_values, train_predicted_ratings)
print(f'RMSE on Test Data: {rmse}')

# Step 3: Recommendations for a User
def recommend(user_id, n=5):
    # Get the predicted ratings for the user
    user_ratings = predicted_ratings_df.loc[user_id]
    
    # Sort movies by predicted rating in descending order
    recommended_movies = user_ratings.sort_values(ascending=False).head(n)
    
    return recommended_movies.index.tolist()

# Get top 5 movie recommendations for user with ID 1
recommended_movies = recommend(1, n=5)
print("\nTop 5 Recommended Movies for User 1:", recommended_movies)
