import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy

# Load the MovieLens dataset (you can download it from https://grouplens.org/datasets/movielens/)
# For this example, we assume you have the 'ratings.csv' file.
ratings = pd.read_csv('ratings.csv')

# Define the rating scale (from 1 to 5 in the MovieLens dataset)
reader = Reader(rating_scale=(1, 5))

# Load the dataset into the Surprise format
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Build and train the collaborative filtering model (SVD)
model = SVD()
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model
accuracy.rmse(predictions)

# Function to get movie recommendations for a given user
def get_top_n_recommendations(predictions, n=10):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))
    
    # Sort the predictions for each user and get the top N recommendations
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n

# Get top N recommendations for a specific user (replace 'user_id' with the desired user ID)
user_id = 1
user_ratings = ratings[ratings['userId'] == user_id]
user_movies = user_ratings['movieId'].tolist()
user_unrated_movies = ratings[~ratings['movieId'].isin(user_movies)]['movieId'].unique()

testset = [(user_id, movie_id, 0) for movie_id in user_unrated_movies]
predictions = model.test(testset)

top_n_recommendations = get_top_n_recommendations(predictions, n=10)

# Print the top N recommendations for the user
print(f"Top 10 Recommendations for User {user_id}:")
for movie_id, est_rating in top_n_recommendations[user_id]:
    print(f"Movie ID: {movie_id}, Estimated Rating: {est_rating}")
