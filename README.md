# 🎬 Movie Recommendation System using SVD

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: JYOTSANA BHARDWAJ

*INTERN ID*: CT08DK599

*DOMAIN*: MACHINE LEARNING

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTOSH

📖 Introduction
This project is a Movie Recommendation System built with Collaborative Filtering and Singular Value Decomposition (SVD). It predicts user ratings for movies that users haven't seen yet, based on existing ratings, and suggests movies accordingly. The system uses the MovieLens dataset (u.data and u.item), which is a well-known dataset for recommendation engines.

The main goal was to implement a simple matrix factorization approach for collaborative filtering using Python and gain practical experience with recommender systems.

🎯 Objective
- Load and preprocess movie rating data  
- Create a user-item matrix for collaborative filtering  
- Apply SVD for matrix factorization  
- Predict missing ratings  
- Calculate RMSE to assess model performance  
- Generate top-N movie recommendations for a user  

🛠 Technologies and Libraries Used
- Python 3  
- NumPy, for numerical operations  
- Pandas, for loading and manipulating data  
- SciPy, for computing sparse matrix SVD  
- Scikit-learn, for splitting data and evaluating the model with RMSE

📁 Dataset Description
The project uses two files from the MovieLens 100K dataset:  
- u.data: Contains user ID, movie ID, rating (1–5), and timestamp  
- u.item: Contains movie metadata, including titles  

These files are combined to link ratings with their respective movie titles. Only the user_id, movie_title, and rating columns are used to build the model. 

🧠 Project Workflow
Data Preprocessing  
- Loaded u.data and u.item files  
- Merged them to include movie titles  
- Built a user-item matrix with users as rows and movie titles as columns  
- Filled missing ratings with 0, which can be improved  

SVD Matrix Factorization  
- Applied SVD using scipy.sparse.linalg.svds() to reduce dimensions and reconstruct the rating matrix  
- Generated predicted ratings for all user-movie combinations 

Model Evaluation
- Conducted an 80-20 train-test split  
- Recomputed SVD on the training matrix  
- Calculated RMSE on the test set to evaluate prediction accuracy

Recommendation Generation
For a given user, the system recommends the top N movies with the highest predicted ratings

🚀 How to Run
Make sure you have Python and pip installed. Run the script:
<pre><code>python RecommendationSystem.py</code></pre>
