# Board Game Recommender

### Goal

1. Build a collaborative recommendation system for current users of boardgamegeek.com, further filtering the results by categories gained from the game descriptions using natural language processing techniques.

2. Build a content based solution for those that are not boardgamegeek.com members using attributes of the games themselves.

### Data collection
boardgamegeek.com (BGG) has an API that allowed me to access XML output for each board game and each user. I used web-scraping techniques to obtain unique ids for every game in the database and from there wrote scripts to call the API, assign variables from a json output, and organize all records into a MongoDB database. I found that there is a 5 second rule on API calls, which meant I had to run my script for a few days. As a result I ran the script on an AWS instance, moved the MongoDB data to EC2, and then pulled the data down locally from there. I made a preliminary cutoff at the top 1000 games rated on BGG to build my current collaborative recommendation system. I am currently using around 14K games in my content-based system.

### Dataset
When putting my data into MongoDB, I organized my data into two datasets:
- Comments/ratings: this included every rating and comment made by users on the first 1000 board games which gave me around 1.2 million records, reviews from about 65,000 unique users, with an average of 18 reviews by each user.

 <img src='https://github.com/Jomonsugi/bgg-exploration/blob/master/plots/ratings_200.png?raw=true'>

- Statistics: (note "statistics" is the name BGG gives this group when calling the API so I am following convention) In total 161 columns, 131 of which were boolean values indicating whether a game belongs to a particular mechanic or category group. I included the board game description as a variable here as well. Examples of other variables are board game complexity, play time, and best number of players.

### Clustering and Distance Matrices
Most of my EDA concerned the second dataset of statistics as I wanted to find variables that could possibly be clustered together or otherwise give an indication of a game someone might like with strong preferences. I used k-modes and k-prototype on various variables and through silhouette scores/plots found promising groups in the mechanic and category segmentations. I then created a hammond distance matrix as a tool to find the most similar games to any particular game based on it's combinations of mechanics and categories. I found it interesting that on further inspection, even though the distance matrix was derived solely on category and mechanic information, trends were visible when looking at average play time, complexity rating, and best number of players. From domain knowledge, I want to note that I believe these distance matrices alone are a powerful way to suggest a game to someone. By giving users the ability to filter results on number of players and play time, this approach alone builds a content based recommendation system. Future work will include exploring the addition of new variables to my distance matrix and giving users other options to customize their recommendations.

* Update: I have added more games to the database for the content-based system to filter through, now totally around 14K. No matter what game is used to find recommendations, I made a cutoff of only returning games rated below 3000 for now. That way if your favorite game is clue, the system can possibly help you find a much more highly rated game with similar play style.

### Spark and Alternating Least Squares
After finding a few promising results from exploring the statistics data set I moved on to the ratings. The user/ratings data that I have allows me to create a utility matrix in which my goal is to predict ratings for games that users have not rated. As 98% of this matrix is null, I am dealing with quite a sparse matrix, however the problem lends itself to collaborative filtering, where I can predict ratings based on previous ratings I already have. I chose alternating least squares as my model to predict for the following reasons:

- it gives good initial results with an RMSE around 1.3
- it has been proved to work well on sparse matrices
- the model is parallelizable which gives me the advantage of using spark to optimize my model and produce recommendations
- I can use ALS-WR to weight the regularization parameter thus making it less dependent on the scale of the dataset

I started by pulling the data from MongoDB directly into a Spark's data structures (RDD and DataFrame). From there I further organized the data and split it into a training, validation, and a test set. I coded a customized grid search to find the optimal regularization parameter, number of iterations, and rank using my training and validation sets, and then tested my model on the holdout test set. This model is fully functioning and is able to produce recommendations for individual users at this point.

### Natural Language Processing
I did want to diversify the output results, with a goal of hopefully making my recommendations more interesting to users. Each game on BGG has a description attached to it. I decided to use natural language processing techniques to see if I could find categories of games. I used a combination of spaCy, Sk-learn, and NLTK to preprocess the data. I filtered out HTML content, removed conventional stop words (and later custom stop words that I found to be domain specific), tokenized, lemmatized, and then transformed all documents to a Tf-idf matrix. From there I started with LDA to look at possible clusters among board game descriptions, using a package called ldaviz to explore my output. Through many iterations I found LDA grouped a large metatopic that I could just call "board gaming" and smaller topics that overlapped each other. I moved to non-negative matrix factorization and immediately found better results. NMF, grouped the game descriptions into intuitive and helpful thematic groups. After testing through options, I settled on grouping games into eight topics. I then went back to my results from my ALS model and setup a script to check the top 8 suggested games for these topics (representing categories of games). If any of the topics were not present I pulled the closest ranked game marked with the topic into the top eight as a way of diversifying the users recommendations.

Here is an example of one topic produced from NMF topic modeling which I called "resource management games"

<img src='https://github.com/Jomonsugi/bgg-exploration/blob/master/plots/documents_basisW1.png?raw=true'>

And here is a before and after top 10 for one particular user, first only using ALS, and then adding in the topic modeling diversity technique:

<img src='https://github.com/Jomonsugi/bgg-exploration/blob/master/screen_shots/Screen%20Shot%202017-06-20%20at%2011.18.19%20PM.png?raw=true'>

The red outlines 3 newcomers to our new list based on diversification of topics gained from the descriptions.
<br><br>
<img src='https://github.com/Jomonsugi/bgg-exploration/blob/master/screen_shots/Screen%20Shot%202017-06-21%20at%208.39.53%20PM.png?raw=true'>


### Flask App
I created a flask app to enable users to interact with both my collaborative and content-based models. The collaborative recommendation portion lets users of boardgamegeek.com enter their using name, and if they choose, pick the number of players they prfer to play with and a min and max play time. The content-based portion allows anyone to enter a game they like (currently rated within BGG's top 1000), and with the same options available as the collaborative model, a recommendation of games similar to the game they have entered is output. I am able to launch the app successfully on AWS, however I do not keep an instance running for public use because of budget constraints. The collaborative system is memory intensive and requires an instance past a free tier to process request.

### Current Work
I'm really excited about this project! At this point I have a scalable workflow where I have automated scripts that make API request, put data into mongodb, pull the data into Spark, optimizes my model with any new data, and allows users to receive recommendations through a web app. As noted above I now have around 14K games in the database that the content-based system is using. From here I would like to pull more users/ratings into my database to expand the functionality of the collaborative model. Further down the road I would like to continue testing model options, to hopefully improve the recommendations I am able to give.

#### Board Game Geek site:

https://boardgamegeek.com/browse/boardgame

#### References

https://link.springer.com/chapter/10.1007%2F978-3-540-68880-8_32

https://satwikkottur.github.io/reports/F14-ML-Report.pdf

http://infolab.stanford.edu/~ullman/mmds/ch9.pdf


#### Code

https://github.com/d4le/recommend

https://github.com/samweaver/mongodb-spark/blob/master/movie-recommendations/src/main/python/movie-recommendations.py
