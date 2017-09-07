## Micah Shanks
#### May 23, 2017

##### Motivation:
Board games have become a hobby of mine over the past few years. There are thousands of games to choose from, with hundreds of solid choices that come out every year. The industry is growing rapidly (documentation to come). I have a closet full of great games, some that I play much more than others, and I'm always looking for the next one deemed worth a purchase. They can be expensive. I have some great games that cost me around $20 or less, while I have spent over $100 on others. There are many considerations I have when looking for a new game. Who is going to play with me? (insert pouty face and foot stomp) How many people is the game best for? What is the play time? What kind of mechanics does it have? What game out there is like one I already have, or like nothing I already have!? So, how do I decide on what to buy? Ratings, reviews, research. One of the sites I use (along with many others) is boardgamegeek.com. For every game I might be interested in, it has a description, stats, categories, ratings, and reviews. Enter my goal. I want to build a recommendation system that not only does the work for me, but does better than I could ever do as there is a wealth of information on all the games out there that I could never have time to exhaustively surmount. With new board games, ratings are a good reference point, but more important are your own personal taste. I want to leverage the game description, ratings comments, stats, and categories, using NLP, reduction techniques, and clustering methods (and whatever else I find useful along the journey), to aid in a table-top gamers next purchase, or maybe even their first. I posit that NLP will come to use as my most powerful, and novel technique for finding useful information.  

##### Presentation:

I've never built a web app, but I'd like to build an interactive tool to recommend a group of games someone might enjoy. I think I would benefit from some interesting visualizations as well.

##### Data:
My data is coming from boardgamegeek.com

https://boardgamegeek.com/browse/boardgame

I have written scripts that both web scrape and make api calls, define variables from stats/categories, obtain all game descriptions, ratings/comments, and then store all the data in a mongodb database. I have organized the schema to be easily pulled into pandas for further work.

I am planning on starting with 14,000 games. I pulled down all ratings that have comments from the first 100 games thus far and have 308,894 reviews, so I will have plenty of data to work with.
