# Fantasy-Football
Live version of logic linked with database and frontend
FYI: This program is currently under development to be more human readable. Due to issues in my machines wifi connection and time constaints preventing me from opening and fixing the connection, database connections were not included in this version and massive amounts of csv's were used. This is so the program could run regardless of hardware failures. 

www.ff-ai.com

This project provides the service of recommending changes to a users fantasy football team's lineup for the upcoming weekend. The user input their team's name-currently restricted to teams made in ESPN(which is the majority of FF participants)-and is outputted recommendation based on players on their team and players not on their team but available for trade in their league's list of free agents. 

In order to make predictions on each players stats for a given week, we scraped data from fantasyfootballdb.com. The file DataScrapeAndSave.py was made to do and will be cleaned and made into a class for import and use by others in the coming weeks.

Once a database was made with appropiate data for players and team, anaylsis to identify which features were stable, correlated(pos or neg), and most impacted in the resulting fantasy football score. 

A few factors that were challenging were the relationship between significance and time of a statistic, the abundance of features relative to the amount of existing data on new players(the curse of too many features), and model determination. Does a players performance 5 years ago hold any weight in how they'll perform next week? How do you fit data with 30 columns and 5 rows into a model for prediction? How can we best transform the data to identify the most significant axis? Is this truly a regression problem? How can we make the regression predictions fit into what is ultimately a logistic prediction? I,e.Predicting rushing yards is regression, but predicting points from rushing yards is logistic classification.

Some of the ways these questions were answered is through trellis plots to identify correlations, mass amounts of experimenting on different models, pca to reduce and identify important axis, an iterative content-based recommender for predictions where data are absent, cumulative, weighted averages for training, power raising for time(still skeptical this is optimal route), scaling for input into training, batching to reduce overfitting, and a whole host of other operations. 

After the NFL has played all the games each week, the predictions are updated and the results are cached in the server. When a user input their team name on the website, we are able to gather the player ids from espn and make the appropiate filtered call to the server and return the best selections. 

Most of the interesting logic for this progra is contain in real.py. I hope nobody reads this sentence before that has been updated to resemble a proper class. 
