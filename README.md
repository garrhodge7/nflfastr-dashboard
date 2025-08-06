# nflfastr-dashboard
NFL Prediction & Analytics Dashboard
This NFL Streamlit App is an interactive web application designed to visualize, explore, and predict outcomes of NFL games using historical play-by-play data and machine learning models. Built with Python and Streamlit, the dashboard provides both casual fans and data-driven analysts with tools to analyze team performance, view matchup statistics, and forecast game results.

Overview
The app leverages NFL play-by-play data (2020–2024) to generate both visual insights and predictive outputs. It supports classification and regression models, as well as similarity searches using nearest neighbor algorithms. All predictions and visualizations are based on engineered team-level metrics aggregated weekly and seasonally.

Features
1. Overview Tab

   View high-level trends and summary statistics.

   Explore team performance over time with customizable visualizations.

   Analyze offensive and defensive metrics, turnovers, efficiency ratings, and more.

   Select by team, season, and metric type to generate comparative charts.

2. Regression Predictions Tab
   
   Predict final scores for upcoming matchups using trained regression models.
   
   Input game details (home team, away team, Vegas spread/total line).
   
   Returns predicted:
   
      Final score for both home and away teams
   
      Total points scored
   
      Whether the game is likely to go over/under the Vegas line
   

3. Classification Predictions Tab
   
   Predict whether the home team will cover the spread.

   Uses binary classification models trained on historical data.

   Input: spread line, team performance metrics, and game context.

   Output: Binary prediction (cover or not) with probability score.

4. Nearest Neighbors Search Tab
   
   Input a spread and total line to find similar historical games.
   
   Uses K-Nearest Neighbors (KNN) to return the 7 closest games by spread and total.
   
   Displays:
   
     Matchups
   
     Final scores
   
     Total result (Over/Under)
   
     Actual spread result (Covered or not)
   
     Helps visualize how similar games have historically played out.

Models Used

LinearRegression and/or RandomForestRegressor for score prediction.

LogisticRegression and/or GradientBoostingClassifier for spread classification.

KNeighborsClassifier or NearestNeighbors for similarity search.
