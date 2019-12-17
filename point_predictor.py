import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Apply simple Ridge Regression for testing purposes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def transform(target):
    # Retrieve columns w/ categorical data
    cat = target.select_dtypes(include=['object']).copy()

    #Return df with onehot encoding applied to the columns which are categorical
    one_hot = pd.get_dummies(target, columns=cat.columns)
    one_hot = one_hot.replace(np.nan, 0.0)
    return one_hot


def process_data(pos):
    if pos == "QB":
        data_initial = pd.read_csv("data/finalized_data/finalized_QB.csv", index_col=0)
        features = ["TD Passes", "Completion Percentage", "Passer Rating", "points", "Player Id", "week"]
    elif pos == "RB":
        data_initial = pd.read_csv("data/finalized_data/finalized_RB.csv", index_col=0)
        features = ["Rushing Attempts", "Receptions", "Yards Per Carry", "points", "Player Id", "week"]
    elif pos == "TE":
        data_initial = pd.read_csv("data/finalized_data/finalized_TE.csv", index_col=0)
        features = ["Yards per Reception", "Yards Per Carry", "points", "Player Id", "week"]
    else:
        data_initial = pd.read_csv("data/finalized_data/finalized_WR.csv", index_col=0)
        features = ["Yards per Reception", "Yards Per Carry", "points", "Player Id", "week"]

    pos_features = data_initial[features]
    pos_features['Player Id'] = pos_features['Player Id'].str.extract('(\d+)', expand=False).astype(int)

    if "points" in features:
        features.remove("points")
    if "Player Id" in features:
        features.remove("Player Id")
    if "week" in features:
        features.remove("week")

    for col in pos_features.keys():
        pos_features[col] = pos_features[col].replace('--', 0).astype("float64")

    return pos_features, features


def predictor(df):

    # Always work on a copy
    final_df = df.copy()

    # Create player id map
    player_idx_map = dict(zip(df.index, df['Player Id']))

    # Get latest week - This is the one we will do predictions on
    latest_week = final_df['week'].max()

    # Train on previous weeks data
    train = final_df[final_df['week'] < latest_week - 1]
    pos_y = train["points"].values

    # Take away week, points and player ID!
    train = train.drop(["week", "Player Id", "points"], axis=1)
    pos_X = train.values

    # Scale/normalize values
    # scaler = StandardScaler()
    # scaler.fit(pos_X)
    # pos_X = scaler.transform(pos_X)

    # Train regression model
    ridge_opt = Ridge(alpha=10, random_state=69420)
    ridge_opt.fit(pos_X, pos_y)

    # Get the dataframe for the latest week to predict scores on
    test = df[df['week'] == latest_week]
    test = test.drop(["week", "Player Id", "points"], axis=1)
    # test = test.drop("Player Id", axis=1)

    # Add the scores for the latest week
    preds = ridge_opt.predict(test.values)
    test["points"] = preds

    # Add back player ids!
    test["Player Id"] = pd.Series()

    for idx in test.index:
        test.loc[idx, "Player Id"] = player_idx_map[idx]

    return test
