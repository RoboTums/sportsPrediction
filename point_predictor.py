import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
# Apply simple Ridge Regression for testing purposes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error

def transform(target):
    # Retrieve columns w/ categorical data
    cat = target.select_dtypes(include=['object']).copy()

    #Return df with onehot encoding applied to the columns which are categorical
    one_hot = pd.get_dummies(target, columns=cat.columns)
    one_hot = one_hot.replace(np.nan, 0.0)
    return one_hot

def process_data(data, pos):
    data_initial = {}
    features = []
    if pos == "qb":
        data_initial = pd.read_csv("data/finalized_data/finalized_QB.csv", index_col=0)
        features = ["TD Passes", "Completion Percentage", "Passer Rating"]
    elif pos == "rb":
        data_initial = pd.read_csv("data/finalized_data/finalized_RB.csv", index_col=0)
        features = ["Rushing Attempts", "Receptions", "Yards Per Carry"]
    elif pos == "te":
        data_initial = pd.read_csv("data/finalized_data/finalized_TE.csv", index_col=0)
        features = ["Yards per Reception", "Yards Per Carry"]
    else:
        data_initial = pd.read_csv("data/finalized_data/finalized_WR.csv", index_col=0)
        features = ["Yards per Reception", "Yards Per Carry"]
    pos_features = data_initial[features]
    for col in pos_features.keys():
        pos_features[col] = pos_features[col].replace('--',0).astype("float64")

    pos_X = transform(pos_features).values
    pos_y = data_initial["points"].values

    # Scale/normalize values
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(pos_X)
    # Apply transform to both the training set and the test set.
    pos_X = scaler.transform(pos_X)
    #X_train, X_test, y_train, y_test= train_test_split(pos_X, pos_y, test_size=0.1,random_state=69420)
    ridge_opt = Ridge(alpha=10, random_state=69420)
    #ridge_opt.fit(X_train, y_train)
    #print(y_train, y_test)
    ridge_opt.fit(pos_X, pos_y)
    data_X = data[features]
    preds = ridge_opt.predict(data_X.values)
    data["points"] = preds
    return data
