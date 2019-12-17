import pandas as pd
import os
from timeSeriesRegressor import TimeSeriesRegressor
from point_predictor import process_data, predictor
import sys

def predict(position, model="rfrg"):
    """ Takes in a finalized scrapped dataframe and produces the prediction for next week """

    # Check that it must be a valid name
    if position not in ["DEF", "QB", "TE", "WR", "RB"]:
        print("Position must be DEF, QB, TE, WR or RB")
        sys.exit(1)

    elif position != "DEF":
        # 1. Process data - turn values into floats and stuff, get important features
        print("===================================================================")
        print("Processing file...")
        print("===================================================================")
        df, key_features = process_data(position)

        # 2. Get future values for each key feature using the TS Regressor
        print("===================================================================")
        print("Starting time series regression...")
        print("===================================================================")
        df['week'] = df['week'].astype(int)
        latest_week = int(df['week'].max()) + 1

        resulting_dfs = []

        for feature in key_features:
            print('---------------------------------------------------------')
            print(f"\nFEATURE: {feature}\n")

            # Select only one feature at a time
            df_train = df[['Player Id', 'week', 'points', feature]]
            ts_reg = TimeSeriesRegressor(df_train)
            new_df = ts_reg.complete_data(df_train)

            # Train on current weeks
            ts_reg.train(new_df, feature, latest_week, extra=False)

            # Predict the next time step
            preds, new_df = ts_reg.predict(new_df, feature, model)
            resulting_dfs.append(new_df)

            print('---------------------------------------------------------')

        # Append to final dataframe
        final_df = resulting_dfs[0]

        for res_df in resulting_dfs[1:]:
            final_df = pd.merge(final_df, res_df, on=["Player Id", "week", "points"])

        print("===================================================================")
        print("Calling predictor...")
        print("===================================================================")
        # 3. Call predictor for points given the features for new week and the features for old weeks
        final_df = predictor(final_df)

        final_df.to_csv(f"final_{position}_output.csv")

        return final_df

    # For DEF, we only use time series score predictor
    else:
        print("===================================================================")
        print("Processing file...")
        print("===================================================================")
        df = pd.read_csv("data/finalized_data/finalized_DEF.csv")
        df_train = df.copy()
        df_train = df_train.rename(columns={"name": "Player Id"})
        df_train = df_train[["Player Id", "week", "points"]]
        df_train["points"] = df_train["points"].replace('--', 0).astype("float64")

        print("===================================================================")
        print("Starting time series regression...")
        print("===================================================================")
        df_train['week'] = df_train['week'].astype(int)
        latest_week = int(df_train['week'].max()) + 1

        # We will only do this for the points feature
        ts_reg = TimeSeriesRegressor(df_train)
        new_df = ts_reg.complete_data(df_train)

        # Train on current weeks
        ts_reg.train(new_df, "points", latest_week, extra=False, defender=True)

        # Predict next week's score
        preds, new_df = ts_reg.predict(new_df, "points", model, defender=True)

        new_df.to_csv("final_DEF_output.csv")

        return new_df


# returns DF of prediction.
# DF: pd dataframe, Position: str. Windowsize: Int
def predictlol(DF, position, windowsize=5):
    portfolio = pd.DataFrame()
    # expected value
    portfolio['mean'] = DF['points'].rolling(windowsize, win_type='boxcar', min_periods=1).mean()
    # standard deviation
    portfolio['std'] = DF['points'].rolling(windowsize, min_periods=1).std()
    # variance
    portfolio['variance'] = DF['points'].rolling(windowsize, min_periods=1).var()
    # X identifier
    portfolio['player'] = DF['Player Id']
    portfolio['position'] = pd.Series([position] * len(portfolio['player']))
    portfolio['cost'] = DF['salary'].rolling(5, min_periods=1).max()

    portfolio = portfolio.dropna()

    return portfolio


def strExcludeMedian(DF):
    newDF = pd.DataFrame()
    # print(DF['position'])

    return newDF


# List of Predictions: List[pd Dataframes]
# outputs whole portfolio of players. 
def generatePortfolioDataSet(listOfPredictions):
    bigKahuna = pd.DataFrame()

    for dataset in listOfPredictions:
        bigKahuna = pd.concat([bigKahuna, dataset])

    joiner = pd.concat([bigKahuna['position'], bigKahuna['player']], axis=1)
    # print('yike:',joiner.columns)

    bigKahuna = bigKahuna.groupby('player')
    joiner = joiner.drop_duplicates(subset=['player'])
    joiner = joiner.sort_values(by=['player'])
    joiner.index = joiner['player']
    yike = bigKahuna.median()
    # print(bigKahuna['mean'].median())
    bigKahuna = pd.concat([yike, joiner], axis=1)

    return bigKahuna


def main():
    output = predict("QB")
    print(output)


# predictionRaw = [x for x in os.listdir() if 'finalized' in x]
#
# #sort predictions for sanity
# predictionRaw.sort()
# predictionClean = [ x[10:].strip('.csv') for x in predictionRaw]
# portfolioData = []
#
# for pos, data in enumerate(predictionRaw):
# 	positionDataframe = pd.read_csv(data,index_col=0)
# 	portfolioData.append(predict(positionDataframe,predictionClean[pos]))
#
# optAlgInput = generatePortfolioDataSet(portfolioData)
#
# #optAlgInput = optAlgInput.drop(['player.1'],axis=1)
# optAlgInput.to_csv('gurobiTime.csv')
# #print(optAlgInput)


main()
