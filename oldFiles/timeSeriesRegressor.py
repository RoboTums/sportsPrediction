''' Usage: python timeSeriesRegressor.py [file_path] [feature separated by underscores] [regressor_model] [use_extra_features]

    python timeSeriesRegressor.py data/finalized_data/finalized_QB.csv Passer_Rating rfrg 0
    @author: dafirebanks '''

from scipy.stats import truncnorm
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
import sys


#### Helper functions
def rmsle(ytrue, ypred):
    return np.sqrt(mean_squared_error(ytrue, ypred))


# Taken from https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy
def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


#####

class TimeSeriesRegressor(object):
    def __init__(self, df):
        # All features important for the position
        self.features = list(df.columns)
        self.features.remove('Player Id')
        self.features.remove('week')

        # All players in the dataframe
        self.players = list(df['Player Id'].unique())

    def generate_distribution(self, df, var, week):
        """ Takes in a dataframe with important features for prediction,
            and produces a distribution per week from each of the variables in it """

        # Get descriptive statistics
        mean = df[var][df['week'] == week].mean()
        std = df[var][df['week'] == week].std()
        low = df[var][df['week'] == week].min()
        high = df[var][df['week'] == week].max()

        #     print(df[var][df['week'] == week].plot.density())

        # TEMPORARY: USE NORMAL DISTRIBUTION
        g = gaussian_kde(df[var][df['week'] == week])
        X = get_truncated_normal(mean=mean, sd=std, low=low, upp=high)
        #     print([g(x)[0] for x in np.linspace(-2,8,10)])
        #         print(list(df[var][df['week'] == week]))
        #         print(X.rvs(10))

        return X

    def complete_data(self, df):
        """ Takes in a dataframe with important features for prediction,
            and returns a dataframe with the same amount of players in every week, for all features"""

        # Make a copy of the df so we don't modify the original one
        new_df = df.copy()

        # For each week:
        for week in range(1, 9):
            # Get the players that don't appear in the week
            cur_players = list(new_df['Player Id'][new_df['week'] == week].unique())
            missing_players = [player for player in self.players if player not in cur_players]
            missing_values = {}  # {player: {feature: value}}

            # For every missing player, add a new row for that week, that feature, sampling a value from the pdf
            for player in missing_players:
                missing_values[player] = {}

                for var in self.features:
                    # Create distribution from statistics for that week, and that feature
                    pdf = self.generate_distribution(new_df, var, week)

                    # Store value
                    missing_values[player][var] = pdf.rvs(1)[0]

            # Now add all the new missing values as rows to the old df
            missing_series = []
            for player, value_dict in missing_values.items():
                value_dict['Player Id'] = player
                value_dict['week'] = week
                missing_series.append(pd.Series(value_dict))

            new_df = new_df.append(pd.DataFrame(missing_series), ignore_index=True, sort=True)

        return new_df

    def _train_preprocess(self, df, feature, extra=False):

        # Work with deep copy of df
        df2 = df.copy()

        # Zero step, strip player ID and week of strings so it's easier to deal with
        df2['week'] = df2['week'].astype(int)

        # First, let's sort by week (and Player Id)!
        df2 = df2.sort_values(['week', 'Player Id'])

        # Then add extra features from the previous week (t-1)
        df2[f'Last_Week_{feature}'] = df2.groupby(['Player Id'])[feature].shift()

        if extra:
            # This would not be useful for a new instance, but maybe we can try for two weeks back?
            df2['Last_Week_Diff'] = df2.groupby(['Player Id'])[f'Last_Week_{feature}'].diff()

            # Extra feature from second to last week
            df2[f'Last-1_Week_{feature}'] = df2.groupby(['Player Id'])[feature].shift(2)

        # TODO Should we drop na or replace with 0s?
        df2 = df2.dropna()
        # print("Train shape", df2.shape)
        return df2

    def _test_preprocessing(self, df, feature):
        df2 = df.copy()

        latest_week = df2['week'].max()
        # print("BRO WEEK", latest_week)

        # We need to add more rows for week 9, and the features from last week
        new_week = []

        for player in self.players:
            new_week.append(pd.Series({'Player Id': player, feature: 0, 'week': latest_week + 1, 'points': 0}))

        df2 = df2.append(pd.DataFrame(new_week), ignore_index=True, sort=True)
        df2 = df2.sort_values(['week', 'Player Id'])

        # print(list(df2['Player Id']))
        # Strip player ID and week of strings so it's easier to deal with
        df2['week'] = df2['week'].astype(int)
        # print(df2.shape)
        # Add the last week feature
        df2[f'Last_Week_{feature}'] = np.roll(df2[feature], len(self.players))
        # print(df2)
        df2 = df2.dropna()

        # print("Test shape", df2.shape)
        return df2

    def train(self, df, feature, max_range, extra=False, defender=False):
        """ Performs a time series regression in df[feature], produces num_tsteps predictions for the future """

        df2 = self._train_preprocess(df, feature, extra)

        # No need for names anymore
        if defender:
            df2 = df2.drop(["Player Id"], axis=1)

        # Instantiate the models
        self.rfrg = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=69420)

        if not defender:
            self.gbrg = LGBMRegressor(n_estimators=1000, learning_rate=0.01)

        # Then, perform regression -> This is to see how it performs over weeks
        mean_error1 = []
        mean_error2 = []

        for week in range(max_range - 5, max_range):
            train = df2[df2['week'] < week]
            val = df2[df2['week'] == week]

            x_train, x_test = train.drop([feature], axis=1), val.drop([feature], axis=1)
            y_train, y_test = train[feature].values, val[feature].values

            self.rfrg.fit(x_train, y_train)
            preds1 = self.rfrg.predict(x_test)
            error1 = rmsle(y_test, preds1)
            print('Week %d - Error for Random Forest %.5f' % (week, error1))

            mean_error1.append(error1)
            if not defender:
                self.gbrg.fit(x_train, np.log1p(y_train))
                preds2 = np.expm1(self.gbrg.predict(x_test))
                error2 = rmsle(y_test, preds2)
                print('Week %d - Error for Gradient Boosting %.5f' % (week, error2))
                mean_error2.append(error2)

            print()
        print()
        print("Feature statistics:")
        print(f"Min value for feature {feature}: {df[feature].min()}")
        print(f"Max value for feature {feature}: {df[feature].max()}")
        print(f"Mean value for feature {feature}: {df[feature].mean()}")
        print(f"Standard deviation for feature {feature}: {df[feature].std()}")
        print()
        print("Results")
        print('Mean Error for Random Forest = %.5f' % np.mean(mean_error1))

        # Note: the final model is trained on every week and stored in self.model!
        final_xtrain = df2.drop([feature], axis=1)
        final_ytrain = df2[feature].values
        self.rfrg.fit(final_xtrain, final_ytrain)

        if not defender:
            print('Mean Error for Gradient Boosting = %.5f' % np.mean(mean_error2))
            self.gbrg.fit(final_xtrain, np.log1p(final_ytrain))

    def predict(self, df, feature, model="rfrg", defender=False):
        """ """
        # print(df.shape)
        # Preprocessing necessary!
        df2 = self._test_preprocessing(df, feature)

        # Remove current feature for predictions
        df3 = df2.drop(feature, axis=1)

        # Remove previous weeks
        df3 = df3.drop(list(range(0, len(df3) - len(self.players))))

        # This is to store the names of the teams
        if defender:
            name_idx_map = dict(zip(df3.index, df3["Player Id"]))
            df3 = df3.drop(["Player Id"], axis=1)

        # Make predictions
        if model == "rfrg":
            preds = self.rfrg.predict(df3)
        elif model == "gbrg":
            preds = self.gbrg.predict(df3)

        if defender:
            # Move back team names
            df3["Player Id"] = pd.Series()
            for idx in df3.index:
                df3.loc[idx, "Player Id"] = name_idx_map[idx]

        # Store predictions
        final_df = self.store_predictions(preds, df2, feature)

        return preds, final_df

    def store_predictions(self, preds, df, feature):
        """ Stores prediction from model as rows to the dataframe """

        prev_values = list(df[feature].iloc[:len(df) - len(self.players)])
        prev_values.extend(preds)

        df[feature] = prev_values

        return df


def shitty_preprocessing(target):
    all_numerical_cols = ["week", "salary"]

    not_def_numerical_cols = ["Games Started", "Rushing Attempts", "Yards Per Carry", "Rushing TDs",
                              "Rushing Yards", "Fumbles", "Fumbles Lost", "Age", "Height (inches)",
                              "Weight (lbs)", "Number"]

    only_qb = ["Passes Completed", "Passes Attempted", "Completion Percentage", "Passing Yards", "Sacked Yards Lost",
               "Passing Yards Per Attempt", "TD Passes", "Ints", "Sacks", "Passer Rating"]

    only_qb.extend(not_def_numerical_cols)
    only_qb.extend(all_numerical_cols)

    for col in only_qb:
        target[col] = target[col].replace("--", "0").astype("float64")

    return target


def main():
    """ Test the TimeSeriesRegressor """

    file_name = sys.argv[1]
    feature = " ".join(sys.argv[2].split("_"))
    model = sys.argv[3]
    extra = int(sys.argv[4])  # 0 if we don't want to use extra features, 1 otherwise

    df = pd.read_csv(file_name, index_col=0)
    df = shitty_preprocessing(df)

    latest_week = int(df['week'].max()) + 1
    df_train = df[['Player Id', 'week', feature]]
    ts_reg = TimeSeriesRegressor(df_train)
    new_df = ts_reg.complete_data(df_train)

    print("--------------------------------------------------------")
    print("Training Regressor...")

    ts_reg.train(new_df, feature, latest_week, extra)

    print("--------------------------------------------------------")
    print("Predicting...")

    # Predict the next time step
    preds, new_df = ts_reg.predict(new_df, feature, model)
    latest_week = int(new_df['week'].max()) + 1

    # Store new dataframe
    print(new_df)
    fout = file_name.split("/")[2]
    new_df.to_csv(f"new_{fout}")


if __name__ == "__main__":
    main()