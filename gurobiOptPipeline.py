import pandas as pd 
import os
#returns DF of prediction.
#DF: pd dataframe, Position: Int. Windowsize: Int
def predict(DF, position, windowsize=5):
	portfolio = pd.DataFrame()
	#expected value	
	portfolio['mean'] = DF['points'].rolling(windowsize,win_type='boxcar', min_periods=1).mean()
	#standard deviation
	portfolio['std'] = DF['points'].rolling(windowsize, min_periods=1).std()
	#variance
	portfolio['variance'] = DF['points'].rolling(windowsize, min_periods=1).var()
	#X identifier
	portfolio['player'] = DF['Player Id']
	portfolio['position'] = pd.Series([position]*len(portfolio['player']))
	portfolio['cost'] = DF['salary'].rolling(5, min_periods=1).max()

	portfolio = portfolio.dropna()



	return portfolio

#List of Predictions: List[pd Dataframes]
# outputs whole portfolio of players. 
def generatePortfolioDataSet(listOfPredictions):

	bigKahuna = pd.DataFrame()

	for dataset in listOfPredictions:
		bigKahuna = pd.concat([bigKahuna, dataset])

	bigKahuna = bigKahuna.groupby('player')
	return bigKahuna.median()

def main():

	predictionRaw = [x for x in os.listdir() if 'finalized' in x] 

	#sort predictions for sanity
	predictionRaw.sort()

	portfolioData = []

	for pos, data in enumerate(predictionRaw):
		positionDataframe = pd.read_csv(data)
		portfolioData.append(predict(positionDataframe,pos))

	optAlgInput = generatePortfolioDataSet(portfolioData)

	#print(optAlgInput)
	 

main()