import pandas as pd 
import os
#returns DF of prediction.
#DF: pd dataframe, Position: str. Windowsize: Int
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

def strExcludeMedian(DF):
	newDF = pd.DataFrame()
	#print(DF['position'])

	return newDF

#List of Predictions: List[pd Dataframes]
# outputs whole portfolio of players. 
def generatePortfolioDataSet(listOfPredictions):

	bigKahuna = pd.DataFrame()

	for dataset in listOfPredictions:
		bigKahuna = pd.concat([bigKahuna, dataset])


	joiner = pd.concat([bigKahuna['position'], bigKahuna['player']],axis=1)
	#print('yike:',joiner.columns)

	bigKahuna = bigKahuna.groupby('player')
	joiner = joiner.drop_duplicates(subset=['player'])
	joiner = joiner.sort_values(by=['player'])
	joiner.index = joiner['player']
	yike = bigKahuna.median()
	#print(bigKahuna['mean'].median())
	bigKahuna = pd.concat([yike,joiner],axis=1)

	return bigKahuna

def main():

	predictionRaw = [x for x in os.listdir() if 'finalized' in x] 

	#sort predictions for sanity
	predictionRaw.sort()
	predictionClean = [ x[10:].strip('.csv') for x in predictionRaw]
	portfolioData = []

	for pos, data in enumerate(predictionRaw):
		positionDataframe = pd.read_csv(data,index_col=0)
		portfolioData.append(predict(positionDataframe,predictionClean[pos]))

	optAlgInput = generatePortfolioDataSet(portfolioData)
	
	#optAlgInput = optAlgInput.drop(['player.1'],axis=1)
	optAlgInput.to_csv('gurobiTime.csv')
	#print(optAlgInput)
	 

main()

