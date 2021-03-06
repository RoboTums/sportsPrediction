
#script that compiles X and Y datasets.

import pandas as pd 
import os 



def findNans(df):
	sum = 0
	for col in df.columns:
		sum += df[col].isna().sum()
	return sum

def mergeXandY(filename):
	#nasty bunch of cases for this 
	if filename == 'QB.csv':
		Xstats = pd.read_csv('./NFL-Statistics-Scrape/Game_Logs_Quarterback.csv')
	elif filename == 'RB.csv':
		Xstats = pd.read_csv('./NFL-Statistics-Scrape/Game_Logs_Runningback.csv')
	elif filename == 'TE.csv' or filename == "WR.csv":
		Xstats = pd.read_csv('./NFL-Statistics-Scrape/Game_Logs_Wide_Receiver_and_Tight_End.csv')
	elif filename == 'DEF.csv':
		defStats = pd.read_csv(filename,index_col=0)
		defStats= defStats.rename({'name':'Player Id'}, axis='columns')
		filename = 'finalized_' + filename
		defStats.to_csv(filename)
		print('\n created CSV for' ,filename)
		print('\n found Nans:', findNans(defStats))
		return
	else:
		print('file:' ,filename,'not found, moving on.')
		return
	#data cleaning of the Y sets. Helps with merge.
	Ystats = pd.read_csv(filename)
	Xstats = Xstats.rename({'Name':'name','Week':'week', 'Opponent':'opponent'},axis='columns')

	basicXstats =  pd.read_csv('./NFL-Statistics-Scrape/Basic_Stats.csv')
	basicXstats = basicXstats.rename({'Name':'name'}, axis='columns')
	Xstats2019 = Xstats[Xstats['Year']==2019]
	Ystats['opponent'] =[x.upper() for x in Ystats['opponent']]


	#gotta merge 3 datasets, involving two merges.
	specializedY = pd.merge(right = Ystats , left = Xstats2019,how='inner') 

	generalizedY = pd.merge(right=  basicXstats, left=specializedY,how='inner')

	print('\n created CSV for' ,filename)
	print('\n found Nans:', findNans(generalizedY))
	filename = 'finalized_' + filename #get rid of .csv
	generalizedY.to_csv(filename)



allCSV = [x for x in os.listdir() if ".csv" in x]

for filename in allCSV:
	mergeXandY(filename)
