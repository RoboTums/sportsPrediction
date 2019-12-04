
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
	if filename == 'RB.csv':
		Xstats = pd.read_csv('./NFL-Statistics-Scrape/Game_Logs_Runningback.csv')
	if filename == 'PK.csv':
		Xstats = pd.read_csv('./NFL-Statistics-Scrape/Game_Logs_Punters.csv')
	if filename == 'TE.csv' or filename == "WR.csv":
		Xstats = pd.read_csv('./NFL-Statistics-Scrape/Game_Logs_Wide_Receiver_and_Tight_End.csv')
	if filename == 'DEF.csv':
		Xstats = pd.read_csv("./NFL-Statistics-Scrape/Game_Logs_Defensive_Lineman.csv")
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
