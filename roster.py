import pandas as pd



# Roster class:

class Roster:
	def __init__(self,Sport,bettingWebsite):
		self.sport = Sport
		self.bettingWebsite = bettingWebsite
		#scoring formula returns a function. 
		self.scoringFormula = scoringFormula(Sport,bettingWebsite) 
	
	def generateRoster():
		#import data 
		#fit predicted scores
		#fit projections as baseline
		# do any ensemble modelling
		#optimize roster 
		return 'TBD'
		
	