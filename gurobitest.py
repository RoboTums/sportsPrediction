import gurobipy
import pandas as pd 
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import sqrt



data = pd.read_csv('gurobiTime.csv', index_col=0)
players = data.index  #think of these like assests.

# calculate basic summary stats for players
player_volatility = data['variance']
player_return = data['mean']
print(player_volatility.head())
#create empty model
m = gp.Model('Markowitz')
#add an empty variable for each player
vars = pd.Series(m.addVars(data.index))

#objective is to minimize risk (squared). this is modeled
#by the covariance matrix, which measures the historical 
#correlation between players

sigma = data.T.cov()

portfolio_risk = sigma.dot(vars).dot(vars)
#print(type(portfolio_risk))	

m.setObjective(portfolio_risk, GRB.MINIMIZE)

# Fix Budget with a constraint
m.addConstr(vars.sum()==1,'Budget')


#Optimize Model to find min risk fantasy team

m.setParam("OutputFlag",0)
m.optimize()


fantasyTeamReturn = player_return.dot(vars)

#Dispay min risk portfolio
print('Minimum Risk Portfolio:\n')
for v in vars:
    if v.x > 0:
        print('\t%s\t: %g' % (v.varname, v.x))
#print(portfolio_risk.getValue())
minrisk_volatility = sqrt(portfolio_risk.getValue())
print('\nVolatility      = %g' % minrisk_volatility)
minrisk_return = fantasyTeamReturn.getValue()
print('Expected Return = %g' % minrisk_return)

#we want to take at least minimum risk vs Reward
target = m.addConstr(fantasyTeamReturn == minrisk_return,'target')

# Solve for efficient frontier by varying target return
frontier = pd.Series()
for r in np.linspace(player_return.min(), player_return.max(), 100):
    target.rhs = r
    m.optimize()
    frontier.loc[sqrt(portfolio_risk.getValue())] = r

frontier.to_csv('lookie.csv')

# Plot volatility versus expected return for individual players
ax = plt.gca()
ax.scatter(x=player_volatility, y=player_return,
           color='Blue', label='Individual Stocks')

#annotation for plotting.

#for i, player in enumerate(players):
#	print(player, player_volatility[i], player_return[i])
	#ax.annotate(player, (player_volatility[i], player_return[i]))

# Plot volatility versus expected return for minimum risk portfolio
ax.scatter(x=minrisk_volatility, y=minrisk_return, color='DarkGreen')
ax.annotate('Minimum\nRisk\nPortfolio', (minrisk_volatility, minrisk_return),
            horizontalalignment='right')

# Plot efficient frontier
frontier.plot(color='DarkGreen', label='Efficient Frontier', ax=ax)
#print(frontier)
# Format and display the final plot
ax.axis([0.005, 20.06, -0.02, 20.025])
ax.set_xlabel('Volatility (standard deviation)')
ax.set_ylabel('Expected Return')
ax.legend()
ax.grid()
plt.savefig('portfolio.png')
print("Plotted efficient frontier to 'portfolio.png'")