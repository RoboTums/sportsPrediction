import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def findOptPortfolio():
	DF = pd.read_csv('gurobiTime.csv',index_col=0)
	DF = DF.drop(['player.1'],axis=1)


	positionDF = []
	num_positions = len(DF['position'].unique())


	for i in DF['position'].unique():
	    positionDF.append( (DF[DF['position'] == i]) )

	    
	bigDict = {}
	for df_iter in range(len(positionDF)):
	    
	    currPosition = positionDF[df_iter]  
	    for player in currPosition.index:
	        returnPlayer = currPosition.loc[player]['mean']
	        playerVariance = currPosition.loc[player]['variance']
	        playerCost = currPosition.loc[player]['cost']
	        position = currPosition.loc[player]['position']
	        bigDict[player] = [returnPlayer, playerVariance, playerCost , position]
	         
	player, returnPlayer, playerVariance, playerCost , position = gp.multidict(bigDict)


	positionPlayerConstraintVars = gp.tuplelist( [(position.select(playr)[0],playr) for playr in player])

	playerDecisionVars = gp.tuplelist( [ (choice, returnPlayer.select(choice)[0], playerVariance.select(choice)[0])  for choice in player ] )

	positionConstraints = gp.tupledict( { 'QB':1, "RB":3,"WR":3, 'TE':1,"DEF":1})#'K':0} )


	try:
	    m = gp.Model('QMIP')
	    #add binary decision variables on whether 
	    
	    
	    #add decision variables
	    players = m.addVars( playerDecisionVars, vtype=GRB.BINARY, name='players') 
	    
	    
	    
	    #add position constraint. We set upper bound to be one and keep the constraint a continuous variable since the 
	    #model will keep integer solutions
	    
	    #positionConstratint = m.addVars(positionConstraints, ub =1 , name='positionConstr')
	    
	    
	    # set objective funcition
	    m.setObjective(  - gp.quicksum( returnPlayer[choice] *players[choice,retrn,var] for choice, retrn, var in players ) 
	                   + gp.quicksum( playerVariance[choice] *players[choice,retrn,var] for choice, retrn, var in players ) 
	                   ,GRB.MINIMIZE)
	    
	    #constraint: sum of each position must be a constraint:
	    #for pos in DF['position'].unique():
	        
	        
	    posConstrs = {} 
	    for pos in DF['position'].unique():
	        players_in_pos = [playr for playr in player if position[playr] == pos]
	        
	        posConstrs[pos] = m.addConstr( gp.quicksum( [ players.select(playr)[0] for playr in player if position[playr] == pos ] ) == positionConstraints[pos] )
	        
	  #  modelPositionConstr =  m.addConstrs(  players.sum(position[choice] ) ==  positionConstraints [ position[choice] ] for choice ,_,_ in players  ) 
	    
	    #salary constraint
	    costConstr = m.addConstr(gp.quicksum( [ playerCost[playr] *players.select(playr)[0] for playr in player ]) <= 50000 )
	    
	    
	    m.update()
	    
	    m.optimize()
	    
	    findPlayers = m.X
	    
	    
	    teamRoster = [ player[i] for i in range(len(player)) if findPlayers[i]==1 ]
	    print('\noptimal team roster: ',teamRoster)
	    
	    sum = 0
	    for i in teamRoster:
	        sum += playerCost[i] 
	    sum 


	    constr = { 'QB':1, "RB":3,"WR":3, 'TE':1,"DEF":1} 

	    for pos in DF['position'].unique():
	        for i in teamRoster:
	            if position[i] == pos:
	                constr[pos] -= 1
	    for key,value in constr.items():
	        if value != 0:
	            print('failure to meet pos constraint:',key)
	    print('valid roster, cost =',sum)    
	    
	    print('\n expected return - risk :',m.objVal)

	except gp.GurobiError as e:
	    print('Error code ' + str(e.errno) + ": " + str(e))


findOptPortfolio()