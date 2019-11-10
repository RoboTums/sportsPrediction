from selenium import webdriver
import pandas as  pd 


#simple script to get the updated weeks. Writes a .CSV file for each position with predicted fantasy points. 
#takes about 20 mins to run for one season. 




#generates fanduel fantasy urls to scrape for a season
#takes in integer for latest week.
#returns list of urls to driver.get()
def generateScrapingUrls(latestWeek):
    output = []
    for i in range(1,latestWeek):
        output.append( 'http://rotoguru1.com/cgi-bin/fyday.pl?week='+str(i)+'&game=fd')
    print(output)
    return output
    
def getDataOfSeason(weeks):
    driver = webdriver.Firefox()

    #set up dataframes per position
    QB = pd.DataFrame(columns=['name' ,'team', 'opponent','points', 'salary','week'])
    RB = pd.DataFrame(columns=['name' ,'team', 'opponent','points', 'salary','week'])
    WR = pd.DataFrame(columns=['name' ,'team', 'opponent','points', 'salary','week'])
    TE = pd.DataFrame(columns=['name' ,'team', 'opponent','points', 'salary','week'])
    PK = pd.DataFrame(columns=['name' ,'team', 'opponent','points', 'salary','week'])
    Def = pd.DataFrame(columns=['name' ,'team', 'opponent','points', 'salary','week'])
    # put Dataframes in list
    DataFrames = [ QB,RB,WR,TE,PK,Def]

    positionsInDataFrames=[0,0,0,0,0,0] # represents each of the last indexs in the dataframes. 
    seasonWeek = 0  #week for dataframes, not the link. 


    for week in weeks: #parse thru urls

        driver.get(week)
        htmlTable = driver.find_element_by_xpath('/html/body/table/tbody/tr/td[3]/p[6]/table/tbody/tr/td[1]/table')
        allRows =htmlTable.find_elements_by_xpath('/html/body/table/tbody/tr/td[3]/p[6]/table/tbody/tr/td[1]/table/tbody/tr')
        # get tables
        seasonWeek += 1 # increment season week
        skipNext = False #helps with parsing
        currentPosition = -1 #reset position

        for row in allRows:
            if skipNext == True: #check if "bad row"
                skipNext = False 
                continue
            if row.text == "Jump to:     QB   |   RB   |   WR   |   TE   |   PK   |   Def   |":  #check if seperator row
                currentPosition += 1
                skipNext = True
                index = 0

            else:
                #add to list, legitimate player. String Parsing. 
                nameList = row.find_elements_by_xpath('td')
                name = nameList[0].text
                team = nameList[1].text
                opponent = nameList[2].text[2:]
                points = nameList[3].text
                salary = nameList[4].text[1:]
                if salary == "/A":
                    salary = 0
                else:
                    salary = nameList[4].text[1:]

                    salary=int(salary.replace(',',''))
                            
                appending = [name,team,opponent,points, salary, seasonWeek]
                
                DataFrames[currentPosition].loc[positionsInDataFrames[currentPosition]] = appending
                
                positionsInDataFrames[currentPosition] += 1 

    dfNames=  ['QB','RB',"WR",'TE',"PK",'DEF']
    for i in range(len( DataFrames)):
        DataFrames[i].to_csv(dfNames[i]+'.csv')

getDataOfSeason( generateScrapingUrls(9) )