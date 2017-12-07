import pandas as pd
import os, errno
import numpy as np


""""
FINAL VERSION OF AI PROJECT FOR SUBMISSION. REALER THAN REAL MY DUDES. NICHOLAS DUGAL
"""

def main():
    for pos in ['rb','wr','te','dst']:
        createAveragedData(pos=pos)

def makePositionsAveraged():
    for pos in ['rb','wr','te','dst','qb']:
        createAveragedData(pos=pos)

# Creates dfs saved as csv for each player that has their average for each week for each year,
# Each level is saved meaning directory structure is: data -> position (csv for all players) -> specificPlayer -> (csv for all years) -> specificYear (csv for all weeks)
def createAveragedData(pos='qb',dataPath='/Users/nickdugal/documents/fantasy-football/data'):

    posPath = dataPath + '/'+pos     # Path to position data
    ensure_dir(posPath)            # Create path if none exists
    os.chdir(posPath)
    updatedQBlist = []                  # This will hold updated df per player to concate and make updated qb DF
    df = pd.read_csv(dataPath+'/qballyears.csv')
    for qbName, data in df.groupby('Name'): #Iterate through df by names, qb=name and data is df for that name
        pPath = posPath+'/' + qbName    # Player Path
        ensure_dir(pPath)                #Create a directory for each qb if none exists to hold their data
        os.chdir(pPath)
        updatedYearList = []            #This will hold each updated df per year to concate & make updated player DF
        for year, subData in data.groupby('Year'): #iterate through each qb to evaluate their data by year
            yPath = pPath  +'/'+str(year)
            ensure_dir(yPath)

            yearDF = makeAverage(year, subData)

            yearDF.to_csv(yPath+'/AveragedData.csv')
            subData.to_csv(yPath+'/ActualData.csv')
            updatedYearList.append(yearDF)
        pDF = pd.concat(updatedYearList)

        pDF.to_csv(pPath+'/AveragedData.csv')
        data.to_csv(pPath+'/ActualData.csv')
        updatedQBlist.append(pDF)
    posDF = pd.concat(updatedQBlist)

    posDF.to_csv(posPath+'/AveragedData.csv')
    df.to_csv(posPath+'/ActualData.csv')

   #Uncomment below after testing
        # updatedYearList = [makeAverage(year,subData) for year, subData in data.groupby('Year')]



# Takes in an individual year's worth of data, return it corrected as averages as DF
def makeAverage(year, yearData):

    columns = purgeAlphas(yearData.columns)
    newDF = pd.DataFrame(columns=yearData.columns, index=yearData.index)

    for column in columns:
        dataVector = yearData[column]
        averagedList = []
        for val in range(len(dataVector)):
            averagedList.append(np.mean(dataVector[0:val]))
        newDF[column] = averagedList

    #We also need to retain non numerical data associated with numericals
    alphaColumns = (set(yearData.columns)).difference(columns)

    for alpha in alphaColumns: #So we run the same as above, except don't do math, just copy data over
        newDF[alpha] = yearData[alpha]
    newDF['Week'] = yearData['Week']
    firstWeek = (newDF['Week'].tolist())[0]
    newDF[newDF.Week == firstWeek] = yearData[yearData.Week == firstWeek]

    return newDF



# Creates a directory for the file path given if none exists
def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    directory = os.path.dirname(file_path)  #there's a race condition - if the directory is created between the os.path.exists
    try:                                    # and the os.makedirs calls, the os.makedirs will fail with an OSError.
        os.makedirs(directory)              # So I'll trap the error and manually inspect embedded error code
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# Takes columns of a DF and removes the currently known nonNumerical values
def purgeAlphas(unCleaned):
    cleaned = (set(list(unCleaned))).difference(['Name', 'Opponent', 'Position', 'Opponent_x', 'Position_x', 'Team', 'Opponent_y', 'Position_y'])
    return cleaned

# Removes the nonNumerical columns of a DF, helper method for purgeAlphas
def removeAlphaData(unCleanedDF):
    return unCleanedDF[list(purgeAlphas(unCleanedDF))]


if __name__ == '__main__': main()
