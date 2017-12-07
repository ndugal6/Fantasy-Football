import pandas as pd
import os, errno
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import metrics


""""
FINAL VERSION OF AI PROJECT FOR SUBMISSION. REALER THAN REAL MY DUDES. NICHOLAS DUGAL.
"""
# 2017 data is in 2017data dir
##Linear regression csv path calls may break when connectioning to database
def main():
    # actualOld = pd.read_csv('Data/' + 'qb' + '/' + '/averagedDataWithDefense.csv',index_col=None)
    # numeric = removeAlphaData(actualOld)
    # print(numeric.describe())
    # numeric2 = numeric.astype(dtype='float32',copy=True,errors='ignore')
    # print(numeric2.head());exit(0)
    # prediction = linearRegression(position='qb', player='Drew Brees', feature='Pass Yards',Future=True)
    pathToQBS = '~/documents/fantasy-football/2017data/qb'
    maybeQBs = os.listdir(pathToQBS)
    defQBS = []
    for maybe in maybeQBs:
        if os.path.isdir(pathToQBS + '/' + maybe):
            defQBS.append(maybe)
    qbPreditionList = []
    



def predict(name):
    actualOld = pd.read_csv('Data/' + 'qb' + '/' + '/actualDataWithDefense.csv')
    actualNew = pd.read_csv('2017Data/' + 'qb' + '/' + '/actualDataWithDefense.csv')
    actual = pd.concat([actualOld, actualNew])
    toDrop = ['Point After', 'Fumble Returns', 'Fumble TD', 'Week', 'Away Games_x', 'Rush 2PT', 'Away Games_y',
              'Safety', 'Receiving 2PT', 'Blocks', 'Pass 2PT', 'Year']
    cols = list(purgeAlphas(actual.columns))
    print(cols)
    features = list(set(cols).difference(toDrop))
    predictionDF = pd.DataFrame(columns=((features.append("Name"))))
    predictionDF['Name'] = ['Drew Brees']
    features.remove('Name')
    print(features)
    for feature in features:
        prediction = linearRegression(position='qb',player='Drew Brees',feature=feature,Future=True)
        print(prediction)
        predictionDF[feature] = prediction.tolist()
    return predictionDF
    print(predictionDF.head())


#Delete. For reference. Get Qbs for certain year
def playersForYear():
    qbs = []
    path = 'Data/' + 'qb'
    files = os.listdir(path)
    for file in files:
        if os.path.isdir(path + '/' + file):
            subFiles = os.listdir(path + '/' + file)
            if '2016' in subFiles:
                qbs.append(path + '/' + file)
#For reference. Should be deleted
def makePredictionDF():
    actual = pd.read_csv('Data/' + 'qb' + '/' + 'Drew Brees' + '/actualDataWithDefense.csv')
    features = list(purgeAlphas(actual.columns))
    predictionDF = pd.DataFrame(columns=((features.append("Name"))))
    predictionDF['Name'] = ['Drew Brees'] * 2
    features.remove('Name')
    for feature in features:
        prediction, mean2, variance = linearRegression(position='qb', player='Drew Brees', feature=feature)
        predictionDF[feature] = prediction.tolist()
    print(predictionDF.head())

#Parameters:    dirtyData - pandas.DataFrame | feature - string | year - int | week - int
# Defaults:     none | 'Pass Yards' | 2016 | 8
# cleans non-numeric data columns, and splits on boundaries
# returns train_X,train_Y,test_X,test_Y DataFrame/Series/DataFrame/Series for input into linear Regression model
def train_test_divider(dirtyData, feature='Pass Yards',year=2016,week=8,Future=False):
    data = removeAlphaData(dirtyData)
    x_features = list(set(list(data.columns)).difference([feature]))
    toDrop = ['Point After', 'Fumble Returns', 'Fumble TD','Week','Away Games_x', 'Rush 2PT','Away Games_y',
              'Safety','Receiving 2PT','Blocks','Pass 2PT','Year']

    x_features = list(set(x_features).difference(toDrop))
    data = dirtyData.copy(deep=True)
    #x_data
    train_data = data[(data.Year <= year) & (data.Week <= week)].copy(deep=True)
    test_data = data[(data.Year >= year) & (data.Week > week)].copy(deep=True)
    #For Future Predictions we do the following
    if Future:
        test_data = data[(data.Year == 2017)][x_features].copy(deep=True)
        x_test = np.mean(test_data)
        x_train = data[x_features].copy(deep=True)
        y_train = data[feature].copy(deep=True)
        return x_train,y_train,x_test


    # test_data = scaler.fit(train_data)
    #

    scaler = StandardScaler()
    #x_data= x_data[:, np.newaxis]
    #x_data = x_data.apply(scaler.fit,axis=1)
    
    x_train = train_data[x_features]
    x_test = test_data[x_features]
    y_test = test_data[feature]
    y_train= train_data[feature]

    return x_train, x_test, y_train, y_test
    # train_data = train_data.as_matrix().astype(np.float)
    # test_data = test_data.as_matrix().astype(np.float)
    x_train = train_data[x_features].as_matrix().astype(np.float)
    # x_train= x_train[:, np.newaxis]
    x_test = test_data[x_features].as_matrix().astype(np.float)
    # x_test = x_test[:, np.newaxis]
    y_train = train_data[feature].as_matrix().astype(np.float)
    # y_train = y_train[:, np.newaxis]
    y_test = test_data[feature].as_matrix().astype(np.float)
    return x_train,x_test,y_train,y_test

#Predicts the stat for a player
#Parameters player: player name as a a string - Default 'Drew Brees'
#           feature: the feature to predict as a string - Default 'Pass Yards'
#           position: string - options 'QB','RB','TE','WR','K'
#Output: feature prediction | mean squared error | explained variance
def linearRegression(position='qb',player='Drew Brees', feature='Pass Yards',Future=False):
    print('Position:\t'+position)
    print('Player:\t'+player)
    print('Feature:\t'+feature)
    # Get actual stats and average stats for input
    actualOld = pd.read_csv('Data/'+position+'/'+player+'/actualDataWithDefense.csv')
    actualNew = pd.read_csv('2017Data/' + 'qb' + '/' +player+ '/actualDataWithDefense.csv')
    actual = pd.concat([actualOld,actualNew])
    averagesOld = pd.read_csv('Data/'+position+'/'+player+'/averagedDataWithDefense.csv')
    averageNew = pd.read_csv('2017Data/' + 'qb' + '/' + player+'/actualDataWithDefense.csv')
    averages = pd.concat([actualOld,actualNew])
    # Get the combo we need
    data = removeAlphaData(inputForFeature(actual,averages,feature))
    data.round(4)
    data = data.apply(lambda x: pd.to_numeric(x,errors='ignore'), axis=1)

    # boom linear model ready to go.. not so difficult :p
    reg = linear_model.LinearRegression()
    #create training and tesitng data
    if Future:
        x_train, y_train, x_test = train_test_divider(dirtyData=data, feature=feature, year=2017, week=10,Future=True)
        model = reg.fit(x_train, y_train)
        prediction = model.predict(x_test.values.reshape(1, -1))
        return prediction
    x_train,x_test,y_train,y_test = train_test_divider(dirtyData=data,feature=feature,year=2017,week=10)

    # X_train = preprocessing.scale(training_data[list(xFeatures)].values.reshape(-1, len(xFeatures)))




    # Fit our line with the data.. Magic happens

    model = reg.fit(x_train, y_train)

    # Predict the value of your dreams
    if(len(x_test) == 0):
        return 0,0,0
    prediction = model.predict(x_test)
    if Future:
        return prediction
    # computes mean squared error
    mean2Err = mean_squared_error(y_test, prediction)

    # explained variance, different than typical variance. Google it if not understood
    varianceScore = r2_score(y_test, prediction)
    predDict = {'Predicted':prediction.tolist(),'Real':y_test.tolist()}
    pred = pd.DataFrame(predDict)

    return prediction, mean2Err, varianceScore



def makePositionsAveraged():
    for pos in ['rb','wr','te','dst','qb','k']:
        createAveragedData(pos=pos)

def other(X_train,X_test,y_train,y_test):
    unscaled_clf = make_pipeline(PCA(n_components=2), GaussianNB())
    unscaled_clf.fit(X_train, y_train)
    pred_test = unscaled_clf.predict(X_test)

    # Fit to data and predict using pipelined scaling, GNB and PCA.
    std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
    std_clf.fit(X_train, y_train)
    pred_test_std = std_clf.predict(X_test)

    # Show prediction accuracies in scaled and unscaled data.
    print('\nPrediction accuracy for the normal test dataset with PCA')
    print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))

    print('\nPrediction accuracy for the standardized test dataset with PCA')
    print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_std)))

    # Extract PCA from pipeline
    pca = unscaled_clf.named_steps['pca']
    pca_std = std_clf.named_steps['pca']

    # Show first principal componenets
    print('\nPC 1 without scaling:\n', pca.components_[0])
    print('\nPC 1 with scaling:\n', pca_std.components_[0])

    # Scale and use PCA on X_train data for visualization.
    scaler = std_clf.named_steps['standardscaler']
    X_train_std = pca_std.transform(scaler.transform(X_train))

    # visualize standardized vs. untouched dataset with PCA performed
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=FIG_SIZE)

    for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
        ax1.scatter(X_train[y_train == l, 0], X_train[y_train == l, 1],
                    color=c,
                    label='class %s' % l,
                    alpha=0.5,
                    marker=m
                    )

    for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
        ax2.scatter(X_train_std[y_train == l, 0], X_train_std[y_train == l, 1],
                    color=c,
                    label='class %s' % l,
                    alpha=0.5,
                    marker=m
                    )

    ax1.set_title('Training dataset after PCA')
    ax2.set_title('Standardized training dataset after PCA')

    for ax in (ax1, ax2):
        ax.set_xlabel('1st principal component')
        ax.set_ylabel('2nd principal component')
        ax.legend(loc='upper right')
        ax.grid()

    plt.tight_layout()

    plt.show()
# Takes DF of average values, DF of actual values
# Returns a dataframe formatted properly for a given Feature
# try df.head() & df[feature] on returned value where you use it... Weird huh
def inputForFeature(actual,average,feature):
    allFeatures = list(average.columns)
    avgFeats = set(allFeatures).difference([feature])
    df = pd.DataFrame(columns=allFeatures)
    for avg in avgFeats:
        df[avg] = average[avg]
    df[feature] = pd.to_numeric(actual[feature], downcast='float')
    df.drop('Unnamed: 0',axis=1,inplace=True)
    return df
# Creates dfs saved as csv for each player that has their average for each week for each year,
# Each level is saved meaning directory structure is: data -> position (csv for all players) -> specificPlayer -> (csv for all years) -> specificYear (csv for all weeks)
def createAveragedData(pos='k',dataPath='/Users/nickdugal/documents/fantasy-football'):

    posPath = dataPath + '/'+pos     # Path to position data
    ensure_dir(posPath)            # Create path if none exists
    os.chdir(posPath)
    updatedPoslist = []                  # This will hold updated df per player to concate and make updated qb DF
    df = pd.read_csv(dataPath+'/'+pos+'2017.csv')
    grouping = 'Name'
    if pos.upper() == 'DST':
        grouping = 'Team'
    for qbName, data in df.groupby(grouping): #Iterate through df by names, qb=name and data is df for that name
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
        updatedPoslist.append(pDF)
    posDF = pd.concat(updatedPoslist)

    posDF.to_csv(posPath+'/AveragedData.csv')
    df.to_csv(posPath+'/ActualData.csv')

   #Uncomment below after testing
        # updatedYearList = [makeAverage(year,subData) for year, subData in data.groupby('Year')]
# Crawling my data directory and and merges offensive data with defensive data
# Not flexible with alternate directory
# Could make it recurse & flexible using a call to check is a file path is actually a dir
def mergeOffenseDefenseAverages(dataPath='/Users/nickdugal/documents/fantasy-football/2017data'):
    actDefense = pd.read_csv(dataPath+'/dst/actualData.csv')
    avgDefense = pd.read_csv(dataPath+'/dst/averagedData.csv')
    positions = ['k','qb','wr','te','rb']

    for pos in positions:


        positionPath = dataPath+'/'+pos
        players = os.listdir(positionPath+'/')
        # os.chdir(playerPath)
        mergeAndSave(positionPath)
        for player in players:
            if player.endswith('.csv') or player.endswith('.DS_Store'):
                continue
            else:
                playerPath = positionPath+'/'+player
                mergeAndSave(playerPath)
                years = os.listdir(playerPath+'/')
                for year in years:
                    if year.endswith('.csv') or year.endswith('.DS_Store'):
                        continue
                    else:
                        yearPath = playerPath + '/' + year
                        mergeAndSave(yearPath,year=year)
# Takes in a path to a directory with actual and average csv, merges with defense data
# hard coded for my directory. Will break if not adjusted
def mergeAndSave(dirPath,year=0,defensePath='/Users/nickdugal/documents/fantasy-football/2017data'):
    dataPath = defensePath
    actDefense = pd.read_csv(dataPath + '/dst/actualdata.csv')
    avgDefense = pd.read_csv(dataPath + '/dst/averagedData.csv')
    if not year == 0:
        actDefense = actDefense[actDefense.Year == int(year)]
        avgDefense = avgDefense[avgDefense.Year == float(year)]
    actual = pd.read_csv(dirPath + '/actualData.csv')
    averaged = pd.read_csv(dirPath + '/averagedData.csv')

    actualCombined = pd.merge(actual, actDefense, left_on=['Year', 'Week', 'Opponent'],
                              right_on=['Year', 'Week', 'Team'])
    averageCombined = pd.merge(averaged, avgDefense, left_on=['Year', 'Week', 'Opponent'],
                               right_on=['Year', 'Week', 'Team'])
    try:
        actualCombined.drop(['Unnamed: 0_x',"Unnamed: 0_y"],axis=1,inplace=True)
        averageCombined.drop(['Unnamed: 0_x', "Unnamed: 0_y"], axis=1, inplace=True)
    except:
        pass
    actualCombined.to_csv(dirPath + '/actualDataWithDefense.csv')
    averageCombined.to_csv(dirPath + '/averagedDataWithDefense.csv')

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
    cleaned = (set(list(unCleaned))).difference(['Name', 'Opponent', 'Position', 'Opponent_x', 'Position_x', 'Team', 'Opponent_y', 'Position_y','Unnamed: 0','Unnamed: 0.1_x',
                                                 'Unnamed: 0.1.1_x','Unnamed: 0.1_y','Unnamed: 0.1.1_y'])
    return cleaned

# Removes the nonNumerical columns of a DF, helper method for purgeAlphas
def removeAlphaData(unCleanedDF):
    return unCleanedDF[list(purgeAlphas(unCleanedDF))]


if __name__ == '__main__': main()
