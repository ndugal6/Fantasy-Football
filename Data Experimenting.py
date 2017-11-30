import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import operator
import argparse
from collections import OrderedDict
from collections import Counter
from sklearn import datasets, linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
#Nicholas Dugal

# ?
## for scaling data before input use
# data = model.Scale(data, data, scale=float(1./256))
# see mnist file line 78

# This is a file for experimenting with Fantasy Foootball data!

# I frequently need to copy paste code for trials in jupyter notebook or python console. This is the reason for many
# methods that seem useless or repetitive. That's on purpose.
# The Method that we are most concerned with

# This returns a data frame of a particular player, set save=True to save it
def getPlayerWithName(pName, position, save=False, sorted=True):
    # change working directory to the correct one. If you're unsure what your current dir is then run (os.getcwd())
    os.chdir('/Users/nickdugal/Documents/Fantasy-Football/data/Updated NFL Data Sets/Indexed Data')
    players_data = pd.read_csv(position + "AllYears.csv")
    players_data.reindex(index=['Year', 'Week', 'Name'])
    playerDataFrame = players_data[players_data.Name == pName]
    if save:
        playerDataFrame.to_csv(pName + ".csv")
    if sorted:
        players_data.sort_index(inplace=True)
    return playerDataFrame


# method name says it all
def combineOffensePositionWithDefense(oPosition, save=True):
    # change working directory to the correct one. If you're unsure what your current dir is then run (os.getcwd())
    os.chdir('/Users/nickdugal/Documents/Fantasy-Football/data/Updated NFL Data Sets/Indexed Data')
    players_data = pd.read_csv(oPosition + "AllYears.csv")
    defense_data = pd.read_csv("dstallyears.csv")
    pd_data = pd.merge(players_data, defense_data, left_on=['Year', 'Week', 'Opponent'],
                       right_on=['Year', 'Week', 'Team'])
    if save:
        pd_data.to_csv(oPosition + "_with_Defense.csv")
    return pd_data


# delete this, I'm keeping for future other stuff
def heatMap():
    heatmap, xedges, yedges = np.histogram2d(X, Y)  # , bins=(64, 64))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.clf()
    plt.imshow(heatmap, extent=extent)
    plt.show()


# Runs linear regression on two variables. I can update if needed. May or may not break.
# X and Y should be two Pandas Series or numpy arrays, maybe also a regular list will work.
def linearRegression(X=None, Y=None):
    os.chdir('data')
    # rb_all = os.listdir(os.getcwd())
    rb_data = pd.read_csv('qbAllYears.csv')
    print(rb_data.columns)
    if (X == None, Y == None):
        Y = rb_data['Pass Yards']
        X = rb_data['Pass Attempts']
    # XSquared = list(map(lam))

    X = X[:, np.newaxis]
    Y = Y[:, np.newaxis]

    # Creates two Dataframes, one for training with random samples consistenting of 80% the total samples, one for testing with the remaining 20%
    X_train, X_test = train_test_split(X, test_size=.2)
    Y_train, Y_test = train_test_split(Y, test_size=.2)

    # Here we create and fit the linear regression model. The call to fit iteratively updates the coeffecients of our model until it's mean squared difference is within
    # a certain threshold
    reg = linear_model.LinearRegression()
    reg.fit(X_train, Y_train)

    # Here we throw the testing data into the model. This returns an array of predicted y values
    PassAttempts_prediction = reg.predict(X_test)

    print("The Coeffecients:\n ", reg.coef_)
    # Gives us the average square distance between predicted values and the actual values
    # Essentially, how far from correct my model is
    mean2Err = mean_squared_error(Y_test, PassAttempts_prediction)
    # Variance ranges from 0 to 1, 1 is peeeeerfect!
    # If I expect myself to get 4 answers wrong, but actually get 10 wrong:high variance(close to 0), high error
    # If I actually get 5 wrong: low variance(close to 1), medium error
    # If I actually get 0 wrong: higher variance, no error
    varianceScore = r2_score(Y_test, PassAttempts_prediction)
    reg.score()
    print("The Mean Squared Error\n", mean2Err)
    # Explained Variance Score: 1 is perfect prediction
    print('Variance Score: %.2f' % varianceScore)
    # Here we plot our testing data to get a look at it. Let's make sure a linear model is the model for us
    plt.scatter(X_test, Y_test, color='black')
    # Here we plot our model's line as defined during training against the testing data
    plt.plot(X_test, PassAttempts_prediction, color='blue', linewidth=3)

    plt.xticks()
    plt.yticks()
    # plt.annotate(('hmm','whatUP'))
    # Here we type code that belongs somewhere else
    logReg = linear_model.LogisticRegression()
    logReg.fit(X_train, Y_train)

    # This shows us what the values for our model is,
    # For N features, you have N+1 coefficients
    coef = reg.coef_[0].tolist()
    # This gives us the min and max values of our graph. Move along..
    xmin, _ = plt.xlim()
    _, ymax = plt.ylim()
    s = str('Our line is defined as \nhaving coefficient values of ' + str(coef) +
            '\nThe Mean Squared Error: ' + str(mean2Err) +
            '\nVariance Score: ' + str(varianceScore))
    plt.text(8, 30, s, rotation=40,
             horizontalalignment='center',
             verticalalignment='center',
             multialignment='center', color='red')

    # plt.scatter(X,Y, s=12,c='red',marker='.')
    # plt.xlim(-5,40)
    # plt.ylim(0,175)                         # xmin, xmax = xlim() use to get current values

    plt.title('Relationship Between Pass Attempts and Pass Completions')

    plt.ylabel('Pass Completions')
    plt.xlabel('Pass Attempts', multialignment='center')

    plt.show()



def multivariateLinearRegression( xFeatures, yfeature, file=None, players_data=None):

    os.chdir('/Users/nickdugal/Documents/Fantasy-Football/data')
    if players_data == None:
        try:
            players_data = pd.read_csv(file)
        except:
            print("Invalid file name or file path passed as argument")

    setList = set(list(players_data.columns))

    if not yfeature in setList:
        raise Exception(yfeature + "Feature doesn't exist. Did you mispell it or pass the from csv")
    if yfeature in set(xFeatures):
        xFeatures = set(xFeatures).difference([yfeature])
    for x in xFeatures:
        if not x in setList:
            raise Exception(x + "Feature doesn't exist. Did you mispell it or pass the from csv")


    X = players_data[list(xFeatures)].values.reshape(-1, len(xFeatures))
    Y = players_data[yfeature]

    # X_train, X_test, Y_train, Y_test = train_test_split(X, preprocessing.scale(Y))

    coeff_titles = list(xFeatures)
    coeff_titles.insert(0, 'Intercept')

    reg = linear_model.LinearRegression()

    model = reg.fit(X_train, Y_train)
    yfeature_prediction = model.predict(X_test)
    print("Predicting: ",yfeature)
    print("The Coeffecients:\n ")
    pattern = "%.2f"
    floatsstrings = [pattern % i for i in list(reg.coef_)]
    print(floatsstrings)


    mean2Err = mean_squared_error(Y_test, yfeature_prediction)
    varianceScore = r2_score(Y_test, yfeature_prediction)
    return (mean2Err, varianceScore)



def realRegression(xFeatures, yfeature,training_data,testing_data,name=None):
    os.chdir('/Users/nickdugal/Documents/Fantasy-Football/squared/allyears/')
    # training_data = pd.read_csv('trainingDF.csv')
    # testing_data = pd.read_csv("testingDF.csv")
    testY = pd.read_csv("testingDF.csv")
    testY = testY[testY.Name == name]

    setList = set(list(training_data.columns))

    if not yfeature in setList:
        raise Exception(yfeature + "Feature doesn't exist. Did you mispell it or pass the from csv")
    if yfeature in set(xFeatures):
        xFeatures = set(xFeatures).difference([yfeature])
    for x in xFeatures:
        if not x in setList:
            raise Exception(x + "Feature doesn't exist. Did you mispell it or pass the from csv")

    X_train = preprocessing.scale(training_data[list(xFeatures)].values.reshape(-1, len(xFeatures)))
    Y_train = training_data[yfeature]
    X_test = preprocessing.scale(testing_data[list(xFeatures)].values.reshape(-1, len(xFeatures)))
    Y_test = testY[yfeature]

    coeff_titles = list(xFeatures)
    coeff_titles.insert(0, 'Intercept')

    reg = linear_model.LinearRegression()

    model = reg.fit(X_train, Y_train)
    yfeature_prediction = model.predict(X_test)
    # print("Predicting: ", yfeature)
    # print("The Coeffecients:\n ")
    # pattern = "%.50f"
    # floatsstrings = [pattern % i for i in list(reg.coef_)]
    # print(floatsstrings)

    mean2Err = mean_squared_error(Y_test, yfeature_prediction)
    varianceScore = r2_score(Y_test, yfeature_prediction)
    if name is None:
        return (mean2Err, varianceScore, yfeature_prediction)
    return (mean2Err, varianceScore, yfeature_prediction,Y_test)



def realMain():
    ourPredictions = open('The Final Predictions.txt', 'w')
    #
    # # These are the stats to predict. All of them have a nonzero coefficient in Fantasy Football Points algorithm for an Offensive player
    stats = ['Fumble TD', 'Fumbles Lost', 'Pass 2PT',
             'Pass Attempts', 'Pass Completions', 'Pass Interceptions', 'Pass TD',
             'Pass Yards', 'Receiving 2PT', 'Receiving TD',
             'Receiving Yards', 'Receptions', 'Rush 2PT', 'Rush Attempts', 'Rush TD',
             'Rush Yards']
    # # These are the stats I'm actually concerned about accurately predicting right now
    statsUsing = stats.copy()
    #
    # # change to stat in stats in order to predict all relevant stats, instead of only the stats I'm currently caring about
    os.chdir('/Users/nickdugal/Documents/Fantasy-Football/squared/allyears')
    training = (pd.read_csv('trainingDF.csv')).groupby('Name')
    testing = pd.read_csv("testingDF.csv")
    predictionList = []
    #
    testSet = set(list(testing.Name))
    actualDataFrame = pd.DataFrame(data=None, index=testSet, columns=stats)
    for name, training_data in training:
        checker = set(list(testing.Name))
        if name in checker:
            testing_data = testing[testing.Name == name]

            trainSet = set((pd.read_csv('trainingDF.csv')).Name.tolist())
            # theNames = trainSet.union(testSet)
            actualDataFrame.loc[name,'Position'] = str(training_data['Position'])
            for stat in stats:
                resultTuples = []
                # for i in range(10):
                    # resultTuples.append(realRegression(stats,stat,training_data,testing_data))
                mean2Errs, variances, prediction, Ytest = realRegression(stats,stat,training_data,testing_data,name)
                    # zip(*resultTuples)

                ourPredictions.write("\n\n\nPredicting: \t" + stat)
                ourPredictions.write(("\nPlayer Name:\t"+ name))
                ourPredictions.write('\nThe prediction\t' + str())
                ourPredictions.write('\nMean^2 Error Avg\t' + str())
                ourPredictions.write('\nVariances Avg\t' + str())
                # predDF['Name']= name
                # predValue = map(float,prediction)
                # print('#'*50)
                # print(prediction)
                # print(Ytest)
                # print(prediction.shape)
                # exit(0)
                actualDataFrame.loc[name,stat] = np.mean(prediction)
                vStr = str(stat+' EV')
                m2eStr = str(stat + ' M2E')
                actualDataFrame.loc[name,m2eStr]= float(mean2Errs)
                actualDataFrame.loc[name,vStr]= float(variances)
        else:
            continue

    actualDataFrame.to_csv('actualDataFrame.csv')
    ourPredictions.close()
    ffPredictions = open('FF Predictions.txt', 'w')
    # exit(0)
    # stats.append('Points')
    actualDataFrame = pd.read_csv('actualDataFrame.csv')
    ffActualDataFrame = pd.DataFrame(data=None, index=testSet, columns=['Name','Points Predicted','Points Actual','mean2Errs','variances'])
    for name, training_data in training:
        checker = set(list(testing.Name))
        if name in checker:
            # resultTuples = []
            # for i in range(10):
            #     resultTuples.append(realRegression(stats, 'Points', training, actualDataFrame))
            mean2Errs, variances, prediction, realPoints = realRegression(stats, 'Points', training_data, actualDataFrame[actualDataFrame.Name == name],name=name)
            ffActualDataFrame.loc[name, 'Points Predicted'] = prediction
            ffActualDataFrame.loc[name,'Points Actual'] = realPoints
            ffActualDataFrame.loc[name,'mean2Errs']= np.true_divide(float(sum(mean2Errs),float(len(mean2Errs))))
            ffActualDataFrame.loc[name,'variances']= np.true_divide(float(sum(variances) , float(len(variances))))
            # ffPredictions.write("\n\n\nPredicting: \t" + 'Points')
            # # ffPredictions.write(("\nPlayer Name:\t" + name))
            # ffPredictions.write('\nThe prediction\t' + str(prediction))
            # ffPredictions.write('\nMean^2 Error Avg\t' + str(mean2Errs))
            # ffPredictions.write('\nVariances Avg\t' + str(variances))
            # predDF['Name'] = name
            # predDF[stat] = (float(sum(prediction) / float(len(prediction))))
            # predDF['MeanSquaredError'] = (float(sum(mean2Errs) / float(len(mean2Errs))))
            # predDF['Explained Variance'] = (float(sum(variances) / float(len(variances))))


def main():
    os.chdir('squared/allyears')
    all = pd.read_csv('SquaredAllPositionsAllYears.csv')
    # print(list(all.columns))
    # all.reindex(index=['Year'])
    inputDF1 = all[all.Year != 2016]
    inputDF2 = all[(all.Year == 2016) & (all.Week <= 8)]
    testing = all[(all.Year == 2016) & (all.Week == 8)]
    # print(inputDF2.head());exit(0)
    breesOG = inputDF2.copy(deep=True)
    inputDF2 = removeAlphaData(inputDF2)

    columns = list(removeAlphaData(inputDF2).columns)
    dfsToConcat = []
    for pos, pData in (breesOG.groupby('Position')):
        for person, personsD in (pData.groupby('Name')):
            means = []
            for col in columns:
                mean = np.mean(personsD[col])
                means.append(mean)
            diction = zip(columns, means)
            newDF = pd.DataFrame()
            newDF['Year'] = personsD['Year']
            newDF['Week'] = personsD['Week']
            newDF['Name'] = person
            newDF['Opponent'] = personsD['Opponent']
            newDF['Position'] =  pos
            for col, me in diction:
                if (col == 'Week'):
                    continue
                if (col == 'Year'):
                    continue
                newDF[col] = me
            dfsToConcat.append(newDF)
    newerDF = pd.concat(dfsToConcat)
    newerDF[newerDF.Week == 8].to_csv('testingDF.csv')
    print(columns)

    # print((len(means)))
    diction = zip(columns,means)
    newDF = pd.DataFrame()
    newDF['Year'] = inputDF2['Year']
    newDF['Week'] = inputDF2['Week']
    newDF['Name'] = breesOG['Name']
    newDF['Opponent'] = breesOG['Opponent']
    newDF['Position'] = breesOG['Position']
    for col,me in diction:
        newDF[col] = me
    newDF.to_csv('testingDF.csv')



    trainingDF = pd.concat([inputDF1,inputDF2])
    trainingDF.to_csv('training_df.csv',index=None)
    # testingDF =

    print(inputDF1.shape)

    print(inputDF2.shape)
    print(inputDF2.head())

    exit(0)



    # squared.to_csv('squaredBrees.csv')

    exit()
    # combineOffensePositionWithDefense('RB')
    createInputData()
    exit()
    multivariateLinearRegression(
        "/Users/nickdugal/Downloads/Fantasy-Football/data/Updated NFL Data Sets/Indexed Data/Drew Brees Data.csv",
        "Pass Attempts")

    os.chdir("/Users/nickdugal/Downloads/Fantasy-Football")
    ourPredictions = open('Drew Brees Pred.txt', 'w')

    # These are the stats to predict. All of them have a nonzero coefficient in Fantasy Football Points algorithm for an Offensive player
    stats = ['Fumble TD', 'Fumbles Lost', 'Pass 2PT',
             'Pass Attempts', 'Pass Completions', 'Pass Interceptions', 'Pass TD',
             'Pass Yards', 'Receiving 2PT', 'Receiving TD',
             'Receiving Yards', 'Receptions', 'Rush 2PT', 'Rush Attempts', 'Rush TD',
             'Rush Yards']
    # These are the stats I'm actually concerned about accurately predicting right now
    statsUsing = ['Pass 2PT', 'Pass Attempts', 'Pass Completions', 'Pass Interceptions', 'Pass TD',
                  'Pass Yards']

    # change to stat in stats in order to predict all relevant stats, instead of only the stats I'm currently caring about
    for stat in statsUsing:
        resultTuples = []
        for i in range(10):
            resultTuples.append(multivariateLinearRegression(
                "/Users/nickdugal/Downloads/Fantasy-Football/data/Updated NFL Data Sets/Indexed Data/Drew Brees With Defense.csv", stat))
        mean2Errs, variances, hmmm = zip(*resultTuples)
        ourPredictions.write("\n\n\nPredicting: \t" + stat)
        ourPredictions.write('\nMean^2 Error Avg\t' + str(sum(mean2Errs) / float(len(mean2Errs))))
        ourPredictions.write('\nVariances Avg\t' + str(sum(variances) / float(len(variances))))

    ourPredictions.close()
#Ignore This
def groupPlayersByNameYear(players_data):
    os.chdir("/Users/nickdugal/Documents/Fantasy-Football/data/Updated NFL Data Sets/Indexed Data/")
    player_data = pd.read_csv('QB_with_Defense.csv')
    # where players_data is a pandas dataframe
    their_stats = players_data.groupby(['Name', 'Year']).agg({'Pass Yards': [np.size, np.mean, np.sum]})
    # Now to get a specific player using this format
    dBrees_data = their_stats.loc[['Drew Brees', ], :]

#Ignore this
    # So here we can get a specific player then group their data into whatever we want, iterate through those groupings and get whatever stats we want on each column
    # Yeah that's kinda a rant
def getSpecifiedStatisticsForSpecificPlayerGrouped():
    player_data = pd.read_csv("QB_with_Defense.csv", index_col=['Year', 'Week'].sort())
    brees_data = player_data[player_data.Name == 'Drew Brees']
    brees_data.drop("Unnamed: 0", axis=1, inplace=True)
    grouped = brees_data.groupby(['Year'])
    for item, value in grouped:
        if item == 2012:
            print(np.mean(value['Pass Yards']))

#IGNore This
    # Grouped is defined the same as in getSpecifiedStatisticsForSpecificPlayerGrouped()
    # This method will iterate through all the features a current player has, remove the non numerical ones, then calculate a descriptive statitistic for each
    # Additional Loop will provided time valued data as needed for multinomimial linear regression
def iterateAndReceiveStats():
    for item, value in grouped:
        xFeatures = set(list(value.columns))
        xFeatures = xFeatures.difference(['Name', 'Opponent', 'Position', 'Opponent_x', 'Position_x',
                                          'Team', 'Opponent_y', 'Position_y'])
        for feature in xFeatures:
            valFeature = value[feature]
            # if valFeature.at[1].isnumeric()==True or valFeature.at[1].isdecimal()==True:
            print(np.mean(valFeature))

            # This does what said needed to be done in iterateAndReceiveStats() method
            # It loops and calculates a rolling mean, returning each new mean value a every week
            # SHOULD CHECK THAT IT STARTS AT WEEK 1, INSTEAD OF COUNTING DOWN
            # Also, these methods need to be returning a list or some usable datastructure instead of printing the results
            # recommend creating empty list and appending where my print statements are

#Ignore this
def cumulativeWeeklyMeans(grouped):
    # item is the feature we grouped on and value is it's assocciated dataframe
    for item, value in grouped:
        if item == 'Drew Brees':
            xFeatures = set(list(value.columns))
            xFeatures = xFeatures.difference(['Name', 'Opponent', 'Position', 'Opponent_x', 'Position_x',
                                              'Team', 'Opponent_y', 'Position_y'])
            print('H')
            for feature in xFeatures:
                valFeature = value[feature]
                print('Hi')
                # if valFeature.at[1].isnumeric()==True or valFeature.at[1].isdecimal()==True:
                # print(np.mean(valFeature))
                # Here we begin finding the effect of time for features
                if feature == 'Pass Yards':
                    print('Weekly Values before means:, ', valFeature)
                    for val in range(0, len(valFeature)):
                        print(np.mean(valFeature[0:val]))


# Takes columns of a DF and removes the currently known nonNumerical values
def purgeAlphas(unCleaned):
    cleaned = (set(list(unCleaned))).difference(['Name', 'Opponent', 'Position', 'Opponent_x', 'Position_x',
                                                 'Team', 'Opponent_y', 'Position_y'])
    return cleaned


# Removes the nonNumerical columns of a DF
def removeAlphaData(unCleanedDF):
    return unCleanedDF[list(purgeAlphas(unCleanedDF))]

#Ignore this
def createInputData():
    os.chdir('data/Updated NFL Data Sets/Indexed Data/')
    qbAllYears = pd.read_csv('QBAllYears.csv', index_col=['Year','Week']).groupby(['Name'])
    qbAllYears.apply(sort_index(inplace=True))
    pd.DataFrame.apply()
    # cumulativeWeeklyMeans(qbAllYears);exit(0)
    for qb, data in qbAllYears:
        if qb == 'Drew Brees':
            updateData(qb,removeAlphaData(data))

def updateData(qb, data):
    print('Name'+qb)
    # print('\nData: ',data)
    for column in list(data.columns):
        columnData = data[column]
        print(columnData)
        return 0


    exit(0)








    # os.chdir("/Users/nickdugal/Documents/Fantasy-Football/data/Updated NFL Data Sets/Indexed Data/")
    # player_data = pd.read_csv("QB_with_Defense.csv", index_col=['Year','Week'])
    # brees_data = player_data[player_data.Name == 'Drew Brees']
    # # brees_data.drop("Unnamed: 0", axis=1, inplace=True)
    # grouped = brees_data.groupby(['Year'])
    # listOfAveragedList = []
    # for year, yearData in grouped:
    #
    #     newValuesAsDict = []
    #
    #     for dataColumn in purgeAlphas(yearData.columns):
    #         dataVector = yearData[dataColumn]
    #         averagedList = []
    #         for val in range(0, len(dataVector)):
    #             averagedList.append(np.mean(dataVector[0:val]))
    #         newValuesAsDict.append(dict({dataColumn: averagedList}))
    #     listOfAveragedList.append((newValuesAsDict))
    #
    # updatedDF = pd.DataFrame(listOfAveragedList)
    # print(updatedDF.head(10))
    # updatedDF.to_csv("Averaged_Brees_Values.csv")

def stuffToSquared():
    # positions = ['QB','RB','K','WR','TE']
    positions = ['RB']
    for pos in positions:
        dBrees = pd.read_csv('Data/' + pos + 'allyears.csv', index_col=None)
        squared = pd.DataFrame()
        breesOG = dBrees.copy(deep=True)
        dBrees = removeAlphaData(dBrees)
        min = dBrees['Year'].min()
        breesGrouped = dBrees.groupby('Year')
        for year, data in breesGrouped:
            power = year - min + 1
            newDF = np.power(data, power)
            newDF['Year'] = dBrees['Year']
            newDF['Week'] = dBrees['Week']
            newDF['Name'] = breesOG['Name']
            newDF['Opponent'] = breesOG['Opponent']
            newDF['Position'] = breesOG['Position']
            newDF.to_csv(str(year) + pos + 'Squared.csv')
def squaredToScaled():
    # positions = ['QB','RB','K','WR','TE']
    positions = ['DST']
    for pos in positions:
        dBrees = pd.read_csv('squared/updated/' + pos + 'allyears.csv', index_col=None)
        # squared = pd.DataFrame()
        # alphas = ['Year','Week','Name','Opponent','Position']
        # for alpha in alphas:
        #     squared[alpha] = dBrees[alpha].copy()
        breesOG = dBrees.copy(deep=True)
        dBrees = removeAlphaData(dBrees)
        newDF = dBrees.apply(preprocessing.scale)
        newDF['Year'] = dBrees['Year']
        newDF['Week'] = dBrees['Week']
        newDF['Team'] = breesOG['Team']
        newDF['Opponent'] = breesOG['Opponent']
        newDF['Position'] = breesOG['Position']
        newDF.to_csv(pos + 'Scaled_Squared.csv')
    exit(0);





if __name__ == "__main__": realMain()
