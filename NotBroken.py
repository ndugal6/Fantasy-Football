import pandas as pd
import os, errno
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn import metrics


""""
FINAL VERSION OF AI PROJECT FOR SUBMISSION. REALER THAN REAL MY DUDES. NICHOLAS DUGAL.
"""
# 2017 data is in 2017data dir
##Linear regression csv path calls may break when connectioning to database
def main():
    pass

def predict(name,position='qb'):
    actualNew = pd.read_csv('2017Data/' + position + '/' + '/actualDataWithDefense.csv')

    toDrop = ['Point After', 'Fumble Returns', 'Fumble TD', 'Week', 'Away Games_x', 'Rush 2PT', 'Away Games_y',
              'Safety', 'Receiving 2PT', 'Blocks', 'Pass 2PT', 'Year']
    cols = list(purgeAlphas(actualNew.columns))
    features = list(set(cols).difference(toDrop))
    predictionDF = pd.DataFrame(columns=((features.append("Name"))))
    predictionDF['Name'] = [name]
    features.remove('Name')
    for feature in features:
        prediction = linearRegression(position=position,player=name,feature=feature,Future=True)
        predictionDF[feature] = prediction
    return predictionDF.copy(deep=True)

def player_pointsOnly(name='drew brees',position='qb'):
    return linearRegression(position=position,player=name,feature = 'Points_x',Future=True)



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
    # Get actual stats and average stats for input
    rookie = False
    try:
        actualOld = pd.read_csv('Data/'+position+'/'+player+'/actualDataWithDefense.csv')
        averagesOld = pd.read_csv('Data/' + position + '/' + player + '/averagedDataWithDefense.csv')
    except:
        rookie = True
        print('WOrking on a Rookie')
    averageNew = pd.read_csv('2017Data/' + position + '/' + player + '/actualDataWithDefense.csv')
    actualNew = pd.read_csv('2017Data/' + position + '/' +player+ '/actualDataWithDefense.csv')
    if rookie:
        actual = actualNew.copy(deep=True)
        averages = averageNew.copy(deep=True)
    else:
        actual = pd.concat([actualOld,actualNew])
        averages = pd.concat([actualOld,actualNew])
    # Get the combo we need
    data = removeAlphaData(inputForFeature(actual,averages,feature))
    data.round(4)
    data = data.apply(lambda x: pd.to_numeric(x,errors='ignore'), axis=1)

    # boom linear model ready to go.. not so difficult :p
    reg = linear_model.LinearRegression()
    #create training and tesitng data
    if Future:
        x_train, y_train, x_test = train_test_divider(dirtyData=data, feature=feature, year=2017, week=8,Future=True)
        model = reg.fit(x_train, y_train)
        prediction = model.predict(x_test.values.reshape(1, -1))
        return prediction
    x_train,x_test,y_train,y_test = train_test_divider(dirtyData=data,feature=feature,year=2017,week=8)

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

    return prediction, mean2Err, varianceScore,y_test



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
