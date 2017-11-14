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

#This is a file for experimenting with Fantasy Foootball data!
# Documentation for pyplot
# https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.html
def main():
    multivariateLinearRegression("AllPositionsAllYears.csv", "Pass Attempts")
    exit(0) #Because I'm weird and don't use breakpoints

    # # linearRegression()
    os.chdir("/Users/nickdugal/Documents/Fantasy-Football")
    ourPredictions = open('predictionsWithPoints.txt','w')
    checkThis = ['Fumble TD', 'Fumbles Lost', 'Pass 2PT',
     'Pass Attempts', 'Pass Completions', 'Pass Interceptions', 'Pass TD',
     'Pass Yards', 'Receiving 2PT', 'Receiving TD',
     'Receiving Yards', 'Receptions', 'Rush 2PT', 'Rush Attempts', 'Rush TD',
     'Rush Yards']

    for checks in checkThis:
        resultTuples = []
        for i in range(10):
        # ourPredictions.write(multivariateLinearRegression("AllPositionsAllYears.csv",checks))
            resultTuples.append(multivariateLinearRegression("AllPositionsAllYears.csv",checks))
        mean2Errs, variances = zip(*resultTuples)
        ourPredictions.write("\n\n\nPredicting: \t"+ checks)
        # ourPredictions.write('\nMean^2 Errors: \t'+list(mean2Errs).__str__())
        ourPredictions.write('\nMean^2 Error Avg\t'+str(sum(mean2Errs)/float(len(mean2Errs))))
        # ourPredictions.write('\nVariances: \t' + list(variances).__str__())
        ourPredictions.write('\nVariances Avg\t' + str(sum(variances) / float(len(variances))))
        # howGoodDF[checks].append(resultTuples)
    #
    # ourPredictions.close()
    # multivariateLinearRegression("AllPositionsAllYears.csv", "Pass Attempts")
def heatMap():

        heatmap, xedges, yedges = np.histogram2d(X, Y)#, bins=(64, 64))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.clf()
        plt.imshow(heatmap, extent=extent)
        plt.show()
def linearRegression():
        os.chdir('data')
        # rb_all = os.listdir(os.getcwd())
        rb_data = pd.read_csv('qbAllYears.csv')
        print(rb_data.columns)

        Y = rb_data['Pass Yards']
        X = rb_data['Pass Attempts']
        # XSquared = list(map(lam))
        #We need to add nonzero column to our data due to matrix and vector maths, and the need for axis intercepts that
        #aren't equal to 0
        #Not explaining all that in comments. Sorry.
        X = X[:, np.newaxis]
        Y = Y[:, np.newaxis]

        #Creates two Dataframes, one for training with random samples consistenting of 80% the total samples, one for testing with the remaining 20%
        X_train, X_test = train_test_split(X,test_size=.2)
        Y_train, Y_test = train_test_split(Y, test_size=.2)

        #Here we create and fit the linear regression model. The call to fit iteratively updates the coeffecients of our model until it's mean squared difference is within
        #a certain threshold
        reg = linear_model.LinearRegression()
        reg.fit(X_train, Y_train)

        #Here we throw the testing data into the model. This returns an array of predicted y values
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
        print("The Mean Squared Error\n", mean2Err)
        # Explained Variance Score: 1 is perfect prediction
        print('Variance Score: %.2f' % varianceScore)
        #Here we plot our testing data to get a look at it. Let's make sure a linear model is the model for us
        plt.scatter(X_test, Y_test, color='black')
        #Here we plot our model's line as defined during training against the testing data
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
        #This gives us the min and max values of our graph. Move along..
        xmin, _ = plt.xlim()
        _, ymax = plt.ylim()
        s = str('Our line is defined as \nhaving coefficient values of ' + str(coef) +
                '\nThe Mean Squared Error: ' + str(mean2Err) +
                '\nVariance Score: ' + str(varianceScore))
        plt.text(8, 30, s, rotation=40,
                 horizontalalignment='center',
                 verticalalignment='center',
                 multialignment='center',color='red')

        # plt.scatter(X,Y, s=12,c='red',marker='.')
        # plt.xlim(-5,40)
        # plt.ylim(0,175)                         # xmin, xmax = xlim() use to get current values

        plt.title('Relationship Between Pass Attempts and Pass Completions')

        plt.ylabel('Pass Completions')
        plt.xlabel('Pass Attempts', multialignment='center')

        plt.show()
def multivariateLinearRegression(file, yfeature):
        #Same exact thing as above except there are waaayyy more features. Above involved only one
        os.chdir('/Users/nickdugal/Documents/Fantasy-Football/data')
        try:
            players_data = pd.read_csv(file)
        except:
            players_data = pd.read_csv('AllPositionsAllYears.csv')

        featuresList = ['Fumble TD', 'Fumbles Lost','Pass 2PT',
       'Pass Attempts', 'Points','Pass Completions', 'Pass Interceptions', 'Pass TD',
       'Pass Yards', 'Receiving 2PT', 'Receiving TD',
       'Receiving Yards', 'Receptions', 'Rush 2PT', 'Rush Attempts', 'Rush TD',
       'Rush Yards']

        #Creates a set, phenominal for membership checking, unions, intersections, etc
        setList = set(featuresList)

        #Test if the yFeature argument is a valid column
        if (yfeature in setList) == False:
            raise Exception('Dude, put in a valid parameter\nPlease try again with a valid feature')
        #Removes the yfeature from the group of Xs
        xFeatures = setList.difference([yfeature])

        Y = players_data[yfeature]
        # Y = preprocessing.scale(Y)
        X = players_data[list(xFeatures)].values.reshape(-1,len(xFeatures))
        # X = preprocessing.scale(X)
        # XSquared = list(map(lam))
        # X = X[:, np.newaxis]
        # Y = Y[:, np.newaxis]

        # X_train, X_test, Y_train, Y_test = train_test_split(preprocessing.scale(X), preprocessing.scale(Y), test_size=.2)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=.2)
        coeff_titles = list(xFeatures)
        coeff_titles.insert(0,'Intercept')


        reg = linear_model.LinearRegression()
        # reg = linear_model.PassiveAggressiveRegressor
        model = reg.fit(X_train, Y_train)

        # model = reg.fit(X_train, Y_train,Y_test)
        PassAttempts_prediction = model.predict(X_test)
        # print("Predicting: ",yfeature)
        print("The Coeffecients:\n ")
        hmmm = dict(zip(coeff_titles,list(reg.coef_)))
        # hmmmDF = pd.DataFrame(hmmm)
        print(hmmm);exit(0)
        mean2Err = mean_squared_error(Y_test, PassAttempts_prediction)
        # print("Score: ",model.score(X_test,Y_test))
        varianceScore = r2_score(Y_test, PassAttempts_prediction)
        # print("The Mean Squared Error\n", mean2Err)
        # Explained Variance Score: 1 is perfect prediction
        # print('Variance Score: %2.2f' % varianceScore)

        # pStr = str("\n\n\nPredicting: \t"+ yfeature)
        # cStr = str('The Coeffecients: %2.2f' %  reg.coef_)
        # sStr = str("\nScore: \t" + str(model.score(X_test,Y_test)))
        # mStr = str("\nMean^2 Error\t" + str(mean2Err))
        # vStr = str('\nVariance Score: \t%2.2f' % varianceScore)
        # return pStr + sStr + mStr + vStr
        return (mean2Err,varianceScore)

if __name__ == "__main__": main()