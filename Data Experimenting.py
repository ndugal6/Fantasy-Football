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
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D

#This is a file for experimenting with Fantasy Foootball data!
# Documentation for pyplot
# https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.html
def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("Something")

    # linearRegression()
    multivariateLinearRegression()

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
        X = X[:, np.newaxis]
        Y = Y[:, np.newaxis]


        X_train = X[:-800]
        X_test = X[-800:]
        Y_train = Y[:-800]
        Y_test = Y[-800:]
        reg = linear_model.LinearRegression()
        reg.fit(X_train, Y_train)
        PassAttempts_prediction = reg.predict(X_test)

        print("The Coeffecients:\n ", reg.coef_)
        mean2Err = mean_squared_error(Y_test, PassAttempts_prediction)
        varianceScore = r2_score(Y_test, PassAttempts_prediction)
        print("The Mean Squared Error\n", mean2Err)
        # Explained Variance Score: 1 is perfect prediction
        print('Variance Score: %.2f' % varianceScore)
        plt.scatter(X_test, Y_test, color='black')
        plt.plot(X_test, PassAttempts_prediction, color='blue', linewidth=3)
        half = int(len(PassAttempts_prediction) / 2)
        plt.xticks()
        plt.yticks()
        # plt.annotate(('hmm','whatUP'))
        logReg = linear_model.LogisticRegression()
        logReg.fit(X_train, Y_train)

        coef = reg.coef_[0].tolist()
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
def multivariateLinearRegression():
        os.chdir('data')
        rb_data = pd.read_csv('qbAllYears.csv')

        Y = rb_data['Pass Points']
        X = rb_data[['Pass Attempts','Pass Completions','Pass Yards']].values.reshape(-1,3)
        # XSquared = list(map(lam))
        # X = X[:, np.newaxis]
        Y = Y[:, np.newaxis]

        X_train = X[:-400]
        X_test = X[-400:]
        Y_train = Y[:-400]
        Y_test = Y[-400:]
        reg = linear_model.LinearRegression()
        model = reg.fit(X_train, Y_train)
        PassAttempts_prediction = model.predict(X_test)

        print("The Coeffecients:\n ", reg.coef_)
        mean2Err = mean_squared_error(Y_test, PassAttempts_prediction)
        varianceScore = r2_score(Y_test, PassAttempts_prediction)
        print("The Mean Squared Error\n", mean2Err)
        # Explained Variance Score: 1 is perfect prediction
        print('Variance Score: %.2f' % varianceScore)
        # plt.scatter(X_test[:,0], X_test[:,1], Y_test, color='black', projection='3d')
        Axes3D.plot(X_test, Y_test,PassAttempts_prediction, color='blue', linewidth=3)

        plt.xticks()
        plt.yticks()
        logReg = linear_model.LogisticRegression()
        logReg.fit(X_train, Y_train)

        coef = reg.coef_[0].tolist()
        xmin, _ = plt.xlim()
        _, ymax = plt.ylim()
        s = str('Our line is defined as \nhaving coefficient values of ' + str(coef) +
                '\nThe Mean Squared Error: ' + str(mean2Err) +
                '\nVariance Score: ' + str(varianceScore))
        plt.text(8, 30, s, rotation=40,
                 horizontalalignment='center',
                 verticalalignment='center',
                 multialignment='center')


        plt.title('Relationship Between Pass Attempts and Pass Completions')

        plt.ylabel('Pass Completions')
        plt.xlabel('Pass Attempts', multialignment='center')

        plt.show()

if __name__ == "__main__": main()