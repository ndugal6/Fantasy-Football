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

#This is a file for experimenting with Fantasy Foootball data!
# Documentation for pyplot
# https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.html
def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("Something")



    os.chdir('data')
    # rb_all = os.listdir(os.getcwd())
    rb_data = pd.read_csv('qbAllYears.csv')

    X = rb_data['Pass Completions']
    Y = rb_data['Pass Attempts']
    X = X[:, np.newaxis]
    Y = Y[:, np.newaxis]
    xy = pd.DataFrame({'Attempts': list(X),
                       'Yards': list(Y)})

    X_train = X[:-20]
    X_test = X[-20:]
    Y_train = Y[:-20]
    Y_test = Y[-20:]
    reg = linear_model.LinearRegression()
    reg.fit(X_train, Y_train)
    RushYards_prediction = reg.predict(X_test)

    print("The Coeffecients:\n ",reg.coef_)
    print("The Mean Squared Error\n",mean_squared_error(Y_test,RushYards_prediction))
    #Explained Variance Score: 1 is perfect prediction
    print('Variance Score: %.2f' % r2_score(Y_test, RushYards_prediction))
    plt.scatter(X_test, Y_test, color='black')
    plt.plot(X_test, RushYards_prediction, color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    exit(0)

    # plt.scatter(X,Y, s=12,c='red',marker='.')
    # plt.xlim(-5,40)
    # plt.ylim(0,175)                         # xmin, xmax = xlim() use to get current values

    plt.title('Relationship Between Rush Yards and Rush Attempts')

    plt.xlabel('Rush Attempts')
    plt.ylabel('Rush Yards')

    plt.show()
    def heatMap():
        heatmap, xedges, yedges = np.histogram2d(X, Y)#, bins=(64, 64))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.clf()
        plt.imshow(heatmap, extent=extent)
        plt.show()
if __name__ == "__main__": main()