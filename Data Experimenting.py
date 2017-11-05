import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import operator
import argparse

#This is a file for experimenting with Fantasy Foootball data!
# Documentation for pyplot
# https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.html
def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("Something")



    os.chdir('data/rb')
    rb_all = os.listdir(os.getcwd())
    rb_data = pd.read_csv(rb_all[0])

    X = rb_data['Rush Attempts']
    Y = rb_data['Rush Yards']


    plt.scatter(X,Y, s=12,c='red',marker='.')


    plt.xlim(-5,40)
    plt.ylim(0,175)                         # xmin, xmax = xlim() use to get current values

    plt.title('Relationship Between Rush Yards and Rush Attempts')

    plt.xlabel('Rush Attempts')
    plt.ylabel('Rush Yards')

    plt.show()

if __name__ == "__main__": main()