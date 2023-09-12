from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import stats
from sklearn import datasets, linear_model
from PyQt5 import QtCore, QtGui, QtWidgets


class regr1():
    def setup1(self, MainWindow):
        data = pd.read_csv("Nation.csv")
        x = data['High']
        y = data['Low']
        slope, intercept, r, p, std_err = stats.linregress(x, y)

        def myfunc(x):
            return slope * x + intercept

        mymodel = list(map(myfunc, x))
        plt.scatter(x, y)
        plt.plot(x, mymodel)
        plt.ylabel('High')
        plt.xlabel('Low')
        plt.title("Линейная регрессия")
        plt.show()


class regr3():
    def setup3(self, MainWindow):
        diabets_data = datasets.load_diabetes()
        di = pd.DataFrame(diabets_data.data)
        di.columns = diabets_data.feature_names
        di['target'] = diabets_data.target
        x = di.drop('target', axis=1)
        rm = linear_model.LinearRegression()
        rm.fit(x, di.target)
        plt.xlabel('High')
        plt.ylabel('Low')
        plt.scatter(di.target, rm.predict(x))
        plt.title('График регрессии №3')
        plt.show()

