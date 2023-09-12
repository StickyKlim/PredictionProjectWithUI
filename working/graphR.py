from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from scipy import stats
import scipy
from scipy.stats import *
import numpy as np
class GraphR():
     def setupUi(self, GraphR):
        df = pd.read_csv("Drinks.csv")
        df = df.fillna(0)
        X = df[['beer_servings']].values
        y = df['spirit_servings'].values
        X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

        regr = LinearRegression()
        regr = regr.fit(X, y)
        y_lin_fit = regr.predict(X_fit)
        linear_r2 = r2_score(y, regr.predict(X))

        plt.scatter(X, y, label='training points', color='red')

        plt.plot(X_fit, y_lin_fit,
                 label='linear (d=1), $R^2={:.2f}$'.format(linear_r2),
                 color='blue',
                 lw=2,
                 linestyle=':')

        plt.xlabel('% lower status of the population [LSTAT]')
        plt.ylabel('Price in $1000\'s [MEDV]')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()