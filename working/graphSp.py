import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import sin, cos
class GraphSp():
    def setupUi(self, GraphSp):
        data = pd.read_csv('Drinks.csv')
        firms = ["Пиво", "Вино", "Спиртное"]
        market_share = [247, 73, 326]
        plt.pie(market_share, labels=firms, shadow=True, startangle=45)
        plt.axis('equal')
        plt.legend(title="Доля алкоголя в России")
        plt.show()