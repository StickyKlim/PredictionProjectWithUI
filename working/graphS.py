import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
class GraphS():
    def setupUi(self, GraphS):
        GraphS.setObjectName("1")
        GraphS.resize(1, 1)
        data = pd.read_csv('Drinks.csv')
        x = data['total_litres_of_pure_alcohol']
        y = data['wine_servings']
        plt.hist(x, 100)
        plt.grid(True)
        plt.xlabel('Вино')
        plt.legend(['Вино'])
        plt.ylabel('Всего литров')
        plt.title('График 4 ')
        plt.show()