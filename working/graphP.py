import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
class GraphP(object):

    def setupUi(self, GraphP):
        GraphP.setObjectName("1")
        GraphP.resize(1, 1)
        data = pd.read_csv('Drinks.csv')
        x = data['total_litres_of_pure_alcohol']
        y1 = data['spirit_servings']
        y2 = data['wine_servings']
        plt.plot(x, y1, 'r o')
        plt.plot(x, y2, 'g ^')
        plt.xlabel('Доля спиртного')
        plt.ylabel('Доля вина')
        plt.title('График 1 ')
        plt.legend(['Спиртное', 'Вино'])
        plt.grid(True)
        plt.show()