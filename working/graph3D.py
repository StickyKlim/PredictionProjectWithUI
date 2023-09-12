import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
class Graph3D():
     def setupUi(self, Graph3D):
        data = pd.read_csv('Drinks.csv')
        x = data['wine_servings']
        y = data['spirit_servings']
        z = data['beer_servings']
        ax = plt.axes(projection='3d')
        ax.scatter3D(x, y, z)
        plt.xlabel('Вино')
        plt.ylabel('Спиртное')
        plt.title('График 3 ')
        plt.show()
