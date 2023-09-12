import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos
class GraphD():
     def setupUi(self, GraphD):
        data = pd.read_csv('Drinks.csv')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        r = 10
        c = 50
        t = data['total_litres_of_pure_alcohol']
        x = r * cos(t)
        y = r * sin(t)
        z = c * t
        plt.xlabel('Общая доля алкоголя')
        plt.ylabel('Количество алкоголя')
        plt.title('График 2 ')
        ax.plot(x, y, z, zdir='z', lw=2)
        plt.show()