import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy import sin, cos

data = pd.read_csv('Nation.csv')

class Graph():
    def setupUi(self, MainWindow):
        x = data['Open']
        y1 = data['High']
        y2 = data['Low']
        plt.plot(x, y1, 'r o')
        plt.plot(x, y2, 'g ^')
        plt.xlabel('Х столб')
        plt.ylabel('Y столб')
        plt.title('График')
        plt.legend(x)
        plt.grid(True)
        plt.show()

class Graph2():
    def setupUi2(self, MainWindow):
        x = data['Open']
        plt.title("Гистограмма")
        plt.xlabel("Данные")
        plt.ylabel("Время")
        plt.hist(x, 10)
        plt.show()

class Graph3():
    def setupUi3(self, MainWindow):
        def f(t):
            return np.exp(-t) * np.cos(2 * np.pi * t)
        t1 = data['High']
        t2 = data['Low']
        plt.figure(1)
        plt.subplot(211)
        plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
        plt.legend(t1)
        plt.ylabel("Я игрик")
        plt.xlabel("Я икс")
        plt.title("High")
        plt.subplot(212)
        plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')
        plt.ylabel("Я игрик")
        plt.xlabel("Я икс")
        plt.title("Low")
        plt.legend(t2)
        plt.show()

class Graph4():
    def setupUi4(self, MainWindow):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.title("3д график")
        plt.ylabel("Игрик")
        plt.xlabel("Икс")
        r = 10
        c = 50
        t = data['High'] #np.linspace(0, 5000, 100)
        x = r * cos(t)
        y = r * sin(t)
        z = c * t
        ax.plot(x, y, z, zdir='z', lw=2)
        plt.show()

class Graph5():
    def setupUi5(self, MainWindow):
        plt.title("Диаграмма")
        firms = data['Symbol']
        market_share = data['Open']
        plt.pie(market_share,labels=firms,shadow=True,startangle=45)
        plt.axis('equal')
        plt.legend(market_share)
        plt.show()

class Graph6():
    def setupUi6(self, MainWindow):
        fig, ax = plt.subplots()
        plt.title("График")
        plt.ylabel("Low")
        plt.xlabel("High")
        x = data['High']
        y = data['Low']
        ax.plot(x, y)
        plt.show()