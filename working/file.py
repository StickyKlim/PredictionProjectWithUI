import sys

from PyQt5 import uic, Qt
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QAction, QApplication, QMainWindow, QTableView, qApp, QFileDialog, QMessageBox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from keras.losses import mean_absolute_error
from pandas.plotting._matplotlib import hist
from tensorflow.python.keras import Sequential

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from PyQt5.QtWidgets import QTableView
from sklearn.model_selection import train_test_split

from cal import *
from spin import *
from radio import *
from graphP import *
from graphSp import *
from graphS import *
from graphD import *
from graph3D import *
from graphR import *

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        SaveAction = QAction(QIcon('save.png'), 'Save', self)
        SaveAction.setShortcut('Ctrl+E')
        SaveAction.triggered.connect(self.save)
        self.toolbar = self.addToolBar('Save')
        self.toolbar.addAction(SaveAction)

        OpenAction = QAction(QIcon('open.png'), 'Open', self)
        OpenAction.setShortcut('Ctrl+O')
        OpenAction.triggered.connect(self.openfile)
        self.toolbar = self.addToolBar('Open')
        self.toolbar.addAction(OpenAction)

        printAction = QAction(QIcon('print.png'), 'Print', self)
        printAction.setShortcut('Ctrl+P')
        printAction.triggered.connect(self.printDialog)
        self.toolbar = self.addToolBar('Print')
        self.toolbar.addAction(printAction)

        RegAction = QAction(QIcon('reg.png'), 'Regression', self)
        RegAction.setShortcut('Ctrl+R')
        RegAction.triggered.connect(self.onClicked)
        self.toolbar = self.addToolBar('Regression')
        self.toolbar.addAction(RegAction)

        taxAction = QAction(QIcon('tax.png'), 'tax', self)
        taxAction.setShortcut('Ctrl+T')
        taxAction.triggered.connect(self.taxSpin)
        self.toolbar = self.addToolBar('Tax')
        self.toolbar.addAction(taxAction)

        calAction = QAction(QIcon('cal.png'), 'calendar', self)
        calAction.setShortcut('Ctrl+C')
        calAction.triggered.connect(self.paintCell)
        self.toolbar = self.addToolBar('Calendar')
        self.toolbar.addAction(calAction)

        helpAction = QAction(QIcon('help.png'), 'Help', self)
        helpAction.setShortcut('Ctrl+H')
        helpAction.triggered.connect(self.help)
        self.toolbar = self.addToolBar('Help')
        self.toolbar.addAction(helpAction)

        exitAction = QAction(QIcon('exit.png'), 'Exit', self)
        exitAction.setShortcut('Escape')
        exitAction.triggered.connect(qApp.quit)
        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(exitAction)

        uic.loadUi("inter.ui", self)
        self.title = 'ИС901 Климов А.С.'
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.statusBar().showMessage('Программа работает')

        self.Button3.clicked.connect(qApp.quit)
        self.action_4.triggered.connect(qApp.quit)

        self.Button.clicked.connect(self.openfile)
        self.action.triggered.connect(self.openfile)

        self.Button2.clicked.connect(self.save)
        self.action_2.triggered.connect(self.save)

        self.action_3.triggered.connect(self.onClicked)

        self.print_action.triggered.connect(self.printDialog)

        self.action_3.triggered.connect(self.onClicked)

        self.tax.triggered.connect(self.taxSpin)

        self.action_6.triggered.connect(self.help)

        self.actionADAM.triggered.connect(self.ADAM)
        self.actionAdadelta.triggered.connect(self.Adadelta)
        self.actionRMSprop.triggered.connect(self.RMSprop)
        self.actionSGD.triggered.connect(self.SGD)
        self.actionAdagard.triggered.connect(self.Adagard)

        self.actionRandom.triggered.connect(self.Random)

        self.action_p.triggered.connect(self.graphP)
        self.action_sp.triggered.connect(self.graphSp)
        self.action_sp.triggered.connect(self.graphSp)
        self.action_s.triggered.connect(self.graphS)
        self.action3d.triggered.connect(self.graph3D)
        self.action_d.triggered.connect(self.graphD)
        self.action_r.triggered.connect(self.graphR)
        self.action_r2.triggered.connect(self.GraphR2)
        self.action_r3.triggered.connect(self.GraphR3)

        self.Button6.clicked.connect(self.desc)

        self.actionPAD.triggered.connect(self.data_gaps_pad)
        self.actionPoly.triggered.connect(self.data_gaps_polynomial)
        self.actionAkima.triggered.connect(self.data_gaps_akima)
        self.actionFillNA.triggered.connect(self.data_gaps_fill)
        self.actionNull.triggered.connect(self.data_gaps_is_null)

        self.SavePredict.clicked.connect(self.predict)

        self.show()

    def predict(self):
        global epochs
        epochs = self.epochs.toPlainText()
        epochs = int(epochs)

        global learning
        learning = self.learning.toPlainText()
        learning = float(learning)

        global dense1
        dense1 = self.dense1.toPlainText()
        dense1 = int(dense1)

        global dense2
        dense2 = self.dense2.toPlainText()
        dense2 = int(dense2)

        global trainsize
        trainsize = self.trainsize.toPlainText()
        trainsize = float(trainsize)

        global batchsize
        batchsize = self.batchsize.toPlainText()
        batchsize = int(batchsize)

        global rho
        rho = self.rho.toPlainText()
        rho = float(rho)

        global beta1
        beta1 = self.beta1.toPlainText()
        beta1 = float(beta1)

        global beta2
        beta2 = self.beta2.toPlainText()
        beta2 = float(beta2)

        global momentum
        momentum = self.momentum.toPlainText()
        momentum = float(momentum)

        global initialacc
        initialacc = self.initialacc.toPlainText()
        initialacc = float(initialacc)

        global epsilon
        epsilon = self.epsilon.toPlainText()
        epsilon = float(epsilon)

        global nesterov
        nesterov = self.nesterov.toPlainText()
        nesterov = str(nesterov)

        global activat
        activat = self.activat.currentText()
        activat = str(activat)

    def onClicked(self):
        global Dialog
        Dialog = QtWidgets.QDialog()
        ui = Radio()
        ui.setupUi(Dialog)
        Dialog.show()

        def Regr1():
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

            plt.xlabel('Пиво')
            plt.ylabel('Спиртное')
            plt.legend(loc='upper right')
            plt.title('Линейная Регрессия')
            plt.grid(True)
            plt.show()
            d = {'Уравнение': ['Линейная (d=3)'], 'Коэф. дет.': ['R^2={:.2f}'.format(linear_r2)]}
            df = pd.DataFrame(data=d)
            model = pandasModel(df)
            view = QTableView()
            view.setModel(model)
            self.tableView_2.setModel(model)

        def Regr2():
            def estimate_coef(x, y):
                n = np.size(x)
                m_x = np.mean(x)
                m_y = np.mean(y)
                SS_xy = np.sum(y * x) - n * m_y * m_x
                SS_xx = np.sum(x * x) - n * m_x * m_x
                b_1 = SS_xy / SS_xx
                b_0 = m_y - b_1 * m_x
                return (b_0, b_1)

            def plot_regression_line(x, y, b):
                plt.scatter(x, y, color="b",
                            marker="o", s=35)
                y_pred = b[0] + b[1] * x
                plt.plot(x, y_pred, color="g")
                plt.xlabel('Вино')
                plt.ylabel('Спиртное')
                plt.title('Многомерная линейная Регрессия')
                plt.legend(['Вино', 'b0 + b1 * x'])
                plt.grid(True)
                plt.show()
                d = {'b0': [format(b[0])], 'b1': [format(b[1])]}
                df = pd.DataFrame(data=d)
                model = pandasModel(df)
                view = QTableView()
                view.setModel(model)
                self.tableView_2.setModel(model)

            def main():
                data = pd.read_csv("Drinks.csv")
                df = data.fillna(0)
                x = data['wine_servings']
                y = data['spirit_servings']
                b = estimate_coef(x, y)
                plot_regression_line(x, y, b)

            if __name__ == "__main__":
                main()

        def Regr3():
            df = pd.read_csv("Drinks.csv")
            df = df.fillna(0)
            X = df[['beer_servings']].values
            y = df['total_litres_of_pure_alcohol'].values

            regr = LinearRegression()

            cubic = PolynomialFeatures(degree=3)
            X_cubic = cubic.fit_transform(X)
            X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
            regr = regr.fit(X_cubic, y)
            y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
            cubic_r2 = r2_score(y, regr.predict(X_cubic))
            plt.scatter(X, y, label='training points', color='blue')
            plt.plot(X_fit, y_cubic_fit,
                     label='cubic (d=3), $R^2={:.2f}$'.format(cubic_r2),
                     color='green',
                     lw=2)
            plt.xlabel('Общее количество')
            plt.title('Кубическая Регрессия')
            plt.grid(True)
            plt.ylabel('Пиво')
            plt.legend(loc='lower right')
            plt.show()

            y = df.iloc[:, -1]
            X = np.array(df['beer_servings']).reshape(-1, 1)
            y = np.array(df['total_litres_of_pure_alcohol']).reshape(-1, 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
            model = LinearRegression()
            model.fit(X_train, y_train)
            model = LinearRegression().fit(X_train, y_train)
            r_sq = model.score(X_train, y_train)
            y_pred = model.predict(y_test)

            auto_types = (y_test)
            auto_df = pd.DataFrame(auto_types, columns=[['Пред. знач.']])
            df1 = pd.DataFrame(data=auto_df)
            model1 = pandasModel(df1)
            view = QTableView()
            view.setModel(model1)
            self.tableView_2.setModel(model1)

        ui.radioButton.clicked.connect(Regr1)
        ui.radioButton_2.clicked.connect(Regr2)
        ui.radioButton_3.clicked.connect(Regr3)

    def save(self):
        response = QFileDialog.getSaveFileName(
            parent=self,
            caption='Сохранение',
            directory='../SAVE/Data_File.csv',
            filter='Data File(*.csv)',
            initialFilter='CSV File (*.csv)'
        )
        print(response)
        return response[0]

    def ADAM(self):
        tf.keras.optimizers.Adam(
            learning_rate=learning,
            beta_1=beta1,
            beta_2=beta2,
            epsilon=epsilon,
            amsgrad=False,
            name='Adam',
        )
        data = pd.read_csv("diabetes.csv")
        dataset = data.values
        X = dataset[:, 0:7]
        X = X.astype('float')
        Y = dataset[:, 8]
        Y = Y.astype('float')
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scale = min_max_scaler.fit_transform(X)
        X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=trainsize)
        X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
        model = Sequential(
            [Dense(dense1, activation=activat), Dense(dense2, activation=activat), Dense(1, activation='sigmoid'), ])

        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
        hist = model.fit(X_train, Y_train, batch_size=batchsize, epochs=epochs,  validation_data=(X_val, Y_val))
        predict = model.predict(X_scale)
        data.insert(0, "prediction_adam", predict)

        y_true = data["Pregnancies"]
        y_pred = data["prediction_adam"]
        mean_error = mean_absolute_error(y_true, y_pred)
        mean_error = round(mean_error, 3)
        mean_error_str = str(mean_error)
        rmse = mean_squared_error(y_true, y_pred)
        rmse = round(rmse, 3)
        rmse = str(rmse)
        self.textEdit.setPlainText(mean_error_str)
        self.textEdit_2.setPlainText(rmse)

        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('Метод Adam')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper right')
        plt.show()

        df = data
        model = pandasModel(df)
        view = QTableView()
        view.setModel(model)
        self.tableView_2.setModel(model)

    def Adadelta(self):
        data = pd.read_csv("diabetes.csv")
        tf.keras.optimizers.Adadelta(
            learning_rate=learning,
            rho=rho,
            epsilon=epsilon,
            name='Adadelta',
        )

        dataset = data.values
        X = dataset[:, 1:2]
        X = X.astype('float')
        Y = dataset[:, 2]
        Y = Y.astype('float')
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scale = min_max_scaler.fit_transform(X)
        X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=trainsize)
        X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
        model = Sequential(
            [Dense(dense1, activation=activat), Dense(dense2, activation=activat),
             Dense(1, activation='sigmoid'), ])
        model.compile(loss='binary_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
        hist = model.fit(X_train, Y_train, batch_size=batchsize, epochs=epochs, validation_data=(X_val, Y_val))

        predict = model.predict(X_scale)
        data.insert(0, "prediction_adadelta", predict)

        y_true = data["Pregnancies"]
        y_pred = data["prediction_adadelta"]
        mean_error = mean_absolute_error(y_true, y_pred)
        mean_error = round(mean_error,3)
        mean_error_str = str(mean_error)
        rmse = mean_squared_error(y_true, y_pred)
        rmse = round(rmse, 3)
        rmse = str(rmse)
        self.textEdit.setPlainText(mean_error_str)
        self.textEdit_2.setPlainText(rmse)

        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('Метод Adadelta')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper right')
        plt.show()
        df = data
        model = pandasModel(df)
        view = QTableView()
        view.setModel(model)
        self.tableView_2.setModel(model)

    def SGD(self):
        data = pd.read_csv("diabetes.csv")
        tf.keras.optimizers.SGD(
            learning_rate=learning,
            nesterov=nesterov,
            name='SGD',
        )

        dataset = data.values
        X = dataset[:, 1:2]
        X = X.astype('float')
        Y = dataset[:, 2]
        Y = Y.astype('float')
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scale = min_max_scaler.fit_transform(X)
        X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=trainsize)
        X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
        model = Sequential(
            [Dense(dense1, activation=activat), Dense(dense2, activation=activat),
             Dense(1, activation='sigmoid'), ])
        model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['accuracy'])
        hist = model.fit(X_train, Y_train, batch_size=batchsize, epochs=epochs, validation_data=(X_val, Y_val))
        predict = model.predict(X_scale)
        data.insert(0, "prediction_SGD", predict)

        y_true = data["Pregnancies"]
        y_pred = data["prediction_SGD"]
        mean_error = mean_absolute_error(y_true, y_pred)
        mean_error = round(mean_error, 3)
        mean_error_str = str(mean_error)
        rmse = mean_squared_error(y_true, y_pred)
        rmse = round(rmse, 3)
        rmse = str(rmse)
        self.textEdit.setPlainText(mean_error_str)

        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('Метод SGD')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper right')
        plt.show()
        df = data
        model = pandasModel(df)
        view = QTableView()
        view.setModel(model)
        self.tableView_2.setModel(model)

    def RMSprop(self):
        data = pd.read_csv("diabetes.csv")
        tf.keras.optimizers.RMSprop(
            learning_rate=learning,
            rho=rho,
            momentum=moment,
            epsilon=epsilon,
            centered=False,
            name='RMSprop',
        )

        dataset = data.values
        X = dataset[:, 1:2]
        X = X.astype('float')
        Y = dataset[:, 2]
        Y = Y.astype('float')
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scale = min_max_scaler.fit_transform(X)
        X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=trainsize)
        X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
        model = Sequential(
            [Dense(dense1, activation=activat), Dense(dense2, activation=activat),
             Dense(1, activation='sigmoid'), ])
        model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
        hist = model.fit(X_train, Y_train, batch_size=batchsize, epochs=epochs, validation_data=(X_val, Y_val))
        predict = model.predict(X_scale)
        data.insert(0, "prediction_RMSprop", predict)

        y_true = data["Pregnancies"]
        y_pred = data["prediction_RMSprop"]
        mean_error = mean_absolute_error(y_true, y_pred)
        mean_error = round(mean_error, 3)
        mean_error_str = str(mean_error)
        rmse = mean_squared_error(y_true, y_pred)
        rmse = round(rmse, 3)
        rmse = str(rmse)
        self.textEdit.setPlainText(mean_error_str)

        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('Метод RMSprpo')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper right')
        plt.show()
        df = data
        model = pandasModel(df)
        view = QTableView()
        view.setModel(model)
        self.tableView_2.setModel(model)

    def Adagard(self):
        data = pd.read_csv("diabetes.csv")
        tf.keras.optimizers.Adagrad(
            learning_rate=learning,
            initial_accumulator_value=initialacc,
            epsilon=epsilon,
            name='Adagrad',
        )

        dataset = data.values
        X = dataset[:, 1:2]
        X = X.astype('float')
        Y = dataset[:, 2]
        Y = Y.astype('float')
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scale = min_max_scaler.fit_transform(X)
        X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=trainsize)
        X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
        model = Sequential(
            [Dense(dense1, activation=activat), Dense(dense2, activation=activat),
             Dense(1, activation='sigmoid'), ])
        model.compile(loss='binary_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
        hist = model.fit(X_train, Y_train, batch_size=batchsize, epochs=epochs, validation_data=(X_val, Y_val))

        predict = model.predict(X_scale)
        data.insert(0, "prediction_adagrad", predict)

        y_true = data["Pregnancies"]
        y_pred = data["prediction_adagrad"]
        mean_error = mean_absolute_error(y_true, y_pred)
        mean_error = round(mean_error, 3)
        mean_error_str = str(mean_error)
        rmse = mean_squared_error(y_true, y_pred)
        rmse = round(rmse, 3)
        rmse = str(rmse)
        self.textEdit.setPlainText(mean_error_str)

        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('Метод AdaGard')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper right')
        plt.show()
        df = data
        model = pandasModel(df)
        view = QTableView()
        view.setModel(model)
        self.tableView_2.setModel(model)

    def GraphR2(self):
        def estimate_coef(x, y):
            n = np.size(x)
            m_x = np.mean(x)
            m_y = np.mean(y)
            SS_xy = np.sum(y * x) - n * m_y * m_x
            SS_xx = np.sum(x * x) - n * m_x * m_x
            b_1 = SS_xy / SS_xx
            b_0 = m_y - b_1 * m_x
            return (b_0, b_1)

        def plot_regression_line(x, y, b):
            plt.scatter(x, y, color="b",
                        marker="o", s=35)
            y_pred = b[0] + b[1] * x
            plt.plot(x, y_pred, color="g")
            plt.xlabel('Вино')
            plt.ylabel('Спиртное')
            plt.title('Многомерная линейная Регрессия')
            plt.legend(['Вино', 'b0 + b1 * x'])
            plt.grid(True)
            plt.show()
            d = {'b0': [format(b[0])], 'b1': [format(b[1])]}
            df = pd.DataFrame(data=d)
            model = pandasModel(df)
            view = QTableView()
            view.setModel(model)
            self.tableView_2.setModel(model)

        def main():
            data = pd.read_csv("Drinks.csv")
            data = data.fillna(0)
            x = data['wine_servings']
            y = data['spirit_servings']
            b = estimate_coef(x, y)
            plot_regression_line(x, y, b)

        if __name__ == "__main__":
            main()

    def openfile(self):
        global fname
        global df
        fname = QFileDialog.getOpenFileName(self, 'Open File', 'c:\\', "Image files (*.csv)")
        f = open(fname[0], 'r')
        df = pd.read_csv(f)
        model = pandasModel(df)
        view = QTableView()
        view.setModel(model)
        self.tableView.setModel(model)

    def help(self):
        msg = QMessageBox()
        msg.setWindowTitle("Помощь")
        msg.setText("Никто не поможет :(")

        x = msg.exec_()

    def GraphR3(self):
        diabets_data = datasets.load_diabetes()
        di = pd.DataFrame(diabets_data.data)
        di.columns = diabets_data.feature_names
        di['target'] = diabets_data.target
        x = di.drop('target', axis=1)
        rm = linear_model.LinearRegression()
        rm.fit(x, di.target)
        plt.title('Полимерная Регрессия')
        plt.xlabel('Пиво')
        plt.ylabel('Спиртное')
        plt.scatter(di.target, rm.predict(x))
        plt.legend(['Отношение'])
        plt.grid(True)
        plt.show()

    def graphP(self):
        global MainWindow
        MainWindow = QtWidgets.QMainWindow()
        ui = GraphP()
        ui.setupUi(MainWindow)

    def graphSp(self):
        global MainWindow
        MainWindow = QtWidgets.QMainWindow()
        ui = GraphSp()
        ui.setupUi(MainWindow)

    def graphS(self):
        global MainWindow
        MainWindow = QtWidgets.QMainWindow()
        ui = GraphS()
        ui.setupUi(MainWindow)

    def graphD(self):
        global MainWindow
        MainWindow = QtWidgets.QMainWindow()
        ui = GraphD()
        ui.setupUi(MainWindow)

    def Random(self):
        data = pd.read_csv("cars_raw.csv")
        dataset = data.values
        X = dataset[:, 6:7]
        X = X.astype('float')
        Y = dataset[:, 10]
        Y = Y.astype('float')
        model = RandomForestRegressor(n_estimators=20, oob_score=True, random_state=1)
        model.fit(X, Y)
        plt.plot(Y, model.predict(X), color='green')
        plt.title('Метод Случайного леса')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper right')
        plt.show()

    def graph3D(self):
        global MainWindow
        MainWindow = QtWidgets.QMainWindow()
        ui = Graph3D()
        ui.setupUi(MainWindow)

    def graphR(self):
        global MainWindow
        MainWindow = QtWidgets.QMainWindow()
        ui = GraphR()
        ui.setupUi(MainWindow)

    def taxSpin(self):
        global MainWindow
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()

        def CalculateTax():
            price = int(ui.price_box.toPlainText())
            tax = (ui.tax_rate.value())
            total_price = price + ((tax / 100) * price)
            total_price_string = "Tax price: " + str(total_price)
            ui.result_window.setText(total_price_string)

        ui.calc_tax_button.clicked.connect(CalculateTax)

    def data_gaps_is_null(self):
        model = pandasModel(df.isnull())
        view = QTableView()
        view.setModel(model)
        self.tableView_2.setModel(model)

    def data_gaps_pad(self):
        model = pandasModel(df.fillna(method='pad'))
        view = QTableView()
        view.setModel(model)
        self.tableView_2.setModel(model)

    def data_gaps_fill(self):
        model = pandasModel(df.fillna(0))
        view = QTableView()
        view.setModel(model)
        self.tableView_2.setModel(model)

    def data_gaps_akima(self):
        model = pandasModel(df.interpolate(method="akima"))
        view = QTableView()
        view.setModel(model)
        self.tableView_2.setModel(model)

    def data_gaps_polynomial(self):
        model = pandasModel(df.interpolate(method="polynomial", order=3))
        view = QTableView()
        view.setModel(model)
        self.tableView_2.setModel(model)

    def desc(self):
        model_new = pandasModel(df.describe())
        view = QTableView()
        view.setModel(model_new)
        self.tableView_2.setModel(model_new)

    def printDialog(self):
        printer = QPrinter(QPrinter.HighResolution)
        dialog = QPrintDialog(printer, self)
        if dialog.exec() == QPrintDialog.Accepted:
            self.textEdit.print(printer)

    def paintCell(self):
        global Dialog
        Dialog = QtWidgets.QDialog()
        ui = Ui_Dialog()
        ui.setupUi(Dialog)
        Dialog.show()

class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    Win = Window()
    app.exec_()