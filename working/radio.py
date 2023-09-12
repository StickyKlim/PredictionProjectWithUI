import scipy
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import stats
from sklearn import datasets, linear_model

class Radio(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        Dialog.setStyleSheet("background-color: rgb(14, 255, 247);")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(60, 20, 321, 51))
        font = QtGui.QFont()
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.radioButton = QtWidgets.QRadioButton(Dialog)
        self.radioButton.toggled.connect(self.Regr1)

        self.radioButton.setGeometry(QtCore.QRect(60, 90, 191, 17))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.radioButton.setFont(font)
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(Dialog)
        self.radioButton_2.setGeometry(QtCore.QRect(60, 130, 82, 17))
        self.radioButton_2.toggled.connect(self.Regr2)

        font = QtGui.QFont()
        font.setPointSize(12)
        self.radioButton_2.setFont(font)
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_3 = QtWidgets.QRadioButton(Dialog)
        self.radioButton_3.setGeometry(QtCore.QRect(60, 180, 181, 17))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.radioButton_3.setFont(font)
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_3.toggled.connect(self.Regr3)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def Regr1(self, selected):
        if selected:
            data = pd.read_csv("Drinks.csv")
            x = data['total_litres_of_pure_alcohol']
            y = data['spirit_servings']
            slope, intercept, r, p, std_err = scipy.stats.linregress(x, y)

            def myfunc(x):
                return slope * x + intercept

            mymodel = list(map(myfunc, x))
            plt.scatter(x, y)
            plt.plot(x, mymodel)
            plt.ylabel('Общее количество алкоголя')
            plt.xlabel('Спиртное')
            plt.show()

    def Regr2(self, selected):
        if selected:
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
                plt.show()

            def main():
                data = pd.read_csv("Drinks.csv")
                x = data['wine_servings']
                y = data['spirit_servings']
                b = estimate_coef(x, y)
                plot_regression_line(x, y, b)

            if __name__ == "__main__":
                main()

    def Regr3(self, selected):
        if selected:
            diabets_data = datasets.load_diabetes()
            di = pd.DataFrame(diabets_data.data)
            di.columns = diabets_data.feature_names
            di['target'] = diabets_data.target
            x = di.drop('target', axis=1)
            rm = linear_model.LinearRegression()
            rm.fit(x, di.target)
            plt.xlabel('Пиво')
            plt.ylabel('Спиртное')
            plt.scatter(di.target, rm.predict(x))
            plt.show()

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "radiobutton"))
        self.label.setText(_translate("Dialog", "Выбрать Регрессию"))
        self.radioButton.setText(_translate("Dialog", "Линейная регрессия"))
        self.radioButton_2.setText(_translate("Dialog", "МЛР"))
        self.radioButton_3.setText(_translate("Dialog", "Кубическая регрессия"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Radio()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
