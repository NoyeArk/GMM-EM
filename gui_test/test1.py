import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class Demo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(500,500)
        self.setWindowTitle('设置窗口背景色')
        self.setObjectName('win')#设置窗口名，相当于CSS中的ID
        self.setStyleSheet('#win{background-color:blue;}')#设置窗口背景颜色

if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = Demo()
    form.show()
    sys.exit(app.exec_())
