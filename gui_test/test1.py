from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon, QPixmap, QPainter

class Ui_Form(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Form")
        self.resize(468, 352)
        self.setWindowIcon(QIcon('bg2.jpg'))


    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = QPixmap("bg1.jpg")
        # 绘制窗口背景，平铺到整个窗口，随着窗口改变而改变
        painter.drawPixmap(self.rect(), pixmap)



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_Form()
    ui.show()
    sys.exit(app.exec_())

