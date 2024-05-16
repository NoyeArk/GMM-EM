import csv
import cv2
import sys
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from datetime import datetime
from matplotlib import pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPalette, QBrush, QPixmap, QPainter, QImage, QIcon
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QGraphicsPixmapItem, QGraphicsScene, QMessageBox

from back_end import em


class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.select_data_label = None
        self.centralwidget = None
        self.output_gragh2d_label = None

        self.row_data = None
        self.handled_data = None
        self.model = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 600)
        MainWindow.setMinimumSize(QtCore.QSize(1000, 600))
        MainWindow.setMaximumSize(QtCore.QSize(1000, 600))

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.title_label = QtWidgets.QLabel(self.centralwidget)
        self.title_label.setGeometry(QtCore.QRect(270, 10, 550, 81))
        self.title_label.setAutoFillBackground(False)
        self.title_label.setStyleSheet("font: bold 28pt \"Calibri\"; \n" "color: #c9701a;")
        self.title_label.setObjectName("title_label")

        self.output_label = QtWidgets.QLabel(self.centralwidget)
        self.output_label.setGeometry(QtCore.QRect(60, 260, 241, 81))
        self.output_label.setStyleSheet("font: 20pt \"AcadEref\";\n""color: rgb(0, 0, 0);")
        self.output_label.setObjectName("output_label")

        self.output_area = QtWidgets.QTextBrowser(self.centralwidget)
        self.output_area.setGeometry(QtCore.QRect(60, 340, 441, 191))
        self.output_area.setStyleSheet("""
                background-color: rgb(255, 255, 255);  /* 背景颜色 */
                color: rgb(0, 0, 0);                    /* 文本颜色 */
                padding: 10px;                          /* 内边距 */
                border-radius: 5px;                     /* 边框圆角 */
                border: 1px solid rgb(200, 200, 200);   /* 边框样式 */
                font-size: 16px;                        /* 字体大小 */
            """)
        self.output_area.setObjectName("output_area")
        self.output_area.setText('> 初始化成功！')

        self.output_gragh2d_label = QtWidgets.QLabel(self.centralwidget)
        self.output_gragh2d_label.setGeometry(QtCore.QRect(560, 260, 161, 81))
        self.output_gragh2d_label.setStyleSheet("font: 20pt \"3ds\";\n""color: rgb(0, 0, 0);")
        self.output_gragh2d_label.setObjectName("output_gragh2d_label")

        self.select_data_label = QtWidgets.QLabel(self.centralwidget)
        self.select_data_label.setGeometry(QtCore.QRect(60, 80, 191, 81))
        self.select_data_label.setStyleSheet("font: 20pt \"AcadEref\";\n" "color: rgb(0, 0, 0);")
        self.select_data_label.setObjectName("select_data_label")
        self.select_data_btn = QtWidgets.QPushButton(self.centralwidget)
        self.select_data_btn.setGeometry(QtCore.QRect(65, 200, 91, 31))
        self.select_data_btn.setObjectName("select_data_btn")
        self.set_btn_style(self.select_data_btn)

        self.data_name_label = QtWidgets.QLabel(self.centralwidget)
        self.data_name_label.setGeometry(QtCore.QRect(70, 140, 150, 61))
        self.data_name_label.setStyleSheet("font: 13pt \"Times New Roman\";\n" "color: rgb(0, 0, 0);")
        self.data_name_label.setObjectName("data_name_label")

        self.process_data_label = QtWidgets.QLabel(self.centralwidget)
        self.process_data_label.setGeometry(QtCore.QRect(300, 80, 200, 81))
        self.process_data_label.setStyleSheet("font: 20pt \"AcadEref\";\n" "color: rgb(0, 0, 0);")
        self.process_data_label.setObjectName("process_data_label")

        self.find_abnormal_btn = QtWidgets.QPushButton(self.centralwidget)
        self.find_abnormal_btn.setGeometry(QtCore.QRect(300, 160, 91, 31))
        self.find_abnormal_btn.setObjectName("find_abnormal_btn")
        self.set_btn_style(self.find_abnormal_btn)

        self.find_deletion_btn = QtWidgets.QPushButton(self.centralwidget)
        self.find_deletion_btn.setGeometry(QtCore.QRect(410, 160, 91, 31))
        self.find_deletion_btn.setObjectName("find_repeat_label")
        self.set_btn_style(self.find_deletion_btn)

        self.find_statistic_btn = QtWidgets.QPushButton(self.centralwidget)
        self.find_statistic_btn.setGeometry(QtCore.QRect(300, 200, 91, 31))
        self.find_statistic_btn.setObjectName("find_statistic_label")
        self.set_btn_style(self.find_statistic_btn)

        self.save_data_btn = QtWidgets.QPushButton(self.centralwidget)
        self.save_data_btn.setGeometry(QtCore.QRect(410, 200, 91, 31))
        self.save_data_btn.setObjectName("save_data_btn")
        self.set_btn_style(self.save_data_btn)

        self.cluster_analysis_label = QtWidgets.QLabel(self.centralwidget)
        self.cluster_analysis_label.setGeometry(QtCore.QRect(560, 80, 191, 81))
        self.cluster_analysis_label.setStyleSheet("font: 20pt \"AcadEref\";\n" "color: rgb(0, 0, 0);")
        self.cluster_analysis_label.setObjectName("cluster_analysis_label")
        self.cluster_num_label = QtWidgets.QLabel(self.centralwidget)
        self.cluster_num_label.setGeometry(QtCore.QRect(560, 130, 171, 81))
        self.cluster_num_label.setStyleSheet("font: 14pt \"AcadEref\";\n" "color: rgb(0, 0, 0);")
        self.cluster_num_label.setObjectName("cluster_num_label")
        self.iter_num_label = QtWidgets.QLabel(self.centralwidget)
        self.iter_num_label.setGeometry(QtCore.QRect(560, 170, 171, 81))
        self.iter_num_label.setStyleSheet("font: 14pt \"AcadEref\";\n" "color: rgb(0, 0, 0);")
        self.iter_num_label.setObjectName("iter_num_label")

        self.cluster_num_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.cluster_num_edit.setGeometry(QtCore.QRect(730, 160, 61, 20))
        self.cluster_num_edit.setObjectName("lineEdit")
        self.iter_num_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.iter_num_edit.setGeometry(QtCore.QRect(730, 200, 61, 20))
        self.iter_num_edit.setObjectName("lineEdit_2")
        self.eps_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.eps_edit.setGeometry(QtCore.QRect(730, 240, 61, 20))
        self.eps_edit.setObjectName("lineEdit_3")

        self.eps_num_label = QtWidgets.QLabel(self.centralwidget)
        self.eps_num_label.setGeometry(QtCore.QRect(560, 210, 171, 81))
        self.eps_num_label.setStyleSheet("font: 14pt \"AcadEref\";\n" "color: rgb(0, 0, 0);")
        self.eps_num_label.setObjectName("eps_num_label")

        self.run_btn = QtWidgets.QPushButton(self.centralwidget)
        self.run_btn.setGeometry(QtCore.QRect(840, 160, 91, 31))
        self.run_btn.setObjectName("run_btn")
        self.set_btn_style(self.run_btn)

        self.output_gragh3d_label = QtWidgets.QLabel(self.centralwidget)
        self.output_gragh3d_label.setGeometry(QtCore.QRect(770, 260, 161, 81))
        self.output_gragh3d_label.setStyleSheet("font: 20pt \"3ds\";\n" "color: rgb(0, 0, 0);")
        self.output_gragh3d_label.setObjectName("output_gragh3d_label")
        self.graphics_view2d = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphics_view2d.setGeometry(QtCore.QRect(550, 340, 191, 192))
        self.graphics_view2d.setObjectName("graphics_view2d")
        self.graphics_view3d = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphics_view3d.setGeometry(QtCore.QRect(770, 340, 191, 192))
        self.graphics_view3d.setObjectName("graphics_view3d")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.translateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.set_event()

        self.eps_edit.setText('0.0001')
        self.iter_num_edit.setText('100')
        self.cluster_num_edit.setText('5')

    def translateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "GMM-EM算法聚类分析"))
        self.title_label.setText(_translate("MainWindow", "高斯混合模型-EM算法聚类分析"))
        self.output_label.setText(_translate("MainWindow", "指令执行运行结果："))
        self.output_gragh2d_label.setText(_translate("MainWindow", "2D-聚类结果"))
        self.select_data_label.setText(_translate("MainWindow", "请选择数据集："))
        self.select_data_btn.setText(_translate("MainWindow", "选择"))
        self.data_name_label.setText(_translate("MainWindow", "…"))
        self.process_data_label.setText(_translate("MainWindow", "查看数据集信息："))
        self.find_abnormal_btn.setText(_translate("MainWindow", "查看异常值"))
        self.find_deletion_btn.setText(_translate("MainWindow", "查看缺失值"))
        self.find_statistic_btn.setText(_translate("MainWindow", "数据统计"))
        self.save_data_btn.setText(_translate("MainWindow", "保存数据"))
        self.cluster_analysis_label.setText(_translate("MainWindow", "聚类分析："))
        self.cluster_num_label.setText(_translate("MainWindow", "输入聚类数量／个："))
        self.iter_num_label.setText(_translate("MainWindow", "输入迭代次数／次："))
        self.eps_num_label.setText(_translate("MainWindow", "输入迭代阈值："))
        self.run_btn.setText(_translate("MainWindow", "开始分析"))
        self.output_gragh3d_label.setText(_translate("MainWindow", "3D-聚类结果"))

    def set_event(self):
        self.select_data_btn.clicked.connect(self.select_dataset)
        self.find_abnormal_btn.clicked.connect(self.check_abnormal)
        self.find_deletion_btn.clicked.connect(self.check_deletion)
        self.find_statistic_btn.clicked.connect(self.check_statistic)
        self.save_data_btn.clicked.connect(self.save_data)
        self.run_btn.clicked.connect(self.run_model)

    def set_btn_style(self, obj):
        obj.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                font-size: 16px;
                font-family: "KaiTi";
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3c923d;
            }
        """)

    def select_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "选择文件", "", "All Files (*)")
        if file_path:
            path_parts = file_path.split("/")
            self.data_name_label.setText(path_parts[-1])

            print('读取数据集')
            self.row_data = pd.read_csv('../dataset/' + path_parts[-1])
            self.handled_data = read_data('../dataset/after_pca_data.csv')
            print('读取数据集成功')
        else:
            print('error')

    def check_abnormal(self):
        outcome = self.row_data.duplicated().sum()  # 检查是否有重复值
        self.output_area.append('> 查看异常值个数：' + str(outcome))

    def check_deletion(self):
        outcome = self.row_data.notnull().any()  # 检查是否有重复值
        self.output_area.append('> 查看缺失值：\n' + str(outcome))

    def check_statistic(self):
        self.output_area.append('> 查看数据集统计信息：')

        outcome = self.row_data.describe()
        name = outcome.columns.tolist()
        value = outcome.values.tolist()
        sense = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

        for i in range(len(name)):
            self.output_area.append('----------列名：' + str(name[i]) + '----------')
            for j in range(len(sense)):
                out = sense[j] + ' '
                out += str(value[j][i]) + ' '
                self.output_area.append(out)
            i += 1  # 每行显示1个列信息

    def run_model(self):
        # 得到用户输入的参数
        cluster_num = int(self.cluster_num_edit.text())
        iter_num = int(self.iter_num_edit.text())
        eps = float(self.eps_edit.text())

        # 初始化模型
        self.model = em.GMM(cluster_num, iter_num, eps)

        self.output_area.append('> 初始化模型...')

        # 训练模型
        y_pred = self.model.fit(self.handled_data)
        self.output_area.append('> 模型训练结束，正在预测...')

        # 预测结果
        self.output_area.append(str(y_pred))
        print(str(y_pred))

        # 绘制2D聚类效果图
        plt.figure(figsize=(8, 8))
        plt.scatter(self.handled_data[:, 0], self.handled_data[:, 1], c=y_pred, cmap='viridis')
        for ii in np.arange(205):
            plt.text(self.handled_data[ii, 0], self.handled_data[ii, 1], s='')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('GMM PCA')
        plt.savefig('2d_outcome.png')

        # 绘制3D聚类效果图
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, projection='3d')
        ax.scatter(self.handled_data[:, 0], self.handled_data[:, 1], self.handled_data[:, 2], c=y_pred, cmap='viridis')

        # 视角转换，转换后更易看出簇群
        ax.view_init(30, 45)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.savefig('3d_outcome.png')

        self.display_graph('2d_outcome.png')
        self.display_graph('3d_outcome.png')

    def display_graph(self, graph_name):
        img = cv2.imread(graph_name)  # 读取图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)  # 创建像素图元
        scene = QGraphicsScene()  # 创建场景
        scene.addItem(item)

        if graph_name == '2d_outcome.png':
            self.graphics_view2d.setScene(scene)
            self.graphics_view2d.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
            self.graphics_view2d.show()
        else:
            self.graphics_view3d.setScene(scene)
            self.graphics_view3d.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
            self.graphics_view3d.show()

    def save_data(self):
        file_name = QFileDialog.getSaveFileName(self, '保存文件', 'data')
        self.output_area.append('> 保存数据中...')
        self.row_data.to_csv(file_name[0] + '.csv', index=False)
        self.output_area.append('> 保存成功！')
        self.output_area.append('> 路径为：' + file_name[0] + '.csv')


def read_data(filename):
    x = []  # 存储读取到的数据

    with open(filename, 'r') as file:
        csv_reader = csv.reader(file, delimiter=' ')
        for row in csv_reader:
            # 将每一行数据转换为浮点数列表
            row_data = [float(val) for val in row]
            x.append(row_data)

    return np.array(x)


if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()  # 创建窗体对象

    ui = Ui_MainWindow()  # 创建PyQt设计的窗体对象
    ui.setupUi(MainWindow)  # 调用PyQt窗体的方法对窗体对象进行初始化设置

    palette = MainWindow.palette()
    palette.setBrush(QPalette.Background, QBrush(QPixmap("../img/background.jpg")))
    MainWindow.setPalette(palette)  # 设置窗口背景颜色
    MainWindow.setWindowIcon(QIcon("../img/logo.png"))

    MainWindow.show()  # 显示窗体
    sys.exit(app.exec_())  # 程序关闭时退出进程
