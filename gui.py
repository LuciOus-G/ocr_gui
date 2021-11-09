import sys
import os
import argparse
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QObject, Qt, pyqtSlot, pyqtSignal, QThread, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QVBoxLayout, QWidget, QShortcut,\
    QPushButton, QTextEdit, QLineEdit, QFormLayout, QHBoxLayout, QCheckBox, QSpinBox, QDoubleSpinBox

from pynput.mouse import Controller

from PIL import ImageGrab
import numpy as np
from screeninfo import get_monitors

QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
import pprint
import easyocr


class App(QMainWindow):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.initUI()
        self.snipWidget = SnipWidget(self)

        self.show()

    def initUI(self):
        self.setWindowTitle("LaTeX OCR")
        # QApplication.setWindowIcon(QtGui.QIcon(':/icons/icon.svg'))
        self.left = 500
        self.top = 300
        self.width = 900
        self.height = 700
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create LaTeX display
        self.webView = QWebEngineView()
        self.webView.setHtml("")
        self.webView.setMinimumHeight(80)

        # Create textbox
        self.textbox = QTextEdit(self)
        # self.textbox.textChanged.connect(self.displayPrediction)
        self.textbox.setMinimumHeight(40)
        self.textbox.setFontPointSize(18)

        # Create snip button
        self.snipButton = QPushButton('Snip [Alt+S]', self)
        self.snipButton.clicked.connect(self.onClick)

        self.shortcut = QShortcut(QKeySequence("Alt+S"), self)
        self.shortcut.activated.connect(self.onClick)

        # Create retry button
        self.retryButton = QPushButton('Retry', self)
        self.retryButton.setEnabled(False)
        self.retryButton.clicked.connect(self.returnSnip)

        # Create layout
        centralWidget = QWidget()
        centralWidget.setMinimumWidth(200)
        self.setCentralWidget(centralWidget)

        lay = QVBoxLayout(centralWidget)
        lay.addWidget(self.webView, stretch=4)
        lay.addWidget(self.textbox, stretch=2)
        buttons = QHBoxLayout()
        buttons.addWidget(self.snipButton)
        buttons.addWidget(self.retryButton)
        lay.addLayout(buttons)
        settings = QFormLayout()
        lay.addLayout(settings)

    @pyqtSlot()
    def onClick(self):
        self.close()
        self.snipWidget.snip()

    def returnSnip(self, img=None):
        # Show processing icon
        self.webView.setHtml('<img src="test.png">',
                             baseUrl=QUrl.fromLocalFile(os.getcwd() + os.path.sep))

        reader = easyocr.Reader(['en'])
        data_new = reader.readtext('test.png')

        list_data = []
        list_result = []

        for dats in data_new:
            list_data.append(dats[-2].replace('x', '*').replace('X', '*').encode('utf-8'))

        for final in list_data:
            x = eval(final)
            list_result.append(f"Hasil dari perhutingan {str(final)} adalah = {str(x)}")
            print(x)
        result = '\n'.join(str(x) for x in list_result)
        self.textbox.setText(result)

        self.snipButton.setEnabled(True)
        self.retryButton.setEnabled(False)

        self.show()
                                                                               #    3 * 3
class SnipWidget(QMainWindow):
    isSnipping = False

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        monitos = get_monitors()
        bboxes = np.array([[m.x, m.y, m.width, m.height] for m in monitos])
        x, y, _, _ = bboxes.min(0)
        w, h = bboxes[:, [0, 2]].sum(1).max(), bboxes[:, [1, 3]].sum(1).max()
        self.setGeometry(x, y, w-x, h-y)

        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()

        self.mouse = Controller()

    def snip(self):
        self.isSnipping = True
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.show()

    def paintEvent(self, event):
        if self.isSnipping:
            brushColor = (0, 180, 255, 100)
            lw = 3
            opacity = 0.3
        else:
            brushColor = (255, 255, 255, 0)
            lw = 3
            opacity = 0

        self.setWindowOpacity(opacity)
        qp = QtGui.QPainter(self)
        qp.setPen(QtGui.QPen(QtGui.QColor('black'), lw))
        qp.setBrush(QtGui.QColor(*brushColor))
        qp.drawRect(QtCore.QRect(self.begin, self.end))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            QApplication.restoreOverrideCursor()
            self.close()
            self.parent.show()
        event.accept()

    def mousePressEvent(self, event):
        self.startPos = self.mouse.position

        self.begin = event.pos()
        self.end = self.begin
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.isSnipping = False
        QApplication.restoreOverrideCursor()

        startPos = self.startPos
        endPos = self.mouse.position

        x1 = min(startPos[0], endPos[0])
        y1 = min(startPos[1], endPos[1])
        x2 = max(startPos[0], endPos[0])
        y2 = max(startPos[1], endPos[1])

        self.repaint()
        QApplication.processEvents()
        img = ImageGrab.grab(bbox=(x1, y1, x2, y2), all_screens=True)
        img.save('test.png')
        QApplication.processEvents()

        self.close()
        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()
        self.parent.returnSnip(img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GUI arguments')
    parser.add_argument('-t', '--temperature', type=float, default=.2, help='Softmax sampling frequency')
    parser.add_argument('-c', '--config', type=str, default='settings/config.yaml', help='path to config file')
    parser.add_argument('-m', '--checkpoint', type=str, default='checkpoints/weights.pth', help='path to weights file')
    parser.add_argument('--no-cuda', action='store_true', help='Compute on CPU')
    parser.add_argument('--no-resize', action='store_true', help='Resize the image beforehand')
    arguments = parser.parse_args()
    latexocr_path = os.path.dirname(sys.argv[0])
    if latexocr_path != '':
        sys.path.insert(0, latexocr_path)
        os.chdir(latexocr_path)
    app = QApplication(sys.argv)
    ex = App(arguments)
    sys.exit(app.exec_())
