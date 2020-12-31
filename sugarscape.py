import src
import pyqt
import sys

if __name__ == "__main__":
    app = pyqt.QApplication(sys.argv)

    window = pyqt.MainWindow()
    window.show()

    sys.exit(app.exec_())