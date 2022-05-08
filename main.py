import sys
import random
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCharts import QChartView, QBarCategoryAxis, QValueAxis, QBarSet, QBarSeries, QLineSeries, QChart
from PySide6.QtCore import Qt, QPoint, QThread, Signal
from PySide6.QtGui import QPainter, QImage, QPixmap, QIcon, QColor
import cv2
from PySide6.QtWidgets import QStyle
import onnxruntime
import numpy as np
import time

session = onnxruntime.InferenceSession("tiny-yolov3-11.onnx")
inname = [input.name for input in session.get_inputs()]
outname = [output.name for output in session.get_outputs()]

def frame_process(frame, input_shape=(416, 416)):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_shape)
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    return image


def get_prediction(image_data, image_size):
    input = {
        inname[0]: image_data,
        inname[1]: image_size
    }
    t0 = time.time()
    boxes, scores, indices = session.run(outname, input)
    predict_time = time.time() - t0

    out_boxes, out_scores, out_classes = [], [], []
    for idx_ in indices[0]:
        out_classes.append(idx_[1])
        out_scores.append(scores[tuple(idx_)])
        idx_1 = (idx_[0], idx_[2])
        out_boxes.append(boxes[idx_1])
        #out_boxes.append(boxes[idx_1])
    return out_boxes, out_scores, out_classes, predict_time

label =["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",
    "knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
    "pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor",
    "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]


class Thread(QThread):
    updateFrame = Signal(QImage)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.status = True
        self.cap = True

    def stop_video(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture("traffic.mp4")

    def run(self):
        self.cap = cv2.VideoCapture("traffic.mp4")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Reading frame in gray scale to process the pattern
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # scaled_img = cv2.resize(color_frame, (416, 416))

            image_data = frame_process(frame, input_shape=(416, 416))
            image_size = np.array([416, 416], dtype=np.float32).reshape(1, 2)
            out_boxes, out_scores, out_classes, predict_time = get_prediction(image_data, image_size)
            # sum_time += predict_time
            # sum_frame += 1
            out_boxes = np.array(out_boxes).tolist()
            out_scores = np.array(out_scores).tolist()
            out_classes = np.array(out_classes).tolist()

            image = cv2.resize(frame, (416, 416))

            for i in range(0, len(out_boxes)):
                if 0 == 0:
                    coor = np.array(out_boxes[i], dtype=np.int32)
                    text = label[out_classes[i]]
                    y, x, h, w = coor
                    cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 1)
                    # cv2.rectangle(image, (x, y + 10), (w, y), (0, 0, 255), 10)
                    font = cv2.FONT_HERSHEY_COMPLEX
                    text = "Class: {0}, Score: {1}".format(label[out_classes[i]], round(out_scores[i], 2))
                    cv2.putText(image, text, (x, y - 5),
                                font, 0.3, (255, 255, 255), 1)

            # Creating and scaling QImage
            h, w, ch = image.shape
            img = QImage(image.data, w, h, ch * w, QImage.Format_RGB888)
            cv2.waitKey(20)
            # Emit signal
            self.updateFrame.emit(img)
        sys.exit(-1)


class MainWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        icon = self.style().standardIcon( QStyle.SP_MediaPlay)
        self.button = QtWidgets.QPushButton(icon, "Start")

        #delete_icon = self.style().standardIcon(QStyle.SP_DialogCancelButton)
        #self.delete_button = QtWidgets.QPushButton(delete_icon, "Delete")
        #self.delete_button.setStyleSheet("background-color: darkred;")
        self.image = QtWidgets.QLabel(self)
        self.image.setFixedSize(416, 416)

        self.layout = QtWidgets.QVBoxLayout(self)
        # self.layout.addWidget(QtWidgets.QLabel("Text"))
        # self.layout.addWidget(QtWidgets.QLineEdit())
        self.layout.addWidget(self.button)
        #self.layout.addWidget(self.delete_button)
        self.layout.addWidget(self.image)

        self.button.clicked.connect(self.start_video)
        self.cam: Thread = None

    @QtCore.Slot()
    def start_video(self):
        self.cam = Thread(self)
        self.cam.start()
        self.cam.updateFrame.connect(self.setImage)

    @QtCore.Slot(QImage)
    def setImage(self, image):
        # print(image)
        self.image.setPixmap(QPixmap.fromImage(image))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    app.setApplicationDisplayName("Python Object Detection")
    styles = QtWidgets.QStyleFactory.keys()
    app.setStyle("Fusion")

    f = open("style.css", "r")
    style = f.read()

    app.setStyleSheet(style)
    widget = MainWidget()
    #widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())