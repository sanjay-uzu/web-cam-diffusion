import sys
import random
import cv2
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from pipeline import Pipeline
# Your existing imports and functions...

class WebcamWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initWebcam()
        self.pipeline=Pipeline()

    def initUI(self):
        # Initialize layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create labels for displaying frames
        self.webcamLabel = QLabel()
        self.processedLabel = QLabel()

        layout.addWidget(self.webcamLabel)
        layout.addWidget(self.processedLabel)

        # Timer for updating frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30)  # Update interval in milliseconds

    def initWebcam(self):
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            sys.exit()

    def update_frames(self):
        # Your existing webcam capture and processing logic...
        ret, frame = self.cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return

        # Update webcam label
        self.display_image(frame, self.webcamLabel)

        # Processing logic and update processed label
        # ... Your image processing code ...
        processed_frame = self.pipeline.process_frame()  # Dummy function for processing
        self.display_image(processed_frame, self.processedLabel)

    def process_image(self, frame):
        # Your image processing code
        # Dummy processing for example
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return processed_frame

    def display_image(self, frame, label):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        # Release resources when closing the window
        self.cap.release()

def main():
    app = QApplication(sys.argv)
    window = WebcamWindow()
    window.setWindowTitle('PyQt5 Webcam Streams with Processing')
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
