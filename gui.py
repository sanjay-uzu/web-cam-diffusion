import sys
import random
import cv2
import os
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from pipeline import Pipeline
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QLineEdit, QVBoxLayout

# Your existing imports and functions...


MODEL_NAME="taureal.safetensors"
LORA_NAME="lcm.safetensors"
VAE_NAME="vae-ft-mse-840000-ema-pruned.vae.pt"
STEPS=7
DENOISE=0.5
CFG=1.5
PROMPT="Man , Blue hoodie ,  (white skinned:1.5) ,  hd , 4k , ultra realistic , hyper realistic , photo realistic\n"



class WebcamWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initWebcam()
        self.pipeline=Pipeline(model_name=MODEL_NAME , lora_name=LORA_NAME , vae_name=VAE_NAME , steps=STEPS , denoise=DENOISE , cfg=CFG , prompt=PROMPT)

    def initUI(self):
        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Layout for settings
        settings_layout = QVBoxLayout()
        main_layout.addLayout(settings_layout)

        # Model Name
        self.modelNameEdit = QLineEdit("taureal.safetensors")
        settings_layout.addWidget(QLabel("Model Name:"))
        settings_layout.addWidget(self.modelNameEdit)

        # Lora Name
        self.loraNameEdit = QLineEdit("lcm.safetensors")
        settings_layout.addWidget(QLabel("Lora Name:"))
        settings_layout.addWidget(self.loraNameEdit)

        # VAE Name
        self.vaeNameEdit = QLineEdit("vae-ft-mse-840000-ema-pruned.vae.pt")
        settings_layout.addWidget(QLabel("VAE Name:"))
        settings_layout.addWidget(self.vaeNameEdit)

        # Steps
        self.stepsEdit = QLineEdit("7")
        settings_layout.addWidget(QLabel("Steps:"))
        settings_layout.addWidget(self.stepsEdit)

        # Denoise
        self.denoiseEdit = QLineEdit("0.5")
        settings_layout.addWidget(QLabel("Denoise:"))
        settings_layout.addWidget(self.denoiseEdit)

        # CFG
        self.cfgEdit = QLineEdit("1.5")
        settings_layout.addWidget(QLabel("CFG:"))
        settings_layout.addWidget(self.cfgEdit)

        # Prompt
        self.promptEdit = QLineEdit("Man, Blue hoodie, (white skinned:1.5), hd, 4k, ultra realistic, hyper realistic, photo realistic")
        settings_layout.addWidget(QLabel("Prompt:"))
        settings_layout.addWidget(self.promptEdit)

        # Webcam layout
        webcam_layout = QHBoxLayout()
        main_layout.addLayout(webcam_layout)

        # Create labels for displaying frames
        self.webcamLabel = QLabel()
        self.processedLabel = QLabel()

        webcam_layout.addWidget(self.webcamLabel)
        webcam_layout.addWidget(self.processedLabel)

        # Timer for updating frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30)

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
