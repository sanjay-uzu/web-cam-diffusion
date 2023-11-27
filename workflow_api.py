import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


import cv2

from pipeline import Pipeline

def main():
    with torch.inference_mode():
        piepeline=Pipeline()
        cap = cv2.VideoCapture(0)

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        try:
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                cv2.imwrite("E:\\ComfyUI\\input\\test.png", frame)

                # If frame is read correctly, ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                gen=piepeline.process_frame()
                # Display the resulting frame
                cv2.imshow('Webcam Feed', frame)
                cv2.imshow('Generated', gen)

                # Break the loop when 'q' is pressed
                if cv2.waitKey(1) == ord('q'):
                    break
        finally:
            # When everything is done, release the capture
            cap.release()
            cv2.destroyAllWindows()
        



if __name__ == "__main__":
    main()
