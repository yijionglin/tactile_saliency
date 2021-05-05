import os
import time
import numpy as np
import pandas as pd

from vsp.video_stream import CvVideoDisplay, CvVideoOutputFile, CvVideoCamera
from vsp.processor import CameraStreamProcessorMT, AsyncProcessor

def make_sensor():
    return AsyncProcessor(CameraStreamProcessorMT(
            camera=CvVideoCamera(source=0,
                                 frame_size=(640, 480),
                                 is_color=True),
            display=CvVideoDisplay(name='preview'),
            writer=CvVideoOutputFile(is_color=True),
        ))

with make_sensor() as sensor:
    frames = sensor.process(num_frames=100)
    print(frames.shape)
