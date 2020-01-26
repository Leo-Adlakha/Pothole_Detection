'''
This file removes frames from a videofile.
The resulting file will look faster when played back at normal speed.
The idea is to create video that can be processed by yolo and look normal speed 
'''

import cv2
import numpy as np


def reduce_frame(input_video, output_video, frame_rate_divider):
    capture = cv2.VideoCapture(input_video)
    size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    codec = cv2.VideoWriter_fourcc(*'DIVX')
    output = cv2.VideoWriter(output_video, codec, 60.0, size)

    i = 0

    while(capture.isOpened()):
        ret, frame = capture.read()
        if ret:
            if i % frame_rate_divider == 0:
                # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
                output.write(frame)
                cv2.imshow('frame', frame)
                i += 1
            else:
                i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    capture.release()
    output.release()
    cv2.destroyAllWindows()
