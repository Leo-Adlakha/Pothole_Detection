import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import subprocess


def process_stream(options, camera_index):
    subprocess.call(["afplay", "connected.wav"])
    colors = [tuple(255 * np.random.rand(3)) for i in range(10)]
    tfnet = TFNet(options)

    capture = cv2.VideoCapture(camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        stime = time.time()
        ret, frame = capture.read()
        results = tfnet.return_predict(frame)
        if ret and len(results) > 0:
            subprocess.call(["afplay", "beep.wav"])

            '''
            The below code is for having the boxes around the 
            predicted Potholes. Now, when deployed we dont need any sort of 
            boxes or confidence levels hence a beep is enough to tell the 
            driver that there is a pothole ahead, hence he should drive slow.
            '''

            for color, result in zip(colors, results):
                top_left = (result['topleft']['x'], result['topleft']['y'])
                bottom_right = (result['bottomright']['x'],
                                result['bottomright']['y'])
                label = result['label']
                confidence = result['confidence']
                text = '{}: {:.0f}%'.format(label, confidence*100)
                frame = cv2.rectangle(frame, top_left, bottom_right, color, 5)
                frame = cv2.putText(frame, text, top_left,
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow('frame', frame)
            print('FPS => {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
