# import numpy as np
# import cv2
# from tracking_modules import select_object
#
# video = "rtsp://admin:admin@192.168.1.18:554/1/h264major"
# cap = cv2.VideoCapture(video)
#
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     door_array = select_object(frame)
#     print(door_array)
#
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import threading
import datetime
import os
import sqlite3
import random
import time
from threading import Thread


class Detector:
    def __init__(self,nname, name, password, ip, width, height, threshold, number, channel):

        self.nname = nname
        self.name = name
        self.password = password
        self.ip = ip
        self.width = width
        self.height = height
        self.threshold = threshold
        self.number = number
        self.channel = channel
        print(nname
              )

    def create_camera(self):  # # rtsp://admin:qwerty123@192.168.88.100:554/ch01/0
        rtsp = "rtsp://" + self.name + ":" + self.password + "@" + self.ip + ":554/" + self.channel + '/' + 'h264major'
        print(rtsp)
        cap = cv2.VideoCapture(rtsp)
        self.width = int(cap.get(3))  # ID number for width is 3
        self.height = int(cap.get(4))  # ID number for height is 480
        cap.set(10, 100)  # ID number for brightness is 10
        while True:
            now = datetime.datetime.now()
            name = now.strftime("%d-%m-%Y-%H-%M-%S")
            _time = now.strftime("%H-%M-%S")
            clock = datetime.datetime.now()
            today = clock.strftime("%Y-%m-%d")

            result = cv2.VideoWriter(os.path.join("temp", str(self.channel) + "-" + self.nname + '.avi'),
                                     cv2.VideoWriter_fourcc(*'MJPG'),
                                     10.0, (self.width, self.height))
            count = 200
            while True:
                if count > 0:
                    ret, frame1 = cap.read()
                    # cv2.imshow('frame', frame1)
                    result.write(frame1)
                    count -= 1
                else:
                    result.release()

        # if cv2.waitKey(1) & 0xFF == ord('p'):
        #     cv2.destroyAllWindows()


# rtsp://admin:qwerty123@192.168.88.100:554/ch01/0
if __name__ == "__main__":
    try:
        for i in range(10):
            name = "Thread #%s" % (i + 1)
            channel1 = Detector(name,'admin', "admin", "192.168.1.18", 1280, 720, 3000, 1, "1")
            ch1 = threading.Thread(target=channel1.create_camera(), daemon=True)
            ch1.start()
            ch1.join()
    # ch = Detector('admin', "admin", "192.168.1.18", 1280, 720, 3000, 1, "1")
    # ch.create_camera()
    except TypeError:
        pass
