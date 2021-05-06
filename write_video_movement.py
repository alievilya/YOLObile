# -*- coding: utf-8 -*-
"""
References/Adapted From:
https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/

Description:
This script runs a motion detector! It detects transient motion in a room
and said movement is large enough, and recent enough, reports that there is
motion!

Run the script with a working webcam! You'll see how it works!
"""

import imutils
import cv2
import numpy as np
import json

# =============================================================================
# USER-SET PARAMETERS
# =============================================================================



# =============================================================================
# CORE PROGRAM
# =============================================================================


# Create capture object
# cap = cv2.VideoCapture(5)  # Flush the stream
# cap.release()

class MoveDetector():
    def __init__(self):
        # Init frame variables
        self.first_frame = None
        self.next_frame = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.delay_counter = 0
        self.movement_persistent_counter = 0

    def set_init(self, cap):
        self.transient_movement_flag = False
        # Read frame
        self.ret, self.frame = cap.read()
        self.text = "Unoccupied"


    def detect_movement(self, config):

        if not self.ret:
            print("CAPTURE ERROR")
            # continue
            return False

        # Resize and save a greyscale version of the image
        self.frame = imutils.resize(self.frame, width=750)
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # Blur it to remove camera noise (reducing false positives)
        self.gray = cv2.GaussianBlur(self.gray, (13, 13), 0)
        # If the first frame is nothing, initialise it
        if self.first_frame is None: self.first_frame = self.gray
        self.delay_counter += 1
        # Otherwise, set the first frame to compare as the previous frame
        # But only if the counter reaches the appriopriate value
        # The delay is to allow relatively slow motions to be counted as large
        # motions if they're spread out far enough
        if self.delay_counter > config["FRAMES_TO_PERSIST"]:
            delay_counter = 0
            self.first_frame = self.next_frame

        # Set the next frame to compare (the current frame)
        self.next_frame = self.gray

        # Compare the two frames, find the difference
        frame_delta = cv2.absdiff(self.first_frame, self.next_frame)
        thresh = cv2.threshold(frame_delta, 20, 255, cv2.THRESH_BINARY)[1]

        # Fill in holes via dilate(), and find contours of the thesholds
        thresh = cv2.dilate(thresh, None, iterations=4)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in cnts:

            # Save the coordinates of all found contours
            (x, y, w, h) = cv2.boundingRect(c)

            # If the contour is too small, ignore it, otherwise, there's transient
            # movement
            if cv2.contourArea(c) > config["MIN_SIZE_FOR_MOVEMENT"]:
                self.transient_movement_flag = True

                # Draw a rectangle around big enough movements
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # The moment something moves momentarily, reset the persistent
        # movement timer.
        if self.transient_movement_flag == True:
            self.movement_persistent_flag = True
            self.movement_persistent_counter = config["MOVEMENT_DETECTED_PERSISTENCE"]

        if self.movement_persistent_counter > 0:
            text = "Movement Detected " + str(self.movement_persistent_counter)
            self.movement_persistent_counter -= 1

        else:
            text = "No Movement Detected"

        cv2.putText(self.frame, str(text), (10, 35), self.font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)
        return self.frame


if __name__ == "__main__":
    with open("cfg/motion_detection_cfg.json") as config_file:
        config = json.load(config_file)

    Motion = [MoveDetector() for _ in range(60)]
    # link ="rtsp://admin:admin@192.168.1.18:554/1/h264major"
    link = config["source"]
    print('opening link: ', link)
    cap = cv2.VideoCapture(link)  # Then start the webcam
    # LOOP!
    while True:
        for i in range(len(Motion)):
            Motion[i].set_init(cap=cap)
            frame = Motion[i].detect_movement(config=config)
            # Splice the two video frames together to make one long horizontal one
            # cv2.imshow("frame", np.hstack((frame_delta, frame)))
            cv2.imshow("frame {}".format(i), frame)

            # Interrupt trigger by pressing q to quit the open CV program
            ch = cv2.waitKey(3)
            if ch & 0xFF == ord('q'):
                break

    # Cleanup when closed
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     cap.release()