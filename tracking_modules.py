from collections import OrderedDict

import cv2
import os
import numpy as np
import itertools


class CountTruth:
    def __init__(self, inside, outside):
        self.inside = inside
        self.outside = outside


def get_truth(video_name):
    with open('data_files/labels_counted.csv', 'r') as file:
        lines = file.readlines()

    TruthArr = CountTruth(0, 0)
    for line in lines:
        line = line.split(",")
        if line[1] == video_name:
            TruthArr.inside = int(line[2])
            TruthArr.outside = int(line[3])
    return TruthArr


class Counter:
    def __init__(self, counter_in, counter_out, track_id):
        self.people_init = OrderedDict()
        self.people_bbox = OrderedDict()
        self.cur_bbox = OrderedDict()
        self.rat_init = OrderedDict()

        self.age_counter = OrderedDict()

        self.lost_ids = set()

        # self.dissappeared_frames = OrderedDict()
        self.counter_in = counter_in
        self.counter_out = counter_out
        self.track_id = track_id

    def obj_initialized(self, track_id):
        self.people_init[track_id] = 0


    def zero_age(self, track_id):
        self.age_counter[track_id] = 0


    def age_increment(self):
        x = None
        for tr in self.age_counter.keys():
            self.age_counter[tr] += 1
            if self.age_counter[tr] >= 20:
                self.lost_ids.add(tr)
                x = tr
        if self.age_counter.get(x):
            del self.age_counter[x]


    def update_identities(self, identities):
        for tr_i in identities:
            if tr_i in self.age_counter.keys() :
                if self.age_counter.get(tr_i) is None:
                    self.age_counter[tr_i] = 0
                else:
                    self.age_counter[tr_i] = 0
            else:
                self.age_counter[tr_i] = 0


    def return_lost_ids(self):
        self.age_increment()
        return self.lost_ids


    def get_age(self, track_id):
        return self.age_counter.get(track_id)


    def get_in(self):
        self.counter_in += 1

    def get_out(self):
        self.counter_out += 1

    def show_counter(self):
        return self.counter_in, self.counter_out

    def return_total_count(self):
        return self.counter_in + self.counter_out


rect_endpoint_tmp = []
rect_bbox = []
drawing = False


def select_object(img):
    """
    Interactive select rectangle ROIs and store list of bboxes.

    Parameters
    ----------
    img :
           image 3-dim.

    Returns
    -------
    bbox_list_rois : list of list of int
           List of bboxes of rectangle rois.
    """

    # mouse callback function
    bbox_list_rois = []

    def draw_rect_roi(event, x, y, flags, param):

        # grab references to the global variables
        global rect_bbox, rect_endpoint_tmp, drawing

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that drawing is being
        # performed. set rect_endpoint_tmp empty list.
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_endpoint_tmp = []
            rect_bbox = [(x, y)]
            drawing = True

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # drawing operation is finished
            rect_bbox.append((x, y))
            drawing = False

            # draw a rectangle around the region of interest
            p_1, p_2 = rect_bbox
            cv2.rectangle(img, p_1, p_2, color=(0, 255, 0), thickness=1)
            cv2.imshow('image', img)

            # for bbox find upper left and bottom right points
            p_1x, p_1y = p_1
            p_2x, p_2y = p_2

            lx = min(p_1x, p_2x)
            ty = min(p_1y, p_2y)
            rx = max(p_1x, p_2x)
            by = max(p_1y, p_2y)

            # add bbox to list if both points are different
            if (lx, ty) != (rx, by):
                bbox = [lx, ty, rx, by]
                bbox_list_rois.append(bbox)

        # if mouse is drawing set tmp rectangle endpoint to (x,y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect_endpoint_tmp = [(x, y)]

    # clone image img and setup the mouse callback function
    img_copy = img.copy()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image', 1200, 600)
    cv2.setMouseCallback('image', draw_rect_roi)

    # keep looping until the 'c' key is pressed
    while True:
        # display the image and wait for a keypress
        if not drawing:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 1200, 600)
            cv2.imshow('image', img)
        elif drawing and rect_endpoint_tmp:
            rect_cpy = img.copy()
            start_point = rect_bbox[0]
            end_point_tmp = rect_endpoint_tmp[0]
            cv2.rectangle(rect_cpy, start_point, end_point_tmp, (0, 255, 0), 1)
            cv2.imshow('image', rect_cpy)

        key = cv2.waitKey(1) & 0xFF
        # if the 'c' key is pressed, break from the loop
        if key == ord('c'):
            break
    # close all open windows
    cv2.destroyAllWindows()

    return bbox_list_rois


def read_door_info(name='doors_info.csv'):
    door_info = {}
    with open(name, 'r') as file:
        lines = file.readlines()
    for line in lines:
        line_l = line.split(";")
        val = line_l[1][2:-3].split(",")
        for i, v in enumerate(val):
            val[i] = int(v)
        door_info[line_l[0]] = val
    return door_info



def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def find_centroid(bbox):
    return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)


class Rectangle:
    def intersection(self, other):
        a, b = self, other
        x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
        y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
        x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
        y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
        if x1 < x2 and y1 < y2:
            return type(self)(x1, y1, x2, y2)

    __and__ = intersection

    def difference(self, other):
        inter = self & other
        if not inter:
            yield self
            return
        xs = {self.x1, self.x2}
        ys = {self.y1, self.y2}
        if self.x1 < other.x1 < self.x2: xs.add(other.x1)
        if self.x1 < other.x2 < self.x2: xs.add(other.x2)
        if self.y1 < other.y1 < self.y2: ys.add(other.y1)
        if self.y1 < other.y2 < self.y2: ys.add(other.y2)
        for (x1, x2), (y1, y2) in itertools.product(
                pairwise(sorted(xs)), pairwise(sorted(ys))
        ):
            rect = type(self)(x1, y1, x2, y2)
            if rect != inter:
                yield rect

    __sub__ = difference

    def __init__(self, x1, y1, x2, y2):
        if x1 > x2 or y1 > y2:
            raise ValueError("Coordinates are invalid")
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def __iter__(self):
        yield self.x1
        yield self.y1
        yield self.x2
        yield self.y2

    def __eq__(self, other):
        return isinstance(other, Rectangle) and tuple(self) == tuple(other)

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return type(self).__name__ + repr(tuple(self))


def pairwise(iterable):
    # https://docs.python.org/dev/library/itertools.html#recipes
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def rect_square(x1, y1, x2, y2):
    return abs(x1 - x2) * abs(y1 - y2)
