import random as rng

import cv2
import numpy as np
import scipy.cluster.hierarchy as hcluster

from base_filters import BaseSliderFilter, SliderInfo
from util import four_point_transform, sobel


class FourPointFilter(BaseSliderFilter):
    filter_params = {'tl_x': SliderInfo(0, 100, 1, 0),
                     'tl_y': SliderInfo(0, 100, 1, 0),
                     'tr_x': SliderInfo(0, 100, 1, 0),
                     'tr_y': SliderInfo(0, 100, 1, 0),
                     'br_x': SliderInfo(0, 100, 1, 0),
                     'br_y': SliderInfo(0, 100, 1, 0),
                     'bl_x': SliderInfo(0, 100, 1, 0),
                     'lb_y': SliderInfo(0, 100, 1, 0)}

    def update(self, src_cv):
        if len(src_cv.shape) == 3:
            h, w, _ = src_cv.shape
        else:
            h, w = src_cv.shape

        tl_x = int(self.widget_list[0].value * w / 100)
        tl_y = int(self.widget_list[1].value * h / 100)
        tr_x = int(self.widget_list[2].value * w / 100)
        tr_y = int(self.widget_list[3].value * h / 100)
        br_x = int(self.widget_list[4].value * w / 100)
        br_y = int(self.widget_list[5].value * h / 100)
        bl_x = int(self.widget_list[6].value * w / 100)
        bl_y = int(self.widget_list[7].value * h / 100)
        return four_point_transform(src_cv, np.array([(tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y)], dtype='float32'))


class LegacyFindSquaresFilter(BaseSliderFilter):
    filter_params = {'size': SliderInfo(1, 11, 2, 3)}

    def update(self, src_gray):
        if len(src_gray.shape) == 3:
            src_gray = cv2.cvtColor(src_gray, cv2.COLOR_BGR2GRAY)

        if src_gray.shape[1] < 800:
            src_gray = cv2.resize(src_gray, (0, 0), 2, 2)

        # Find contours
        all_squares = self.find_squares(src_gray)
        print(all_squares)

        # Find centers
        all_centers = []
        for sq in all_squares:
            center, _ = cv2.minEnclosingCircle(sq)
            all_centers.append([center[0], center[1]])

        print(all_centers)
        # Find clusters
        thresh = 30
        cluster_indicies = hcluster.fclusterdata(all_centers, thresh, criterion="distance")

        # Create groups of clusters, keyed by original index
        groups = {}
        for i, cluster_id in enumerate(cluster_indicies):
            if cluster_id in groups:
                groups[cluster_id].append(i)
            else:
                groups[cluster_id] = [i]

        contours = [all_squares[groups[k][0]] for k in groups.keys()]
        centers = [all_centers[groups[k][0]] for k in groups.keys()]
        bounds = [cv2.boundingRect(c) for c in contours]

        # Draw contours
        # color = (255, 255, 255)
        # color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))

        cluster_n = 130
        colors = np.zeros((1, cluster_n, 3), np.uint8)
        colors[0, :] = 255
        colors[0, :, 0] = np.arange(0, 180, 180.0 / cluster_n)
        colors = cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)[0]

        drawing = np.zeros((src_gray.shape[0], src_gray.shape[1], 3), dtype=np.uint8)

        for i in range(len(contours)):
            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            pt1 = (bounds[i][0], bounds[i][1])
            pt2 = (bounds[i][0] + bounds[i][2], bounds[i][1] + bounds[i][3])
            cv2.rectangle(drawing, pt1, pt2, (255, 255, 255))
            cv2.drawContours(drawing, contours, i, color)

        # Draw centers
        centers = np.int32(np.vstack(centers))
        for (x, y), label in zip(centers, cluster_indicies.ravel()):
            c = list(map(int, colors[label]))
            cv2.circle(drawing, (x, y), 2, c, -1)

        return drawing

    def find_squares(self, gray):
        blur = 3
        min_area = gray.shape[0] * gray.shape[1] / 200
        squares = []

        bin1 = sobel(gray, blur)

        otsu, _ = cv2.threshold(bin1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, bin1 = cv2.threshold(bin1, int(otsu / 2), 255, cv2.THRESH_BINARY)

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        bin1 = cv2.morphologyEx(bin1, cv2.MORPH_ERODE, element)

        contours, _ = cv2.findContours(bin1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda _c: cv2.contourArea(_c), reverse=True)
        for orig_contour in sorted_contours:
            area = cv2.contourArea(orig_contour)
            if min_area < area < 5 * min_area:
                br = cv2.boundingRect(orig_contour)
                br_width, br_height = br[2], br[3]
                r = br_width / br_height
                if r > 1.0:
                    r = br_height / br_width
                if r > 0.95:
                    bra = br_width * br_height
                    if area > bra * 0.9:
                        squares.append(orig_contour)
        return squares
