import random as rng

import cv2 as cv
import numpy as np
import scipy.cluster.hierarchy as hcluster

from base_filters import BaseSliderFilter
from opencv_filters import sobel


class FindSquaresFilter(BaseSliderFilter):
    filter_params = {'size': {'min_val': 1, 'max_val': 11, 'step_val': 2}}

    def update(self, src_gray):
        if len(src_gray.shape) == 3:
            src_gray = cv.cvtColor(src_gray, cv.COLOR_BGR2GRAY)

        if src_gray.shape[1] < 800:
            src_gray = cv.resize(src_gray, (0, 0), 2, 2)

        # Find contours
        all_squares = self.find_squares(src_gray)
        print(all_squares)

        # Find centers
        all_centers = []
        for sq in all_squares:
            center, _ = cv.minEnclosingCircle(sq)
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
        bounds = [cv.boundingRect(c) for c in contours]

        # Draw contours
        # color = (255, 255, 255)
        # color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))

        cluster_n = 130
        colors = np.zeros((1, cluster_n, 3), np.uint8)
        colors[0, :] = 255
        colors[0, :, 0] = np.arange(0, 180, 180.0 / cluster_n)
        colors = cv.cvtColor(colors, cv.COLOR_HSV2BGR)[0]

        drawing = np.zeros((src_gray.shape[0], src_gray.shape[1], 3), dtype=np.uint8)

        for i in range(len(contours)):
            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            pt1 = (bounds[i][0], bounds[i][1])
            pt2 = (bounds[i][0] + bounds[i][2], bounds[i][1] + bounds[i][3])
            cv.rectangle(drawing, pt1, pt2, (255, 255, 255))
            cv.drawContours(drawing, contours, i, color)

        # Draw centers
        centers = np.int32(np.vstack(centers))
        for (x, y), label in zip(centers, cluster_indicies.ravel()):
            c = list(map(int, colors[label]))
            cv.circle(drawing, (x, y), 2, c, -1)

        return drawing

    def find_squares(self, gray):
        blur = 3
        min_area = gray.shape[0] * gray.shape[1] / 200
        squares = []

        bin1 = sobel(gray, blur)

        otsu, _ = cv.threshold(bin1, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        _, bin1 = cv.threshold(bin1, int(otsu / 2), 255, cv.THRESH_BINARY)

        element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
        bin1 = cv.morphologyEx(bin1, cv.MORPH_ERODE, element)

        contours, _ = cv.findContours(bin1, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda _c: cv.contourArea(_c), reverse=True)
        for orig_contour in sorted_contours:
            area = cv.contourArea(orig_contour)
            if min_area < area < 5 * min_area:
                br = cv.boundingRect(orig_contour)
                br_width, br_height = br[2], br[3]
                r = br_width / br_height
                if r > 1.0:
                    r = br_height / br_width
                if r > 0.95:
                    bra = br_width * br_height
                    if area > bra * 0.9:
                        squares.append(orig_contour)
        return squares
