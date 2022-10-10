from collections import Counter

# <cough>
import config
import os
# </cough>

import cv2
import numpy as np

from base_filters import BaseFilter, BaseSliderFilter, SliderInfo
from util import sobel, angle_cos


class AdaptiveThresholdFilter(BaseSliderFilter):
    filter_params = {'threshold': SliderInfo(0, 255, 1, 30),
                     'threshold_type': SliderInfo(0, 1, 1, 0),
                     'block_size': SliderInfo(3, 9, 2, 3),
                     'C': SliderInfo(-5, 5, 1, 0)}
    operators = [
        (cv2.THRESH_BINARY, 'BINARY'),
        (cv2.THRESH_BINARY_INV, 'BINARY_INV')]

    def get_display_callback(self, _filter):
        return {
            'type': self.get_operator_display,
        }.get(_filter, lambda x: str(x))

    def get_operator_code(self, x):
        return self.operators[x][0]

    def get_operator_display(self, x):
        return self.operators[x][1]

    def update(self, src_cv):
        threshold = int(self.widget_list[0].value)
        threshold_type = self.get_operator_code(int(self.widget_list[1].value))
        block_size = int(self.widget_list[2].value)
        C = int(self.widget_list[3].value)

        return cv2.adaptiveThreshold(src_cv, threshold, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type, block_size, C)


class AddWeightedFilter(BaseFilter):
    def update(self, src_cv):
        return cv2.addWeighted(src_cv[0], 1, src_cv[1], 0.75, 0)


class BilateralFilter(BaseSliderFilter):
    filter_params = {'kernel': SliderInfo(1, 51, 2, 3),
                     'sigma_color': SliderInfo(3, 255, 2, 75),
                     'sigma_space': SliderInfo(3, 255, 2, 75)}

    def update(self, src_cv):
        kernel = int(self.widget_list[0].value)
        sigma_color = int(self.widget_list[1].value)
        sigma_space = int(self.widget_list[2].value)
        return cv2.bilateralFilter(src_cv, kernel, sigma_color, sigma_space)


class BitwiseAndFilter(BaseFilter):
    filter_params = {}

    def update(self, src_cv):
        return cv2.bitwise_and(src_cv[0], src_cv[1])


class BitwiseNotFilter(BaseFilter):
    filter_params = {}

    def update(self, src_cv):
        return cv2.bitwise_not(src_cv)


class BitwiseOrFilter(BaseFilter):
    filter_params = {}

    def update(self, src_cv):
        return cv2.bitwise_or(src_cv[0], src_cv[0])


class BlurFilter(BaseSliderFilter):
    filter_params = {'size': SliderInfo(1, 15, 2, 3)}

    def update(self, src_cv):
        val = int(self.widget_list[0].value)
        return cv2.blur(src_cv, (val, val))


class CannyFilter(BaseSliderFilter):
    filter_params = {'thresh1': SliderInfo(0, 1000, 10, 0),
                     'thresh2': SliderInfo(0, 1000, 10, 100),
                     'aperture_size': SliderInfo(3, 7, 2, 3),
                     'l2_gradient': SliderInfo(0, 1, 1, 0)}

    def update(self, src_cv):
        thresh1 = int(self.widget_list[0].value)
        thresh2 = int(self.widget_list[1].value)
        aperture_size = int(self.widget_list[2].value)
        l2_gradient = int(self.widget_list[3].value) == 1
        return cv2.Canny(src_cv, thresh1, thresh2, edges=None, apertureSize=aperture_size, L2gradient=l2_gradient)


class ChannelFilter(BaseSliderFilter):
    filter_params = {'channel': SliderInfo(0, 2, 1, 0),
                     'thresh': SliderInfo(0, 255, 1, 0),
                     'type': SliderInfo(0, 2, 1, 0)}

    def update(self, src_cv):
        channel = int(self.widget_list[0].value)
        val = int(self.widget_list[1].value)
        oper_int = int(self.widget_list[2].value)
        src_cv = src_cv[:, :, channel]
        return cv2.threshold(src_cv, val, 255, oper_int)[1]


class ClipFilter(BaseSliderFilter):
    filter_params = {'clip': SliderInfo(0, 20, 1, 0)}

    def update(self, src_cv):
        val = int(self.widget_list[0].value)

        return src_cv[val:src_cv.shape[1] - val, val:src_cv.shape[0] - val]


class ContoursFilter(BaseSliderFilter):
    operators = [
        (cv2.RETR_EXTERNAL, 'RETR_EXTERNAL'),
        (cv2.RETR_LIST, 'RETR_LIST'),
        (cv2.RETR_CCOMP, 'RETR_CCOMP'),
        (cv2.RETR_TREE, 'RETR_TREE'),
        (cv2.RETR_FLOODFILL, 'RETR_FLOODFILL'),
    ]

    filter_params = {'lower': SliderInfo(0, 150000, 50, 0),
                     'upper': SliderInfo(0, 150000, 50, 500),
                     "method": SliderInfo(0, 4, 1, 0)}

    def get_display_callback(self, _filter):
        return {
            'method': self.get_operator_display,
        }.get(_filter, lambda x: str(x))

    def get_operator_code(self, x):
        return self.operators[x][0]

    def get_operator_display(self, x):
        return self.operators[x][1]

    def update(self, src_cv):
        lower = int(self.widget_list[0].value)
        upper = int(self.widget_list[1].value)
        method_int = int(self.widget_list[2].value)

        contours, _ = cv2.findContours(src_cv, method_int, cv2.CHAIN_APPROX_SIMPLE)

        output_cv = np.zeros((src_cv.shape[0], src_cv.shape[1], 3), dtype=np.uint8)
        contours_to_draw = [c for c in contours if lower < cv2.contourArea(c) < upper]

        if len(contours_to_draw) > 0:
            for contour in contours_to_draw:
                color = (255, 255, 255)  # (rng.randint(65, 256), rng.randint(65, 256), rng.randint(65, 256))
                cv2.drawContours(output_cv, [contour], -1, color, thickness=1, lineType=cv2.LINE_AA)

        return output_cv


class CropFilter(BaseSliderFilter):
    filter_params = {
        'left': SliderInfo(0, 100, 1, 0),
        'top': SliderInfo(0, 100, 1, 0),
        'right': SliderInfo(0, 100, 1, 100),
        'bottom': SliderInfo(0, 100, 1, 100)}

    def update(self, src_cv):
        left = int(self.widget_list[0].value)
        top = int(self.widget_list[1].value)
        right = int(self.widget_list[2].value)
        bottom = int(self.widget_list[3].value)
        h, w, _ = src_cv.shape
        return src_cv[int(h * top / 100.0):int(h * bottom / 100.00), int(w * left / 100.0):int(w * right / 100.0)]


class DiffOfGaussianBlurFilter(BaseSliderFilter):
    filter_params = {'size1': SliderInfo(1, 15, 2, 3),
                     'size2': SliderInfo(1, 15, 2, 5)}

    def update(self, src_cv):
        size1 = int(self.widget_list[0].value)
        size2 = int(self.widget_list[1].value)
        return cv2.GaussianBlur(src_cv, (size1, size1), 0) - cv2.GaussianBlur(src_cv, (size2, size2), 0)


class EqualizeHistogramFilter(BaseFilter):
    filter_params = {}

    def update(self, src_cv):
        return cv2.equalizeHist(src_cv)


class FillRectFilter(BaseSliderFilter):
    filter_params = {
        'left': SliderInfo(0, 100, 1, 0),
        'top': SliderInfo(0, 100, 1, 0),
        'right': SliderInfo(0, 100, 1, 100),
        'bottom': SliderInfo(0, 100, 1, 100)}

    def update(self, src_cv):
        left = int(self.widget_list[0].value)
        top = int(self.widget_list[1].value)
        right = int(self.widget_list[2].value)
        bottom = int(self.widget_list[3].value)
        h, w, _ = src_cv.shape
        src_cv[int(h * top / 100.0):int(h * bottom / 100.00), int(w * left / 100.0):int(w * right / 100.0)] = 0
        return src_cv


class FindSquaresFilter(BaseSliderFilter):
    filter_params = {'arc_length': SliderInfo(0, 0.5, 0.001, 0.24)}

    def update(self, src_cv):
        arc_length = self.widget_list[0].value
        img = src_cv
        img = cv2.GaussianBlur(img, (5, 5), 0)
        squares = []
        for gray in cv2.split(img):
            for threshold in range(0, 255, 26):
                if threshold == 0:
                    _bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                    _bin = cv2.dilate(_bin, None)
                else:
                    _retval, _bin = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                contours, _hierarchy = cv2.findContours(_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cnt_len = cv2.arcLength(cnt, True)
                    cnt = cv2.approxPolyDP(cnt, arc_length * cnt_len, True)
                    if len(cnt) >= 4 and cv2.contourArea(cnt) > 1000:
                        cnt = cnt.reshape(-1, 2)
                        max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
                        if max_cos < 0.2:
                            squares.append(cnt)

        output_cv = np.zeros((src_cv.shape[0], src_cv.shape[1], 3), dtype=np.uint8)
        for contour in squares:
            color = (128, 255, 255)  # (rng.randint(65, 256), rng.randint(65, 256), rng.randint(65, 256))
            cv2.drawContours(output_cv, [contour], -1, color, thickness=1, lineType=cv2.LINE_AA)

        return output_cv


class GaussianBlurFilter(BaseSliderFilter):
    filter_params = {'size': SliderInfo(1, 15, 2, 3)}

    def update(self, src_cv):
        val = int(self.widget_list[0].value)
        return cv2.GaussianBlur(src_cv, (val, val), 0)


class GrayscaleFilter(BaseFilter):
    filter_params = {}

    def update(self, src_cv):
        return cv2.cvtColor(src_cv, cv2.COLOR_BGR2GRAY)


class HarrisCornerFilter(BaseSliderFilter):
    filter_params = {'block_size': SliderInfo(1, 5, 1, 3),
                     'ksize': SliderInfo(1, 25, 2, 3),
                     'k': SliderInfo(0, 1, 0.01, 0),
                     'threshold': SliderInfo(0, 1, 0.01, 0)
                     }

    def update(self, src_cv):
        block_size = int(self.widget_list[0].value)
        ksize = int(self.widget_list[1].value)
        k = self.widget_list[2].value
        threshold = self.widget_list[3].value

        gray = cv2.cvtColor(src_cv, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, block_size, ksize, k)
        dst = cv2.dilate(dst, None)
        src_cv[dst > threshold * dst.max()] = [0, 255, 255]
        return src_cv


class HiLoThresholdFilter(BaseSliderFilter):
    filter_params = {'lower': SliderInfo(0, 255, 1, 0),
                     'upper': SliderInfo(1, 255, 1, 255)}

    def update(self, src_cv):
        lower = int(self.widget_list[0].value)
        upper = int(self.widget_list[1].value)
        return cv2.inRange(src_cv, lower, upper)


class HiLo2ThresholdFilter(BaseSliderFilter):
    filter_params = {'thresh': SliderInfo(0, 255, 1, 0),
                     'neighborhood': SliderInfo(1, 128, 2, 3)}

    def update(self, src_cv):
        thresh = int(self.widget_list[0].value)
        neighborhood = int(self.widget_list[1].value)
        return cv2.inRange(src_cv, max(0, thresh - neighborhood), min(255, thresh + neighborhood))


from sklearn import cluster


def rt2xy(line):
    mx = 2000
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = x0 + mx * (-b)
    y1 = y0 + mx * (a)
    x2 = x0 - mx * (-b)
    y2 = y0 - mx * (a)
    return x1, y1, x2, y2


class HoughLinesFilter(BaseSliderFilter):
    filter_params = {'thresh': SliderInfo(10, 200, 10, 200),
                     'max_lines': SliderInfo(20, 200, 20, 100),
                     'cluster_size': SliderInfo(1, 20, 1, 2)}

    def update(self, src_cv):

        gray = cv2.cvtColor(src_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 140, apertureSize=3)
        lines_ = np.array([])
        h, w = src_cv.shape[:2]
        valid_calib_rect = (int(w / 2) - r2p4, int(h / 2) - r2p5, 2 * r2p4, r2p5 + r2p8)

        thresh = int(self.widget_list[0].value)
        max_lines = int(self.widget_list[1].value)

        t = 15
        while True:
            lines_ = cv2.HoughLines(edges, 1, np.pi / 180, thresh)
            lines_ = lines_.reshape(-1, 2)
            lines_ = [l for l in lines_ if l[1] * 57 < (90 - t) or l[1] * 57 > (90 + t)]
            # if len(lines_) > max_lines:
            #     thresh += 10
            #     continue
            #

            lines_l = [l for l in lines_ if l[1] * 57 < (90 - t)]
            lines_r = [l for l in lines_ if l[1] * 57 > (90 + t)]
            if not lines_l or not lines_r:
                thresh -= 10
                continue

            break

        print(f'thresh:{thresh}  # lines:{len(lines_)}')

        lines = [line for line in lines_ if segment_rect_intersect(rt2xy(line), valid_calib_rect)]

        cluster_size = int(self.widget_list[2].value)
        candidates = []
        while cluster_size < 15:
            if len(lines) > cluster_size:
                c = cluster.KMeans(cluster_size)
                c.fit(np.array(lines).reshape(-1, 2))
                # print(cluster_size, int(c.inertia_))
                if c.inertia_ > 1800000:
                    cluster_size += 2
                    continue

                for line in c.cluster_centers_:
                    x1, y1, x2, y2 = rt2xy(line)
                    cv2.line(src_cv, np.int32((x1, y1)), np.int32((x2, y2)), (0, 0, 255), 1)
                    candidates.append((x1, y1, x2, y2))
                break
            cluster_size += 1

        pts = [pt for pt in find_intersections(candidates) if point_in_rect(pt, valid_calib_rect)]
        if len(pts) > 0:
            vp = tuple(np.int32(np.mean(pts, axis=0)))
            cv2.circle(src_cv, vp, color=(0, 255, 255), radius=5, thickness=2)

        # src_cv_tmp = src_cv.copy()
        # inside_ct = 0
        # for pt in pts:
        #     if point_in_rect(pt, valid_calib_rect):
        #         cv2.circle(src_cv_tmp, np.int32(pt), color=(0, 255, 255), radius=5, thickness=2)
        #         inside_ct += 1
        #
        # if inside_ct > 0:
        #     print(f'intersections found:{len(pts)}')
        #     print(f'inside_ct:{inside_ct}')
        #     for x1, y1, x2, y2 in candidates:
        #         cv2.line(src_cv_tmp, np.int32((x1,y1)), np.int32((x2,y2)), (255, 255, 0), 2)
        #
        #     src_cv = src_cv_tmp

        cv2.rectangle(src_cv, (int(w / 2) - r2p4, int(h / 2) - r2p5), (int(w / 2) + r2p4, int(h / 2) + r2p8), (255, 255, 255), 2)

        return src_cv


class HoughLinesPFilter(BaseSliderFilter):
    filter_params = {
        'threshold': SliderInfo(1, 250, 2, 101),
        'min_line_length': SliderInfo(1, 500, 1, 10),
        'max_gap': SliderInfo(0, 250, 5, 5),
        'min_slope': SliderInfo(0, np.pi / 4, 0.01, 0)
    }

    def update(self, src_cv):
        threshold = int(self.widget_list[0].value)
        min_line_length = int(self.widget_list[1].value)
        max_gap = int(self.widget_list[2].value)
        min_slope = self.widget_list[3].value

        output_cv = np.zeros((src_cv.shape[0], src_cv.shape[1], 3), dtype=np.uint8)

        lines = cv2.HoughLinesP(src_cv, 1, np.pi / 180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_gap)
        for line in lines:
            if filter_slope(line[0], min_slope):
                x1, y1, x2, y2 = line[0]
                cv2.line(output_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return output_cv


class HSVFilter(BaseSliderFilter):
    filter_params = {'h-lo': SliderInfo(0, 180, 1, 0),
                     'h-hi': SliderInfo(0, 180, 1, 180),
                     's-lo': SliderInfo(0, 255, 1, 0),
                     's-hi': SliderInfo(0, 255, 1, 255),
                     'v-lo': SliderInfo(0, 255, 1, 0),
                     'v-hi': SliderInfo(0, 255, 1, 255)}

    def update(self, src_cv):
        h_lower = int(self.widget_list[0].value)
        h_upper = int(self.widget_list[1].value)
        s_lower = int(self.widget_list[2].value)
        s_upper = int(self.widget_list[3].value)
        v_lower = int(self.widget_list[4].value)
        v_upper = int(self.widget_list[5].value)
        return cv2.inRange(cv2.cvtColor(src_cv, cv2.COLOR_BGR2HSV), (h_lower, s_lower, v_lower), (h_upper, s_upper, v_upper))


class MaskFilter(BaseFilter):
    def update(self, src_cv):
        _, mask = cv2.threshold(cv2.cvtColor(src_cv[1], cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        src_cv[0][mask > 0] = 0
        src_cv[0] += src_cv[1] * (mask > 0)
        return src_cv[0]


class MedianBlurFilter(BaseSliderFilter):
    filter_params = {'size': SliderInfo(1, 15, 2, 3)}

    def update(self, src_cv):
        val = int(self.widget_list[0].value)
        return cv2.medianBlur(src_cv, val)


class MorphFilter(BaseSliderFilter):
    operators = [
        (cv2.MORPH_ERODE, 'MORPH_ERODE'),
        (cv2.MORPH_DILATE, 'MORPH_DILATE'),
        (cv2.MORPH_OPEN, 'MORPH_OPEN'),
        (cv2.MORPH_CLOSE, 'MORPH_CLOSE'),
        (cv2.MORPH_GRADIENT, 'MORPH_GRADIENT'),
        (cv2.MORPH_TOPHAT, 'MORPH_TOPHAT'),
        (cv2.MORPH_BLACKHAT, 'MORPH_BLACKHAT'),
    ]

    elements = [
        (cv2.MORPH_RECT, 'MORPH_RECT'),
        (cv2.MORPH_CROSS, 'MORPH_CROSS'),
        (cv2.MORPH_ELLIPSE, 'MORPH_ELLIPSE'),
    ]

    filter_params = {'size_x': SliderInfo(1, 150, 2, 3),
                     'size_y': SliderInfo(1, 150, 2, 3),
                     'iters': SliderInfo(1, 9, 1, 1),
                     'operator': SliderInfo(0, len(operators) - 1, 1, 0),
                     'element': SliderInfo(0, len(elements) - 1, 1, 0)}

    def get_display_callback(self, _filter):
        return {
            'operator': self.get_operator_display,
            'element': self.get_element_display,
        }.get(_filter, lambda x: str(x))

    def get_operator_code(self, x):
        return self.operators[x][0]

    def get_operator_display(self, x):
        return self.operators[x][1]

    def get_element_code(self, x):
        return self.elements[x][0]

    def get_element_display(self, x):
        return self.elements[x][1]

    def update(self, src_cv):
        morph_size_x = int(self.widget_list[0].value)
        morph_size_y = int(self.widget_list[1].value)
        iterations = int(self.widget_list[2].value)
        oper_int = self.get_operator_code(int(self.widget_list[3].value))
        elem_int = self.get_element_code(int(self.widget_list[4].value))
        element = cv2.getStructuringElement(elem_int, (morph_size_x, morph_size_y))
        return cv2.morphologyEx(src_cv, oper_int, element, iterations=iterations)


class PassthroughFilter(BaseFilter):
    def update(self, src_cv):
        return src_cv


class RGBFilter(BaseSliderFilter):
    filter_params = {'r_min': SliderInfo(0, 255, 1, 0),
                     'r_max': SliderInfo(0, 255, 1, 255),
                     'g_min': SliderInfo(0, 255, 1, 0),
                     'g_max': SliderInfo(0, 255, 1, 255),
                     'b_min': SliderInfo(0, 255, 1, 0),
                     'b_max': SliderInfo(0, 255, 1, 255)}

    def update(self, src_cv):
        r_min = int(self.widget_list[0].value)
        r_max = int(self.widget_list[1].value)
        g_min = int(self.widget_list[2].value)
        g_max = int(self.widget_list[3].value)
        b_min = int(self.widget_list[4].value)
        b_max = int(self.widget_list[5].value)

        return cv2.inRange(src_cv, (b_min, g_min, r_min), (b_max, g_max, r_max))


class SimpleBlobDetectorFilter(BaseSliderFilter):
    filter_params = {
        'minThreshold': SliderInfo(0, 255, 1, 10),
        'maxThreshold': SliderInfo(0, 255, 1, 100),
        'minArea': SliderInfo(500, 100000, 100, 500),
        'maxArea': SliderInfo(1000, 100000, 100, 1000),
        'minCircularity': SliderInfo(0, 1, 0.1, 0),
        'maxCircularity': SliderInfo(0, 1, 0.1, 0),
        'minConvexity': SliderInfo(0, 1, 0.1, 0),
        'maxConvexity': SliderInfo(0, 1, 0.1, 0),
        'minInertia': SliderInfo(0, 1, 0.1, 0),
        'maxInertia': SliderInfo(0, 1, 0.1, 0),
        'minDistBetweenBlobs': SliderInfo(1, 100, 1, 10),
        'minRepeatability': SliderInfo(1, 100, 1, 1),
        'thresholdStep': SliderInfo(1, 127, 1, 1),
    }

    def update(self, src_cv):
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = self.widget_list[0].value
        params.maxThreshold = self.widget_list[1].value
        params.thresholdStep = self.widget_list[12].value

        # Filter by Area.
        params.filterByArea = True
        params.minArea = self.widget_list[2].value
        params.maxArea = self.widget_list[3].value

        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = self.widget_list[4].value
        params.maxCircularity = self.widget_list[5].value

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = self.widget_list[6].value
        params.maxConvexity = self.widget_list[7].value

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = self.widget_list[8].value
        params.maxInertiaRatio = self.widget_list[9].value

        params.minDistBetweenBlobs = self.widget_list[10].value
        params.minRepeatability = self.widget_list[11].value

        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(src_cv)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob

        src_copy = src_cv.copy()
        print("kps:" + str(len(keypoints)))
        for kp in keypoints:
            x, y = kp.pt
            cv2.circle(src_copy, (int(x), int(y)), color=(0, 0, 255), radius=int(kp.size), thickness=3)
        return src_copy
        # return cv2.drawKeypoints(src_cv, keypoints, np.array([]), (0,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


class SobelFilter(BaseSliderFilter):
    filter_params = {'ksize': SliderInfo(1, 15, 2, 3)}

    def update(self, src_cv):
        return sobel(src_cv, int(self.widget_list[0].value))


class ThresholdFilter(BaseSliderFilter):
    operators = [
        (cv2.THRESH_BINARY, 'BINARY'),
        (cv2.THRESH_BINARY_INV, 'BINARY_INV'),
        (cv2.THRESH_BINARY + cv2.THRESH_OTSU, 'BINARY+OTSU'),
        (cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU, 'BINARY_INV+OTSU')
    ]
    filter_params = {'thresh': SliderInfo(0, 255, 1, 0),
                     'type': SliderInfo(0, 3, 1, 0)}

    def get_display_callback(self, _filter):
        return {
            'type': self.get_operator_display,
        }.get(_filter, lambda x: str(x))

    def get_operator_code(self, x):
        return self.operators[x][0]

    def get_operator_display(self, x):
        return self.operators[x][1]

    def update(self, src_cv):
        val = self.widget_list[0].value
        oper_int = self.get_operator_code(int(self.widget_list[1].value))
        return cv2.threshold(src_cv, val, 255, oper_int)[1]


# line: [[x1,y1], [x2,y2]]
def line_line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), round(y)


def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


# Return true if line segments AB and CD intersect
def segments_intersect(a, b, c, d):
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


# rect(x,y,w,h)
def point_in_rect(pt, rect):
    return (rect[0] < pt[0] < rect[0] + rect[2]) and (rect[1] < pt[1] < rect[1] + rect[3])


def segment_rect_intersect(line, rect):
    rx, ry, rw, rh = rect
    left = segments_intersect(line[:2], line[2:], (rx, ry), (rx, ry + rh))
    right = segments_intersect(line[:2], line[2:], (rx + rw, ry), (rx + rw, ry + rh))
    top = segments_intersect(line[:2], line[2:], (rx, ry), (rx + rw, ry))
    bottom = segments_intersect(line[:2], line[2:], (rx, ry + rh), (rx + rw, ry + rh))

    return left or right or top or bottom


# find all intersections between the two sets of lines (x1,y1,x2,y2)
def find_intersections(lines):
    if not lines:
        return []

    intersections = []
    for i, line_l in enumerate(lines[:-1]):
        for line_r in lines[i + 1:]:
            intersect = line_line_intersection([line_l[:2], line_l[2:]], [line_r[:2], line_r[2:]])
            if intersect:
                intersections.append(intersect)

    return intersections


# find all intersections between the two sets of lines (x1,y1,x2,y2)
def find_intersections_in_sets(lines_l, lines_r):
    if not lines_l or not lines_r:
        return []

    intersections = []
    for line_l in lines_l:
        for line_r in lines_r:
            intersect = line_line_intersection([line_l[:2], line_l[2:]], [line_r[:2], line_r[2:]])
            if intersect:
                intersections.append(intersect)

    return intersections


def draw_lines(lines, output_cv) -> None:
    for x1, y1, x2, y2 in lines:
        cv2.line(output_cv, (x1, y1), (x2, y2), (0, 128, 255), 1, cv2.LINE_AA)
        cv2.line(output_cv, (x1, y1), (x2, y2), (0, 128, 255), 1, cv2.LINE_AA)


def group_segments(lines, middle_frame_x):
    lines_r = []
    lines_l = []
    for line in lines:
        x1, y1, x2, y2 = line
        if x1 < middle_frame_x and x2 < middle_frame_x:
            if (y1 - y2) * (x1 - x2) < 0:
                lines_l.append(line)

        if middle_frame_x <= x1 and middle_frame_x <= x2:
            if (y1 - y2) * (x1 - x2) > 0:
                lines_r.append(line)

    return lines_l, lines_r


def filter_slope(_line, slope_threshold) -> bool:
    x1, y1, x2, y2 = _line
    if x1 == x2:
        return True
    m = (y2 - y1) / (x2 - x1)
    return not -slope_threshold < m < slope_threshold


def find_vp_candidates(segments, frame_width, min_slope, valid_calib_rect):
    if segments is None or len(segments) == 0:
        return [], [], []

    segments = np.array(segments)[:, 0]

    lines_l, lines_r = group_segments(segments, int(frame_width / 2))
    lines_l = segments_to_lines(lines_l)
    lines_r = segments_to_lines(lines_r)
    lines_l = [line for line in lines_l if filter_slope(line, min_slope)]
    lines_r = [line for line in lines_r if filter_slope(line, min_slope)]
    lines_l = [line for line in lines_l if segment_rect_intersect(line, valid_calib_rect)]
    lines_r = [line for line in lines_r if segment_rect_intersect(line, valid_calib_rect)]
    intersections = find_intersections_in_sets(lines_l, lines_r)
    return intersections, lines_l, lines_r


def segments_to_lines(segments):
    lines = []
    for line in segments:
        x1, y1, x2, y2 = line
        if x2 == x1 or y1 == y2:
            continue
        m = (y2 - y1) / (x2 - x1)

        xf = 100
        _x1 = -int(-x2 + xf * (y2 - y1) / m)
        _y1 = -int(m * xf * (x2 - x1) - y2)

        _x2 = int(x1 + xf * (y2 - y1) / m)
        _y2 = int(m * xf * (x2 - x1) + y1)

        lines.append((_x1, _y1, _x2, _y2))

    return lines


OVERLAY_LINE_HEIGHT = 30


def hi_contrast_text(frame, text, pos):
    cv2.putText(frame, text, (pos[0] + 2, pos[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (1, 1, 1), 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def radians_to_pixels(degree):
    return int(910 * degree / (180 / np.pi))


r2p4, r2p5, r2p8 = radians_to_pixels(4), radians_to_pixels(5), radians_to_pixels(8)


class VanishingPointFilter(BaseSliderFilter):
    filter_params = {
        'threshold': SliderInfo(1, 250, 1, 101),
        'min_line_length': SliderInfo(1, 250, 1, 10),
        'max_gap': SliderInfo(2, 350, 1, 20),
        'min_slope': SliderInfo(0, 1, 0.01, 0)
    }

    def estimate_vp_kmeans(self, intersections):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        n = np.float32(intersections).reshape((-1, 2))
        ret, label, center = cv2.kmeans(n, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        largest_cluster = Counter(label.ravel())
        ll = [(int(l[0]), int(l[1])) for l in center]
        self.all_candidates_counter.update(ll)

        vp = np.int32(center[largest_cluster.most_common(1)[0][0]])
        return vp[0], vp[1]

    def calculate_vp(self, candidates, previous_vp):
        vp = previous_vp
        if len(candidates) > 0:
            candidates = np.array(candidates)
            p = np.array(previous_vp)
            distances = np.linalg.norm(candidates - p, axis=1)
            min_index = np.argmin(distances)
            vp = candidates[min_index]
        return tuple(vp)

    def calculate_vp2(self, candidates):
        vp = None
        if len(candidates) > 0:
            self.nc = len(candidates)
            self.cur_candidates_counter = Counter(candidates)
            self.all_candidates_counter.update(np.array(self.cur_candidates_counter.most_common(10), dtype=object)[:, 0])
            most_common = self.all_candidates_counter.most_common(4)
            most_common = np.array(most_common, dtype=object)
            ws = most_common[:, 1] / sum(most_common[:, 1])
            mean_x = int(np.average([pt[0] for pt in most_common[:, 0]], axis=0, weights=ws))
            mean_y = int(np.average([pt[1] for pt in most_common[:, 0]], axis=0, weights=ws))
            vp = (mean_x, mean_y)
        return vp

    def draw_overlay(self, src_cv, vp) -> np.ndarray:
        if self.overlay is None:
            self.overlay = np.full(src_cv.shape, fill_value=0, dtype=np.uint8)

        overlay_to_display = self.overlay.copy()

        if vp:
            cv2.circle(overlay_to_display, vp, 11, (0, 255, 255), 2)
            overlay_to_display = cv2.add(overlay_to_display, src_cv)

            self.vps = np.append(self.vps, vp)
            mse = np.mean(np.mean((vp[0] - self.vps) ** 2)) ** 0.5
            hi_contrast_text(overlay_to_display, f'vp:{vp}', (0, OVERLAY_LINE_HEIGHT * 2))
            hi_contrast_text(overlay_to_display, f'mse:{int(mse)}', (0, OVERLAY_LINE_HEIGHT * 4))

        hi_contrast_text(overlay_to_display, f'none_found_30_100:{self.nic_0}', (0, OVERLAY_LINE_HEIGHT * 5))
        hi_contrast_text(overlay_to_display, f'none_found_1030:{self.nic_1}', (0, OVERLAY_LINE_HEIGHT * 6))
        hi_contrast_text(overlay_to_display, f'top candidates:{self.all_candidates_counter.most_common(5)}', (0, OVERLAY_LINE_HEIGHT * 7))
        hi_contrast_text(overlay_to_display, f'cur candidates:{self.cur_candidates_counter.most_common(5)}', (0, OVERLAY_LINE_HEIGHT * 8))
        hi_contrast_text(overlay_to_display, f'candidates:{self.nc}', (0, OVERLAY_LINE_HEIGHT * 9))
        hi_contrast_text(overlay_to_display, f'frame:{self.frame_num}', (0, OVERLAY_LINE_HEIGHT * 10))

        # add the current vp to the overlay of previous vp data
        _overlay = self.overlay.copy()
        cv2.circle(_overlay, vp, 1, (0, 255, 255), -1)
        alpha = 0.25
        self.overlay = cv2.addWeighted(_overlay, alpha, self.overlay, 1 - alpha, 0)

        return overlay_to_display

    overlay = None
    all_candidates_counter = Counter()
    cur_candidates_counter = Counter()
    vps = np.array([])
    nic_0 = 0
    nic_1 = 0
    previous_vp = None
    nc = 0
    frame_num = -1

    def on_new_frame(self):
        self.frame_num += 1

    def update(self, src_cv):
        threshold = int(self.widget_list[0].value)
        min_line_length = int(self.widget_list[1].value)
        max_gap = int(self.widget_list[2].value)
        min_slope = self.widget_list[3].value

        frame_height, frame_width = src_cv.shape[:2]
        output_cv = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        mid_x, mid_y = round(frame_width / 2), round(frame_height / 2)
        if not self.previous_vp:
            self.previous_vp = mid_x, mid_y
        valid_calib_rect = (mid_x - r2p4, mid_y - r2p5, 2 * r2p4, r2p5 + r2p8)

        canny = cv2.Canny(src_cv, 30, 100, edges=None, apertureSize=3, L2gradient=False)
        lines = cv2.HoughLinesP(canny, 1, np.pi / 180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_gap)
        intersections, lines_l, lines_r = find_vp_candidates(lines, frame_width, min_slope, valid_calib_rect)

        if len(intersections) == 0:
            self.nic_0 += 1
            canny = cv2.Canny(src_cv, 10, 30, edges=None, apertureSize=3, L2gradient=False)
            lines = cv2.HoughLinesP(canny, 1, np.pi / 180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_gap)
            intersections, lines_l, lines_r = find_vp_candidates(lines, frame_width, min_slope, valid_calib_rect)

        if len(intersections) == 0:
            self.nic_1 += 1
        else:
            draw_lines(lines_l, output_cv)
            draw_lines(lines_r, output_cv)
        if len(intersections) < 50:
            vp = self.calculate_vp(intersections, self.previous_vp)
        else:
            vp = self.estimate_vp_kmeans(intersections)
        if vp:
            self.previous_vp = vp
        return self.draw_overlay(output_cv, vp)


class GTFilter(BaseSliderFilter):
    filter_params = {
        'flip_pitch_yaw': SliderInfo(0, 1, 1, 0)
    }

    def __init__(self, params):
        super().__init__(params)
        self.gt_index = -1
        input_file = config.current_media_path

        path, _ = os.path.splitext(input_file)
        path = path + '.txt'
        self.gt = np.loadtxt(path)
        self.overlay = None

    def on_new_frame(self):
        self.gt_index += 1

    def update(self, src_cv):
        height, width = src_cv.shape[:2]
        if self.widget_list[0].value:
            yaw, pitch = self.gt[self.gt_index]
        else:
            pitch, yaw = self.gt[self.gt_index]

        if self.overlay is None:
            self.overlay = np.full(src_cv.shape, fill_value=0, dtype=np.uint8)

        overlay_to_display = self.overlay.copy()
        if not np.isnan(yaw) and not np.isnan(pitch):
            vp = (int(width / 2 + 910 * yaw), int(height / 2 - 910 * pitch))
            cv2.circle(overlay_to_display, vp, 11, (0, 255, 0), 2)

            mid_x, mid_y = round(width / 2), round(height / 2)
            cv2.rectangle(overlay_to_display, (mid_x - r2p4, mid_y - r2p5), (mid_x + r2p4, mid_y + r2p8), (255, 255, 255), 2)

            _overlay = self.overlay.copy()
            cv2.circle(_overlay, vp, 1, (0, 255, 0), -1)
            alpha = 0.25
            self.overlay = cv2.addWeighted(_overlay, alpha, self.overlay, 1 - alpha, 0)

            hi_contrast_text(overlay_to_display, f'gt:{vp}', (0, OVERLAY_LINE_HEIGHT * 1))

        return overlay_to_display
