from collections import Counter

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
