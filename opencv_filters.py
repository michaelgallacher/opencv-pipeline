import cv2 as cv
import numpy as np

from base_filters import BaseFilter, BaseSliderFilter


# Thanks Adrian @ https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example!
def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (max_width, max_height))
    # return the warped image
    return warped


def sobel(src_cv, sobel_blur):
    ddepth = cv.CV_16S
    ksize = sobel_blur

    # [reduce_noise]
    # Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    src_cv = cv.GaussianBlur(src_cv, (ksize, ksize), 0)
    # [reduce_noise]

    # [convert_to_gray]
    # Convert the image to grayscale
    # gray = cv.cvtColor(src_cv, cv.COLOR_BGR2GRAY)
    gray = src_cv
    # [convert_to_gray]

    # [sobel]
    # Gradient-X
    grad_x = cv.Sobel(gray, ddepth, 1, 0)

    # Gradient-Y
    grad_y = cv.Sobel(gray, ddepth, 0, 1)
    # [sobel]

    # [convert]
    # converting back to uint8
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    # [convert]

    # [blend]
    # Total Gradient (approximate)
    grad = cv.addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0)

    return grad


class AdaptiveThresholdFilter(BaseSliderFilter):
    filter_params = {'threshold': {'min_val': 0, 'max_val': 255, 'step_val': 1},
                     'threshold_type': {'min_val': 0, 'max_val': 1, 'step_val': 1},
                     'block_size': {'min_val': 3, 'max_val': 9, 'step_val': 2},
                     'C': {'min_val': -5, 'max_val': 5, 'step_val': 1}}
    opers = [
        (cv.THRESH_BINARY, 'BINARY'),
        (cv.THRESH_BINARY_INV, 'BINARY_INV')]

    def get_display_callback(self, filter):
        return {
            'type': self.get_oper_display,
        }.get(filter, lambda x: str(x))

    def get_oper_code(self, x):
        return self.opers[x][0]

    def get_oper_display(self, x):
        return self.opers[x][1]

    def update(self, src_cv):
        threshold = int(self.widget_list[0].value)
        threshold_type = self.get_oper_code(int(self.widget_list[1].value))
        block_size = int(self.widget_list[2].value)
        C = int(self.widget_list[3].value)

        return cv.adaptiveThreshold(src_cv, threshold, cv.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type, block_size, C)


class BilateralFilter(BaseSliderFilter):
    filter_params = {'kernel': {'min_val': 1, 'max_val': 51, 'step_val': 2},
                     'sigma_color': {'min_val': 3, 'max_val': 93, 'step_val': 2},
                     'sigma_space': {'min_val': 3, 'max_val': 93, 'step_val': 2},
                     'sigmas': {'min_val': 3, 'max_val': 93, 'step_val': 2}}

    def update(self, src_cv):
        kernel = int(self.widget_list[0].value)
        sigma_color = int(self.widget_list[1].value)
        sigma_space = int(self.widget_list[2].value)
        return cv.bilateralFilter(src_cv, kernel, sigma_color, sigma_space)


class BitwiseAndFilter(BaseFilter):
    filter_params = {}

    def update(self, src_cv):
        return cv.bitwise_and(src_cv[0], src_cv[1])


class BitwiseNotFilter(BaseFilter):
    filter_params = {}

    def update(self, src_cv):
        return cv.bitwise_not(src_cv)


class BitwiseOrFilter(BaseFilter):
    filter_params = {}

    def update(self, src_cv):
        return cv.bitwise_or(src_cv[0], src_cv[1])


class BlurFilter(BaseSliderFilter):
    filter_params = {'size': {'min_val': 1, 'max_val': 15, 'step_val': 2}}

    def update(self, src_cv):
        val = int(self.widget_list[0].value)
        return cv.blur(src_cv, (val, val))


class CannyFilter(BaseSliderFilter):
    filter_params = {'thresh1': {'min_val': 0, 'max_val': 1000, 'step_val': 1},
                     'thresh2': {'min_val': 0, 'max_val': 1000, 'step_val': 1},
                     'aperture_size': {'min_val': 3, 'max_val': 7, 'step_val': 2},
                     'l2_gradient': {'min_val': 0, 'max_val': 1, 'step_val': 1}}

    def update(self, src_cv):
        thresh1 = int(self.widget_list[0].value)
        thresh2 = int(self.widget_list[1].value)
        aperture_size = int(self.widget_list[2].value)
        l2_gradient = int(self.widget_list[3].value) == 1
        return cv.Canny(src_cv, thresh1, thresh2, edges=None, apertureSize=aperture_size, L2gradient=l2_gradient)


class ChannelFilter(BaseSliderFilter):
    filter_params = {'channel': {'min_val': 0, 'max_val': 2, 'step_val': 1},
                     'thresh': {'min_val': 0, 'max_val': 255, 'step_val': 1},
                     'type': {'min_val': 0, 'max_val': 2, 'step_val': 1}}

    def update(self, src_cv):
        channel = int(self.widget_list[0].value)
        val = int(self.widget_list[1].value)
        oper_int = int(self.widget_list[2].value)
        src_cv = src_cv[:, :, channel]
        return cv.threshold(src_cv, val, 255, oper_int)[1]


class ClipFilter(BaseSliderFilter):
    filter_params = {'clip': {'min_val': 0, 'max_val': 20, 'step_val': 1}}

    def update(self, src_cv):
        val = int(self.widget_list[0].value)

        return src_cv[val:src_cv.shape[1] - val, val:src_cv.shape[0] - val]


class ContoursFilter(BaseSliderFilter):
    opers = [
        (cv.RETR_EXTERNAL, 'RETR_EXTERNAL'),
        (cv.RETR_LIST, 'RETR_LIST'),
        (cv.RETR_CCOMP, 'RETR_CCOMP'),
        (cv.RETR_TREE, 'RETR_TREE'),
        (cv.RETR_FLOODFILL, 'RETR_FLOODFILL'),
    ]

    filter_params = {'lower': {'min_val': 0, 'max_val': 150000, 'step_val': 50},
                     'upper': {'min_val': 0, 'max_val': 150000, 'step_val': 50},
                     "method": {'min_val': 0, 'max_val': 4, 'step_val': 1}}

    def get_display_callback(self, filter):
        return {
            'method': self.get_oper_display,
        }.get(filter, lambda x: str(x))

    def get_oper_code(self, x):
        return self.opers[x][0]

    def get_oper_display(self, x):
        return self.opers[x][1]

    def update(self, src_cv):
        lower = int(self.widget_list[0].value)
        upper = int(self.widget_list[1].value)
        method_int = int(self.widget_list[2].value)

        contours, _ = cv.findContours(src_cv, method_int, cv.CHAIN_APPROX_SIMPLE)

        output_cv = np.zeros((src_cv.shape[0], src_cv.shape[1], 3), dtype=np.uint8)
        contours_to_draw = [c for c in contours if lower < cv.contourArea(c) < upper]

        if len(contours_to_draw) > 0:
            for contour in contours_to_draw:
                color = (255, 255, 255)  # (rng.randint(65, 256), rng.randint(65, 256), rng.randint(65, 256))
                cv.drawContours(output_cv, [contour], -1, color, thickness=1, lineType=cv.LINE_AA)

        return output_cv


class CropFilter(BaseSliderFilter):
    filter_params = {
        'left': {'min_val': 0, 'max_val': 100, 'step_val': 1},
        'top': {'min_val': 0, 'max_val': 100, 'step_val': 1},
        'right': {'min_val': 0, 'max_val': 100, 'step_val': 1},
        'bottom': {'min_val': 0, 'max_val': 100, 'step_val': 1}}

    def update(self, src_cv):
        left = int(self.widget_list[0].value)
        top = int(self.widget_list[1].value)
        right = int(self.widget_list[2].value)
        bottom = int(self.widget_list[3].value)
        h, w, _ = src_cv.shape
        return src_cv[int(h * top / 100.0):int(h * bottom / 100.00), int(w * left / 100.0):int(w * right / 100.0)]


class EqualizeHistogramFilter(BaseFilter):
    filter_params = {}

    def update(self, src_cv):
        return cv.equalizeHist(src_cv)


class FourPointFilter(BaseSliderFilter):
    filter_params = {'tl_x': {'min_val': 0, 'max_val': 100, 'step_val': 1},
                     'tl_y': {'min_val': 0, 'max_val': 100, 'step_val': 1},
                     'tr_x': {'min_val': 0, 'max_val': 100, 'step_val': 1},
                     'tr_y': {'min_val': 0, 'max_val': 100, 'step_val': 1},
                     'br_x': {'min_val': 0, 'max_val': 100, 'step_val': 1},
                     'br_y': {'min_val': 0, 'max_val': 100, 'step_val': 1},
                     'bl_x': {'min_val': 0, 'max_val': 100, 'step_val': 1},
                     'lb_y': {'min_val': 0, 'max_val': 100, 'step_val': 1}}

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


class GaussianBlurFilter(BaseSliderFilter):
    filter_params = {'size': {'min_val': 1, 'max_val': 15, 'step_val': 2}}

    def update(self, src_cv):
        val = int(self.widget_list[0].value)
        return cv.GaussianBlur(src_cv, (val, val), 0)


class GrayscaleFilter(BaseFilter):
    filter_params = {}

    def update(self, src_cv):
        return cv.cvtColor(src_cv, cv.COLOR_BGR2GRAY)


class HiLoThresholdFilter(BaseSliderFilter):
    filter_params = {'lower': {'min_val': 0, 'max_val': 255, 'step_val': 1},
                     'upper': {'min_val': 1, 'max_val': 255, 'step_val': 1}}

    def update(self, src_cv):
        lower = int(self.widget_list[0].value)
        upper = int(self.widget_list[1].value)
        return cv.inRange(src_cv, lower, upper)


class HiLo2ThresholdFilter(BaseSliderFilter):
    filter_params = {'thresh': {'min_val': 0, 'max_val': 255, 'step_val': 1},
                     'neighborhood': {'min_val': 1, 'max_val': 128, 'step_val': 2}}

    def update(self, src_cv):
        thresh = int(self.widget_list[0].value)
        neighborhood = int(self.widget_list[1].value)
        return cv.inRange(src_cv, max(0, thresh - neighborhood), min(255, thresh + neighborhood))


class HSVFilter(BaseSliderFilter):
    filter_params = {'h-lo': {'min_val': 0, 'max_val': 180, 'step_val': 1},
                     'h-hi': {'min_val': 0, 'max_val': 180, 'step_val': 1},
                     's-lo': {'min_val': 0, 'max_val': 255, 'step_val': 1},
                     's-hi': {'min_val': 0, 'max_val': 255, 'step_val': 1},
                     'v-lo': {'min_val': 0, 'max_val': 255, 'step_val': 1},
                     'v-hi': {'min_val': 0, 'max_val': 255, 'step_val': 1}}

    def update(self, src_cv):
        h_lower = int(self.widget_list[0].value)
        h_upper = int(self.widget_list[1].value)
        s_lower = int(self.widget_list[2].value)
        s_upper = int(self.widget_list[3].value)
        v_lower = int(self.widget_list[4].value)
        v_upper = int(self.widget_list[5].value)
        return cv.inRange(cv.cvtColor(src_cv, cv.COLOR_BGR2HSV), (h_lower, s_lower, v_lower), (h_upper, s_upper, v_upper))


class MedianBlurFilter(BaseSliderFilter):
    filter_params = {'size': {'min_val': 1, 'max_val': 15, 'step_val': 2}}

    def update(self, src_cv):
        val = int(self.widget_list[0].value)
        return cv.medianBlur(src_cv, val)


class MorphFilter(BaseSliderFilter):
    opers = [
        (cv.MORPH_ERODE, 'MORPH_ERODE'),
        (cv.MORPH_DILATE, 'MORPH_DILATE'),
        (cv.MORPH_OPEN, 'MORPH_OPEN'),
        (cv.MORPH_CLOSE, 'MORPH_CLOSE'),
        (cv.MORPH_GRADIENT, 'MORPH_GRADIENT'),
        (cv.MORPH_TOPHAT, 'MORPH_TOPHAT'),
        (cv.MORPH_BLACKHAT, 'MORPH_BLACKHAT'),
    ]

    elements = [
        (cv.MORPH_RECT, 'MORPH_RECT'),
        (cv.MORPH_CROSS, 'MORPH_CROSS'),
        (cv.MORPH_ELLIPSE, 'MORPH_ELLIPSE'),
    ]

    filter_params = {'size_x': {'min_val': 1, 'max_val': 150, 'step_val': 2},
                     'size_y': {'min_val': 1, 'max_val': 150, 'step_val': 2},
                     'iters': {'min_val': 1, 'max_val': 9, 'step_val': 1},
                     'operator': {'min_val': 0, 'max_val': len(opers) - 1, 'step_val': 1},
                     'element': {'min_val': 0, 'max_val': len(elements) - 1, 'step_val': 1}}

    def get_display_callback(self, filter):
        return {
            'operator': self.get_oper_display,
            'element': self.get_element_display,
        }.get(filter, lambda x: str(x))

    def get_oper_code(self, x):
        return self.opers[x][0]

    def get_oper_display(self, x):
        return self.opers[x][1]

    def get_element_code(self, x):
        return self.elements[x][0]

    def get_element_display(self, x):
        return self.elements[x][1]

    def update(self, src_cv):
        morph_size_x = int(self.widget_list[0].value)
        morph_size_y = int(self.widget_list[1].value)
        iterations = int(self.widget_list[2].value)
        oper_int = self.get_oper_code(int(self.widget_list[3].value))
        elem_int = self.get_element_code(int(self.widget_list[4].value))
        element = cv.getStructuringElement(elem_int, (morph_size_x, morph_size_y))
        return cv.morphologyEx(src_cv, oper_int, element, iterations=iterations)


class PassthroughFilter(BaseFilter):
    def update(self, src_cv):
        return src_cv


class RGBFilter(BaseSliderFilter):
    filter_params = {'r_min': {'min_val': 0, 'max_val': 255, 'step_val': 1},
                     'r_max': {'min_val': 0, 'max_val': 255, 'step_val': 1},
                     'g_min': {'min_val': 0, 'max_val': 255, 'step_val': 1},
                     'g_max': {'min_val': 0, 'max_val': 255, 'step_val': 1},
                     'b_min': {'min_val': 0, 'max_val': 255, 'step_val': 1},
                     'b_max': {'min_val': 0, 'max_val': 255, 'step_val': 1}}

    def update(self, src_cv):
        r_min = int(self.widget_list[0].value)
        r_max = int(self.widget_list[1].value)
        g_min = int(self.widget_list[2].value)
        g_max = int(self.widget_list[3].value)
        b_min = int(self.widget_list[4].value)
        b_max = int(self.widget_list[5].value)

        return cv.inRange(src_cv, (b_min, g_min, r_min), (b_max, g_max, r_max))


class SimpleBlobDetectorFilter(BaseSliderFilter):
    filter_params = {
        'minThreshold': {'min_val': 0, 'max_val': 255, 'step_val': 1},
        'maxThreshold': {'min_val': 0, 'max_val': 255, 'step_val': 1},
        'minArea': {'min_val': 500, 'max_val': 100000, 'step_val': 100},
        'maxArea': {'min_val': 1000, 'max_val': 100000, 'step_val': 100},
        'minCircularity': {'min_val': 0, 'max_val': 1, 'step_val': 0.1},
        'maxCircularity': {'min_val': 0, 'max_val': 1, 'step_val': 0.1},
        'minConvexity': {'min_val': 0, 'max_val': 1, 'step_val': 0.1},
        'maxConvexity': {'min_val': 0, 'max_val': 1, 'step_val': 0.1},
        'minInertia': {'min_val': 0, 'max_val': 1, 'step_val': 0.1},
        'maxInertia': {'min_val': 0, 'max_val': 1, 'step_val': 0.1},
        'minDistBetweenBlobs': {'min_val': 1, 'max_val': 100, 'step_val': 1},
        'minRepeatability': {'min_val': 1, 'max_val': 100, 'step_val': 1},
        'thresholdStep': {'min_val': 1, 'max_val': 127, 'step_val': 1},
    }

    def update(self, src_cv):
        # Setup SimpleBlobDetector parameters.
        params = cv.SimpleBlobDetector_Params()

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
        # OLD: detector = cv2.SimpleBlobDetector(params)
        detector = cv.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(src_cv)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob

        src_copy = src_cv.copy()
        print("kps:" + str(len(keypoints)))
        for kp in keypoints:
            x, y = kp.pt
            cv.circle(src_copy, (int(x), int(y)), color=(0, 0, 255), radius=int(kp.size), thickness=3)
        return src_copy
        # return cv.drawKeypoints(src_cv, keypoints, np.array([]), (0,0,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


class SobelFilter(BaseSliderFilter):
    filter_params = {'ksize': {'min_val': 1, 'max_val': 15, 'step_val': 2}}

    def update(self, src_cv):
        return sobel(src_cv, int(self.widget_list[0].value))


class ThresholdFilter(BaseSliderFilter):
    opers = [
        (cv.THRESH_BINARY, 'BINARY'),
        (cv.THRESH_BINARY_INV, 'BINARY_INV'),
        (cv.THRESH_BINARY + cv.THRESH_OTSU, 'BINARY+OTSU'),
        (cv.THRESH_BINARY_INV + cv.THRESH_OTSU, 'BINARY_INV+OTSU')
    ]
    filter_params = {'thresh': {'min_val': 0, 'max_val': 255, 'step_val': 1},
                     'type': {'min_val': 0, 'max_val': 3, 'step_val': 1}}

    def get_display_callback(self, filter):
        return {
            'type': self.get_oper_display,
        }.get(filter, lambda x: str(x))

    def get_oper_code(self, x):
        return self.opers[x][0]

    def get_oper_display(self, x):
        return self.opers[x][1]

    def update(self, src_cv):
        val = self.widget_list[0].value
        oper_int = self.get_oper_code(int(self.widget_list[1].value))
        return cv.threshold(src_cv, val, 255, oper_int)[1]
