import random as rng
import cv2 as cv
import numpy as np

from base_filters import BaseFilter, BaseSliderFilter


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
    filter_params = {'lower': {'min_val': 0, 'max_val': 15000, 'step_val': 10},
                     'upper': {'min_val': 0, 'max_val': 15000, 'step_val': 10}}

    def update(self, src_cv):
        lower = int(self.widget_list[0].value)
        upper = int(self.widget_list[1].value)

        contours, _ = cv.findContours(src_cv, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        output_cv = np.zeros((src_cv.shape[0], src_cv.shape[1], 3), dtype=np.uint8)
        contours_to_draw = [c for c in contours if lower < cv.contourArea(c) < upper]
        if len(contours_to_draw) > 0:
            for contour in contours_to_draw:
                color = (255, 255, 255)  # (rng.randint(65, 256), rng.randint(65, 256), rng.randint(65, 256))
                cv.drawContours(output_cv, [contour], -1, color, thickness=1, lineType=cv.LINE_8)

        return output_cv


class EqualizeHistogramFilter(BaseFilter):
    filter_params = {}

    def update(self, src_cv):
        return cv.equalizeHist(src_cv)


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
        return cv.inRange(src_cv, (h_lower, s_lower, v_lower), (h_upper, s_upper, v_upper))


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

    filter_params = {'size_x': {'min_val': 1, 'max_val': 50, 'step_val': 2},
                     'size_y': {'min_val': 1, 'max_val': 50, 'step_val': 2},
                     'iters': {'min_val': 1, 'max_val': 3, 'step_val': 1},
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
