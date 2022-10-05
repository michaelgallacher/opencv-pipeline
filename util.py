import cv2
import numpy as np
from kivy.graphics.texture import Texture


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


def cv_to_kivy_texture(src_cv):
    if len(src_cv.shape) == 2:
        out_cv = cv2.cvtColor(src_cv, cv2.COLOR_GRAY2BGR)
    else:
        out_cv = src_cv

    frame = (out_cv.shape[1], out_cv.shape[0])
    buf = out_cv.tostring()
    texture = Texture.create(size=(frame[0], frame[1]), colorfmt='bgr')
    texture.flip_vertical()
    texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    return texture


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

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    # return the warped image
    return warped


def sobel(src_cv, sobel_blur):
    ddepth = cv2.CV_16S
    ksize = sobel_blur

    # Remove noise by blurring with a Gaussian filter
    src_cv = cv2.GaussianBlur(src_cv, (ksize, ksize), 0)

    # Convert the image to grayscale
    # gray = cv2.cvtColor(src_cv, cv2.COLOR_BGR2GRAY)
    gray = src_cv

    # Gradient-X
    grad_x = cv2.Sobel(gray, ddepth, 1, 0)

    # Gradient-Y
    grad_y = cv2.Sobel(gray, ddepth, 0, 1)

    # converting back to uint8
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # Total Gradient (approximate)
    grad = cv2.addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0)

    return grad
