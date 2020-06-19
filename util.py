import cv2 as cv
from kivy.graphics.texture import Texture


def cv_to_kivy_texture(src_cv):
    if len(src_cv.shape) == 2:
        out_cv = cv.cvtColor(src_cv, cv.COLOR_GRAY2BGR)
    else:
        out_cv = src_cv

    frame = (out_cv.shape[1], out_cv.shape[0])
    buf = out_cv.tostring()
    texture = Texture.create(size=(frame[0], frame[1]), colorfmt='bgr')
    texture.flip_vertical()
    texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    return texture
