import argparse
import importlib
import json
import os
import traceback
from collections import namedtuple

## This line MUST BE ABOVE all kivy import statements
os.environ['KIVY_NO_ARGS'] = '1'

from kivy.app import App
from kivy.uix.accordion import Accordion
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.splitter import Splitter

from opencv_filters import *
from util import cv_to_kivy_texture

SliderInfo = namedtuple('SliderInfo', 'name min_val max_val init_val step_val')
SliderInfo2 = namedtuple('SliderInfo', 'min_val max_val step_val')


class Pipeline(Accordion):
    preview = Image(allow_stretch=True, keep_ratio=True)

    def __init__(self):
        super().__init__(orientation='vertical')
        self.src_cv = None
        self.anim_duration = 0.1

    def on_collapsed(self, instance, value):
        if not value and instance.preview:
            self.preview.texture = instance.preview.texture
            self.update()

    def on_update(self, instance, value):
        self.update()

    def add_filter(self, filter_widget):
        filter_widget.value_changed = self.invalidated
        filter_widget.bind(collapse=self.on_collapsed)
        filter_widget.bind(is_enabled=self.on_update)
        self.add_widget(filter_widget)

    # noinspection PyBroadException
    def update(self):
        images_with_tids = {}
        next_image = self.src_cv
        for _filter in reversed(self.children):
            if _filter.is_enabled:
                try:
                    # use the custom input image(s), if any
                    if _filter.input:
                        next_image = images_with_tids[_filter.input]
                    elif _filter.inputs:
                        next_image = list(map(lambda i: images_with_tids[i], _filter.inputs))

                    # process the image and update the id table, if necessary.
                    next_image = _filter.update(next_image)

                    if _filter.tid:
                        images_with_tids[_filter.tid] = next_image

                    # set the preview if the widget isn't collapsed
                    if not _filter.collapse:
                        self.preview.texture = cv_to_kivy_texture(next_image)

                    # clear any error
                    _filter.error = ''
                except Exception:
                    print(traceback.format_exc())
                    _filter.error = traceback.format_exc()

        return next_image


class PipelineApp(App):
    def __init__(self, **kwargs):
        super().__init__()
        filename = kwargs.get('image_path')
        if not os.path.exists(filename):
            print(f'file {filename} not found.')
            exit(1)
        self.pipeline_name = kwargs.get('pipeline_path')
        if not os.path.exists(self.pipeline_name):
            print(f'file {self.pipeline_name} not found.')
            exit(1)

        self.src_cv = cv.imread(filename)
        self.src_cv = cv.flip(self.src_cv, 0)

        self.src_image = Image(allow_stretch=True, keep_ratio=True)
        self.src_image.texture = cv_to_kivy_texture(self.src_cv)
        self.dest_image = Image(allow_stretch=True, keep_ratio=True)
        self.pipeline_widgets = Pipeline()

        self.pipeline_widgets.src_cv = self.src_cv
        self.pipeline_widgets.invalidated = self.update
        self.pipeline_widgets.preview = self.src_image

    def load_pipeline(self, file):
        f = open(file)
        json_pipeline = f.read()

        pipeline = json.loads(json_pipeline)
        for f in pipeline:
            module = importlib.import_module('main')
            class_ = getattr(module, f['filter'])
            params = f.get('params', None)
            filter_instance = class_() if params is None else class_(params)
            filter_instance.display_name = f.get('tid') if f.get('tid', None) else filter_instance.__class__.__name__
            filter_instance.is_enabled = f.get('enabled', False)
            filter_instance.input = f.get('input', None)
            filter_instance.inputs = f.get('inputs', None)
            filter_instance.tid = f.get('tid', None)

            self.pipeline_widgets.add_filter(filter_instance)

    def build(self):
        root = BoxLayout(orientation='vertical')

        main_box = BoxLayout(orientation='horizontal')
        # source image
        splitter = Splitter(sizable_from='right', min_size=100, strip_size='6pt')
        main_box.add_widget(splitter)

        splitter.add_widget(self.src_image)

        # pipeline
        splitter = Splitter(sizable_from='right', min_size=100, strip_size='6pt')
        main_box.add_widget(splitter)

        splitter.add_widget(self.pipeline_widgets)

        # dest image
        main_box.add_widget(self.dest_image)

        root.add_widget(main_box)

        return root

    def on_start(self):
        self.load_pipeline(self.pipeline_name)
        self.update()

    def update(self):
        self.dest_image.texture = cv_to_kivy_texture(self.pipeline_widgets.update())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image_path', required=False, default='test1.png', help='path to input image')
    ap.add_argument('-p', '--pipeline_path', required=False, default='test_pipeline.json', help='path to json describing pipeline')
    args = vars(ap.parse_args())

    PipelineApp(image_path=args['image_path'], pipeline_path=args['pipeline_path']).run()
    cv.destroyAllWindows()
