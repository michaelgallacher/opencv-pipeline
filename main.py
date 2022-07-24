import argparse
import importlib
import json
import os
import traceback
from collections import namedtuple

# To enable command-line arguments, this line MUST BE ABOVE all kivy import statements
os.environ['KIVY_NO_ARGS'] = '1'

# To enable a maximized window, this line MUST BE ABOVE all kivy import statements...except the previous one.
from kivy.config import Config
Config.set('graphics', 'window_state', 'maximized')

from kivy.core.image import Image as CoreImage
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
from kivy.uix.actionbar import ActionBar, ActionView, ActionButton, ActionPrevious
from kivy.properties import ObjectProperty
from kivy.app import App
from kivy.core.window import Window
from kivy.modules import inspector
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.uix.splitter import Splitter

from DragDrop import DraggableBoxLayoutBehavior
from opencv_filters import *
from util import cv_to_kivy_texture

SliderInfo = namedtuple('SliderInfo', 'name min_val max_val init_val step_val')
SliderInfo2 = namedtuple('SliderInfo', 'min_val max_val step_val')


class DraggableAccordionLayout(DraggableBoxLayoutBehavior, GridLayout):
    def handle_drag_release(self, index, drag_widget):
        self.add_widget(drag_widget, index)


class Pipeline(DraggableAccordionLayout):
    preview = Image(allow_stretch=True, keep_ratio=True)

    selected_item = ObjectProperty(None)

    def __init__(self):
        super().__init__()
        self.src_cv = None
        self.cols = 1
        self.size_hint_y = None
        self.spacing = 30
        self.bind(minimum_height=self.setter('height'))

    def on_selected_item(self, instance, value):
        if value and value.filter_preview:
            self.preview.texture = value.filter_preview.texture
            self.update()

    def on_update(self, instance, value):
        self.update()

    def on_touch_down(self, touch):
        for child in self.children:
            if child.title.collide_point(*touch.pos) and child != self.selected_item:
                if self.selected_item:
                    self.selected_item.is_selected = False
                child.is_selected = True
                self.selected_item = child
                break

        return super().on_touch_down(touch)

    def add_filter(self, filter_widget):
        filter_widget.value_changed = self.invalidated
        filter_widget.bind(is_enabled=self.on_update)
        filter_widget.update_ui()
        self.add_widget(filter_widget)

    # noinspection PyBroadException
    def update(self):
        images_with_tids = {}
        next_image = self.src_cv.copy()
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

                    # clear any error
                    _filter.error = ''

                    # set the preview if the widget is selected
                    if _filter.is_selected:
                        self.preview.texture = cv_to_kivy_texture(next_image)
                        # the preview for the pipeline is set so skip the rest of the filters.
                        break

                except Exception:
                    print(traceback.format_exc())
                    _filter.error = traceback.format_exc()

        return next_image


class TiledBackgroundImage(Image):
    def __init__(self, **kwargs):
        super(TiledBackgroundImage, self).__init__(**kwargs)

    def resize(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            texture = CoreImage("tile.png").texture
            texture.wrap = 'repeat'
            nx = float(self.width) / texture.width
            ny = float(self.height) / texture.height
            Rectangle(pos=self.pos, size=self.size, texture=texture,
                      tex_coords=(0, 0, nx, 0, nx, ny, 0, ny))

    on_size = resize


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

        extension = filename[-3:].lower()
        if extension in ('jpg', 'png'):
            self.video_capture = None
            self.src_cv = cv.imread(filename)
        elif extension in ('mp4', 'mov'):
            self.video_capture = cv.VideoCapture(filename)
            ret_, self.src_cv = self.video_capture.read()
        else:
            print(f'file type {extension} is not supported')
            exit(1)

        self.src_image = TiledBackgroundImage(allow_stretch=True, keep_ratio=True)
        self.src_image.texture = cv_to_kivy_texture(self.src_cv)
        self.dest_image = TiledBackgroundImage(allow_stretch=True, keep_ratio=True)
        self.pipeline_widgets = Pipeline()

        self.pipeline_widgets.src_cv = self.src_cv
        self.pipeline_widgets.invalidated = self.update
        self.pipeline_widgets.preview = self.dest_image

        self.interval_event = None

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

        action_bar = ActionBar()

        action_view = ActionView()
        action_view.use_separator = True

        action_previous = ActionPrevious()
        action_previous.title = 'OpenCV Pipeline'
        action_previous.with_previous = False
        action_view.add_widget(action_previous)

        action_button = ActionButton()
        action_button.text = "File"
        action_view.add_widget(action_button)

        action_bar.add_widget(action_view)
        root.add_widget(action_bar)

        main_box = BoxLayout(orientation='horizontal')
        # source image
        splitter = Splitter(sizable_from='right', min_size=100, strip_size='6pt')
        main_box.add_widget(splitter)

        splitter.add_widget(self.src_image)

        # pipeline
        splitter = Splitter(sizable_from='right', min_size=100, strip_size='6pt')
        main_box.add_widget(splitter)

        sv = ScrollView(size_hint=(1, None))
        main_box.bind(size=sv.setter('size'))
        sv.bar_color = [1, 1, 1, 1]
        sv.bar_width = 10
        sv.scroll_distance = '10dp'
        sv.scroll_timeout = 100
        sv.scroll_wheel_distance = '15dp'
        sv.smooth_scroll_end = 20
        sv.add_widget(self.pipeline_widgets)
        splitter.add_widget(sv)

        # dest image
        main_box.add_widget(self.dest_image)

        root.add_widget(main_box)

        Window.bind(on_key_down=self.on_key_down)

        return root

    def on_key_down(self, window, key, scancode, codepoint, modifier):
        if not self.video_capture:
            return

        if codepoint == ' ':
            if self.interval_event:
                self.interval_event.cancel()
                self.interval_event = None
            else:
                self.interval_event = Clock.schedule_interval(self.on_interval, 1.0 / 33.0)
        elif codepoint == '.':
            if self.interval_event:
                self.interval_event.cancel()
                self.interval_event = None
            else:
                self.interval_event = None
                Clock.schedule_once(self.on_interval)

    def on_start(self):
        self.load_pipeline(self.pipeline_name)
        Clock.schedule_once(self.on_interval)

        inspector.create_inspector(Window, self)

    # Called when a video is in use.
    def on_interval(self, dt):
        if self.video_capture:
            ret_, self.src_cv = self.video_capture.read()
            if not ret_:
                return

        self.update()

    # Called after 'src_cv' is updated.
    def update(self):
        self.src_image.texture = cv_to_kivy_texture(self.src_cv)
        self.pipeline_widgets.src_cv = self.src_cv
        self.dest_image.texture = cv_to_kivy_texture(self.pipeline_widgets.update())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image_path', required=False, default='test1.png', help='path to input image')
    ap.add_argument('-p', '--pipeline_path', required=False, default='test_pipeline.json', help='path to json describing pipeline')
    args = vars(ap.parse_args())

    PipelineApp(image_path=args['image_path'], pipeline_path=args['pipeline_path']).run()
    cv.destroyAllWindows()
