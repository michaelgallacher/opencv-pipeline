from kivy.animation import Animation
from kivy.lang import Builder
from kivy.properties import BooleanProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.slider import Slider

from DragDrop import DraggableBoxLayoutBehavior, DraggableController, DraggableObjectBehavior

drag_controller = DraggableController()


class DragFilter(DraggableObjectBehavior, BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs, drag_controller=drag_controller)


Builder.load_string('''
<BaseFilter>:
    value_changed: None
    display_name: ''
    title: title
    title_layout: title_layout
    enabled_button: enabled_button
    collapsed_button: collapsed_button
    # controls: controls
    controls_holder: controls_holder
    orientation: 'vertical'
    size_hint_y: None
    height: title_layout.height
    canvas.after:
        Color: 
            rgba: (1,1,1,1) if root.is_selected else (0,0,0,0)
        Line:
            points: [root.x,root.y, root.right,root.y, root.right,root.top, root.x,root.top, root.x,root.y]
            width: 3  
    BoxLayout:
        id: title_layout
        orientation: 'horizontal'
        size_hint_y: None
        height: title.height
        Button:
            id: collapsed_button
            text: 'V'
            on_press: root.on_collapsed_pressed(self)
            background_normal: ''
            background_down: ''
            background_color: 0.1,0.4,1,1
            size_hint: (None, None)
            size: (40, title.height)
            pos_hint: {'top': 1, 'x': 1}
        Label:
            canvas.before:
                Color: 
                    rgba: (0.1,0.4,1,1) 
                Rectangle:
                    size: self.size
                    pos: self.pos
            id: title
            text: ''
            markup: True
            size_hint_y: None
            height: self.texture_size[1]+10
        Button:
            id: enabled_button
            text: 'X'
            on_press: root.on_enabled_pressed(self)
            background_normal: ''
            background_down: ''
            size_hint: (None, None)
            size: (40, title.height)
            pos_hint: {'top': 1, 'right': 1}
    BoxLayout:
        id: controls_holder
        orientation: 'vertical'
        BoxLayout:
            id: controls
            size_hint_y: None
            height: 0
            orientation: 'vertical'
       
''')


# Note to self: don't try this again...tuples are immutable.
# SliderInfo = namedtuple('SliderInfo', 'min_val max_val step_val init_val')
class SliderInfo:
    def __init__(self, min_val, max_val, step_val, init_val):
        self.min_val = min_val
        self.max_val = max_val
        self.step_val = step_val
        self.init_val = init_val


class BaseFilter(BoxLayout):
    is_enabled = BooleanProperty(defaultvalue=False)
    is_selected = BooleanProperty(defaultvalue=False)
    is_collapsed = BooleanProperty(defaultvalue=True)

    error = StringProperty()

    # This is set in the subclass implementation
    filter_preview = Image(allow_stretch=True, keep_ratio=True)

    def initiate_drag(self):
        # during a drag, we remove the widget from the original location
        self.parent.remove_widget(self)

    def __init__(self):
        super().__init__()
        self.controls = BoxLayout(orientation='vertical', size_hint_y=None, height=0)

    def on_enabled_pressed(self, instance):
        self.is_enabled = not self.is_enabled

    def on_collapsed_pressed(self, instance):
        self.is_collapsed = not self.is_collapsed
        self.update_ui()

    def on_complete(self, animation, widget):
        widget.controls_holder.add_widget(widget.controls)

    def on_is_collapsed(self, instance, value):
        if self.controls not in self.controls_holder.children:
            anim = Animation(height=self.title_layout.height + self.controls.height, duration=0.05)
            anim.bind(on_complete=self.on_complete)
            anim.start(self)
        else:
            self.controls_holder.remove_widget(self.controls)
            Animation(height=self.title_layout.height, duration=0.05).start(self)

    def update_ui(self, *_):
        if self.error:
            self.enabled_button.background_color = (0.6, 0, 0, 1)
            self.title.text = f'[color=#ff0000]{self.display_name}[/color]'
        else:
            self.enabled_button.background_color = (0, 0.6, 0, 1) if self.is_enabled else (0, 0.4, 0, 1)
            color_str = 'ffffff' if self.is_enabled else '888888'
            self.title.text = f'[color=#{color_str}]{self.display_name}[/color]'

        if len(self.controls.children) == 0:
            self.collapsed_button.text = ''
            self.height = self.title_layout.height
        elif self.is_collapsed:
            self.collapsed_button.text = 'V'
            self.height = self.title_layout.height
        else:
            self.collapsed_button.text = '^'
            self.height = self.title_layout.height + self.controls.height

    def on_error(self, obj, value):
        self.update_ui()

    def on_is_enabled(self, obj, value):
        self.update_ui()
        if self.value_changed is not None:
            self.value_changed()

    def add_filter_widget(self, widget):
        self.controls.add_widget(widget)
        self.controls.height += widget.height


class BaseSliderFilter(BaseFilter):
    def __init__(self, init_values):
        super().__init__()
        self.widget_list = []
        box_layout = BoxLayout(orientation='vertical')
        box_layout.size_hint_y = None
        box_layout.height = 0
        all_params = self.filter_params

        # set the initial values as specified by the pipeline
        for key, init_value in init_values.items():
            all_params[key].init_val = init_value

        # create all the widgets
        for name, params in all_params.items():
            # Add slider to widgets
            slider_widget = Slider(min=params.min_val, max=params.max_val, value=params.init_val, step=params.step_val)
            slider_widget.label_name = name
            slider_widget.bind(value=self._on_value_changed)
            slider_widget.size_hint_y = None
            slider_widget.height = '32sp'
            self.widget_list.append(slider_widget)

            # create label and value with widgets underneath
            widget_box_layout = BoxLayout(orientation='vertical')
            widget_box_layout.size_hint_y = None
            widget_box_layout.height = 0
            widget_label = Label(size_hint=(1, None))
            widget_label.label_name = name
            widget_label.display_callback = self.get_display_callback(name)
            widget_label.text = self.get_widget_display_text(widget_label, str(int(params.init_val)))
            widget_label.texture_update()
            widget_label.height = widget_label.texture_size[1]
            slider_widget.label_widget = widget_label

            # put it all together
            widget_box_layout.add_widget(widget_label)
            widget_box_layout.add_widget(slider_widget)
            widget_box_layout.padding = (5, -5)

            box_layout.add_widget(widget_box_layout)

            widget_box_layout.height = widget_label.height + slider_widget.height + 5
            box_layout.height += widget_box_layout.height

        self.size_hint_y = None
        self.height = box_layout.height
        self.add_filter_widget(box_layout)

    def on_touch_down(self, touch):
        if self.title.collide_point(*touch.pos) and self.is_collapsed:
            self.is_collapsed = False
        return super().on_touch_down(touch)

    # the default is to show the value being used
    def get_display_callback(self, _):
        return lambda x: str(x)

    def get_widget_display_text(self, instance, value):
        return instance.label_name + ': ' + instance.display_callback(int(value))

    def _on_value_changed(self, instance, value):
        instance.label_widget.text = self.get_widget_display_text(instance.label_widget, value)
        self.value_changed()
