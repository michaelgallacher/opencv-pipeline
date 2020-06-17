from kivy.lang import Builder
from kivy.properties import BooleanProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.slider import Slider

from DragDrop import DraggableBoxLayoutBehavior, DraggableController, DraggableObjectBehavior
from MyAccordionItemTitle import MyAccordionItemTitle

hold_ref_so_auto_cleanup_doesnt_remove_the_import = MyAccordionItemTitle.__class__.__name__

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
    controls: controls
    controls_holder: controls_holder
    orientation: 'vertical'
    size_hint_y: None
    height: controls.height + title_layout.height
               
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
            size_hint: (None, None)
            size: (40, title.height)
            pos_hint: {'top': 1, 'x': 1}
        Label:
            canvas.before:
                Color: 
                    rgba: 0.1,0.2,0.5,1
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


class BaseFilter(BoxLayout):
    is_enabled = BooleanProperty(defaultvalue=False)
    error = StringProperty()

    collapsed = BooleanProperty(defaultvalue=False)

    # This is set in the subclass implementation
    preview = Image(allow_stretch=True, keep_ratio=True)

    last_height = 0

    def initiate_drag(self):
        # during a drag, we remove the widget from the original location
        self.parent.remove_widget(self)

    def __init__(self):
        super().__init__()

    def on_touch_down(self, touch):
        if self.title.collide_point(*touch.pos):
            return
        return super().on_touch_down(touch)

    def on_enabled_pressed(self, instance):
        if self.error:
            return
        self.is_enabled = not self.is_enabled

    def on_collapsed_pressed(self, instance):
        self.collapsed = not self.collapsed
        self.update_ui()

    def on_collapsed(self, instance, value):
        if len(self.controls_holder.children) == 0:
            self.controls_holder.add_widget(self.controls)
            self.controls_holder.height = self.controls.height
            self.height += self.controls.height
        else:
            self.controls_holder.remove_widget(self.controls)
            self.controls_holder.height = 0
            self.height -= self.controls.height

    def update_ui(self, *_):
        if self.error:
            self.enabled_button.background_color = (1, 0, 0, 1)
            self.title.text = f'[color=#ff0000]{self.display_name}[/color]'
            self.controls.disabled = True
        else:
            self.controls.disabled = not self.is_enabled
            self.enabled_button.background_color = (0, 1, 0, 1) if self.is_enabled else (0, 0.5, 0, 1)
            color_str = 'ffffff' if self.is_enabled else '888888'
            self.title.text = f'[color=#{color_str}]{self.display_name}[/color]'

        if len(self.controls.children) == 0:
            self.collapsed_button.background_color = (0.1, 0.2, 0.5, 1)
            self.collapsed_button.text = ''
        elif self.collapsed:
            self.collapsed_button.background_color = (0.1, 0.2, 0.5, 1)
            self.collapsed_button.text = 'V'
        else:
            self.collapsed_button.background_color = (0.1, 0.2, 0.5, 1)
            self.collapsed_button.text = '^'

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
            all_params[key]['init_val'] = init_value

        # create all the widgets
        for name, params in all_params.items():
            # Add slider to widgets
            slider_widget = Slider(min=params['min_val'], max=params['max_val'], value=params['init_val'], step=params['step_val'])
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
            widget_label.text = self.get_widget_display_text(widget_label, str(int(params['init_val'])))
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

    # the default is to show the value being used
    def get_display_callback(self, _):
        return lambda x: str(x)

    def get_widget_display_text(self, instance, value):
        return instance.label_name + ': ' + instance.display_callback(int(value))

    def _on_value_changed(self, instance, value):
        instance.label_widget.text = self.get_widget_display_text(instance.label_widget, value)
        self.value_changed()
