from kivy.clock import Clock
from kivy.properties import BooleanProperty, StringProperty
from kivy.uix.accordion import AccordionItem
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.slider import Slider

from MyAccordionItemTitle import MyAccordionItemTitle

hold_ref_so_auto_cleanup_doesnt_remove_the_import = MyAccordionItemTitle.__class__.__name__


class BaseFilter(AccordionItem):
    is_enabled = BooleanProperty(defaultvalue=False)
    error = StringProperty()

    # This is set in the subclass implementation
    preview = Image(allow_stretch=True, keep_ratio=True)

    def __init__(self):
        super().__init__()

        self.title_template = 'MyAccordionItemTitle'

        self.enabled_button = Button(text='X', size_hint=(1, 1))
        self.enabled_button.bind(on_press=self.on_enabled_pressed)
        self.enabled_button.background_normal = ''
        self.enabled_button.size_hint = (None, None)
        self.enabled_button.size = (40, 40)
        self.enabled_button.pos_hint = {'top': 1, 'right': 1}
        super(FloatLayout, self).add_widget(self.enabled_button)

        self.controls = BoxLayout(orientation='vertical')

        self.add_widget(self.controls)

        self.value_changed = None

        self.bind(is_enabled=self.on_enabled)
        self.bind(error=self.on_error)
        Clock.schedule_once(self.update_ui, 1)

    # The AccordionItem eats all touches because it doesn't
    # expect anything but the Label to be there.
    def on_touch_down(self, touch):
        if self.enabled_button.collide_point(*touch.pos):
            return self.enabled_button.on_touch_down(touch)
        return super(BaseFilter, self).on_touch_down(touch)

    def on_enabled_pressed(self, instance):
        if self.error:
            return
        self.is_enabled = not self.is_enabled

    def on_collapse(self, instance, value):
        super().on_collapse(instance, value)

    def update_ui(self, *_):
        if self.error:
            self.enabled_button.background_color = (1, 0, 0, 1)
            self.title = f'[color=#ff0000]{self.display_name}[/color]'
            self.controls.disabled = True
        else:
            self.controls.disabled = not self.is_enabled
            self.enabled_button.background_color = (0, 1, 0, 1) if self.is_enabled else (0, 0, 0, 0)
            self.title = f'[color=#ffffff]{self.display_name}[/color]'

    def on_error(self, obj, value):
        self.update_ui()

    def on_enabled(self, obj, value):
        self.update_ui()
        if self.value_changed is not None:
            self.value_changed()

    def add_filter_widget(self, widget):
        self.controls.add_widget(widget)


class BaseSliderFilter(BaseFilter):
    def __init__(self, init_values):
        super().__init__()
        self.widget_list = []
        box_layout = BoxLayout(orientation='vertical')
        box_layout.padding = (20, 20)
        self.add_filter_widget(box_layout)
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
            self.widget_list.append(slider_widget)

            # create label and value with widgets underneath
            widget_box_layout = BoxLayout(orientation='vertical')
            widget_label = Label(size_hint=(1, 0.2))
            widget_label.label_name = name
            widget_label.display_callback = self.get_display_callback(name)
            widget_label.text = self.get_widget_display_text(widget_label,  str(int(params['init_val'])))
            slider_widget.label_widget = widget_label

            # put it all together
            widget_box_layout.add_widget(widget_label)
            widget_box_layout.add_widget(slider_widget)
            box_layout.add_widget(widget_box_layout)

    # the default is to show the value being used
    def get_display_callback(self, _):
        return lambda x: str(x)

    def get_widget_display_text(self, instance, value):
        return instance.label_name + ': ' + instance.display_callback(int(value))

    def _on_value_changed(self, instance, value):
        instance.label_widget.text = self.get_widget_display_text(instance.label_widget, value)
        self.value_changed()
