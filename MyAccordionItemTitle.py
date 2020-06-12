from kivy.lang import Builder
from kivy.uix.label import Label

Builder.load_string('''
[MyAccordionItemTitle@Label]:
    markup: True
    text: ctx.title
    normal_background: ctx.item.background_normal if ctx.item.collapse else ctx.item.background_selected
    disabled_background: ctx.item.background_disabled_normal if ctx.item.collapse else ctx.item.background_disabled_selected
    canvas.before:
        BorderImage:
            source: self.disabled_background if self.disabled else self.normal_background
            pos: self.pos
            size: self.size
        PushMatrix
        Translate:
            xy: self.center_x, self.center_y
        Rotate:
            angle: 90 if ctx.item.orientation == 'horizontal' else 0
            axis: 0, 0, 1
        Translate:
            xy: -self.center_x, -self.center_y
    canvas.after:
        PopMatrix
''')


class MyAccordionItemTitle(Label):
    pass
