import matplotlib.pyplot as plt
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

# Convert an .svg to .png using svglib and reportlab
def convert_svg_to_png(input_svg, output_png):
    drawing = svg2rlg(input_svg)
    renderPM.drawToFile(drawing, output_png, fmt='PNG')

# Usage
input_svg = '/home/amir/inverse_prob/PP.svg'
output_png = '/home/amir/inverse_prob/PP_1.png'

convert_svg_to_png(input_svg, output_png)