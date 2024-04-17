from datetime import datetime
import dash
import dash_bootstrap_components as dbc
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d
import pandas as pd
import math
external_stylesheets = [dbc.themes.BOOTSTRAP]  
from dash import Dash, dcc, html,callback
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
# bootstrap theme
# https://bootswatch.com/lux/
# external_stylesheets = [dbc.themes.SANDSTONE]


app = Dash(__name__, suppress_callback_exceptions=True,external_stylesheets=external_stylesheets) #external_stylesheets=external_stylesheets
server = app.server
# building the navigation bar
# https://github.com/facultyai/dash-bootstrap-components/blob/master/examples/advanced-component-usage/Navbars.py

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="assets/Images/logo_espol.png", height="460px"),className="col1_1"),
                        dbc.Col(html.H6("PROYECTO CALCULO VECTORIAL"),style={"color":"white","font-size":"20vw"},className="Titulo"),
                        dbc.Col(html.H6("GRUPO 6"),style={"color":"white","font-size":"10vw"},className="col1")
                    ],
                    align='center',
                    justify='center'

                ),
                className="row_header g-0"
            ),
        ]
    ),
    style={'justifyContent':'space-around'},
    color="dark",
    dark=True,
)
navbar2 = dbc.Row(
            [
                dbc.Col(html.Img(src="assets/images/logo_espol.png", height="41px",className="col1_1")),
                dbc.Col(html.H5("PREDICTION OF MALIGNANT OR BENIGN CANCER"),style={"color":"white","font-size":"10vw"},className="col1_1"),
                dbc.Col(html.H5("ANGELO ZURITA"),style={"color":"white","font-size":"10vw"},className="col1_1")
            ],
            align='center',
            justify='space-between',    
            style={'justifyContent':'space-around'},
            className="Header g-0"
) 

image = dbc.Row(
    dbc.Col(
        [
            html.Img(src='assets/images/Tumores.png',className='imagen_tumor border'),
            html.P("CENTRO INTERNACIONAL DE CÁNCER")
        ],className='text-fuente'
),className='content-center align-content mt-4')

info_radius = {
    'title': 'Radius',
    'text': 'La media de las distancias desde el centro hasta los puntos del perímetro. '
            'Esta variable mide el tamaño promedio del tumor. '
            'Los tumores malignos a menudo tienen un tamaño mayor comparado con los benignos.'
}

info_texture = {
    'title': 'Texture',
    'text': 'Desviación estándar de los valores en escala de grises. '
            'La textura mide la variación en la intensidad de los píxeles en la imagen, que puede indicar '
            'irregularidad en la distribución de las células dentro del tumor. '
            'Los tumores malignos suelen mostrar mayor irregularidad en la textura.'
}

info_perimeter = {
    'title': 'Perimeter',
    'text': 'La longitud del borde del tumor. Los tumores con bordes irregulares y más largos suelen ser '
            'indicativos de malignidad.'
}

info_area = {
    'title': 'Area',
    'text': 'El área del tumor se calcula a partir de su contorno. '
            'Un área mayor puede ser una señal de malignidad, aunque por sí sola no es un indicador definitivo.'
}

info_smoothness = {
    'title': 'Smoothness',
    'text': 'Variación local en las longitudes del radio. '
            'Esta medida evalúa qué tan suaves o irregulares son los bordes del tumor. '
            'Los bordes más irregulares pueden ser un signo de un tumor maligno.'
}

info_compactness = {
    'title': 'Compactness',
    'text': 'Perímetro al cuadrado dividido por el área menos uno. '
            'La compactidad evalúa qué tan densamente están empaquetadas las células en el tumor. '
            'Un valor más alto sugiere un tumor más denso, lo cual es común en los tumores malignos.'
}

info_concavity = {
    'title': 'Concavity',
    'text': 'Gravedad de las partes cóncavas del contorno. '
            'La concavidad mide las indentaciones en el contorno del tumor. '
            'Los tumores malignos a menudo presentan concavidades más pronunciadas debido a su crecimiento desigual.'
}

info_concave_points = {
    'title': 'Concave Points',
    'text': 'Número de partes cóncavas del contorno. Similar a la concavidad, este indicador cuenta el número de '
            'indentaciones en el borde del tumor. Un número mayor de puntos cóncavos está asociado con la malignidad.'
}

info_symmetry = {
    'title': 'Symmetry',
    'text': 'Cuán simétrica es la forma del tumor. Los tumores malignos a menudo son asimétricos.'
}

info_fractal_dimension = {
    'title': 'Fractal Dimension',
    'text': 'Aproximación de la línea costera" menos uno. '
            'Esta medida se relaciona con la complejidad del contorno del tumor, comparándola con una línea costera. '
            'Un contorno más fractal puede indicar un tumor maligno.'
}

# Lista que contiene todos los diccionarios de información
infos = [
    info_radius, info_texture, info_perimeter, info_area, info_smoothness,
    info_compactness, info_concavity, info_concave_points, info_symmetry,
    info_fractal_dimension
]

def card(info):
    return dbc.Col(
        html.Div(
            [
                html.H2(info['title'], className='card-title'),
                html.Div(info['text'], className='card-text')
            ],
            className='card',
        ),
    )

div_interpretación = html.Div(
    [
        html.Div(
            [
                html.Div(className='circle'),
                html.H5('INTERPRETACIÓN',className='SanSerif text-margin'),
                html.Div([
                    dbc.Row([
                        card(infos[0]),
                        card(infos[1]),
                        card(infos[2]),
                    ],className='row_info'),
                    dbc.Row([
                        card(infos[3]),
                        card(infos[4]),
                        card(infos[5]),
                    ],className='row_info'),
                    dbc.Row([
                        card(infos[6]),
                        card(infos[7]),
                        card(infos[8]),
                    ],className='row_info'),
                    dbc.Row([
                        card(infos[9]),
                    ],className='row_info2')
                ],className='spacing')
            ],
            className='container contentx-center'
        ),
    ]
)



app.layout = html.Div([
    navbar2,image,div_interpretación,
],className='body all')


if __name__ == '__main__':
    app.run_server(host='127.0.0.1',port=8020 ,debug=True)
