import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html,callback,dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from scipy.special import expit 
from sklearn.tree import plot_tree
import os
import warnings
import random as rd

from sklearn import tree
import graphviz
from IPython.display import Image
import pydotplus
import base64

external_stylesheets = [dbc.themes.BOOTSTRAP]  
warnings.filterwarnings('ignore')
# bootstrap theme
# https://bootswatch.com/lux/
# external_stylesheets = [dbc.themes.SANDSTONE]


app = Dash(__name__, suppress_callback_exceptions=True,external_stylesheets=external_stylesheets)
server = app.server

Titulo = dbc.Row(
    html.H1('Prediction of malignant and benign tumors in breast cancer'.upper()),class_name='margintTitle' 
)

navbar2 = dbc.Row(
            [
                dbc.Col(html.Img(src="assets/images/logo_espol.png", height="41px",className="col1_1")),
                dbc.Col(html.H5(""),style={"color":"white","font-size":"10vw"},className="col1_1"),
                dbc.Col(html.H5("ANGELO ZURITA"),style={"color":"white","font-size":"10vw"},className="col1_1")
            ],
            align='center',
            justify='space-between',    
            style={'justifyContent':'space-around'},
            className="Header g-0"
) 

links = html.Div([
    html.A('STATISTICS', href='#seccion1',className='link'),  
    html.A('FEATURES', href='#features',className='link'),
    html.A('MODEL', href='#modelo',className='link')
], className='row_links')


gender_info = {'FEMALE':['1 in 8','assets/images/female.png'],'MALE':['1 in 1000','assets/images/male.png']}
def cardInfo(gender):
    card = html.Div(className='card', children=[
        html.Img(src=gender_info[gender][1], alt='gender',className='image_card'),
        html.Div(className='card__content', children=[
            html.P(gender, className='card__title'),
            html.P(gender_info[gender][0], className='card__description')
        ])
    ])
    return card

div_know = html.Div([
    html.H1('Did You Know?'.upper(),className='title-fuente'),
    html.Div(
        [    
            html.H3('The average risk of being diagnosed with breast cancer at some point in their lives is',className='text'),
        ],className='subtitle-div-know'),
    html.Div([
                html.Div([cardInfo('FEMALE')]),
                html.Div([cardInfo('MALE')]),
            ],className='card_row'),
],className='div_know')

buttons = html.Div([
    html.Div([
        html.P('INCIDENCE'),
        html.Div([
            dbc.Button('BothSex',id='inci-bothsex',className='buttons'),
            dbc.Button('Female',id='inci-female',className='buttons')
        ],className='spacing-buttons')
    ],className='col-iguales'),
    html.Div([
        html.P('MORTALITY'),
        html.Div([
            dbc.Button('BothSex',id='mort-bothsex',className='buttons'),
            dbc.Button('Female',id='mort-female',className='buttons')
        ],className='spacing-buttons')
    ],className='col-iguales')
],className='container_botones')

buttons_ecuador = html.Div([
    html.Div([
        html.P('INCIDENCE'),
        html.Div([
            dbc.Button('BothSex',id='inci-bothsex_ecuador',className='buttons'),
            dbc.Button('Female',id='inci-female_ecuador',className='buttons')
        ],className='spacing-buttons')
    ],className='col-iguales'),
    html.Div([
        html.P('MORTALITY'),
        html.Div([
            dbc.Button('BothSex',id='mort-bothsex_ecuador',className='buttons'),
            dbc.Button('Female',id='mort-female_ecuador',className='buttons')
        ],className='spacing-buttons')
    ],className='col-iguales')
],className='container_botones')


columns  = ['Label','Total']
incidense_both2022 = pd.read_csv('assets/Data/Data_BreastCancer/incidences2022_bothSex.csv')[columns]
incidense_female2022 = pd.read_csv('assets/Data/Data_BreastCancer/incidense2022_females.csv')[columns]
mort_both2022 = pd.read_csv('assets/Data/Data_BreastCancer/mort_2022_bothSex.csv')[columns]
mort_female2022 = pd.read_csv('assets/Data/Data_BreastCancer/mort_2022_females.csv')[columns]

ecuador_incidense_both2022 = pd.read_csv('assets/Data/Data_BreastCancer/incidences_ecuador_bothsex.csv')[columns]
ecuador_incidense_female2022 = pd.read_csv('assets/Data/Data_BreastCancer/incidences _ecuador_females.csv')[columns]
ecuador_mort_both2022 = pd.read_csv('assets/Data/Data_BreastCancer/mort_ecuador_bothsex.csv')[columns]
ecuador_mort_female2022 = pd.read_csv('assets/Data/Data_BreastCancer/mort_ecuador_female.csv')[columns]


def initial_figure():
    df = incidense_both2022
    y_title = 'INCIDENSE'
    title = 'INCIDENSE - BOTH SEX'
    top10 = df.sort_values('Total', ascending=False).head(10)
    colors = px.colors.qualitative.Pastel
    color_map = {label: colors[0] for label in top10['Label'] if label != 'Breast'}
    color_map['Breast'] = 'rgb(78, 108, 138)'
    fig = px.bar(top10, x='Label', y='Total', text_auto=True, labels={'Total': y_title},
                color='Label', title=title, color_discrete_map=color_map)
    fig.update_traces(showlegend=False)
    return fig


def initial_figure_ecuador():
    df = ecuador_incidense_both2022
    y_title = 'INCIDENSE'
    title = 'INCIDENSE - BOTH SEX - ECUADOR'
    top10 = df.sort_values('Total', ascending=False).head(10)
    colors = px.colors.qualitative.Pastel
    color_map = {label: colors[0] for label in top10['Label'] if label != 'Breast'}
    color_map['Breast'] = 'rgb(78, 108, 138)'
    fig = px.bar(top10, x='Label', y='Total', text_auto=True, labels={'Total': y_title},
                color='Label', title=title, color_discrete_map=color_map)
    fig.update_traces(showlegend=False)
    return fig


def CardPosition(type,numero,country='WORLD'):
    msg = ''
    if type=='INCIDENSE':
        msg = 'With the most incidents'
    else : 
        msg = 'Deadliest'
    return html.Div([
        html.P(country,className='title'),
        html.P(numero,className='Number'),
        html.P(msg,className='msg')
    ],className='columnscenter margin-left')

div_stadistics = html.Div([
    html.Div(className='circle'),
    html.Div(html.Img(src='assets/images/ecuador.png',className='imagen_ecuador'),className='bandera_ecuador'),
    html.H3('STATISTICS IN THE WORLD 2022'),
    buttons,
    html.Div([dcc.Graph(id='grafico', figure=initial_figure()),
            CardPosition('INCIDENSE',2)]
    ,id='contenidoGrafico',className='contenidoGrafico'),
    html.Div([
        html.H3('MOST COMMON CANCER IN DIFFERENT COUNTRIES'),
        html.Img(src='assets/images/Mapa.svg')
    ],className='col-mapa'),
    html.H3('STATISTICS IN ECUADOR 2022'),
    buttons_ecuador,
    html.Div([dcc.Graph(id='grafico2', figure=initial_figure_ecuador()),
            CardPosition('INCIDENSE',1,'ECUADOR')]
    ,id='contenidoGrafico_ecuador',className='contenidoGrafico'),
],className='center-divstadistics container')



@app.callback(
    Output('contenidoGrafico', 'children'),
    [
        Input('inci-bothsex', 'n_clicks'),
        Input('inci-female', 'n_clicks'),
        Input('mort-bothsex', 'n_clicks'),
        Input('mort-female', 'n_clicks'),
    ],
    prevent_initial_call=True
)
def graphInicial(incibothsex, incifemale, mortbothsex, mortfemale):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    df = ''
    y_title = ''
    title = ''
    number = 2
    if button_id == 'inci-female':
        df = incidense_female2022
        y_title = 'INCIDENSE'
        title = 'INCIDENSE - FEMALE'
        number = 1
    elif button_id == 'mort-bothsex':
        df = mort_both2022
        y_title = 'MORTALITY'
        title = 'MORTALITY - BOTH SEX'
        number = 4
    elif button_id == 'mort-female':
        df = mort_female2022
        y_title = 'MORTALITY'
        title = 'MORTALITY - FEMALE'
        number = 1
    else:
        df = incidense_both2022
        y_title = 'INCIDENSE'
        title = 'INCIDENSE - BOTH SEX'
        number = 2
    
    top10 = df.sort_values('Total', ascending=False).head(10)
    colors = px.colors.qualitative.Pastel
    color_map = {label: colors[0] for label in top10['Label'] if label != 'Breast'}
    color_map['Breast'] = 'rgb(78, 108, 138)' 
    fig = px.bar(top10, x='Label', y='Total', text_auto=True, labels={'Total': y_title},
                color='Label', title=title, color_discrete_map=color_map)
    fig.update_traces(showlegend=False)
    return [dcc.Graph(figure=fig),CardPosition(y_title,number)]


@app.callback(
    Output('contenidoGrafico_ecuador', 'children'),
    [
        Input('inci-bothsex_ecuador', 'n_clicks'),
        Input('inci-female_ecuador', 'n_clicks'),
        Input('mort-bothsex_ecuador', 'n_clicks'),
        Input('mort-female_ecuador', 'n_clicks'),
    ],
    prevent_initial_call=True
)
def graphInicialEcuador(incibothsex, incifemale, mortbothsex, mortfemale):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    df = ''
    y_title = ''
    title = ''
    number = 1
    if button_id == 'inci-female_ecuador':
        df = ecuador_incidense_female2022
        y_title = 'INCIDENSE'
        title = 'INCIDENSE - FEMALE - ECUADOR'
        number = 1
    elif button_id == 'mort-bothsex_ecuador':
        df = ecuador_mort_both2022
        y_title = 'MORTALITY'
        title = 'MORTALITY - BOTH SEX - ECUADOR'
        number = 4
    elif button_id == 'mort-female_ecuador':
        df = ecuador_mort_female2022
        y_title = 'MORTALITY'
        title = 'MORTALITY - FEMALE - ECUADOR'
        number = 1
    else:
        df = ecuador_incidense_both2022
        y_title = 'INCIDENSE'
        title = 'INCIDENSE - BOTH SEX - ECUADOR'
        number = 1
    
    top10 = df.sort_values('Total', ascending=False).head(10)
    colors = px.colors.qualitative.Pastel
    color_map = {label: colors[0] for label in top10['Label'] if label != 'Breast'}
    color_map['Breast'] = 'rgb(78, 108, 138)' 
    figEcua = px.bar(top10, x='Label', y='Total', text_auto=True, labels={'Total': y_title},
                color='Label', title=title, color_discrete_map=color_map)
    figEcua.update_traces(showlegend=False)
    return [dcc.Graph(figure=figEcua),CardPosition(y_title,number,'ECUADOR')]


image = dbc.Row(
    dbc.Col(
        [
            html.Img(src='assets/images/Tumores.png',className='imagen_tumor border'),
        ],className='text-fuente'
),className='content-center align-content mt-4')

info_radius = {
    'title': 'Radius',
    'text': [
        'The average of distances from the center to points on the perimeter.',
        'This variable measures the average size of the tumor.',
        'Malignant tumors often have a larger size compared to benign ones.'
    ]
}

info_texture = {
    'title': 'Texture',
    'text': [
        'Standard deviation of grayscale values.',
        'Texture measures the variation in pixel intensity, which can indicate irregularity in the cell distribution within the tumor.',
        'Malignant tumors usually show more irregularity in texture.'
    ]
}

info_perimeter = {
    'title': 'Perimeter',
    'text': [
        'The length of the edge of the tumor.',
        'Tumors with irregular and longer edges are usually indicative of malignancy.'
    ]
}

info_area = {
    'title': 'Area',
    'text': [
        'The area of the tumor is calculated from its contour.',
        'A larger area can be a sign of malignancy, although by itself it is not a definitive indicator.'
    ]
}

info_smoothness = {
    'title': 'Smoothness',
    'text': [
        'Local variation in the lengths of the radii.',
        'This measure assesses how smooth or irregular the edges of the tumor are.',
        'More irregular edges can be a sign of a malignant tumor.'
    ]
}

info_compactness = {
    'title': 'Compactness',
    'text': [
        'Perimeter squared divided by the area minus one.',
        'Compactness assesses how densely the cells are packed in the tumor.',
        'A higher value suggests a denser tumor, which is common in malignant tumors.'
    ]
}

info_concavity = {
    'title': 'Concavity',
    'text': [
        'Severity of the concave parts of the contour.',
        'Concavity measures the indentations in the tumor contour.',
        'Malignant tumors often have more pronounced concavities due to their uneven growth.'
    ]
}

info_concave_points = {
    'title': 'Concave Points',
    'text': [
        'Number of concave portions of the contour. Similar to concavity,',
        'this indicator counts the number of indentations on the edge of the tumor.',
        'A higher number of concave points is associated with malignancy.'
    ]
}

info_symmetry = {
    'title': 'Symmetry',
    'text': [
        'How symmetrical the shape of the tumor is.',
        'Malignant tumors are often asymmetrical.'
    ]
}

info_fractal_dimension = {
    'title': 'Fractal Dimension',
    'text': [
        'Approximation of the coastline minus one.',
        'This measure relates to the complexity of the tumor contour, comparing it to a coastline.',
        'A more fractal contour may indicate a malignant tumor.'
    ]
}

# Lista que contiene todos los diccionarios de información
infos = [
    info_radius, info_texture, info_perimeter, info_area, info_smoothness,
    info_compactness, info_concavity, info_concave_points, info_symmetry,
    info_fractal_dimension
]
def card(info):
    return dbc.AccordionItem(
        title=info['title'],
        children=html.Ul([html.Li(text) for text in info['text']])  # Crea un elemento Li para cada texto en la lista
    )

accordion = dbc.Accordion([card(info) for info in infos])  # Esto generará una lista de AccordionItems

div_interpretación = html.Div(
    [
        html.Div(
            [
                html.Div(className='circle'),
                html.H3('Understanding Tumor Characteristics'.upper(),className='SanSerif text-margin'),
                html.Div(accordion, className='accordion')
            ],
            className='container contentx-center',
            id='features'
        ),
    ]
)

## TABLE    
df = pd.read_csv('assets/Data/data.csv')
df_means = df[['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 
            'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']]
lista = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean','concavity_mean','concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
df_groupby = df_means.groupby('diagnosis')[lista].mean().T
index = df_groupby.index
B = df_groupby.reset_index()[['B','index']].reset_index(drop=True)
B.columns=['B','Characteristic']
M = df_groupby.reset_index()[['index','M']].reset_index(drop=True)
M.columns=['Characteristic','M']

df_unido = pd.merge(B, M, on='Characteristic', how='inner')
df_unido['M'] =df_unido['M'].round(3)
df_unido['B'] =df_unido['B'].round(3)



infoTable = """
Benign (B): Represents the average measurements for benign tumors.;
Malignant (M): Reflects the average measurements for malignant tumors.;
The table clearly shows that across all characteristics, the values associated with malignant tumors are systematically higher in comparison to benign ones.
"""
benigno_maligno = html.Div([
    html.H3('BENIGN VS MALIGNANT TUMORS'),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df_unido.columns],
        data=df_unido.to_dict('records'),
        style_table={'height': '350px', 'overflowY': 'auto'},
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold'
        },
        style_cell={
            'textAlign': 'center',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'minWidth': '300px', 'width': '180px', 'maxWidth': '180px',
        },
        style_data_conditional=[
                # Resaltar cuando B es mayor que M
                {
                    'if': {
                        'filter_query': '{B} >= {M}',
                        'column_id': 'B'
                    },
                    'backgroundColor': 'rgb(26, 77, 128)',
                    'color': 'white'
                },
                # Resaltar cuando M es mayor que B
                {
                    'if': {
                        'filter_query': '{M} >= {B}',
                        'column_id': 'M'
                    },
                    'backgroundColor': 'rgb(128, 26, 26)',
                    'color': 'white'
                },
                # Resaltar ambas celdas cuando M y B son iguales
                # Ajustamos las consultas para considerar el margen de error
            ]
    )], className='div_table'
)

descrip_tabla = html.Div([
    html.Div(
        [
            html.Div(className='circle2'),
            html.Div([
                html.Ul([html.Li(text) for text in infoTable.split(';')])
            ], className='div_info')
        ],
        className='container2 '
        )
],className='margin-b')

row_table = html.Div([
    benigno_maligno, descrip_tabla
],className='Row_table')


def scatter_df(col,medida='MM'):
    Name_label = f" AVERAGE {col.split('_')[0].upper()} [{medida}]".upper()
    Title = f"AVERAGE TUMOR {col.split('_')[0].upper()}"
    fig = px.scatter(
                                df,x=col,color='diagnosis',
                                labels={'diagnosis':'TIPO',col:Name_label,'index':''},
                                title=Title,width=700,color_discrete_map={'B':'rgb(26, 77, 128)','M':'rgb(128, 26, 26)'})
    fig.for_each_trace(
        lambda trace: trace.update(name="MALIGNANT" if trace.name == "M" else "BENIGN")
    )
    return fig

div_figure = html.Div([
        html.H3('Data Distribution'.upper()),
        html.Div([
            dcc.Graph(figure=scatter_df(lista[0]), style={'gridColumn': '1', 'gridRow': '1'}),
            dcc.Graph(figure=scatter_df(lista[1]), style={'gridColumn': '2', 'gridRow': '1'}),
            dcc.Graph(figure=scatter_df(lista[2]), style={'gridColumn': '1', 'gridRow': '2'}),
            dcc.Graph(figure=scatter_df(lista[3],medida='MM2'), style={'gridColumn': '2', 'gridRow': '2'}),
            dcc.Graph(figure=scatter_df(lista[4]), style={'gridColumn': '1', 'gridRow': '3'}),
            dcc.Graph(figure=scatter_df(lista[5]), style={'gridColumn': '2', 'gridRow': '3'}),
            dcc.Graph(figure=scatter_df(lista[6]), style={'gridColumn': '1', 'gridRow': '4'}),
            dcc.Graph(figure=scatter_df(lista[7]), style={'gridColumn': '2', 'gridRow': '4'}),
            dcc.Graph(figure=scatter_df(lista[8]), style={'gridColumn': '1', 'gridRow': '5'}),
            dcc.Graph(figure=scatter_df(lista[9]), style={'gridColumn': '2', 'gridRow': '5'}),
        ],style={
        'display': 'grid',
        'gridTemplateColumns': '1fr 1fr',  # Define dos columnas
        'gridGap': '10px',  # Espacio entre las celdas de la grilla
    })
    ],className='center_fig'
)

def input_Characteristic(title, id=None):
    if id is None:
        id = title  # If no ID is provided, use title as the ID
    return html.Div([
        html.Label(title, htmlFor=id), 
        dcc.Input(
            id=id,  # Set a unique ID for each input for callback purposes
            type='number',
            step=0.00001,  
            placeholder='Enter value',
        ),
    ], className='input_div') 



nuevo =html.Div(
                    [
                        html.Div(
                            [
                                html.Div(input_Characteristic("RADIUS",'radius-input')),
                                html.Div(input_Characteristic("TEXTURE",'texture-input')),
                                html.Div(input_Characteristic("PERIMETER",'perimeter-input')),
                                html.Div(input_Characteristic("AREA",'area-input')),
                                html.Div(input_Characteristic("SMOOTHESS",'smoothess-input')),
                            ],className='',style={'gridColumn': '1', 'gridRow': '1'}),
                        html.Div([
                                html.Div(input_Characteristic("COMPACTNESS",'compactness-input')),
                                html.Div(input_Characteristic("CONCAVITY",'concavity-input')),
                                html.Div(input_Characteristic("CONCAVE POINTS","concavePoint-input")),
                                html.Div(input_Characteristic("SYMMTRY",'symetry-input')),
                                html.Div(input_Characteristic("FRACTAL DIMENSION","fractalDimension-input")),
                            ],className='', style={'gridColumn': '2', 'gridRow': '1'})
                    ],
                    style={
                            'display': 'grid',
                            'gridTemplateColumns': '1fr 1fr', 
                            'gridGap': '50px',  
                            'gridColumn': '1', 'gridRow': '1'
                        }
                )

model_path = 'assets/models/ModelMean.sav'
with open(model_path, 'rb') as model_file:
    gbc = pickle.load(model_file)

featureImportance_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                'smoothness_mean', 'compactness_mean', 'concavity_mean', 
                'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

importances = gbc.feature_importances_

df_feature_importances = pd.DataFrame({'Feature': featureImportance_names, 'Importance': importances})
df_feature_importances = df_feature_importances.sort_values('Importance', ascending=False)
df_feature_importances = df_feature_importances.round(3)
feature_importances = px.bar(df_feature_importances, 
                x='Importance', 
                y='Feature', 
                title='Feature Importances',
                text_auto = True,
                orientation='h',
                color='Feature',
                color_discrete_sequence = px.colors.qualitative.Pastel,
                width=1000
            )
feature_importances.update_layout(yaxis={'categoryorder':'total ascending'})

div_modelo = html.Div(
    [
        html.Div(
            [
                html.Div(className='circle'),
                html.Div([
                        html.Div(
                        [
                            html.Div(className='circle_nota', id='circle_nota'),
                            html.Div(html.P('Values in millimeters only the area in square millimeters', className='center_text'),
                                    className='circle_notaTexto', id='circle_notaTexto'),
                        ],
                        className='circle-hover'
                    ),
                ],className='circle_div'),
                html.H3('MODEL',className='SanSerif text-margin'),
                html.Div([
                    dcc.Graph(figure=feature_importances)
                ]),
                html.Div([
                    nuevo,
                    html.Div(id='contenido_nuevo',className='div_probabilidad',style={'margin-top':'25px'})
                ],style={
                            'display': 'grid',
                            'gridTemplateColumns': '1fr 1fr', 
                            'gridGap': '50px',  
                            'gridColumn': '1', 'gridRow': '1',
                            'margin-top':'50px'
                        }),
                html.Div([
                    dbc.Button('PREDICT',id='button-predict',className='button-pred'),
                    html.Div([
                        html.Label('Predetermined values'.upper()),
                        html.Div([
                                    dcc.Dropdown(
                                        options=[
                                            {'label': 'MALIGNANT', 'value': 'MALIGNANT'},
                                            {'label': 'BENIGN', 'value': 'BENIGN'},
                                            {'label': 'USER', 'value': 'USER'},
                                            {'label': 'RANDOM', 'value': 'RANDOM'}
                                        ],
                                        value='USER',
                                        id='dropdown-pred',
                                        className='dropdown'  # Esta es la clase que definiste en tu archivo CSS.
                                    )
                        ])
                    ],className='predeterminedDiv')
                ],className='div_button'),
                html.Div(id='div_howWork',className='div-work')
            ],
            className='container contentx-center'
        ),
    ],className='margin-10 mb-20',
    id='modelo'
)

model_scalerpath = 'assets/models/scaler_ModelMean.sav'
with open(model_scalerpath, 'rb') as model_file:
    scaler = pickle.load(model_file)

@callback(
    [   
        Output('contenido_nuevo', 'children',allow_duplicate=True),
        Output('div_howWork', 'children',allow_duplicate=True),
    ],
    [
        State('radius-input', 'value'),
        State('texture-input', 'value'),
        State('perimeter-input', 'value'),
        State('area-input', 'value'),
        State('smoothess-input', 'value'),
        State('compactness-input', 'value'),
        State('concavity-input', 'value'),
        State('concavePoint-input', 'value'),
        State('symetry-input', 'value'),
        State('fractalDimension-input', 'value'),
    ],
    Input('button-predict', 'n_clicks'),
    prevent_initial_call=True
)
def update_output(radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points, symmetry, fractal_dimension, n_clicks):
    if n_clicks != None and  n_clicks > 0:
        lista_elemento = [radius or 0.0, texture or 0.0, perimeter or 0.0, area or 0.0, 
                        smoothness or 0.0, compactness or 0.0, concavity or 0.0, 
                        concave_points or 0.0, symmetry or 0.0, fractal_dimension or 0.0]
        new_data = pd.DataFrame([lista_elemento], columns=featureImportance_names)
        new_data.loc[0] = lista_elemento
        X_train = scaler.transform(new_data)
        y_proba = gbc.predict_proba(X_train)
        prob_maligna = np.round(y_proba[0, 1],3)
        prob_benigna = np.round(y_proba[0, 0],3)
        data = pd.DataFrame({
            'Categoria': ['BENIGN', 'MALIGNANT'],
            'Probabilidad': [prob_benigna, prob_maligna]
        })
        import plotly.express as px
        # Creando el gráfico de pastel
        fig_probabilidad = px.pie(data, values='Probabilidad', names='Categoria', title='PREDICTION',color='Categoria',
                                color_discrete_map={'BENIGN':'rgb(26, 77, 128)','MALIGNANT':'rgb(128, 26, 26)'})
        
        html1 =  html.Div([
                    dcc.Graph(figure=fig_probabilidad)
                ],style={
                            'gridColumn': '2', 'gridRow': '1'
                        }),
        staged_decision_scores = [score for score in gbc.staged_decision_function(X_train)]
        staged_probabilities = [expit(score) for score in staged_decision_scores]
        df_probabilities = pd.DataFrame(columns=['BENIGN','MALIGNANT'])
        df_probabilities['MALIGNANT'] = [value[0][0] for value in staged_probabilities]
        df_probabilities['BENIGN'] = 1 - df_probabilities['MALIGNANT']
        df_probabilities.loc[180] = [prob_benigna,prob_maligna]
        df_probabilities = df_probabilities.round(3)
        df_probabilities_iter = df_probabilities.reset_index()
        df_probabilities_iter.columns = ['ITERATION','BENIGN','MALIGNANT']
        df_probabilities_iter = df_probabilities_iter.melt(id_vars='ITERATION', value_vars=['BENIGN', 'MALIGNANT'], var_name='TYPE', value_name='PROBABILITY')
        fig_probabilty = px.line(df_probabilities_iter, x='ITERATION', y='PROBABILITY',color='TYPE', markers=True,
                    color_discrete_map={'BENIGN':'rgb(26, 77, 128)','MALIGNANT':'rgb(128, 26, 26)'},width=1000,
                    hover_name='TYPE', hover_data={'TYPE':False,'ITERATION': True})
        # Mejorar la presentación del gráfico
        fig_probabilty.update_layout(
            title='Probability Evolution for BENIGN and MALIGNANT'.upper(),
            xaxis_title='Iteration'.upper(),
            yaxis_title='Probability'.upper(),
            legend_title='TYPE'
        )
        html2 = [
                    html.Div([
                        html.H2("How does this work?".upper())
                    ],style={'margin-top':'20px'}),
                    html.Div([
                        html.Div(dcc.Graph(figure=fig_probabilty)),
                    ],className='div-work'),
                    html.Div(id='tree-iterator'),
                ]
        return html1,html2
    return html.Div(),html.Div()

@callback(
    [
        Output('radius-input', 'value',allow_duplicate=True),
        Output('texture-input', 'value',allow_duplicate=True),
        Output('perimeter-input', 'value',allow_duplicate=True),
        Output('area-input', 'value',allow_duplicate=True),
        Output('smoothess-input', 'value',allow_duplicate=True),
        Output('compactness-input', 'value',allow_duplicate=True),
        Output('concavity-input', 'value',allow_duplicate=True),
        Output('concavePoint-input', 'value',allow_duplicate=True),
        Output('symetry-input', 'value',allow_duplicate=True),
        Output('fractalDimension-input', 'value',allow_duplicate=True),
    ],
    Input('button-predict', 'n_clicks'),
    [
        State('radius-input', 'value'),
        State('texture-input', 'value'),
        State('perimeter-input', 'value'),
        State('area-input', 'value'),
        State('smoothess-input', 'value'),
        State('compactness-input', 'value'),
        State('concavity-input', 'value'),
        State('concavePoint-input', 'value'),
        State('symetry-input', 'value'),
        State('fractalDimension-input', 'value'),
    ],
    prevent_initial_call=True
)
def update_input_values(n_clicks, radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points, symmetry, fractal_dimension):
    if n_clicks != None and n_clicks > 0:
        inputs = [radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points, symmetry, fractal_dimension]
        return [0.00 if v == None else v for v in inputs]
    return [radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points, symmetry, fractal_dimension]

df_random = pd.read_csv('assets/Data/RandomValuesOriginal.csv')
@callback(
    [
        Output('radius-input', 'value'),
        Output('texture-input', 'value'),
        Output('perimeter-input', 'value'),
        Output('area-input', 'value'),
        Output('smoothess-input', 'value'),
        Output('compactness-input', 'value'),
        Output('concavity-input', 'value'),
        Output('concavePoint-input', 'value'),
        Output('symetry-input', 'value'),
        Output('fractalDimension-input', 'value'),
        Output('contenido_nuevo', 'children'),
        Output('div_howWork', 'children'),
    ],
    Input('dropdown-pred', 'value'),
)
def update_input_values(value):
    if value == 'RANDOM' and value != None:
        id_random = rd.randint(0,len(df_random))
        lista_return = df_random.loc[id_random].to_list()[0:10]
        lista_return.append(html.Div())
        lista_return.append(html.Div())
        return lista_return
    elif value != 'USER' and value != None:
        inputs = []
        if(value == 'MALIGNANT'):
            inputs = df_unido['M'].to_list()
        else:
            inputs = df_unido['B'].to_list()
        inputs.append(html.Div())
        inputs.append(html.Div())
        return inputs
    lista_error = ([None]*10)
    lista_error.append(html.Div())
    lista_error.append(html.Div())
    return lista_error



fuentes = html.Div([
        html.H3('RESOURCES',className='title-fuente'),
        html.Div([
        html.Ol([
            html.Li("https://www.breastcancer.org/es/tipos/cancer-de-mama-en-hombres"),
            html.Li("https://gco.iarc.fr/today/online-analysis-map"),
            html.Li("https://www.kaggle.com/datasets/mragpavank/breast-cancer/data"),
            html.Li("https://www.verywellhealth.com/what-does-malignant-and-benign-mean-514240"),
            html.Li("https://gco.iarc.who.int/today/en/dataviz/pie?mode=cancer&group_populations=1&populations=900&sexes=2"),
        ],className='div_lista')
    ],className='fuentes_div')
],className='container contentx-center ')

app.layout = html.Div([
    navbar2,links,div_know,div_stadistics,Titulo,image,div_interpretación,row_table,div_figure,div_modelo,fuentes
],className='body all')


if __name__ == '__main__':
    app.run_server(debug=True)
