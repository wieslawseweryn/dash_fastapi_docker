import math
import os.path
import time

from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
import dash_tabulator
import numpy as np
from datetime import datetime as dt, timedelta
import plotly.graph_objects as go
import pandas as pd

from util import badge, get_default_color, trace_options, read_kolory, subtitle, \
    get_settings, session_or_die, list_to_options, is_logged_in, session2settings, \
    file_exists, logit, filter_data, get_trace_name, fields_to_options, \
    defaults2session, dict_to_options, get_template
from plotly.subplots import make_subplots
import constants as constants
from constants import top_opt_class, top_row_class, top_row_class_inner
from constants import mapbox_styles
import plotly.express as px

from config import session

start_server_date = str(dt.today())[:16]

kolory = read_kolory()

nomargins = 'ml-0 pl-0'


def imgButton(icon_name, _id, multi=1, color='gray'):
    add_attr = ' '
    if multi > 1:
        add_attr = ' fa-' + str(multi) + 'x'
    return html.I(className="m-2 fa " + icon_name + add_attr,
                  id=_id,
                  style={'cursor': 'pointer', 'color': color})


@logit
def appinfo():
    ret_val = \
        html.Div(
            children=[
                'Źródła danych: ',
                dcc.Link('CSSE',
                         href='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'),
                ', ',
                dcc.Link('Our World in Data', href='https://covid.ourworldindata.org/data/owid-covid-data.csv'),
                ', ',
                dcc.Link('Covid-19 w Polsce - Michał Rogalski',
                         href='https://docs.google.com/spreadsheets/d/1ierEhD6gcq51HAm433knjnVwey4ZE5DCnu1bW7PRG3E/edit#gid=1309014089'),
                'Wersja danych: ' + str(start_server_date),
                '  (c) docent ',
                dcc.Link('poczta.docent@gmail.com', href='mailto:poczta.docent@gmail.com'),
                html.Br(),
            ])
    return ret_val


def layout_app_header():
    ret_val = \
        html.Div(
            children=[
                html.Div(
                    dbc.Row([
                        dbc.Col([
                            html.Div(children=["covide.pl"], style={'color': 'orange'}, className='pl-4'),
                        ], width=6),
                        dbc.Col([
                            'Źródła danych: ',
                            dcc.Link('CSSE',
                                     href='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'),
                            ', ',
                            dcc.Link('Our World in Data',
                                     href='https://covid.ourworldindata.org/data/owid-covid-data.csv'),
                            ', ',
                            dcc.Link('Covid-19 w Polsce',
                                     href='https://docs.google.com/spreadsheets/d/1ierEhD6gcq51HAm433knjnVwey4ZE5DCnu1bW7PRG3E/edit#gid=1309014089'),
                            ', Wersja danych: ' + str(dt.today())[:16],
                            ' (c) docent ',
                            dcc.Link('poczta.covide@gmail.com', href='mailto:poczta.covid@gmail.com'),
                        ], width=6, className='text-right'),
                    ]),
                    className="app-header--title")
            ]
        )
    return ret_val


def layout_header(txt):
    ret_val = \
        html.Div(children=[
            txt,
        ], style={'background-color': 'rgb(189, 183, 107)',
                  'width': '100%',
                  'font-weight': 'bold',
                  'font-size': '22px',
                  'text-align': 'center',
                  'color': 'black'})
    return ret_val


def layout_footer():
    ret_val = \
        html.Div(children=[
            html.A(href='https://twitter.com/docent_ws', children='https://twitter.com/docent_ws', target='_blank'), html.Br(),
            html.A(href='https://docent.space', children='https://docent.space', target='_blank'),
        ], style={'width': '100%', 'text-align': 'right', 'padding-right': '15px', 'position': 'absolute',
                  'bottom': '0%', 'color': 'red'})
    return ret_val


@logit
def layout_main(DF):
    ret_val = \
        [
            layout_app_header(),
            html.Div(id="hidden_div_for_redirect_callback_stat"),
            dbc.Row(className='mr-0',
                    children=[
                        dbc.Col(className='ml-4 mr-1 border-right border-danger',
                                style={'background': 'rgb(0,43,53)'},
                                children=layout_main_controls(), width=3),
                        dbc.Col(id='charts_id',
                                style={'background': session_or_die('color_1', default='rgb(0,43,53)')},
                                children=[
                                    dcc.Loading(id="loading_id", type="circle",
                                                children=html.Div(id="loading-output_id")),
                                    dbc.Tabs(style={'cursor': 'pointer'}, className='ml-0 mr-0',
                                             children=[
                                                 dbc.Tab(label="Oś czasu", tab_id="tab-1",
                                                         children=[
                                                             layout_timeline_controls(),
                                                             dbc.Row(id='timeline_charts_id',
                                                                     style={'height': '75vh', 'overflow-y': 'auto',
                                                                            'background': constants.get_defa()[
                                                                                'color_0']},
                                                                     className='ml-0 mr-0 mt-0 mb-0 pt-1 pb-0')
                                                         ], className='mr-4'),
                                                 dbc.Tab(label="Ranking", tab_id="tab-2",
                                                         children=[
                                                             layout_rankings_controls(DF),
                                                             dbc.Row(id='rankings_charts_id',
                                                                     style={'height': '81vh', 'overflow-y': 'auto',
                                                                            'background': constants.get_defa()[
                                                                                'color_0']},
                                                                     className='ml-0 mr-0')
                                                         ]),
                                                 dbc.Tab(label="Dynamika", tab_id="tab-3",
                                                         children=[
                                                             layout_dynamics_controls(),
                                                             dbc.Row(id='dynamics_charts_id',
                                                                     style={'height': '75vh', 'overflow-y': 'auto',
                                                                            'background': constants.get_defa()[
                                                                                'color_0']},
                                                                     className='ml-0 mr-0')
                                                         ]),
                                                 dbc.Tab(label="Mapa", tab_id="tab-5",
                                                         children=[
                                                             layout_map_controls(DF),
                                                             dbc.Row(id='map_id',
                                                                     style={'height': '84vh', 'overflow-y': 'auto',
                                                                            'background': constants.get_defa()[
                                                                                'color_0']},
                                                                     className='ml-0 mr-0 mt-0 mb-0')
                                                         ], className='mr-4'),
                                                 dbc.Tab(label="Korelacja", tab_id="tab-5a",
                                                         children=[
                                                             layout_core_controls(DF),
                                                             dbc.Row(id='core_id',
                                                                     style={'height': '75vh', 'overflow-y': 'auto',
                                                                            'background': constants.get_defa()[
                                                                                'color_0']},
                                                                     className='ml-0 mr-0 mt-0 mb-0')
                                                         ], className='mr-4'),
                                                 dbc.Tab(label="Tabela", tab_id="tab-6",
                                                         children=[
                                                             layout_table_controls(),
                                                             dbc.Row(id='table_id',
                                                                     style={'width': '95%', 'height': '75vh',
                                                                            'overflow-y': 'auto'},
                                                                     className='pl-0 pr-0 ml-0 mr-0 mt-0 mb-0')
                                                         ], className='mr-4'),
                                                 dbc.Tab(label="Przegląd", tab_id="tab-6b",
                                                         children=[
                                                             layout_overview_controls(),
                                                             dbc.Row(id='overview_id',
                                                                     style={'width': '99%', 'height': '80vh',
                                                                            'overflow-y': 'auto',
                                                                            'background': constants.get_defa()[
                                                                                'color_0']},
                                                                     className='pl-0 pr-0 ml-0 mr-0 mt-0 mb-0')
                                                         ], className='mr-0 pr-0'),
                                                 dbc.Tab(label="Raporty", tab_id="tab-6a",
                                                         children=[
                                                             #  'graph', 'cube', 'circle', 'dot' or 'default'
                                                             dcc.Loading(id="loading2_id", type="cube",
                                                                         children=html.Div(id="loading2-output_id")),
                                                             layout_more_controls(),
                                                             dbc.Row(id='more_id',
                                                                     style={'width': '99%', 'height': '80vh',
                                                                            'overflow-y': 'auto',
                                                                            'background': session_or_die('color_0', default='rgb(0,43,53)')},
                                                                     className=constants.nomargins)
                                                         ], className='mr-0 pr-0'),
                                                 dbc.Tab(label="Info", tab_id="tab-7",
                                                         children=[
                                                             html.Div(id='info_id')
                                                         ]),
                                                 dbc.Tab(label="Ustawienia", tab_id="tab-8",
                                                         children=[
                                                             html.Div(id='settings_id')
                                                         ]),
                                             ], id="tabs_id", active_tab="tab-1")
                                ])
                    ]
                    ),
        ]
    return ret_val


@logit
def layout_dropdowns():
    ret_val = \
        [
            dbc.DropdownMenu(
                html.Div(layout_collapse_data(), style=constants.dropdown_style), label="Dane", direction='top', size='sm',
                color="secondary", className="m-1"
            ),
            dbc.DropdownMenu(
                html.Div(layout_collapse_font(), style=constants.dropdown_style), label="Czcionki", direction='top', size='sm', color="secondary",
                className="m-1"
            ),
            dbc.DropdownMenu(
                html.Div(layout_collapse_color(), style=constants.dropdown_style), label="Kolory", direction='top', size='sm', color="secondary",
                className="m-1"
            ),
            dbc.DropdownMenu(
                html.Div(layout_collapse_layout(), style=constants.dropdown_style), label="Układ", direction='top', size='sm', color="secondary",
                className="m-1"
            ),
            dbc.DropdownMenu(
                html.Div(layout_collapse_draw(), style=constants.dropdown_style), label="Grafika", direction='top', size='sm', color="secondary",
                className="m-1"
            ),
            dbc.DropdownMenu(
                html.Div(layout_collapse_graph(), style=constants.dropdown_style), label="Wykresy", direction='top', size='sm', color="secondary",
                className="m-1"
            ),
            dbc.DropdownMenu(
                html.Div(layout_collapse_map(), style=constants.dropdown_style), label="Mapa", direction='top', size='sm', color="secondary",
                className="m-1"
            ),
        ]
    return ret_val


@logit
def layout_main_controls():
    ret_val = \
        [
            html.Div(dbc.Button('but', id='buttler'), style={'display': 'none'}),
            html.Div(dbc.Input(id='go', value=True), style={'display': 'none'}),
            html.Div(dbc.Input(id='go1', value=True), style={'display': 'none'}),   # parametry
            html.Div(dbc.Input(id='go2', value=True), style={'display': 'none'}),   # daty
            html.Div(dbc.Input(id='go3', value=True), style={'display': 'none'}),   # kolory
            html.Div(id='modal'),
            html.Div(id='modal1'),  # tabela download
            html.Div(id='modal2'),  # moje
            html.Div(id='modal3'),  # GitHub

            html.Div(id='pop', children=
            dbc.Popover([dbc.PopoverHeader("Pomoc programu COVID-19"), dbc.PopoverBody('body'), ],
                        id='popover', is_open=False, target='buttler')),

            dbc.Row([
                dbc.Col([
                    html.Div(id='login_data_id')
                ], width=9),
                dbc.Col([
                    html.Div(id="hidden_div_for_redirect_callback"),
                    dbc.Button(children=layout_login(), id='login_button_id', size='sm', className='mr-0')
                ], width=3, className='text-right'),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.RadioItems(
                        options=[
                            {"label": "województwa", 'value': 'poland'},
                            {"label": "powiaty", 'value': 'cities'},
                            {"label": "świat", 'value': 'world'},
                        ],
                        value='poland',
                        id={'type': 'params', 'index': "scope_id"},
                        inline=True
                    ),
                ], width=12),
                dbc.Col([
                    html.Div([
                        dbc.Select(id={'type': 'params', 'index': "subscope_id"},
                                   options=[
                                       {"label": "Wszystkie kraje", 'value': 'world'},
                                       {"label": "Europa", 'value': 'Europa'},
                                       {"label": "Azja", 'value': 'Azja'},
                                       {"label": "Afryka", 'value': 'Afryka'},
                                       {"label": "Ameryka Północna", 'value': 'Ameryka Północna'},
                                       {"label": "Ameryka Południowa", 'value': 'Ameryka Południowa'},
                                       {"label": "Oceania", 'value': 'Oceania'},
                                   ],
                                   value='world',
                                   # persistence=True,
                                   className='p-0',
                                   ),
                    ], style={'visibility': 'visible'}),
                ], width=6, className='text-right'),
            ], className=' ml-1 mt-0 mb-0', style={'backgroundColor': get_default_color(11)}),
            # html.Hr(style={'border': '1px'}),
            dbc.Row([
                dbc.Col([
                    badge('Lokalizacja', 'lokalizacja'),
                    dcc.Dropdown(id={'type': 'params', 'index': 'locations_id'},
                                 multi=True,
                                 placeholder='Kliknij aby wybrać lokalizację ...',
                                 style={'backgroundColor': get_default_color(12),
                                        'borderColor': get_default_color(12)},
                                 ),
                ], width=9),
                dbc.Col([
                    html.Img(src='assets/covid.png', width='100%', height='50%',
                             className='mt-0 mb-0 pt-0 pb-0 ml-0 mr-0 pl-0 pr-0',
                             style={'background-color': 'transparent'}),
                ], width=3),
            ]),
            badge('Szybki wybór grupy lokalizacji', 'grupa', color='Grey'),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Select(id='select_group_id',
                               options=[],
                               value='',
                               persistence=False
                               ),
                ], width=6),
                dbc.Col([
                    dbc.RadioItems(
                        id='as_sum_id',
                        options=[
                            {"label": "pojedynczo", 'value': 'single'},
                            {"label": "sumarycznie", 'value': 'sum'},
                        ],
                        labelStyle={'font-size': '16px'},
                        value='single',
                        persistence=False, inline=False
                    ),
                ], width=3),
                dbc.Col([
                    dbc.Button("Dodaj", id='group_button_id', size='sm', className='mr-1')
                ], width=3, className='text-right')
            ]),
            badge('Dane', 'dane'),
            html.Div([
                dbc.Checklist(
                    id={'type': 'params', 'index': 'chart_type_data_id'},
                    className='checklist-dash',
                    options=trace_options('poland', 'data'),
                    labelStyle={'font-size': '16px'},
                    value=[], labelCheckedStyle={"color": "orange"},
                    switch=False, persistence=False, inline=True
                )
            ], className='twoColumns'),
            dbc.Row([
                dbc.Col([
                    dbc.RadioItems(
                        options=[
                            {"label": "bez modyfikacji", "value": 1},
                            {"label": "na 100 000 osób", "value": 2},
                            {"label": "na 1000 km2", "value": 3},
                        ],
                        value=1,
                        id={'type': 'params', 'index': "data_modifier_id"},
                        style={'font-size': '12px', 'color': 'white'},
                        inline=True
                    ),
                ], className=top_opt_class, width=6),
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            badge('Od daty: ', 'oddaty'),
                        ], className=top_opt_class, width=4),
                        dbc.Col([
                            dcc.DatePickerSingle(id={'type': 'params', 'index': 'from_date_id'},
                                                 display_format='DD-MM-Y',
                                                 date=dt.today(),
                                                 persistence=False),
                        ], className='pt-2', width=8),
                        dbc.Col([
                            badge('Do daty: ', 'oddaty'),
                        ], className=top_opt_class, width=4),
                        dbc.Col([
                            dcc.DatePickerSingle(id={'type': 'params', 'index': 'to_date_id'},
                                                 display_format='DD-MM-Y',
                                                 date=dt.today(),
                                                 persistence=False),
                        ], className='pt-2', width=8),
                    ]),
                ]),
            ], className=top_row_class_inner + ' ml-1 mt-0 mb-0', style={'backgroundColor': get_default_color(11)}),
            dbc.Row([
                dbc.Col([
                    badge('Dzielone przez', 'map_dzielna'),
                    dbc.Select(id={'type': 'params', 'index': 'dzielna_id'},
                               options=[{"label": "<brak>", "value": '<brak>'}] + fields_to_options(),
                               value='<brak>',
                               persistence=False,
                               ),
                ], className=top_opt_class, width=6),
                dbc.Col([
                    badge('Agregacja', 'agregacja'),
                    dbc.Select(id={'type': 'params', 'index': 'table_mean_id'},
                               options=dict_to_options(constants.table_mean),
                               value='daily',
                               persistence=False,
                               )
                ], className=top_opt_class, width=6),
            ], className=top_row_class_inner + ' ml-1 mt-0 mb-0', style={'backgroundColor': get_default_color(11)}),
            # badge('Wskaźniki', 'wskazniki'),
            html.Div([
                dbc.Checklist(
                    id={'type': 'params', 'index': 'chart_type_calculated_id'},
                    className='checklist-dash',
                    options=trace_options('poland', 'calculated'),
                    labelStyle={'font-size': '16px'},
                    value=[], labelCheckedStyle={"color": "orange"},
                    switch=False, persistence=False, inline=True
                )
            ], className='twoColumns'),
            html.Br(),
            html.Div(id='dropdowns',
                     children=layout_dropdowns(),
                     style={"display": "flex", "flexWrap": "wrap"}, className='fixed-bottom'),
        ]
    return html.Div(ret_val, style=constants.dropdown_style)


@logit
def layout_collapse_moje():
    ret_val = \
        [
            dbc.Row([
                dbc.Col([
                    badge('Moje ustawienia', 'mojeustawienia'),
                    dbc.Button("Zapisz ustawienia", id='write_settings_id', color='warning', block=True),
                    dbc.Button("Wczytaj ustawienia", id='read_settings_id', color='warning', block=True),
                    dbc.Button("Przywróć ustawienia domyślne", id='reset_settings_id', color='danger', block=True),
                ], width=6),
            ])
        ]
    return ret_val


@logit
def layout_collapse_data():
    average_days = get_settings('average_days')
    smooth = get_settings('smooth')
    smooth_method = get_settings('smooth_method')
    total_min = get_settings('total_min')
    duration_d = get_settings('duration_d')
    duration_r = get_settings('duration_r')
    win_type = get_settings('win_type')
    rounding = get_settings('rounding')
    options_yesno = get_settings('options_yesno')

    ret_val = \
        [
            badge('Opcje danych', 'opcjedanych'),
            dbc.Row([
                dbc.Col([
                    dbc.Checklist(
                        id={'type': 'params', 'index': 'options_yesno_id'},
                        options=[
                            {"label": "Pokaż dokładne wartości przy wygładzaniu lub uśrednianiu", 'value': 'points'},
                        ],
                        value=options_yesno, labelCheckedStyle={"color": "orange"},
                        switch=False, persistence=False, inline=False,
                        className='mt-2'
                    ),
                ]),
            ]),
            # html.Br(),
            dbc.Row([
                dbc.Col([
                    badge('Średni czas trwania choroby zakończonej śmiercią (dni)', 'duration_d'),
                ], width=8, className='text-right'),
                dbc.Col([
                    dbc.Input(id={'type': 'params', 'index': "duration_d_id"},
                              type='number', value=duration_d, size='5',
                              persistence=False),
                ], width=3),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Średni czas trwania choroby zakończonej wyzdrowieniem (dni)', 'duration_r'),
                ], width=8, className='text-right'),
                dbc.Col([
                    dbc.Input(id={'type': 'params', 'index': "duration_r_id"},
                              type='number', value=duration_r, size='5',
                              persistence=False),
                ], width=3),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Od liczby przypadków', 'lprzypadkow'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Input(id={'type': 'params', 'index': "total_min_id"},
                              type='number', value=total_min, size='5', step=10,
                              persistence=False),
                ], width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Uśrednianie (dni)', 'usrednienie')
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Input(id={'type': 'params', 'index': "average_days_id"},
                              type='number', min=1, value=average_days,
                              persistence=False)],
                    width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Algorytm uśredniania:', 'win_type'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Select(id={'type': 'params', 'index': 'win_type_id'},
                               options=list_to_options([
                                   'równe wagi', 'boxcar', 'triang', 'gaussian', 'hamming', 'bartlett', 'bohman']),
                               value=win_type)],
                    width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Zaokrąglanie:', 'rounding'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Select(id={'type': 'params', 'index': 'rounding_id'},
                               options=list_to_options([
                                   'bez zaokrąglenia', 'int', '0', '1', '2', '3', '4']),
                               value=rounding)],
                    width=4),
            ]),
            badge('Algorytm wygładzania', 'algorytm'),
            dbc.RadioItems(
                id={'type': 'params', 'index': 'smooth_method_id'},
                className='checklist-dash',
                options=[
                    {"label": "Regresja liniowa (metoda wielomianowa)", 'value': 'wielomiany'},
                    {"label": "Metoda Savitzky'ego-Golaya", 'value': 'sawicki'},
                ],
                value=smooth_method,
                persistence=False, inline=True
            ),

            badge('Stopień wygładzania wykresu', 'stopienwygladzenia'),
            dcc.RadioItems(
                id={'type': 'params', 'index': 'smooth_id'},
                className='radio-dash',
                options=[{'label': 'brak', 'value': 0},
                         {'label': '3 stopnia', 'value': 3},
                         {'label': '5 stopnia', 'value': 5},
                         {'label': '7 stopnia', 'value': 7},
                         {'label': '9 stopnia', 'value': 9},
                         {'label': '11 stopnia', 'value': 11}],
                value=smooth,
                persistence=False,
                labelStyle={'display': 'inline-block', }),
        ]
    return ret_val


@logit
def layout_collapse_font():
    font_size_xy = get_settings('font_size_xy')
    font_size_title = get_settings('font_size_title')
    font_size_anno = get_settings('font_size_anno')
    font_size_legend = get_settings('font_size_legend')
    ret_val = \
        [
            badge('Wielkość czcionki:', 'fontsize'),
            dbc.Row([
                dbc.Col([
                    dbc.Input(id={'type': 'params', 'index': "font_size_xy_id"},
                              type='number', min=1, value=font_size_xy, size='7', style={'margin-left': '20px'},
                              persistence=False)],
                    width=5),
                dbc.Col(dbc.Label('Czcionka opisu osi'),
                        width=7),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Input(id={'type': 'params', 'index': "font_size_title_id"},
                              type='number', min=1, value=font_size_title, size='7', style={'margin-left': '20px'},
                              persistence=False)],
                    width=5),
                dbc.Col(dbc.Label('Czcionka tytułu'),
                        width=7)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Input(id={'type': 'params', 'index': "font_size_anno_id"},
                              type='number', min=1, value=font_size_anno, size='7', style={'margin-left': '20px'},
                              persistence=False)],
                    width=5),
                dbc.Col(dbc.Label('Czcionka etykiet (adnotacji)'),
                        width=7),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Input(id={'type': 'params', 'index': "font_size_legend_id"},
                              type='number', min=1, value=font_size_legend, size='7', style={'margin-left': '20px'},
                              persistence=False)],
                    width=5),
                dbc.Col(dbc.Label('Czcionka legendy'),
                        width=7),
            ]),
        ]
    return ret_val


@logit
def layout_collapse_color():
    ret_val = \
        [
            badge('Paleta kolorów:', 'colorobject'),
            dbc.Select(id={'type': 'params', 'index': 'template_id'},
                       options=[{'label': 'Domyślny', 'value': 'default'},
                                {'label': 'Beżowe', 'value': 'beige'},
                                {'label': 'Błękit', 'value': 'blue'},
                                {'label': 'Niebieski', 'value': 'darkblue'},
                                {'label': 'Zgniła zieleń', 'value': 'green'},
                                {'label': 'Ceglasty', 'value': 'brick'},
                                {'label': 'Jasne', 'value': 'white'},
                                # {'label': 'Przezroczyste tło', 'value': 'transparent'},
                                ],
                       value='default',
                       ),
            badge('Wybór kolorów:', 'colorobject'),
            dbc.Select(id={'type': 'params', 'index': 'colors_id'},
                       options=[{'label': 'Tło podkładu (0)', 'value': 'color_0'},
                                {'label': 'Tło wykresu (1)', 'value': 'color_1'},
                                {'label': 'Osie współrzędnych (2)', 'value': 'color_2'},
                                {'label': 'Czcionka opisu osi (3)', 'value': 'color_3'},
                                {'label': 'Czcionka etykiet i adnotacji (4)', 'value': 'color_4'},
                                {'label': 'Adnotacje (wykresy) (5)', 'value': 'color_5'},
                                {'label': 'Tło legendy (6)', 'value': 'color_6'},
                                {'label': 'Tytuł (7)', 'value': 'color_7'},
                                {'label': 'Słupki (ranking) (8)', 'value': 'color_8'},
                                {'label': 'Wyróżnienie (9)', 'value': 'color_9'},
                                {'label': 'Czcionka legendy (10)', 'value': 'color_10'},
                                {'label': 'Dodatkowy 1 (13)', 'value': 'color_13'},
                                {'label': 'Dodatkowy_2 (14)', 'value': 'color_14'},
                                ],
                       value='color_1',
                       ),
            html.Div(children=layout_palettes()),
        ]
    return ret_val


@logit
def layout_palettes():
    ret_val = []
    for scale in constants.color_scales.keys():
        if '_r' not in scale and len(constants.color_scales[scale]) == 12:
            for x in constants.color_scales[scale]:
                element = html.Button(className='square',
                                      id={'type': 'color', 'index': str(x)},
                                      style={'background-color': str(x)})
                ret_val.append(element)
            ret_val.append(html.Br())
    return html.Div(id='palettes_id', children=ret_val)
    # return html.Div(id='palettes_id', children=ret_val, style={'height': '600px', 'overflow-y': 'auto'})


@logit
def layout_collapse_layout():
    margint = get_settings('margint')
    marginb = get_settings('marginb')
    marginl = get_settings('marginl')
    marginr = get_settings('marginr')
    titlexpos = get_settings('titlexpos')
    titleypos = get_settings('titleypos')
    copyrightxpos = get_settings('copyrightxpos')
    copyrightypos = get_settings('copyrightypos')
    annoxpos = get_settings('annoxpos')
    annoypos = get_settings('annoypos')
    ret_val = \
        [
            subtitle('Marginesy'),
            dbc.Row([
                dbc.Col([
                    badge('Margines górny:', 'margint'),
                ], width=4, className='text-right pl-0 pr-0'),
                dbc.Col([
                    dcc.Slider(id={'type': 'params', 'index': 'margint_id'},
                               min=0, max=200,
                               step=10,
                               value=margint,
                               persistence=False,
                               marks={x: str(x) for x in range(0, 200, 25)},
                               ),
                ], width=8, className='pl-0 pr-0'),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Margines dolny:', 'marginb'),
                ], width=4, className='text-right pl-0 pr-0'),
                dbc.Col([
                    dcc.Slider(id={'type': 'params', 'index': 'marginb_id'},
                               min=0, max=100,
                               step=10,
                               value=marginb,
                               persistence=False,
                               marks={x: str(x) for x in range(0, 100, 25)},
                               ),
                ], width=8, className='pl-0 pr-0'),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Margines lewy:', 'marginl'),
                ], width=4, className='text-right pl-0 pr-0'),
                dbc.Col([
                    dcc.Slider(id={'type': 'params', 'index': 'marginl_id'},
                               min=0., max=100.,
                               step=10,
                               value=marginl,
                               persistence=False,
                               marks={x: str(x) for x in range(0, 100, 25)},
                               ),
                ], width=8, className='pl-0 pr-0'),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Margines prawy:', 'marginr'),
                ], width=4, className='text-right pl-0 pr-0'),
                dbc.Col([
                    dcc.Slider(id={'type': 'params', 'index': 'marginr_id'},
                               min=0, max=100,
                               step=10,
                               value=marginr,
                               persistence=False,
                               marks={x: str(x) for x in range(0, 100, 25)},
                               ),
                ], width=8, className='pl-0 pr-0'),
            ]),
            subtitle('Pozycjonowanie tytułu wykresu'),
            dbc.Row([
                dbc.Col([
                    badge('Od lewej:', 'titlexpos'),
                ], width=4, className='text-right pl-0 pr-0'),
                dbc.Col([
                    dcc.Slider(id={'type': 'params', 'index': 'titlexpos_id'},
                               min=0., max=0.99,
                               step=0.01,
                               value=titlexpos,
                               persistence=False,
                               marks={x: str(x) for x in np.arange(0., 1., 0.25)},
                               ),
                ], width=8, className='pl-0 pr-0'),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Od podstawy:', 'titleypos'),
                ], width=4, className='text-right pl-0 pr-0'),
                dbc.Col([
                    dcc.Slider(id={'type': 'params', 'index': 'titleypos_id'},
                               min=0., max=0.99,
                               step=0.01,
                               value=titleypos,
                               persistence=False,
                               marks={x: str(x) for x in np.arange(0., 1., 0.25)},
                               ),
                ], width=8, className='pl-0 pr-0'),
            ]),
            subtitle('Pozycjonowanie adnotacji'),
            dbc.Row([
                dbc.Col([
                    badge('Od lewej:', 'annoxpos'),
                ], width=4, className='text-right pl-0 pr-0'),
                dbc.Col([
                    dcc.Slider(id={'type': 'params', 'index': 'annoxpos_id'},
                               min=-0.25, max=1.,
                               step=0.01,
                               value=annoxpos,
                               persistence=False,
                               marks={x: str(x) for x in np.arange(-0.25, 1., 0.25)},
                               ),
                ], width=8, className='pl-0 pr-0'),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Od podstawy:', 'copyrightypos'),
                ], width=4, className='text-right pl-0 pr-0'),
                dbc.Col([
                    dcc.Slider(id={'type': 'params', 'index': 'annoypos_id'},
                               min=-0.5, max=1.,
                               step=0.01,
                               value=annoypos,
                               persistence=False,
                               marks={x: str(x) for x in np.arange(-0.5, 1., 0.25)},
                               ),
                ], width=8, className='pl-0 pr-0'),
            ]),
            subtitle('Pozycjonowanie copyright'),
            dbc.Row([
                dbc.Col([
                    badge('Od lewej:', 'titlexpos'),
                ], width=4, className='text-right pl-0 pr-0'),
                dbc.Col([
                    dcc.Slider(id={'type': 'params', 'index': 'copyrightxpos_id'},
                               min=-0.25, max=1.,
                               step=0.01,
                               value=copyrightxpos,
                               persistence=False,
                               marks={x: str(x) for x in np.arange(-0.25, 1., 0.25)},
                               ),
                ], width=8, className='pl-0 pr-0'),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Od podstawy:', 'copyrightypos'),
                ], width=4, className='text-right pl-0 pr-0'),
                dbc.Col([
                    dcc.Slider(id={'type': 'params', 'index': 'copyrightypos_id'},
                               min=-0.5, max=1.,
                               step=0.01,
                               value=copyrightypos,
                               persistence=False,
                               marks={x: str(x) for x in np.arange(-0.5, 1., 0.25)},
                               ),
                ], width=8, className='pl-0 pr-0'),
            ]),
        ]
    return ret_val


@logit
def layout_collapse_draw():
    linedraw = get_settings('linedraw')
    color_order = get_settings('color_order')
    linewidth_basic = get_settings('linewidth_basic')
    linewidth_thin = get_settings('linewidth_thin')
    linewidth_thick = get_settings('linewidth_thick')
    ret_val = \
        [
            subtitle('Wykresy na osi czasu'),
            dbc.Row([
                dbc.Col([
                    badge('Grubość linii podstawowej', 'linewidth_basic'),
                ], width=12, className='ml-2 mr-2'),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Slider(id={'type': 'params', 'index': 'linewidth_basic_id'},
                               min=0.1, max=10.,
                               step=0.1,
                               value=linewidth_basic,
                               persistence=False,
                               marks={x: str(x) for x in np.arange(0.5, 10., 1.)},
                               ),
                ], width=12, className='mt-1 ml-2 mr-2'),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Grubość linii wyróżnionej', 'linewidth_thick'),
                ], width=12, className='ml-2 mr-2'),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Slider(id={'type': 'params', 'index': 'linewidth_thick_id'},
                               min=0.1, max=10.,
                               step=0.1,
                               value=linewidth_thick,
                               persistence=False,
                               marks={x: str(x) for x in np.arange(0.5, 10., 1.)},
                               ),
                ], width=12, className='mt-1 ml-2 mr-2'),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Grubość linii pomocniczej (dane dokładne)', 'linewidth_thin'),
                ], width=12, className='ml-2 mr-2'),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Slider(id={'type': 'params', 'index': 'linewidth_thin_id'},
                               min=0.1, max=10.,
                               step=0.1,
                               value=linewidth_thin,
                               persistence=False,
                               marks={x: str(x) for x in np.arange(0.5, 10., 1.)},
                               ),
                ], width=12, className='mt-1 ml-2 mr-2'),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Styl linii:', 'linedraw'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Select(id={'type': 'params', 'index': 'linedraw_id'},
                               options=[
                                   {'label': 'tylko linie', 'value': 'lines'},
                                   {'label': 'linie + markery', 'value': 'lines+markers'},
                               ],
                               value=linedraw,
                               persistence=False,
                               ),
                ], width=5),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Kolorystyka (wykres liniowy)', 'kolorystykal'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Select(id={'type': 'params', 'index': 'color_order_id'},
                               options=list_to_options(sorted(constants.color_scales.keys())),
                               value=color_order,
                               persistence=False,
                               )
                ], width=5),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id='kolorki_id',
                             style={'text-align': 'right'},
                             className='pl-2'),
                ], width=12),
            ]),
        ]
    return ret_val


@logit
def layout_collapse_graph():
    options_graph_format = get_settings('options_graph_format')
    legend_place = get_settings('legend_place')
    annotations = get_settings('annotations')
    anno_form = get_settings('anno_form')
    line_dash = get_settings('line_dash')
    bar_gap = get_settings('bar_gap')
    bar_fill = get_settings('bar_fill')
    bar_frame = get_settings('bar_frame')
    bar_mode = get_settings('bar_mode')
    options_rank = get_settings('options_rank')
    arrangement = get_settings('arrangement')
    plot_height = get_settings('plot_height')
    radio_scale = get_settings('radio_scale')
    table_rows = get_settings('table_rows')

    ret_val = \
        [
            subtitle('Wykresy na osi czasu'),
            dbc.Row([
                dbc.Col([
                    badge('Układ wielu wykresów', 'uklad'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Select(id={'type': 'params', 'index': 'arrangement_id'},
                               options=[{'label': 'pełny ekran', 'value': 'big'},
                                        {'label': 'auto', 'value': 'smart'},
                                        ],
                               value=arrangement,
                               persistence=False,
                               )
                ], width=5),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Wysokość', 'wysokosctimeline'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Select(id={'type': 'params', 'index': 'plot_height_id'},
                               options=[
                                   {'label': 'x 0.5', 'value': 0.5},
                                   {'label': 'x 0.6', 'value': 0.6},
                                   {'label': 'x 0.7', 'value': 0.7},
                                   {'label': 'x 0.8', 'value': 0.8},
                                   {'label': 'x 0.9', 'value': 0.9},
                                   {'label': 'x 1', 'value': 1},
                                   {'label': 'x 1.1', 'value': 1.1},
                                   {'label': 'x 1.2', 'value': 1.2},
                                   {'label': 'x 1.3', 'value': 1.3},
                                   {'label': 'x 1.4', 'value': 1.4},
                                   {'label': 'x 1,5', 'value': 1.5},
                                   {'label': 'x 2', 'value': 2},
                               ],
                               value=plot_height,
                               persistence=False,
                               )
                ], width=5),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Skala (dane kumulowane)', 'skalay'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Select(id={'type': 'params', 'index': 'radio_scale_id'},
                               options=[{'label': 'liniowa', 'value': 'linear'},
                                        {'label': 'logarytmiczna', 'value': 'log'}
                                        ],
                               value=radio_scale,
                               persistence=False,
                               )
                ], width=5),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Adnotacje', 'adnotacje'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Select(id={'type': 'params', 'index': 'annotations_id'},
                               options=[{'label': 'maksimum', 'value': 'max'},
                                        {'label': 'ostatni', 'value': 'last'},
                                        {'label': 'maksimum + ostatni', 'value': 'max last'},
                                        {'label': 'wszystkie', 'value': 'all'},
                                        {'label': 'co 1', 'value': 'x01'},
                                        {'label': 'co 2', 'value': 'x02'},
                                        {'label': 'co 5', 'value': 'x05'},
                                        {'label': 'co 10', 'value': 'x10'},
                                        {'label': 'co 20', 'value': 'x20'},
                                        {'label': 'brak', 'value': 'none'},
                                        {'label': 'heatmap', 'value': 'anno_heatmap'}
                                        ],
                               value=annotations,
                               persistence=False,
                               )
                ], width=5),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Format adnotacji', 'adnotacje_format'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Select(id={'type': 'params', 'index': 'anno_form_id'},
                               options=[{'label': 'liczba', 'value': 'num'},
                                        {'label': 'nazwa', 'value': 'name'},
                                        {'label': 'nazwa+liczba', 'value': 'namenum'},
                                        {'label': 'nazwa+data+liczba', 'value': 'namedatenum'},
                                        ],
                               value=anno_form,
                               persistence=False,
                               )
                ], width=5),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Położenie legendy:', 'legendplace'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Select(id={'type': 'params', 'index': 'legend_place_id'},
                               options=[
                                   {'label': 'poziomo na górze', 'value': 'poziomo'},
                                   {'label': 'pionowo po lewej', 'value': 'lewo'},
                                   {'label': 'pionowo po prawej', 'value': 'prawo'},
                               ],
                               value=legend_place,
                               persistence=False,
                               ),
                ], width=5),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Rodzaj linii:', 'gridlines'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Select(id={'type': 'params', 'index': 'line_dash_id'},
                               options=[
                                   {'label': 'solid', 'value': 'solid'},
                                   {'label': 'dashes', 'value': 'dashes'},
                               ],
                               value=line_dash,
                               persistence=False,
                               ),
                ], width=5),
            ]),
            subtitle('Wykres słupkowy'),
            dbc.Row([
                dbc.Col([
                    badge('Odstęp między słupkami', 'wysokoscrankings'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Select(id={'type': 'params', 'index': 'bar_gap_id'},
                               options=list_to_options([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]),
                               value=bar_gap,
                               persistence=False)
                ], width=5),
                dbc.Col([
                    badge('Wypełnienie słupków', 'wysokoscrankings'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Select(id={'type': 'params', 'index': 'bar_fill_id'},
                               options=list_to_options(['', '/', '\\', 'x', '-', '|', '+', '.']),
                               value=bar_fill,
                               persistence=False)
                ], width=5),
                dbc.Col([
                    badge('Grubość ramki słupka', 'wysokoscrankings'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Select(id={'type': 'params', 'index': 'bar_frame_id'},
                               options=list_to_options([0, 0.5, 0.7, 1., 1.2, 1.6]),
                               value=bar_frame,
                               persistence=False)
                ], width=5),
                dbc.Col([
                    badge('Rodzaj wykresu', 'wysokoscrankings'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Select(id={'type': 'params', 'index': 'bar_mode_id'},
                               options=list_to_options(['group', 'stack']),
                               value=bar_mode,
                               persistence=False)
                ], width=5),
            ]),
            subtitle('Ranking'),
            dbc.Row([
                dbc.Col([
                    badge('Opcje', 'optrankings'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Checklist(
                        id={'type': 'params', 'index': 'options_rank_id'},
                        options=[
                            {"label": "Pokaż pozycję 'pozostałe'", 'value': 'rest'},
                        ],
                        value=options_rank, labelCheckedStyle={"color": "orange"},
                        switch=False, persistence=False, inline=False,
                        className='mt-2'
                    ),
                ], width=5),
            ]),
            subtitle('Tabela'),
            dbc.Row([
                dbc.Col([
                    badge('Liczba wierszy', 'tablerows'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Input(id={'type': 'params', 'index': "table_rows_id"},
                              type='number', value=table_rows, size='2', min=1, max=1000,
                              persistence=True)
                ], width=5),
            ]),
            subtitle('Opcje wspólne'),
            dbc.Row([
                dbc.Col([
                    badge('Format zapisu grafiki', 'formatgrafiki'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Select(id={'type': 'params', 'index': 'options_graph_format_id'},
                               options=list_to_options(['svg', 'png', 'webp', 'jpeg']),
                               value=options_graph_format,
                               persistence=False,
                               ),
                ], width=5),
            ]),
        ]
    return ret_val


@logit
def layout_collapse_map():
    map_color_scale = get_settings('map_color_scale')
    map_mapbox = get_settings('map_mapbox')
    map_opacity = get_settings('map_opacity')
    map_height = get_settings('map_height')
    map_options = get_settings('map_options')
    map_opt = get_settings('map_opt')
    map_palette = get_settings('map_palette')
    map_cut = get_settings('map_cut')

    ret_val = \
        [
            subtitle('Mapa'),
            dbc.Row([
                dbc.Col([
                    badge('Kolorystyka', 'mapkolorystyka'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Select(id={'type': 'params', 'index': 'map_color_scale_id'},
                               options=list_to_options(['Greys', 'YlGnBu', 'Greens', 'YlOrRd', 'Bluered', 'RdBu',
                                                        'Reds', 'Blues', 'Picnic', 'Rainbow', 'Portland', 'Jet', 'Hot',
                                                        'Blackbody', 'Earth', 'Electric', 'Viridis', 'Cividis', 'River Nights']),
                               value=map_color_scale,
                               persistence=False,
                               )
                ], width=5),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Wysokość', 'wysokoscmapa'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Select(id={'type': 'params', 'index': 'map_height_id'},
                               options=list_to_options([250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850,
                                                        900, 950, 1000, 1100, 1200]),
                               value=map_height,
                               persistence=False)
                ], width=5),
            ]),
            dbc.Row([
                dbc.Col([
                    badge('Podkład mapy', 'mappodklad'),
                ], width=7, className='text-right'),
                dbc.Col([
                    dbc.Select(id={'type': 'params', 'index': 'map_mapbox_id'},
                               options=list_to_options(mapbox_styles.keys()),
                               value=map_mapbox,
                               persistence=False,
                               )
                ], width=5),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    badge('Krycie wypełnienia:', 'mapkrycie'),
                ], width=6, className='text-right pl-0 pr-0'),
                dbc.Col([
                    dcc.Slider(id={'type': 'params', 'index': 'map_opacity_id'},
                               min=0., max=1.,
                               step=0.1,
                               value=map_opacity,
                               marks={0: '0%', 0.5: '50%', 1: '100%'},
                               persistence=False,
                               className='mr-2'
                               ),
                ], width=6, className='pl-0 pr-0'),
            ]),
            badge('Opcje', 'mapoptions'),
            dbc.Checklist(
                id={'type': 'params', 'index': 'map_options_id'},
                options=[
                    {"label": "odwróć kolory", 'value': 'reversescale'},
                    {"label": "linie konturów", 'value': 'drawborder'},
                    {"label": "adnotacje", 'value': 'annotations'},
                    {"label": "większa dokładność", 'value': 'quality'},
                    {"label": "tylko liczba", 'value': 'number'},
                    {"label": "0", 'value': '0c'},
                    {"label": "1", 'value': '1c'},
                    {"label": "2", 'value': '2c'},
                    {"label": "3", 'value': '3c'},
                ],
                value=map_options,
                labelCheckedStyle={"color": "orange"},
                switch=False,
                persistence=False,
                inline=True,
                className='mt-6 ml-2'
            ),

            badge('Rodzaj', 'maparodzaj'),
            dbc.RadioItems(
                id={'type': 'params', 'index': 'map_opt_id'},
                className='checklist-dash',
                options=[
                    {"label": "skala logarytmiczna", 'value': 'log'},
                    {"label": "skala liniowa", 'value': 'linear'},
                ],
                value=map_opt,
                persistence=False, inline=True
            ),
            badge('Pasek kolorów', 'maparodzaj'),
            dbc.RadioItems(
                id={'type': 'params', 'index': 'map_palette_id'},
                className='checklist-dash',
                options=[
                    {"label": "Ciągły", 'value': 'ciągły'},
                    {"label": "Dyskretny", 'value': 'dyskretny'},
                ],
                value=map_palette,
                persistence=False, inline=True
            ),
            dbc.RadioItems(
                id={'type': 'params', 'index': 'map_cut_id'},
                className='checklist-dash',
                options=[
                    {"label": "Przedziały równe", 'value': 'równe'},
                    {"label": "Przedziały kwantylowe", 'value': 'kwantyle'},
                    {"label": "Przedziały własne", 'value': 'własne'},
                ],
                value=map_cut,
                persistence=False, inline=True
            )

        ]
    return ret_val


@logit
def layout_timeline_controls():
    radio_type = get_settings('radio_type')
    timeline_opt = get_settings('timeline_opt')
    timeline_view = get_settings('timeline_view')
    radio_class = get_settings('radio_class')
    radio_flow = get_settings('radio_flow')

    ret_val = \
        dbc.Row(children=[
            dbc.Col([
                badge('Rodzaj wy\u00ADkre\u00ADsu', 'rodzajwykresu'),
                dbc.Select(id={'type': 'params', 'index': 'radio_class_id'},
                           options=[{'label': 'Oddzielnie kategorie', 'value': 'types'},
                                    {'label': 'Oddzielnie lokalizacje', 'value': 'places'}],
                           value=radio_class,
                           persistence=False,
                           style={'cursor': 'pointer'},
                           ),
            ], className=top_opt_class, width=2),
            dbc.Col([
                badge('Skalowanie', 'rodzajwykresu'),
                dbc.Select(id={'type': 'params', 'index': 'radio_flow_id'},
                           options=[{'label': 'Rzeczywiste wartości', 'value': 'real'},
                                    {'label': 'Porównanie przebiegów', 'value': 'proportional'}],
                           value=radio_flow,
                           persistence=False,
                           style={'cursor': 'pointer'},
                           ),
            ], className=top_opt_class, width=2),
            dbc.Col([
                badge('Typ', 'typwykresutimeline'),
                dbc.Select(id={'type': 'params', 'index': 'radio_type_id'},
                           options=[{'label': 'liniowy', 'value': 'scatter'},
                                    {'label': 'słupkowy', 'value': 'bar'}],
                           value=radio_type,
                           persistence=False,
                           ),
            ], className=top_opt_class, width=1),
            dbc.Col([
                badge('Wyróżnienie', 'wyroznienie'),
                dbc.Select(id={'type': 'params', 'index': 'timeline_highlight_id'},
                           options=[
                               {'label': 'Bez wyróżnienia', 'value': 'Brak'},
                           ],
                           value='Brak'
                           )
            ], className=top_opt_class, width=2),
            # dbc.Col([
            #     badge('Od daty', 'oddaty'),
            #     html.Br(),
            #     dcc.DatePickerSingle(id={'type': 'params', 'index': 'from_date_id'},
            #                          display_format='DD-MM-Y',
            #                          date=dt.today(),
            #                          persistence=False),
            # ], className=top_opt_class, width=1),
            dbc.Col([
                badge('Widok', 'widok'),
                dbc.Checklist(
                    id={'type': 'params', 'index': 'timeline_view_id'},
                    className='checklist-dash',
                    options=[
                        {"label": "legenda", 'value': 'legenda'},
                        {"label": "suwak", 'value': 'suwak'},
                    ],
                    value=timeline_view, labelCheckedStyle={"color": "orange"},
                    switch=False, persistence=False, inline=False
                )
            ], className=top_opt_class, width=1),
            dbc.Col([
                badge('Dane', 'timelinedane'),
                dbc.Checklist(
                    id={'type': 'params', 'index': 'timeline_opt_id'},
                    className='checklist-dash',
                    options=[
                        {"label": "Uśrednianie", 'value': 'usrednianie'},
                        {"label": "Średnia okresowa", 'value': 'usrednianieo'},
                        {"label": "Wygładzanie", 'value': 'wygladzanie'},
                    ],
                    value=timeline_opt, labelCheckedStyle={"color": "orange"},
                    switch=False, persistence=False, inline=True
                )
            ], className=top_opt_class, width=2),
        ], className=top_row_class, style={'backgroundColor': session_or_die('color_11', get_default_color(11))})

    return ret_val


@logit
def layout_rankings_controls(DF):
    sess = session_or_die('session', 'no-session')
    if sess == 'no-session':
        date_picker = dt.today() - timedelta(days=1)
    else:
        df = DF.get[sess]
        date_picker = max(df.date)

    ret_val = \
        dbc.Row([
            dbc.Col([
                badge('Wyróżnienie', 'wyroznienie'),
                dbc.Select(id={'type': 'params', 'index': 'rankings_highlight_id'},
                           options=[
                               {'label': 'Bez wyróżnienia', 'value': 'Brak'},
                           ],
                           value='Brak'
                           )
            ], className=top_opt_class, width=2),
            dbc.Col([
                badge('Liczba pozycji:', 'pozostale'),
                dbc.Select(id={'type': 'params', 'index': 'max_cut_id'},
                           options=[{'label': 'Pokaż wszystkie', 'value': '0'}] +
                                   [{'label': str(i), 'value': str(i)} for i in range(1, 20)],
                           value='16')
            ], className=top_opt_class, width=2),
        ], className=top_row_class, style={'backgroundColor': session_or_die('color_11', get_default_color(11))})

    return ret_val


@logit
def layout_dynamics_controls():
    dynamics_scaling = get_settings('dynamics_scaling')
    columns = get_settings('columns')
    dynamics_chart_height = get_settings('dynamics_chart_height')

    ret_val = \
        dbc.Row([
            dbc.Col(children=[
                badge('Skalowanie Y', 'skalowaniey'),
                dbc.Select(id={'type': 'params', 'index': 'dynamics_scaling_id'},
                           options=list_to_options([
                               'Zachowanie proporcji',
                               'Porównanie kształtów',
                           ]),
                           value=dynamics_scaling,
                           persistence=False,
                           )], className=top_opt_class, width=2),
            dbc.Col(children=[
                badge('Liczba kolumn', 'liczbakolumn'),
                dbc.Input(id={'type': 'params', 'index': "columns_id"},
                          type='number', value=columns, size='2', min=1, max=20,
                          persistence=columns)
            ], className=top_opt_class, width=2),
            dbc.Col(children=[
                badge('Wyso\u00ADkość', 'wysokoscdynamics'),
                dbc.Select(
                    id={'type': 'params', 'index': 'dynamics_chart_height_id'},
                    options=list_to_options([200, 250, 300, 350, 400]),
                    value=dynamics_chart_height,
                    persistence=False,
                )], className=top_opt_class, width=2),
        ], className=top_row_class, style={'backgroundColor': session_or_die('color_11', get_default_color(11))})

    return ret_val


@logit
def layout_map_controls(DF):
    sess = session_or_die('session', 'no-session')
    if sess == 'no-session':
        map_date = dt.today() - timedelta(days=1)
    else:
        df = DF[sess]
        map_date = max(df.date)

    map_opt = get_settings('map_opt')

    ret_val = \
        dbc.Row([
            dbc.Col([
                badge('Data', 'datamapa'),
                # html.Br(),
                imgButton('fa-backward', 'map_date_prev_id'),
                dcc.DatePickerSingle(id={'type': 'params', 'index': 'map_date_id'},
                                     display_format='DD-MM-Y',
                                     date=map_date,
                                     persistence=False),
                imgButton('fa-forward', 'map_date_next_id'),
            ], className=top_opt_class + ' pt-4', width=3),

        ], className=top_row_class, style={'backgroundColor': session_or_die('color_11', get_default_color(11))})
    return ret_val


@logit
def layout_core_controls(DF):
    core_opt = get_settings('core_opt')
    core_agg_x = get_settings('core_agg_x')
    core_agg_y = get_settings('core_agg_y')
    core_view = get_settings('core_view')
    ret_val = \
        dbc.Row(children=[
            dbc.Col([
                badge('Rodzaj agregacji X', 'rodzajwykresu'),
                dbc.Select(id={'type': 'params', 'index': 'core_agg_x_id'},
                           options=[{'label': 'średnia', 'value': 'mean'},
                                    {'label': 'suma', 'value': 'sum'},
                                    {'label': 'maksimum', 'value': 'max'},
                                    {'label': 'minimum', 'value': 'min'},
                                    {'label': 'mediana', 'value': 'median'},
                                    {'label': 'jeden dzień', 'value': 'oneday'}
                                    ],
                           value=core_agg_x,
                           persistence=False,
                           style={'cursor': 'pointer'},
                           ),
            ], className=top_opt_class, width=2),
            dbc.Col([
                badge('Rodzaj agregacji Y', 'rodzajwykresu'),
                dbc.Select(id={'type': 'params', 'index': 'core_agg_y_id'},
                           options=[{'label': 'średnia', 'value': 'mean'},
                                    {'label': 'suma', 'value': 'sum'},
                                    {'label': 'maksimum', 'value': 'max'},
                                    {'label': 'minimum', 'value': 'min'},
                                    {'label': 'mediana', 'value': 'median'},
                                    {'label': 'jeden dzień', 'value': 'oneday'}
                                    ],
                           value=core_agg_y,
                           persistence=False,
                           style={'cursor': 'pointer'},
                           ),
            ], className=top_opt_class, width=2),
            dbc.Col([
                badge('Wyróżnienie', 'wyroznienie'),
                dbc.Select(id={'type': 'params', 'index': 'core_highlight_id'},
                           options=[
                               {'label': 'Bez wyróżnienia', 'value': 'Brak'},
                           ],
                           value='Brak'
                           )
            ], className=top_opt_class, width=2),
            dbc.Col([
                badge('Data', 'oddaty'),
                html.Br(),
                dcc.DatePickerSingle(id={'type': 'params', 'index': 'core_date_id'},
                                     display_format='DD-MM-Y',
                                     date=dt.today(),
                                     persistence=False),
            ], className=top_opt_class, width=1),
            dbc.Col([
                badge('Widok', 'widok'),
                dbc.Checklist(
                    id={'type': 'params', 'index': 'core_view_id'},
                    className='checklist-dash',
                    options=[
                        {"label": "suwak", 'value': 'suwak'},
                        {"label": "linia regresji", 'value': 'regresja'},
                        {"label": "linie błędów", 'value': 'errors'},
                    ],
                    value=core_view, labelCheckedStyle={"color": "orange"},
                    switch=False, persistence=False, inline=False
                )
            ], className=top_opt_class, width=1),
            dbc.Col([
                badge('Dane', 'timelinedane'),
                dbc.Checklist(
                    id={'type': 'params', 'index': 'core_opt_id'},
                    className='checklist-dash',
                    options=[
                        {"label": "Zamiana osi", 'value': 'flipaxes'},
                        {"label": "'0' X", 'value': 'tozerox'},
                        {"label": "'0' Y", 'value': 'tozeroy'},
                    ],
                    value=core_opt, labelCheckedStyle={"color": "orange"},
                    switch=False, persistence=False, inline=True
                )
            ], className=top_opt_class, width=2),
        ], className=top_row_class, style={'backgroundColor': session_or_die('color_11', get_default_color(11))})
    return ret_val


@logit
def layout_table_controls():
    ret_val = \
        dbc.Row(children=[
            html.I(className="fa fa-question-circle-o fa-2x",
                   id={'type': 'button_html', 'index': 'timeline'},
                   style={'cursor': 'pointer', 'margin-top': '25px', 'margin-right': '15px'}),
            dbc.Col([
                badge('Do daty', 'tabledodaty'),
                dcc.DatePickerSingle(id={'type': 'params', 'index': 'table_to_date_id'},
                                     display_format='Y-MM-DD',
                                     date=dt.today(),
                                     persistence=False,
                                     className='ml-2'),
            ], className=top_opt_class, width=2),
            dbc.Col([
                badge('tmp/', 'downloadtable'),
                dcc.Input(
                    id='download_format_id',
                    type='text',
                    placeholder='nazwa pliku',
                    value='out_file.csv'
                ),
                html.Div([
                    dbc.Button("Pobierz", id='download_button_id', size='sm', className='mr-1'),
                    dcc.Download(id="download")
                ]),
            ], className=top_opt_class, width=3),
            dbc.Col([
                badge('tmp/', 'downloadtable'),
                dcc.Input(
                    id='download_pivot_format_id',
                    type='text',
                    placeholder='nazwa pliku',
                    value='out_file_pivot.csv'
                ),
                html.Div([
                    dbc.Button("Pobierz pivot", id='download_pivot_button_id', size='sm', className='mr-1'),
                    dcc.Download(id="download_pivot")
                ]),
            ], className=top_opt_class, width=3),
        ], className=top_row_class, style={'backgroundColor': session_or_die('color_11', get_default_color(11))})

    return ret_val


@logit
def layout_info(df_w, df_p, df_c, t_start):
    file_w = constants.data_files['world']['data_fn']
    file_p = constants.data_files['poland_all']['data_fn']
    file_c = constants.data_files['cities_cases']['data_fn']
    # send_gmail()
    info = ' '
    sources = ''
    ret_val = [
        dbc.Row([
            dbc.Col([
                html.H1(),
                html.H2('covide.pl'),
                html.H6('app info:  0.81 (31.08.2020)' + info),
                html.H6(sources),
                html.H6('Serwer działa od ' + t_start.strftime('%y-%m-%d %H:%M:%S')),
                html.H6('Użytkownik: ' + str(session.get('user_name')) + ',  e-mail: ' + str(session.get('user_email')))
            ]),
        ], className='ml-4 pt-4'),
        dbc.Row([
            dbc.Col([
                html.Br(),
                html.H5('Baza danych światowych'),
                html.Br(),
                html.H6('Zakres danych: od ' + df_w.date.min() + ' do ' + df_w.date.max()),
                html.H6('Liczba lokalizacji: ' + str(len(df_w.location.unique()))),
                html.H6(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(file_w))))
            ], width=4),
            dbc.Col([
                html.Br(),
                html.H5('Baza danych Polska - województwa'),
                html.Br(),
                html.H6('Zakres danych: od ' + df_p.date.min() + ' do ' + df_p.date.max()),
                html.H6('Liczba lokalizacji: ' + str(len(df_p.location.unique()))),
                html.H6(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(file_p))))
            ], width=4),
            dbc.Col([
                html.Br(),
                html.H5('Baza danych Polska - miasta i powiaty'),
                html.Br(),
                html.H6('Zakres danych: od ' + df_c.date.min() + ' do ' + df_c.date.max()),
                html.H6('Liczba lokalizacji: ' + str(len(df_c.location.unique()))),
                html.H6(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(file_c))))
            ], width=4),
        ]),
        dbc.Row([
            dbc.Col([
                html.P('Źródła danych:'),
                html.H6('''Dane z Polski pochodzą ze strony "Covid-19 w Polsce" 
                        prowadzonej przez Michała Rogalskiego, który wraz z zespołem wolontariuszy analizuje 
                        i przenosi do arkuszy Google dane podawane
                        przez Ministerstwo Zdrowia oraz regionalne Stacje Sanitarno-Epidemiologiczne.'''),
                html.H6('Dane światowe pochodzą z dwóch źródeł:'),
                html.H6('''1. Our World in Data - organizacji non-profit, której ideą przewodnią jest zrozumienie
                        wielkich problemów naszego świata i stawienie im czoła.'''),
                html.H6('2. Center for System Science and Engineering (CSSE) przy Uniwersytecie Johnsa Hopkinsa.'),
                html.H6(''),
                html.H6('Serwis covide.pl jest wykonany z wykorzystaniem frameworku Plotly Dash.')
            ])
        ])
    ]

    return ret_val


def calc_results(number, df_p):
    dows = {
        0: 'poniedziałek',
        1: 'wtorek',
        2: 'środa',
        3: 'czwartek',
        4: 'piątek',
        5: 'sobota',
        6: 'niedziela',
    }
    df = df_p[df_p.date >= '2021-01-01'][['date', 'new_cases']].copy()
    df['dow'] = pd.to_datetime(df.date).dt.dayofweek
    df['year'] = pd.to_datetime(df.date).dt.year
    df['week'] = pd.to_datetime(df.date).dt.week
    df = df[(df.week >= 2)].copy()
    total = df.new_cases.sum()
    shares = df.groupby('dow')['new_cases'].sum() / total
    shares_dict = {i: shares[i] for i in range(len(shares))}
    n_week = {i: int(int(number) * shares[i]) for i in range(7)}
    dow_txt = []
    share_txt = []
    nweek_txt = []
    for i in range(7):
        dow_txt.append(dows[i])
        dow_txt.append(html.Br())
        share_txt.append(str(round(shares[i]*100, 2))+'%')
        share_txt.append(html.Br())
        nweek_txt.append(str(n_week[i]))
        nweek_txt.append(html.Br())
    ret_val = [ \
        dbc.Col([
        ], width=1),
        dbc.Col([
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.H5('Dzień tygodnia:'),
            html.Div(dow_txt),
        ], width=2),
        dbc.Col([
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.H5('Udział:'),
            html.Div(share_txt),
        ], width=1),
        dbc.Col([
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.H5('Ilość:'),
            html.Div(nweek_txt),
        ], width=1)
    ]
    return ret_val


@logit
def layout_hrefs(context='polska'):
    if context == 'polska':
        ret_val = \
            html.Div([
                html.Img(src='assets/covid.png', width='50%', height='50%', className='mb-4',
                         style={'background-color': 'transparent'}),
                # html.Br(),
                dbc.Button(
                    "Aktualności",
                    id="stat-collapse-button_id",
                    className="mb-3",
                    color="primary",
                ),
                html.Div(
                    id='stat-collapse-div_id',
                    children=[
                        dbc.Collapse(
                            dbc.Card(dbc.CardBody(dcc.Markdown(constants.news_text))),
                            id="stat-collapse-text_id",
                        ),
                    ],
                    style={'overflow-y': 'auto'}
                ),
                html.A(href='#link_crd1_id',
                       children='Polska - przyrosty dzienne'),
                html.Br(),
                html.A(href='#link_crd2_id',
                       children='Polska - przyrosty (średnia z 28 dni)'),
                html.Br(),
                html.A(href='#link_crd3_id',
                       children='Polska - dane sumaryczne'),
                html.Br(),
                html.A(href='#link_crd4_id',
                       children='Polska - wskaźniki epidemiczne'),
                html.Br(),
                html.A(href='#link_tp_id',
                       children='Polska - tabela porównawcza województw'),
                html.Br(),
                html.A(href='#link_rt_id',
                       children='Polska - wskaźnik reprodukcji R(t)'),
                html.Br(),
                html.A(href='#link_mort_id',
                       children='Wykres porównanawczy liczby zgonów'),
                html.Br(),
                html.A(href='#link_pczprz_id',
                       children='Polska - wykresy (przyrosty dzienne)'),
                html.Br(),
                html.A(href='#link_pczsum_id',
                       children='Polska - wykresy (dane sumaryczne)'),
                html.Br(),
                html.A(href='#link_pczwsk_id',
                       children='Polska - wykresy czasowe wskaźników epidemicznych'),
                html.Br(),
                html.A(href='#link_pmap_id',
                       children='Polska - mapa'),
                html.Br(),
                html.A(href='#link_ptdp_id',
                       children='Polska - tabela danych źródłowych'),
                html.Br(),
                html.A(href='#link_wczpl_id',
                       children='Polska - wykresy czasowe danych źródłowych'),
                html.Br(),
                html.A(href='#link_wczeu_id',
                       children='Europa - wykresy czasowe danych źródłowych'),
                html.Br(),
                html.A(href='#link_wczwr_id',
                       children='Świat - wykresy czasowe danych źródłowych'),
                html.Br(),
            ])
    else:
        ret_val = \
            html.Div([
                html.Img(src='assets/covid.png', width='50%', height='50%', className='mb-4'),
                html.Br(),
                html.A(href='#link_crd1_id',
                       children=context + ' - przyrosty dzienne'),
                html.Br(),
                html.A(href='#link_crd2_id',
                       children=context + ' - przyrosty (średnia z 28 dni)'),
                html.Br(),
                html.A(href='#link_crd3_id',
                       children=context + ' - dane sumaryczne'),
                html.Br(),
                html.A(href='#link_crd4_id',
                       children=context + ' - wskaźniki epidemiczne'),
                html.Br(),
                html.A(href='#link_wczprz_id',
                       children=context + ' - wykresy (przyrosty dzienne)'),
                html.Br(),
                html.A(href='#link_wczsum_id',
                       children=context + ' - wykresy (dane sumaryczne)'),
                html.Br(),
                html.A(href='#link_wczwsk_id',
                       children=context + ' - wykresy czasowe wskaźników epidemicznych'),
                html.Br(),
                html.A(href='#link_wmap_id',
                       children=context + ' - mapa'),
                html.Br(),
                html.A(href='#link_wtdp_id',
                       children=context + ' - tabela danych źródłowych'),
                html.Br(),
                html.A(href='#link_wczwoj_id',
                       children=context + ' - wykresy czasowe danych źródłowych'),
                html.Br(),
            ])
    return ret_val


opis = '''
### Wizualizacja danych epidemicznych SARS-CoV-2

 Analiza i graficzna prezentacja danych epidemicznych pandemii koronawirusa SARS-CoV-2. System narzędzi 
 do przetwarzania aktualnych danych epidemicznych na postać graficzną (wykresy, mapy, tabele). 

##### Zakres geograficzny danych:
- wszystkie kraje świata, kontynenty, regiony, grupy krajów
- Polska w rozbiciu na województwa, miasta i powiaty.
##### Rodzaje prezentowanych danych źródłowych:
- infekcje razem
- śmiertelne razem
- wyzdrowienia razem
- aktywne razem
- testy razem
- infekcje nowe
- śmiertelne nowe
- wyzdrowienia nowe
- testy nowe
##### Wskaźniki epidemiczne:
- okres podwojenia (dni)
- zapadalność (na 100 000 mieszkańców)
- umieralność (na 100 000 mieszkańców)
- śmiertelność (%)
- wykrywalność (%)
##### Rodzaje prezentacji:
- wykresy na osi czasu
- ranking
- dynamika
- bilans
- mapa
- tabela
##### Możliwość dostosowania wyglądu prezentacji do indywidualnych potrzeb:
- dobór kolorystyki, wielkości czcionki, odstępów itp.
- indywidualna parametryzacja układu wykresu
- możliwość zmiany indywidualnych cech wykresu
- możliwość zapisania wykresu (mapy) do lokalnego pliku w formacie wektorowym lub rastrowym
##### Dostosowanie danych do prezentacji:
- uśrednianie kroczące (bez zmiany odstępu czasu między próbkami
- uśrednianie okresowe
- wygładzanie danych z regulowanym stopniem wielomianu 
'''


@logit
def layout_settings():
    data = ''
    v1 = session2settings()
    v2 = data
    v3 = constants.get_defa()
    keys = sorted(v1.keys())
    set1 = []
    set2 = []
    set3 = []
    for key in keys:
        badge1 = ''
        badge2 = ''
        badge3 = ''
        char = ' ___________'
        if key.startswith('color_') and len(key) == 7:
            badge1 = html.Span(char, style=dict(color=v1.get(key)))
            if type(v2) == dict:
                badge2 = html.Span(char, style={'color': v2.get(key)})
            badge3 = html.Span(char, style={'color': v3.get(key)})
        color1 = 'orange'
        color2 = 'red'
        if type(v2) == dict:
            if str(v1.get(key)) == str(v2.get(key)) == str(v3.get(key)):
                color1 = 'white'
                color2 = 'white'
        else:
            if str(v1.get(key)) == str(v3.get(key)):
                color1 = 'white'
                color2 = 'white'
        set1.append(html.H6(
            [html.Span(str(key), style={'color': color1}), ': ', html.Span(str(v1.get(key)), style={'color': color2}),
             badge1]))
        if type(v2) == dict:
            set2.append(html.H6([html.Span(str(key), style={'color': color1}), ': ',
                                 html.Span(str(v2.get(key)), style={'color': color2}), badge2]))
        set3.append(html.H6(
            [html.Span(str(key), style={'color': color1}), ': ', html.Span(str(v3.get(key)), style={'color': color2}),
             badge3]))

    ret_val = [
        dbc.Row([
            dbc.Col([
                html.H3('Ustawienia aktualne:'),
                html.Div(id='set1', children=set1, style={'height': '600px', 'overflow-y': 'scroll'}),
            ], width=3),
            dbc.Col([
                html.H3('Moje ustawienia:'),
                html.Div(id='set2', children=set2, style={'height': '600px', 'overflow-y': 'scroll'}),
            ], width=3),
            dbc.Col([
                html.H3('Ustawienia domyślne:'),
                html.Div(id='set3', children=set3, style={'height': '600px', 'overflow-y': 'scroll'}),
            ], width=3),
            dbc.Col([
                html.H3('Skale kolorów:'),
                html.Div(style={'height': '600px', 'overflow-y': 'scroll'},
                         children=dcc.Graph(figure=px.colors.sequential.swatches())),
            ], width=3),
        ], className='ml-4 pt-4'),
    ]
    return ret_val


def layout_title(text, font_size, color, posx=0.5, posy=0.0, pad_t=20, pad_r=0, pad_b=50, pad_l=0, xanchor='center'):
    ret_val = dict(
        x=posx,
        y=posy,
        xanchor=xanchor,
        yanchor='top',
        pad={'t': pad_t, 'b': pad_b, 'l': pad_l, 'r': pad_r, },
        text=text,
        font=dict(size=font_size, color=color),
    )
    return ret_val


def layout_legend(y=0, legend_place='lewo'):
    if legend_place == 'lewo':
        orientation = 'v'
        x = 0
    elif legend_place == 'prawo':
        orientation = 'v'
        x = 1
    else:  # legend_place == 'poziomo':
        orientation = 'h'
        x = 0
    ret_val = dict(
        x=x,
        y=y,
        orientation=orientation,
        itemclick='toggleothers',
    )
    return ret_val


@logit
def layout_login():
    if session_or_die('mode', 'demo') == 'demo':
        ret_val = ['Login']
    else:
        if is_logged_in():
            ret_val = ['Logout', dcc.Location(pathname="/logout", id="someid3")]
        else:
            ret_val = ['Login', dcc.Location(pathname="/login", id="someid2")]

    return ret_val


@logit
def layout_kolory():
    if session_or_die('color_order', 'brak') == 'brak':
        scale = constants.color_scales[constants.get_defa()['color_order']]
    else:
        scale = constants.color_scales[session.get('color_order')]
    ret_val = []
    for x in scale:
        element = html.Button(className='square', style={'background-color': str(x)})
        ret_val.append(element)
    return ret_val


###########################
# Update set for location
###########################

@logit
def layout_set(DF, location, scope):
    template = get_template()
    settings = session2settings()
    traces0 = constants.trace_names_list['data'] + constants.trace_names_list['calculated']
    traces = []
    for chart in traces0:
        if scope == 'poland' and constants.trace_props[chart[0]]['disable_pl']:
            continue
        if scope == 'world' and constants.trace_props[chart[0]]['disable_world']:
            continue
        if scope == 'cities' and constants.trace_props[chart[0]]['disable_cities']:
            continue
        traces.append(chart)
    cols = 4
    row_height = 250
    rows = int((len(traces) - 1) / cols) + 1
    titles = [x[1] for x in traces]
    figure = make_subplots(rows=rows,
                           cols=cols,
                           subplot_titles=titles,
                           print_grid=False,
                           )
    row = 1
    col = 0
    for chart in traces:
        filtered = filter_data(DF,
                               loc=location,
                               trace=chart[0],
                               total_min=1,
                               scope=scope,
                               data_modifier=1,
                               from_date=settings['from_date'],
                               to_date=settings['to_date'],
                               duration_d=14,
                               duration_r=21,
                               win_type='równe wagi'
                               )
        filtered.sort_values(by=['location', 'date'], inplace=True)
        filtered.reset_index(drop=True, inplace=True)
        xvals = list(filtered['date'])

        # Obliczenie uśredniedniania

        yvals = list(filtered[chart[0]].rolling(7, min_periods=1).mean())
        yvals = [0 if math.isnan(i) else i for i in yvals]
        yvals2 = list(filtered[chart[0]])
        yvals2 = [0 if math.isnan(i) else i for i in yvals2]
        fig_data = go.Scatter(x=xvals,
                              y=yvals,
                              opacity=1,
                              line=dict(color='yellow', width=1),
                              name=chart[1])
        fig_data2 = go.Bar(x=xvals,
                           y=yvals2,
                           opacity=0.3,
                           name=chart[1])
        col += 1
        if col > cols:
            row += 1
            col = 1
        figure.add_trace(fig_data, row=row, col=col)
        figure.add_trace(fig_data2, row=row, col=col)

    height = 100 + row_height * row
    figure.update_layout(height=height,
                         template=template,
                         title=dict(text=location + ' - wykresy danych źródłowych (7-dniowa średnia ruchoma)'),
                         showlegend=False),
    config = {
        'responsive': True,
        'displayModeBar': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id=location, figure=figure, config=config)
    return dbc.Col(fig, width=12)


#####################################################
# Prosty wykres dla jednego typu i wielu lokalizacji
#####################################################


@logit
def layout_timeline_type(DF, locations, data_type, scope,
                         from_date='', opt={}, height=300):
    traces = []
    errors = []
    defaults2session()
    beautify = opt.get('beautify', 'None')
    modify = opt.get('modify', 'none')
    back_color = opt.get('back_color', constants.get_defa()['color_1'])
    font_color = opt.get('font_color', constants.get_defa()['color_3'])
    font_size = opt.get('font_size', constants.get_defa()['font_size_xy'])
    title_color = opt.get('title_color', constants.get_defa()['color_3'])
    titlexpos = float(opt.get('titlexpos', constants.get_defa()['titlexpos']))
    titleypos = float(opt.get('titleypos', constants.get_defa()['titleypos']))
    title_size = opt.get('title_size', constants.get_defa()['font_size_title'])
    chart_type = opt.get('chart_type', 'bar')
    chart_colors = opt.get('chart_colors', constants.get_defa()['color_order'])
    annotations = opt.get('annotations', constants.get_defa()['annotations'])

    tail = ''
    if modify.lower() == 'population':
        data_modifier = 2
    else:
        data_modifier = 1
    if beautify.lower() == 'smooth':
        timeline_opt = ['wygladzanie']
    elif beautify.lower() == 'mean':
        timeline_opt = ['usrednianie']
    else:
        timeline_opt = []
    anno = []
    for location in locations:
        if constants.trace_props[data_type]['category'] == 'data':
            tail = {1: '', 2: ' (na 100 000 osób)'}[data_modifier]
        else:
            tail = ''
        if from_date == '':
            from_date = DF[scope]['date'].min()
        filtered_df = filter_data(DF,
                                  loc=location,
                                  trace=data_type,
                                  total_min=0,
                                  scope=scope,
                                  data_modifier=data_modifier,
                                  timeline_opt=timeline_opt,
                                  from_date=from_date,
                                  duration_d=1,
                                  duration_r=1,
                                  win_type='równe wagi'
                                  )
        if len(filtered_df) == 0:
            errors.append('Brak danych dla: ' + scope + ', ' + location + ', ' + data_type)

        # konstrukcja nitki wykresu

        anno_1 = {}
        if annotations == 'min':
            row = filtered_df.loc[filtered_df['data'].idxmin()]
            anno_1['x'] = row['date']
            anno_1['y'] = row['data']
            anno_1['text'] = location + '<br><b>' + str(row['data'])
        elif annotations == 'max':
            row = filtered_df.loc[filtered_df['data'].idxmax()]
            anno_1['x'] = row['date']
            anno_1['y'] = row['data']
            anno_1['text'] = location + '<br><b>' + str(row['data'])
        elif annotations == 'last':
            row = filtered_df.iloc[len(filtered_df) - 1]
            anno_1['x'] = row['date']
            anno_1['y'] = row['data']
            anno_1['text'] = location + '<br><b>' + str(row['data'])
        fig_data = dict(
            x=list(filtered_df['date'].astype('datetime64[ns]')),
            y=list(filtered_df['data']),
            type=chart_type,
            name=location,
            opacity=1
        )
        anno.append(anno_1)
        traces.append(fig_data)
    if len(errors) > 0:
        return errors[0]

    if len(locations) == 1:
        head = locations[0] + '<br><sup>'
    else:
        head = ''
    figure = {
        'data': traces,
        'layout': dict(
            margin=dict(l=50, r=10, b=50, t=10, pad=4),
            template='plotly_dark',
            xaxis={'tickfont': {'size': font_size, 'color': font_color}},
            yaxis={'tickfont': {'size': font_size, 'color': font_color}},
            itemclick='toggleothers',
            annotations=[x for x in anno],
            legend=dict(x=0.02, y=0.90,
                        orientation='v',
                        itemclick='toggleothers',
                        font=dict(size=14, color='white')),
            paper_bgcolor=back_color,
            plot_bgcolor=back_color,
            colorway=chart_colors,
            title=layout_title(text=head + get_trace_name(data_type).capitalize() + '<br><sup>' + tail,
                               font_size=title_size,
                               color=title_color,
                               xanchor='center',
                               posx=titlexpos,
                               posy=titleypos,
                               ),
        )
    }
    config = {
        'displaylogo': False,
        'responsive': True,
        'locale': 'pl-PL',
    }
    if height == 'auto':
        children = dcc.Graph(id=data_type, config=config, figure=figure, style={'height': '100vh'})
    else:
        children = dcc.Graph(id=data_type, config=config, figure=figure, style={'height': height})

    return children


#####################################################
# Prosty wykres dla jednej lokalizacji i wielu typów
#####################################################

# @logit
def layout_timeline_location(DF, location, data_types, scope,
                             from_date='', opt={}, height=600):
    template = get_template()
    settings = session2settings()
    traces = []
    errors = []
    beautify = opt.get('beautify', 'None')
    modify = opt.get('modify', 'none')
    chart_type = opt.get('chart_type', 'line')
    chart_colors = opt.get('chart_colors', constants.get_defa()['color_order'])
    if modify.lower() == 'population':
        data_modifier = 2
    else:
        data_modifier = 1
    anno = []
    for data_type in data_types:
        if from_date == '':
            from_date = DF[scope]['date'].min()
        filtered_df = filter_data(DF,
                                  loc=location,
                                  trace=data_type,
                                  total_min=0,
                                  scope=scope,
                                  data_modifier=data_modifier,
                                  from_date=from_date,
                                  duration_d=1,
                                  duration_r=1,
                                  win_type='równe wagi'
                                  )
        if len(filtered_df) == 0:
            errors.append('Brak danych dla: ' + scope + ', ' + location + ', ' + chart_type)

        # konstrukcja nitki wykresu

        anno_1 = {}
        row = filtered_df.iloc[len(filtered_df) - 1]
        anno_1['x'] = row['date']
        anno_1['y'] = row['data']
        anno_1['text'] = location + '<br><b>' + str(row['data'])
        fig_data = dict(
            x=list(filtered_df['date'].astype('datetime64[ns]')),
            y=list(filtered_df['data']),
            type=chart_type,
            name=get_trace_name(data_type),
            opacity=1
        )
        anno.append(anno_1)
        traces.append(fig_data)
    if len(errors) > 0:
        return errors[0]

    figure = {
        'data': traces,
        'layout': dict(
            margin=dict(l=50, r=10, b=50, t=120, pad=4),
            template=template,
            itemclick='toggleothers',
            annotations=[x for x in anno],
            legend=dict(x=0.02, y=1.1,
                        orientation='h',
                        itemclick='toggleothers',
                        font=dict(size=14, color='white')),
            colorway=chart_colors,
            title=dict(text=location),
        )
    }
    config = {
        'displaylogo': False,
        'responsive': True,
        'locale': 'pl-PL',
    }
    if height == 'auto':
        children = dcc.Graph(id=location, config=config, figure=figure, style={'height': '90vh'})
    else:
        children = dcc.Graph(id=location, config=config, figure=figure, style={'height': height})

    return children


#############################################################
# Wykres multiple-axes dla wielu typów i jednej lokalizacji
#############################################################

@logit
def layout_timeline_multi_axes(DF, location, chart_types, scope, height=300):
    import plotly.graph_objects as go

    df = DF[scope]
    filtered_df = df[df['location'].str.lower() == location.lower()][['date', 'location'] + chart_types].copy()

    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
    figure = go.Figure()
    yaxis_n = 0
    for chart_type in chart_types:
        yaxis_n += 1

        # konstrukcja nitki wykresu

        if 'new_' in chart_type:
            filtered_df[chart_type] = round(filtered_df[chart_type].rolling(7, min_periods=1).mean(), 1)
        if yaxis_n == 1:
            figure.add_trace(go.Scatter(
                x=list(filtered_df['date'].astype('datetime64[ns]')),
                y=list(filtered_df[chart_type]),
                line=dict(color=colors[0]),
                name=constants.trace_props[chart_type]['title'],
            ))
            figure.update_layout(
                yaxis=dict(
                    title=constants.trace_props[chart_type]['title'],
                    # line=dict(color=colors[0]),
                    titlefont=dict(color=colors[0]),
                    tickfont=dict(color=colors[0])
                )
            )
        else:
            figure.add_trace(go.Scatter(
                x=list(filtered_df['date'].astype('datetime64[ns]')),
                y=list(filtered_df[chart_type]),
                line=dict(color=colors[yaxis_n - 1]),
                name=constants.trace_props[chart_type]['title'],
                yaxis="y" + str(yaxis_n)
            ))
            if yaxis_n == 2:
                figure.update_layout(
                    yaxis2=dict(
                        title=constants.trace_props[chart_type]['title'],
                        titlefont=dict(color=colors[1]),
                        tickfont=dict(color=colors[1]),
                        anchor="free",
                        overlaying="y",
                        side="left",
                        position=0.15
                    ),
                )
            elif yaxis_n == 3:
                figure.update_layout(
                    yaxis3=dict(
                        title=constants.trace_props[chart_type]['title'],
                        titlefont=dict(color=colors[2]),
                        tickfont=dict(color=colors[2]),
                        anchor="x",
                        overlaying="y",
                        side="right",
                    ),
                )
            elif yaxis_n == 4:
                figure.update_layout(
                    yaxis4=dict(
                        title=constants.trace_props[chart_type]['title'],
                        titlefont=dict(color=colors[3]),
                        tickfont=dict(color=colors[3]),
                        anchor="free",
                        overlaying="y",
                        side="right",
                        position=0.85
                    ),
                )

    figure.update_layout(
        xaxis=dict(domain=[0.3, 0.7]),
        # title_text="multiple y-axes example",
        legend=dict(orientation='h', y=1.15),
        # width=800,
        margin=dict(l=10, r=10, b=20, t=40, pad=4),
        template='plotly_dark',
        paper_bgcolor=constants.get_defa()['color_1'],
        plot_bgcolor=constants.get_defa()['color_1'],
    )
    config = {
        'responsive': True,
        'displayModeBar': False,
        'locale': 'pl-PL',
    }
    if height == 'auto':
        children = dcc.Graph(id=location, config=config, figure=figure, style={'height': '90vh'})
    else:
        children = dbc.Col(dcc.Graph(id=location, figure=figure, config=config))

    return children


# Wykres z 2 osiami Y

#################
# Prosta tabela
#################
@logit
def layout_table(DF, locations, chart_types, scope):
    df = DF[scope]
    filtered_df = df[df.location.isin(locations)][['date', 'location'] + chart_types].copy()
    if scope == 'poland':
        filtered_df['location'] = filtered_df['location'].str.replace('Polska', ' Polska', regex=False)
    if scope == 'poland':
        disabled = 'disable_pl'
    elif scope == 'cities':
        disabled = 'disable_cities'
    else:
        disabled = 'disable_world'
    data = filtered_df.to_dict('records')
    columns = [
                  {"title": "Data",
                   "field": "date",
                   "hozAlign": "left"},
                  {"title": "Lokalizacja",
                   "field": "location",
                   'headerFilter': 'select',
                   'headerFilterPlaceholder': "Wybierz lokalizację...",
                   'headerFilterParams': {'values': True},
                   'headerFilterFunc': "="},
              ] + [{"title": [constants.trace_props[i]['title']], "field": i} for i in chart_types
                   if not constants.trace_props[i][disabled]]
    options = dict(
        height='60vh',
        initialSort=[
            {'column': 'date', 'dir': 'desc'},
            {'column': 'location', 'dir': 'asc'},
        ],
        # initialFilter=[{'field': "location", 'type': "=", 'value': "Polska"}],
    )
    figure = html.Div([
        dash_tabulator.DashTabulator(
            id='tabulator',
            columns=columns,
            data=data,
            options=options,
        ),
    ], id='example-table-theme')

    return figure


######################
# Tabela porównawcza
######################
@logit
def layout_table_rank(DF, locations, chart_types, scope, height='60vh'):
    df = DF[scope]
    date_max = df['date'].max()
    # filtered_df = df[(df['location'].isin(locations))][['date', 'location', 'population']+chart_types].copy()
    filtered_df = df[(df['date'] == date_max) & (df['location'].isin(locations))][
        ['date', 'location', 'population'] + chart_types].copy()
    chart_types_1 = {}
    for type in chart_types:
        if 'total_' in type or 'new_' in type:
            type_1 = type + '_100'
            filtered_df[type_1] = round(filtered_df[type] / filtered_df['population'] * 100000, 2)
            chart_types_1[type] = type_1

    data = filtered_df.to_dict('records')
    columns = [
                  {"title": "Data",
                   "field": "date",
                   "hozAlign": "left",
                   'headerFilter': 'select',
                   'headerFilterPlaceholder': "Data ...",
                   },
                  {"title": "Lokalizacja",
                   "field": "location"},
                  {"title": "Ludność",
                   "field": "population"},
              ] + [{"title": [constants.trace_props[i]['title']], "field": i} for i in chart_types] + \
              [{"title": [constants.trace_props[i]['title'] + ' / 100000'], "field": chart_types_1[i]} for i in
               list(chart_types_1.keys())]

    options = dict(
        height=height,
        initialSort=[
            {'column': 'location', 'dir': 'asc'},
            {'column': 'date', 'dir': 'desc'},
        ],
    )
    figure = html.Div([
        dash_tabulator.DashTabulator(
            id='tabulator',
            columns=columns,
            data=data,
            options=options,
        ),
    ], id='example-table-theme-1')

    return figure


@logit
def layout_more_controls():
    ret_val = \
        dbc.Row(children=[
            dbc.Col([
                badge('Klasa prezentacji', 'klasaprezentacji'),
                dbc.Select(id='more_class_id',
                           options=[
                               {'label': '--wybierz kategorię prezentacji--', 'value': 'none'},

                               {'label': '--- podstawowe', 'value': 'podstawowe'},
                               {'label': '--- ANALIZY', 'value': 'analizy'},
                               {'label': '--- SZCZEPIENIA (MZ)', 'value': 'szczepienia MZ'},
                               {'label': '--- SZCZEPIENIA (ECDC)', 'value': 'szczepienia ECDC'},
                               {'label': '--- HIT', 'value': 'HIT'},
                               {'label': '--- ZGONY NADMIAROWE', 'value': 'nadmiarowe'},
                               {'label': '--- APPLE', 'value': 'Apple'},
                               {'label': '--- INNE', 'value': 'inne'},
                               {'label': '--- ZGONY BASIW', 'value': 'BASIW'},
                               {'label': '--- INFEKCJE BASIW', 'value': 'cases BASIW'},
                               {'label': '--- LC, PIMS', 'value': 'LC, PIMS'},
                               {'label': '--- MZ infekcje po szczepieniu', 'value': 'reinf MZ'},
                               {'label': '--- POZOSTAŁE', 'value': 'pozostałe'},
                           ],
                           value='--wybierz kategorię prezentacji--',
                           persistence=False,
                           )
            ], className=top_opt_class, width=2),
            dbc.Col([
                badge('Rodzaj prezentacji', 'rodzajprezentacji'),
                dbc.Select(id='more_type_id',
                           options=[{'label': '--wybierz prezentację--'}],
                           value='--wybierz prezentację--',
                           persistence=False,
                           )
            ], className=top_opt_class, width=4),
            dbc.Col(id='more_params_1_id', className=top_opt_class, width=2),
            dbc.Col(id='more_params_2_id', className=top_opt_class, width=2),
            dbc.Col(id='more_params_3_id', className=top_opt_class, width=2),
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ], className=top_row_class, style={'backgroundColor': session_or_die('color_11', get_default_color(11))})

    return ret_val


@logit
def layout_overview_controls():
    ret_val = \
        dbc.Row(children=[
            dbc.Col([
                badge('Rodzaj informacji', 'rodzajinformacji'),
                dbc.Select(id='overview_type_id',
                           options=[
                               {'label': '--wybierz rodzaj informacji--', 'value': 'none'},

                               {'label': 'Wykresy danych źródłowych dla wybranego państwa (wdzp)', 'value': 'wdzp'},
                               {'label': 'Wykresy danych źródłowych dla wybranego województwa (wdzw)', 'value': 'wdzw'},
                               {'label': 'Wykresy danych źródłowych dla wybranego powiatu (wdzc)', 'value': 'wdzc'},
                               {'label': 'Tabela danych źródłowych dla wybranego państwa (tdzp)', 'value': 'tdzp'},
                               {'label': 'Tabela danych źródłowych dla wybranego województwa (tdzp)', 'value': 'tdzw'},
                               {'label': 'Wykorzystanie zasobów (łóżka, respiratory) (wz)', 'value': 'wz'},
                           ],
                           value='none',
                           persistence=False,
                           )
            ], className=top_opt_class, width=5),
            dbc.Col(id='overview_params_1_id', className=top_opt_class, width=2),
            dbc.Col(id='overview_params_2_id', className=top_opt_class, width=2),
            dbc.Col(id='overview_params_3_id', className=top_opt_class, width=2),
            html.Div(id='overview_1_id'),
            html.Div(id='overview_2_id'),
            html.Div(id='overview_3_id'),
        ], className=top_row_class, style={'backgroundColor': session_or_die('color_11', get_default_color(11))})

    return ret_val


