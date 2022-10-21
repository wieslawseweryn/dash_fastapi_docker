import dash_bootstrap_components as dbc
from datetime import datetime as dt
from dash import html

from layouts import layout_set, layout_table, layout_timeline_location
from util import read_kolory, logit

start_server_date = str(dt.today())[:16]

kolory = read_kolory()

nomargins = 'ml-0 pl-0'

def layout_title(text, font_size, color, posx=0.5, posy=0.0, pad_t=20, pad_r=0, pad_b=50, pad_l=0, xanchor='center'):
    ret_val = dict(
        x=posx,
        y=posy,
        xanchor=xanchor,
        yanchor='top',
        pad={'t': pad_t, 'b': pad_b, 'l': pad_l, 'r': pad_r,},
        text=text,
        font=dict(size=font_size, color=color),
    )
    return ret_val


##################################################
# Wykres danych źródłowych dla wybranego państwa
##################################################

@logit
def layout_overview_wdzp(DF, location, _1, _2):
    children = dbc.Col(layout_set(DF, location, 'world')),
    return children


#####################################################
# Wykres danych źródłowych dla wybranego województwa
#####################################################

@logit
def layout_overview_wdzw(DF, location, _1, _2):
    children = dbc.Col(layout_set(DF, location, 'poland')),
    return children


#####################################################
# Wykres danych źródłowych dla wybranego powiatu
#####################################################

@logit
def layout_overview_wdzc(DF, location, _1, _2):
    children = dbc.Col(layout_set(DF, location, 'cities')),
    return children


##################################################
# Tabela danych źródłowych dla wybranego państwa
##################################################

@logit
def layout_overview_tdzp(DF, location, _1, _2):
    children = dbc.Col([
                    html.H5(id='link_wtdp_id', children=location + ' - tabela podstawowych danych źródłowych (mogą być niepełne)'),
                    html.Div([
                        layout_table(DF, DF['world'][DF['world']['location'] == location]['location'].unique(),
                                     ['new_cases', 'total_cases', 'new_deaths', 'total_deaths',
                                      'zapadalnosc', 'umieralnosc', 'smiertelnosc'],
                                     'world'),
                    ]),
                ], width=12),
    return children


##################################################
# Tabela danych źródłowych dla wybranego państwa
##################################################

@logit
def layout_overview_tdzp(DF, location, _1, _2):
    children = dbc.Col([
                    html.H5(id='link_wtdp_id', children=location + ' - tabela podstawowych danych źródłowych (mogą być niepełne)'),
                    html.Div([
                        layout_table(DF, DF['world'][DF['world']['location'] == location]['location'].unique(),
                                     ['new_cases', 'total_cases', 'new_deaths', 'total_deaths',
                                      'zapadalnosc', 'umieralnosc', 'smiertelnosc'],
                                     'world'),
                    ]),
                ], width=12),
    return children


######################################################
# Tabela danych źródłowych dla wybranego województwa
######################################################

@logit
def layout_overview_tdzw(DF, location, _1, _2):
    children = dbc.Col([
                    html.H5(id='link_wtdp_id', children=location + ' - tabela podstawowych danych źródłowych (mogą być niepełne)'),
                    html.Div([
                        layout_table(DF, DF['cities'][DF['cities']['wojew'] == location.lower()]['location'].unique(),
                                     ['new_cases', 'total_cases', 'new_deaths', 'total_deaths',
                                      'zapadalnosc', 'umieralnosc', 'smiertelnosc'],
                                     'cities'),
                    ]),
                ], width=12),
    return children


##########################
# Wykorzystanie zasobów
##########################

@logit
def layout_overview_wz(DF, location, _1, _2):
    if location == 'Polska':
        children = [
            dbc.Col([
                html.H5(id='link_pzl_id', children='Zajętość łóżek'),
                layout_timeline_location(DF, 'Polska', ['hosp_patients', 'total_beds'], 'resources'),
            ], width=4, className=nomargins),
            dbc.Col([
                html.H5(id='link_pzr_id', children='Zajętość respiratorów'),
                layout_timeline_location(DF, 'Polska', ['icu_patients', 'total_resp'], 'resources'),
            ], width=4, className=nomargins),
            dbc.Col([
                html.H5(id='link_puz_id', children='Użycie zasobów (%)'),
                layout_timeline_location(DF, 'Polska', ['used_resp', 'used_beds'], 'resources'),
            ], width=4, className=nomargins),
        ]
    else:
        children = [
            dbc.Col([
                html.H5(id='link_wzl_id', children='Zajętość łóżek'),
                layout_timeline_location(DF, location, ['hosp_patients', 'total_beds'], 'resources'),
            ], width=4, className=nomargins),
            dbc.Col([
                html.H5(id='link_wzr_id', children='Zajętość respiratorów'),
                layout_timeline_location(DF, location, ['icu_patients', 'total_resp'], 'resources'),
            ], width=4, className=nomargins),
            dbc.Col([
                html.H5(id='link_wuz_id', children='Użycie zasobów (%)'),
                layout_timeline_location(DF, location, ['used_resp', 'used_beds'], 'resources'),
            ], width=4, className=nomargins),
        ]
    return children


