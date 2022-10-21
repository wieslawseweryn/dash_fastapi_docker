from shutil import copyfile

import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc

from fastapi import FastAPI, Request
from fastapi.middleware.wsgi import WSGIMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import FileResponse

import uvicorn

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,
                external_stylesheets=[external_stylesheets, dbc.themes.SOLAR],
                prevent_initial_callbacks=True,
                )

from config import session

import os
from dash.dependencies import Input, Output, State, ALL
import pandas as pd
from figures import update_figures_map_mapbox, update_figures_rankings, update_figures_dynamics, \
    update_figures_timeline_types, update_figures_timeline_places, update_figures_table, \
    update_figures_core
from layouts_more import layout_more_wplz, layout_more_tlzp, layout_more_wrrw, layout_more_nzwall, \
    layout_more_tlzgw, \
    layout_more_cmpf, layout_more_ecdc_vacc, layout_more_ecdc_vacc3, \
    layout_more_mz_vacc1, layout_more_mz_nop, layout_more_ireland, \
    layout_more_ecdc_vacc0, layout_more_ecdc_vacc5, layout_more_ecdc_vacc6, \
    layout_more_ecdc_vacc7, layout_more_ecdc_vacc8, layout_more_mz_vacc2, layout_more_ecdc_vacc9, \
    layout_more_mz_vacc2a, layout_more_mz_vacc3, layout_more_mz_vacc4, \
    layout_more_mz_vacc5, layout_more_mz_psz, layout_more_mz_psz2, layout_more_hit1, layout_more_mz_psz3, \
    layout_more_mz_psz2t, layout_more_mz_vacc2b, layout_more_mz_vacc6, \
    layout_more_ecdc_vacc11, layout_more_hit2, layout_more_hit3, layout_more_nzwcmp, layout_more_hit4, \
    layout_more_mz_cases1, layout_more_mz_cases3, layout_more_heatmap, layout_more_mzage2, layout_more_2axes, \
    layout_more_mz_cases5, layout_more_mz_reinf, layout_more_cmpf_2, layout_more_mz_cases6, layout_more_mz_reinf2, \
    layout_more_mz_cases7, layout_more_nzwcmp_1, layout_more_hit5, layout_more_multimap_woj, layout_more_multimap_pow, \
    layout_more_analiza_poznan, layout_more_mz_reinf3, layout_more_mz_reinf4, \
    layout_more_mz_reinf5, layout_more_analiza_2, layout_more_chronomap_woj, layout_more_chronomap_pow, \
    layout_more_mz_cases8, layout_more_wplz0, layout_more_nz_heatmap, layout_more_mzage3, \
    layout_more_mz_cases10, layout_more_mzage4, layout_more_mz_cases11, layout_more_mz_psz5, layout_more_mzage1, \
    layout_more_mz_cases2, layout_more_mz_cases4, layout_more_mz_cases12, layout_more_histogram, \
    layout_more_map_discrete_woj, layout_more_map_discrete_pow, layout_more_cfr1, layout_more_cfr2, layout_more_cfr3, \
    layout_more_mz_cases13, layout_more_mz_cases14, layout_more_mz_cases15, layout_more_lc1, layout_more_lc2, \
    layout_more_ecdc_vacc1, layout_more_ecdc_vacc2, layout_more_nzwcmp_2, layout_more_wplz1, \
    layout_more_analiza_dni_tygodnia, layout_more_wplz2, layout_more_wplz3
from layouts_overview import layout_overview_wdzp, layout_overview_wdzw, layout_overview_tdzp, \
    layout_overview_tdzw, layout_overview_wz, layout_overview_wdzc
import constants
from constants import continents, eu_countries_list, wojewodzkie_list, pl_icase, \
    arab_countries_list, latin_america_list, wojewodztwa_list, subgroups, demoludy_list, skandynawskie_list, data_files, \
    wojewodztwa_pn_zach, wojewodztwa_pd_wsch
from util import list_to_options, trace_options, read_kolory, defaults2session, modal_message, \
    get_help, reset_colors, is_logged_in, controls2session, logit, debug, badge, \
    fields_to_options

from layouts import layout_main, layout_info, layout_settings, layout_kolory, layout_hrefs
from datetime import timedelta, datetime
import datetime
import random
import uuid

kolory = read_kolory()
t_start = datetime.datetime.now()
persistence = True
soft_hyphen = '\u00AD'

path_df_w = data_files['world']['data_fn']
path_df_p = data_files['poland_all']['data_fn']
path_df_c = data_files['cities_cases']['data_fn']
path_df_r = data_files['poland_resources']['data_fn']
path_df_rt = data_files['poland_rt']['data_fn']
path_df_m = data_files['mortality_eurostat']['data_fn']

df_w = pd.read_csv(path_df_w)
df_p = pd.read_csv(path_df_p)
df_c = pd.read_csv(path_df_c)
df_r = pd.read_csv(path_df_r)
df_m = pd.read_csv(path_df_m)
df_rt = pd.read_csv(path_df_rt)

DF = {'world': df_w,
      'poland': df_p,
      'cities': df_c,
      'resources': df_r,
      'rt': df_rt,
      'mort': df_m,
      }

layout = html.Div(id='main_layout', children=layout_main(DF))

@app.callback(
    Output("modal_body", "is_open"),
    [Input("modal_close", "n_clicks")],
    [State("modal_body", "is_open")],
)
@logit
def on_modal_close(n1, is_open):
    if n1:
        return not is_open
    return is_open


@app.callback(
    [
        Output({'type': 'paleta', 'index': 'paleta_id'}, 'children'),
    ],
    [
        Input({'type': 'params', 'index': 'colors_id'}, 'value'),
        Input({'type': 'params', 'index': 'scale_id'}, 'value'),
    ]
)
@logit
def on_colors_change(_1, scale):
    ctx = dash.callback_context
    ret_val = dash.no_update
    context = ctx.triggered[0]['prop_id']
    className = 'threeColumns'
    if context in ['.', '{"index":"scale_id","type":"params"}.value']:
        items = []
        if scale != '--none--':
            if scale == '--all--':
                for skala in kolory:
                    for kolor in kolory.get(skala):
                        items.append(
                            dbc.DropdownMenuItem(kolor['label'],
                                                 id={'type': 'paleta', 'index': kolor['value']},
                                                 style={'background-color': kolor['value']}))
                className = 'sixColumns'
            else:
                for kolor in kolory.get(scale):
                    items.append(
                        dbc.DropdownMenuItem(kolor['label'],
                                             id={'type': 'paleta', 'index': kolor['value']},
                                             style={'background-color': kolor['value']}))
        ret_val = html.Div(children=items, className=className)
        return [ret_val]

    return [ret_val]


# Zmiana palety kolorów
#
@app.callback(
    [
        Output('go', 'value'),
    ],
    [
        Input({'type': 'params', 'index': 'template_id'}, 'value'),
    ],
)
@logit
def on_change_template(template):
    session['template'] = template
    session['template_change'] = True
    for key in constants.user_templates[template].keys():
        session[key] = constants.user_templates[template].get(key)
    ret_val = random.random()
    return [ret_val]


#####################
# popover help close
#####################

@app.callback(
    [Output('popover', 'is_open')],
    [Input('close_popover', 'n_clicks')]
)
@logit
def on_popover_close(_1):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == '.':
        return [dash.no_update]
    return [False]


#####################
# choose color2
#####################

@app.callback(
    [
        Output('palettes_id', 'style'),
        # on_update_figures
        Output('go3', 'value')
    ],
    [Input({'type': 'color', 'index': ALL}, 'n_clicks')],
    [State({'type': 'params', 'index': 'colors_id'}, 'value')]
)
@logit
def on_choose_color_2(_1, colors2):
    ret_val_go2 = dash.no_update
    ctx = dash.callback_context
    if 'rgb' in ctx.triggered[0]['prop_id']:
        chosen_color = eval(ctx.triggered[0]['prop_id'].replace('.n_clicks', ''))['index']
        session[colors2] = chosen_color
        ret_val_go2 = random.random()
    return [dash.no_update, ret_val_go2]


#####################
# popover help open
#####################

@app.callback(
    [Output('pop', 'children')],
    [Input({'type': 'button_help', 'index': ALL}, 'n_clicks')],
    [State('popover', 'is_open')]
)
@logit
def on_toggle_popover_help(_1, _2):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == '.':
        return [dash.no_update]
    if session.get('prevent_help'):
        session['prevent_help'] = False
        return [dash.no_update]
    wzorzec = ctx.triggered[0]['prop_id']
    _id = wzorzec.replace('{"index":"', '')
    _id = _id.replace('","type":"button"}.n_clicks', '')
    ret_val = [dbc.Popover(
        [
            dbc.PopoverHeader("Pomoc programu COVID-19"),
            dbc.PopoverBody([
                dcc.Markdown(get_help(_id), style={'color': 'white'}),
                # dcc.Markdown(help_text.get(_id), style={'color': 'white'}),
                html.Div(className='text-right', children=[
                    html.I(className="fa fa-sign-out fa-2x", id='close_popover',
                           style={'cursor': 'pointer', 'color': 'yellow'}),
                ]),
            ]),
        ],
        id='popover',
        is_open=True,
        target='buttler',
        style={'background-color': '#367dd9', 'font-size': '16px'}
    )]

    return [ret_val]

# kolory Adama
# background: #a9bdbd i czcionka - white lub black
# background: #879799 i czcionka - white lub black
# background: #367dd9 i czcionka (color) - white


###############
# Date Buttons
###############
@app.callback(
    [
        Output({'type': 'params', 'index': 'map_date_id'}, 'date'),
        Output('go2', 'value'),
    ],
    [
        Input('rank_date_prev_id', 'n_clicks'),
        Input('rank_date_next_id', 'n_clicks'),
        Input('map_date_prev_id', 'n_clicks'),
        Input('map_date_next_id', 'n_clicks'),
    ],

    [
        State({'type': 'params', 'index': 'map_date_id'}, 'date'),
    ]
)
@logit
def on_date_button(_1, _2, _5, _6, m_date):
    ctx = dash.callback_context.triggered[0]['prop_id']
    ret_val_m = dash.no_update
    ret_val_go2 = dash.no_update
    if ctx == 'map_date_next_id.n_clicks':
        ret_val_m = datetime.datetime.strptime(m_date[:10], '%Y-%m-%d') + timedelta(days=1)
    elif ctx == 'map_date_prev_id.n_clicks':
        ret_val_m = datetime.datetime.strptime(m_date[:10], '%Y-%m-%d') - timedelta(days=1)
    return [ret_val_m, ret_val_go2]


# ###############
# # Download
# ###############
@app.callback([
        Output("download", "data"),
    ],
    [
        Input("download_button_id", "n_clicks")
    ],
    [
        State("download_format_id", "value")
    ],
    )
@logit
def on_download(_1, file_format):
    ctx = dash.callback_context.triggered[0]['prop_id']
    ret_val_d = dash.no_update
    if ctx == 'download_button_id.n_clicks':
        src = 'tmp/table_data.csv'
        dest = 'tmp/' + file_format
        copyfile(src, dest)
        src = 'tmp/table_data_pivot.csv'
        ret_val_d = dcc.send_file(dest)
    return [ret_val_d]


# #################
# # Download pivot
# #################
@app.callback([
        Output("download_pivot", "data"),
    ],
    [
        Input("download_pivot_button_id", "n_clicks")
    ],
    [
        State("download_pivot_format_id", "value")
    ],
    )
@logit
def on_download_pivot(_1, file_format):
    ctx = dash.callback_context.triggered[0]['prop_id']
    ret_val_d = dash.no_update
    if ctx == 'download_pivot_button_id.n_clicks':
        src = 'tmp/table_data_pivot.csv'
        dest = 'tmp/' + file_format
        copyfile(src, dest)
        ret_val_d = dcc.send_file(dest)
    return [ret_val_d]


###############
# Switch tab
###############
@app.callback(
    [
        Output('hidden_div_for_redirect_callback_stat', 'children'),
        Output('info_id', 'children'),
        Output('settings_id', 'children'),
     ],
    [Input('tabs_id', 'active_tab')]
)
@logit
def on_switch_tab(active_tab):
    session['active_tab'] = active_tab
    if active_tab == 'tab-7':
        return [dash.no_update, layout_info(df_w, df_p, df_c, t_start), dash.no_update]
    elif active_tab == 'tab-8':
        return [dash.no_update, dash.no_update, layout_settings()]
    elif active_tab == 'tab-6c':
        return [dash.no_update, dash.no_update, dash.no_update]
    else:
        return [dash.no_update, dash.no_update, dash.no_update]


###############
# Login button
###############
@app.callback(
    [Output('hidden_div_for_redirect_callback', 'children'),
     ],
    [Input('login_button_id', 'n_clicks')]
)
@logit
def on_login_button(n):
    if is_logged_in():
        # logout
        return [dcc.Location(pathname="/logout", id="someid2")]
    else:
        # login
        return [dcc.Location(pathname="/login", id="someid2")]


############################
# Przełączenie bazy danych
# wyzwalacze:
# - scope_id
# - subscope_id
# - przycisk dodania grupy group_button_id
# zmienia:
# - opcje, wartość i placeholder pola lokalizacji
# - opcje wyboru typu danych i wskaźników
# - pole od daty wykresów czasowych
# - opcje modyfikatora danych
# - opcje i wartość wyboru grup
# - opcje i wartość pola markera mapy
# - opcje i wartość pola koloru konturu i markera mapy
# - wartość od daty i do daty tabeli
# - styl pola wyboru podbazy
#
# callback jest użyty też do inicjacji sesji użytkownika oraz
# logowania (nazwa użytkownika i napis na przycisku)

############################

@app.callback(
    [
    Output({'type': 'params', 'index': 'locations_id'}, 'options'),
    Output({'type': 'params', 'index': 'locations_id'}, 'value'),
    Output({'type': 'params', 'index': 'locations_id'}, 'placeholder'),
    Output({'type': 'params', 'index': 'chart_type_data_id'}, 'options'),
    Output({'type': 'params', 'index': 'chart_type_data_id'}, 'value'),
    Output({'type': 'params', 'index': 'chart_type_calculated_id'}, 'options'),
    Output({'type': 'params', 'index': 'chart_type_calculated_id'}, 'value'),
    Output({'type': 'params', 'index': 'from_date_id'}, 'date'),
    Output({'type': 'params', 'index': 'data_modifier_id'}, 'options'),
    Output({'type': 'params', 'index': 'data_modifier_id'}, 'value'),
    Output('select_group_id', 'options'),
    Output('select_group_id', 'value'),
    Output({'type': 'params', 'index': 'core_date_id'}, 'date'),
    Output({'type': 'params', 'index': "subscope_id"}, 'style'),
    ],
    [
    Input({'type': 'params', 'index': "scope_id"}, 'value'),
    Input({'type': 'params', 'index': "subscope_id"}, 'value'),
    Input('group_button_id', 'n_clicks'),
    ],
    [
    State('select_group_id', 'value'),
    State('as_sum_id', 'value'),
    State({'type': 'params', 'index': 'locations_id'}, 'value'),
    ],
    prevent_initial_call=False
    )
@logit
def on_change_scope(scope, subscope, _1, select_group, as_sum, old_loc_val):
    session['refresh'] = True
    session['scope'] = scope
    session['log'] = ''
    dash.callback_context.response.set_cookie('dash_uuid4', str(uuid.uuid4()))

    ctx = dash.callback_context.triggered[0]['prop_id']

    # inicjacja

    if ctx == '.':
        debug('Init session .........')
        if not is_logged_in():
            session['mode'] = 'demo'
        else:
            session['mode'] = 'login'
        reset_colors()
        defaults2session()
        session['active_tab'] = 'tab-1'
    loc_opt = dash.no_update
    loc_val = dash.no_update
    loc_placeholder = dash.no_update

    if scope in ['poland', 'cities']:
        style_sub = {'visibility': 'hidden'}
    else:
        style_sub = {'visibility': 'visible'}
        scope = subscope
    if scope:
        loc_val = []
    data_modifier = [
        {"label": "bez modyfikacji", "value": 1},
        {"label": "na 100 000 osób", "value": 2},
        {"label": "na 1000 km2", "value": 3},
    ]
    groups = []
    group_val = dash.no_update
    if scope == 'world':
        choices = sorted(df_w['location'].unique(), key=pl_icase)
        loc_opt = list_to_options(choices)
        loc_placeholder = 'Kliknij aby wybrać lokalizacje...'
        groups = list_to_options([
            "Cały świat",
            "Kontynenty",
            "Europa",
            "Unia Europejska",
            "Europa spoza UE",
            "Dawne demoludy",
            "Kraje skandynawskie",
            "Azja",
            "Afryka",
            "Ameryka Północna",
            "Ameryka Południowa",
            "Oceania",
            "Ameryka Łacińska",
            "Kraje arabskie"
        ])
        other = list(subgroups.keys())
        others = [{'label': x, 'value': x} for x in other]
        groups += others
        group_val = "Cały świat"
    elif scope in continents:
        choices = sorted(df_w[
                             (df_w['Continent_Name'].isin([scope])) |
                             (df_w['location'] == scope)]['location'].unique(), key=pl_icase)
        loc_opt = list_to_options(choices)
        loc_placeholder = 'Kliknij aby wybrać kraj...'
        groups = list_to_options([
            "Wszystkie kraje",
        ])
        group_val = "Wszystkie kraje"
    elif scope == 'poland':
        choices = sorted(df_p['location'].unique(), key=pl_icase)
        loc_opt = list_to_options(choices)
        loc_placeholder = 'Kliknij aby wybrać województwo...'
        groups = list_to_options(
            [
                'Wszystkie województwa',
                'Województwa północno-zachodnie',
                'Województwa południowo-wschodnie'
            ])
        group_val = "Wszystkie województwa"
        data_modifier = [
            {"label": "bez modyfikacji", "value": 1},
            {"label": "na 100 000 osób", "value": 2},
            {"label": "na 1000 km2", "value": 3},
        ]
    elif scope == 'cities':
        choices = sorted(df_c['location'].unique(), key=pl_icase)
        loc_opt = list_to_options(choices)
        loc_placeholder = 'Kliknij aby wybrać powiat lub miasto...'
        group_val = "Wszystkie powiaty"
        data_modifier = [
            {"label": "bez modyfikacji", "value": 1},
            {"label": "na 100 000 osób", "value": 2},
            {"label": "na 1000 km2", "value": 3},
        ]
        groups = list_to_options(
            ["Wszystkie powiaty",
             "Wszystkie miasta",
             "Powiaty ziemskie",
             "Miasta wojewódzkie"] + sorted(wojewodztwa_list,
                                            key=pl_icase))

    # Dodanie grupy do loc_val
    if ctx == 'group_button_id.n_clicks':
        new_loc_val = []
        if scope == 'world':
            if select_group == 'Cały świat':
                if as_sum == 'single':
                    new_loc_val = sorted(df_w[df_w['Continent_Name'].isin(continents)]['location'].unique(),
                                         key=pl_icase)
                else:
                    new_loc_val = ['Świat']
            elif select_group == 'Kontynenty':
                if as_sum == 'single':
                    new_loc_val = continents
                else:
                    new_loc_val = ['Świat']
            elif select_group == 'Unia Europejska':
                if as_sum == 'single':
                    new_loc_val = sorted(eu_countries_list, key=pl_icase)
                else:
                    new_loc_val = ['Unia Europejska']
            elif select_group == 'Europa spoza UE':
                if as_sum == 'single':
                    new_loc_val = \
                        sorted(df_w[(df_w['Continent_Name'] == 'Europa') &
                                    (~df_w['location'].isin(eu_countries_list))]['location'].unique(),
                               key=pl_icase)
                else:
                    new_loc_val = ['Europa spoza Unii Europejskiej']
            elif select_group == 'Ameryka Łacińska':
                if as_sum == 'single':
                    new_loc_val = sorted(latin_america_list, key=pl_icase)
                else:
                    new_loc_val = ['Ameryka Łacińska']
            elif select_group == 'Kraje arabskie':
                if as_sum == 'single':
                    new_loc_val = sorted(arab_countries_list, key=pl_icase)
                else:
                    new_loc_val = ['Kraje arabskie']
            elif select_group == 'Dawne demoludy':
                if as_sum == 'single':
                    new_loc_val = sorted(demoludy_list, key=pl_icase)
                else:
                    new_loc_val = ['Dawne demoludy']
            elif select_group == 'Kraje skandynawskie':
                if as_sum == 'single':
                    new_loc_val = sorted(skandynawskie_list, key=pl_icase)
                else:
                    new_loc_val = ['Kraje skandynawskie']
            elif select_group in continents:
                if as_sum == 'single':
                    new_loc_val = df_w[df_w['Continent_Name'] == select_group]['location'].unique()
                else:
                    new_loc_val = [select_group]
            elif select_group in subgroups:
                list_codes = subgroups[select_group]
                if as_sum == 'single':
                    new_loc_val = df_w[df_w['iso_code'].isin(list_codes)]['location'].unique()
                else:
                    new_loc_val = [select_group]
        elif scope in continents:
            if select_group == 'Wszystkie kraje':
                if as_sum == 'single':
                    new_loc_val = df_w[df_w['Continent_Name'] == scope]['location'].unique()
                else:
                    new_loc_val = [scope]
        elif scope == 'poland':
            if select_group == 'Wszystkie województwa':
                if as_sum == 'single':
                    new_loc_val = sorted(df_p[df_p['Continent_Name'] == 'Polska']['location'].unique(), key=pl_icase)
                else:
                    new_loc_val = ['Polska']
            elif select_group == 'Województwa północno-zachodnie':
                if as_sum == 'single':
                    new_loc_val = sorted(wojewodztwa_pn_zach, key=pl_icase)
                else:
                    new_loc_val = ['Województwa północno-zachodnie']
            elif select_group == 'Województwa południowo-wschodnie':
                if as_sum == 'single':
                    new_loc_val = sorted(wojewodztwa_pd_wsch, key=pl_icase)
                else:
                    new_loc_val = ['Województwa południowo-wschodnie']
        elif scope == 'cities':
            if select_group == 'Wszystkie miasta':
                if as_sum == 'single':
                    new_loc_val = sorted(df_c[df_c['Continent_Name'] == 'miasto']['location'].unique(), key=pl_icase)
                else:
                    new_loc_val = ['Wszystkie miasta']
            elif select_group == 'Powiaty ziemskie':
                if as_sum == 'single':
                    new_loc_val = sorted(df_c[df_c['Continent_Name'] == 'powiat']['location'].unique(), key=pl_icase)
                else:
                    new_loc_val = ['Powiaty ziemskie']
            elif select_group == 'Miasta wojewódzkie':
                if as_sum == 'single':
                    new_loc_val = sorted(wojewodzkie_list, key=pl_icase)
                else:
                    new_loc_val = ['Miasta wojewódzkie']
            elif select_group == 'Wszystkie powiaty':
                if as_sum == 'single':
                    new_loc_val = sorted(df_c[df_c['Continent_Name'] != 'Grupa']['location'].unique(), key=pl_icase)
                else:
                    new_loc_val = ['Wszystkie powiaty']
            elif select_group in df_c['wojew'].unique():
                if as_sum == 'single':
                    new_loc_val = sorted(df_c[df_c['wojew'] == select_group]['location'].unique(), key=pl_icase)
                else:
                    new_loc_val = [select_group]
        loc_val = sorted(list(set(new_loc_val) | set(old_loc_val)), key=pl_icase)

    if scope == 'poland':
        from_date = min(df_p['date'])
        core_date = max(df_p['date'])
    elif scope == 'cities':
        from_date = min(df_c['date'])
        core_date = max(df_c['date'])
    else:
        from_date = min(df_w['date'])
        core_date = max(df_w['date'])

    options_data = trace_options(session.get('scope'), 'data')
    options_calculated = trace_options(session.get('scope'), 'calculated')
    if ctx == '{"index":"scope_id","type":"params"}.value':
        values_data = []
        values_calculated = []
    else:
        values_data = dash.no_update
        values_calculated = dash.no_update

    return [loc_opt, loc_val, loc_placeholder, options_data, values_data,
            options_calculated, values_calculated, from_date,
            data_modifier, 1, groups, group_val, core_date, style_sub]


###########################
# Set variable selects
###########################

@app.callback([
    Output({'type': 'params', 'index': 'rankings_highlight_id'}, 'options'),
    Output({'type': 'params', 'index': 'rankings_highlight_id'}, 'value'),
    Output({'type': 'params', 'index': 'timeline_highlight_id'}, 'options'),
    Output({'type': 'params', 'index': 'timeline_highlight_id'}, 'value'),
    Output({'type': 'params', 'index': 'core_highlight_id'}, 'options'),
    Output({'type': 'params', 'index': 'core_highlight_id'}, 'value'),
],
    [
        Input({'type': 'params', 'index': 'locations_id'}, 'value'),
    ],
    [State({'type': 'params', 'index': "scope_id"}, 'value'),
     ]
)
@logit
def on_change_locations(locations, scope):
    rankings_highlight_value = 'Brak'
    timeline_highlight_value = 'Brak'
    core_highlight_value = 'Brak'

    if rankings_highlight_value in ['Brak'] + locations:
        rankings_highlight_value = dash.no_update
    if timeline_highlight_value in ['Brak'] + locations:
        timeline_highlight_value = dash.no_update
    if core_highlight_value in ['Brak'] + locations:
        core_highlight_value = dash.no_update
    rankings_highlight_opt = [{'label': 'Bez wyróżnienia', 'value': 'Brak'}] + list_to_options(locations)
    timeline_highlight_opt = [{'label': 'Bez wyróżnienia', 'value': 'Brak'}] + list_to_options(locations)
    core_highlight_opt = [{'label': 'Bez wyróżnienia', 'value': 'Brak'}] + list_to_options(locations)

    return [
        rankings_highlight_opt, rankings_highlight_value,
        timeline_highlight_opt, timeline_highlight_value,
        core_highlight_opt, core_highlight_value,
        ]

###########################
# on change parameters
###########################

@app.callback(
    [
    Output('go1', 'value'),
    Output("kolorki_id", "children"),
    ],
    [
        Input({'type': 'params', 'index': ALL}, 'value'),
        Input({'type': 'params', 'index': ALL}, 'date'),
    ],
    [
        State({'type': 'params', 'index': "scope_id"}, 'value'),
    ]
)
@logit
def on_change_parameters(_1, _2, _3):
    ret_click = dash.no_update
    ctx = dash.callback_context
    no_ref = ['{"index":"colors_id","type":"params"}.value',
              '{"index":"scale_id","type":"params"}.value']
    if ctx.triggered[0]['prop_id'] in no_ref or not session.get('refresh'):
        go_val = dash.no_update
    else:
        go_val = random.random()

    # skopiowanie do session zmienionych parametrów

    controls2session(ctx)
    session['prevent_help'] = True
    return [go_val, layout_kolory()]


@app.callback([
    Output('timeline_charts_id', 'children'),
    Output('rankings_charts_id', 'children'),
    Output('dynamics_charts_id', 'children'),
    Output('map_id', 'children'),
    Output('core_id', 'children'),
    Output('table_id', 'children'),
    Output('loading-output_id', 'children'),
    Output('modal', 'children'),
],
    [
        Input('tabs_id', 'active_tab'),
        Input('go', 'value'),
        Input('go1', 'value'),  # zm. parametrów
        Input('go2', 'value'),
        Input('go3', 'value'),  # zm. kolorów
    ],
    # [State({'type': 'params', 'index': 'refresh_id'}, 'style')]
)
@logit
def on_update_figures(current_tab, _go, _go1, _go2, _go3):
    no_data = ''

    # Wyjście, jeśli jest wyłączone odświeżanie lub nie ma wystarczających danych do utworzenia wykresu

    if not session.get('chart_type') or len(session.get('locations')) == 0:
        if current_tab in ['tab-1', 'tab-2', 'tab-3', 'tab-4', 'tab-6']:
            return ['Czekam ...'] * 6 + [dash.no_update] * 2
        if current_tab in ['tab-5'] and len(session.get('locations')) == 0:
            return ['Czekam ...'] * 6 + [dash.no_update] * 2
    figure = None
    timeline_charts_children = no_data
    rankings_charts_children = no_data
    dynamics_charts_children = no_data
    balance_charts_children = no_data
    map_children = no_data
    core_children = no_data
    table_children = no_data
    loading_output_children = dash.no_update
    modal_children = dash.no_update

    if current_tab == 'tab-1':
        if session.get('radio_class') == 'types':
            figure = update_figures_timeline_types(DF)
        else:
            figure = update_figures_timeline_places(DF)
        timeline_charts_children = figure
    elif current_tab == 'tab-2':
        figure = update_figures_rankings(DF)
        rankings_charts_children = figure
    elif current_tab == 'tab-3':
        figure = update_figures_dynamics(DF)
        dynamics_charts_children = figure
    elif current_tab == 'tab-5':
        figure = update_figures_map_mapbox(DF)
        map_children = figure
    elif current_tab == 'tab-5a':
        figure = update_figures_core(DF)
        core_children = figure
    elif current_tab == 'tab-6':
        figure = update_figures_table(DF)
        table_children = figure

    # wyświetlenie komunikatu

    if type(figure) == str:
        modal_children = modal_message(figure)

    if modal_children != dash.no_update:
        return [no_data] * 6 + [dash.no_update] + [modal_children]
    else:
        return [
                    timeline_charts_children,
                    rankings_charts_children,
                    dynamics_charts_children,
                    map_children,
                    core_children,
                    table_children,
                    loading_output_children,
                    modal_children]

@app.callback(
    Output('map_id', 'style'),
    [Input('mapa', 'clickData')])
@logit
def display_click_data(custom_data):
    debug(custom_data)
    return dash.no_update

@app.callback(
    Output("hidden_div_for_redirect_callback_2", "children"),
    [Input("go_on_id", "n_clicks")],
)
@logit
def go_on(n):
    return [dcc.Location(pathname="/covid", id="someid3")]


@app.callback(
    [
        Output("stat-collapse-text_id", "is_open"),
        Output("stat-collapse-div_id", "style"),
    ],
    [Input("stat-collapse-button_id", "n_clicks")],
    [State("stat-collapse-text_id", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return [not is_open, {'overflow-y': 'auto'}]
    return [is_open, {'height': '20vh', 'overflow-y': 'auto'}]


@app.callback(
    Output("mort_content_id", "children"),
    [
        Input("select_mort_location_id", "value"),
        Input("select_mort_sex_id", "value"),
        Input("select_mort_year_id", "value")
    ],
)
@logit
def go_mortality(loc, sex, year):
    return [layout_more_wplz(DF, loc, sex, year)]

# Wybór klasy prezentacji
@app.callback(
    [
        Output("more_type_id", "options"),
        Output("more_type_id", "value")
    ],
    [
        Input("more_class_id", "value")
    ],
)
def choose_more_class(n):
    ret_val = ['']
    if n == 'podstawowe':
        ret_val = [
            {'label': '--wybierz prezentację--'},
            {'label': 'Heatmap (heatmap)', 'value': 'heatmap'},
            {'label': 'Mapa discrete województwa (map_discrete_woj)', 'value': 'map_discrete_woj'},
            {'label': 'Mapa discrete powiaty (map_discrete_pow)', 'value': 'map_discrete_pow'},
            {'label': 'Wykres 2 osie (2axes)', 'value': '2axes'},
            {'label': 'Histogram (histogram)', 'value': 'histogram'},
            {'label': 'Multimapa województwa (multimap_woj)', 'value': 'multimap_woj'},
            {'label': 'Multimapa powiatów (multimap_pow)', 'value': 'multimap_pow'},
            {'label': 'Chronomapa województwa (chronomap_woj)', 'value': 'chronomap_woj'},
            {'label': 'Chronomapa powiaty (chronomap_pow)', 'value': 'chronomap_pow'},
        ]
    elif n == 'analizy':
        ret_val = [
            {'label': '--wybierz prezentację--'},
            {'label': 'Szpital Tymczasowy Poznań MTP (analiza_poznan)', 'value': 'analiza_poznan'},
            {'label': 'Analiza dni tygodnia (analiza_dni_tygodnia)', 'value': 'analiza_dni_tygodnia'},
            {'label': 'Synchroniczne porównanie wielu przebiegów (analiza_2)', 'value': 'analiza_2'},
        ]
    elif n == 'szczepienia MZ':
        ret_val = [
            {'label': '--wybierz prezentację--'},
            {'label': 'Wykres podstawowy *dane MZ* (mz_vacc1)', 'value': 'mz_vacc1'},
            {'label': 'Udział grup wiekowych w tygodniowych szczepieniach *dane MZ* (mz_vacc2)', 'value': 'mz_vacc2'},
            {'label': 'Udział grup wiekowych w dziennych szczepieniach *dane MZ* (mz_vacc2a)', 'value': 'mz_vacc2a'},
            {'label': 'Dynamika szczepień w grupach wiekowych *dane MZ* (mz_vacc2b)', 'value': 'mz_vacc2b'},
            {'label': 'Bilans szczepień w powiatach i miastach *dane MZ* (mz_vacc3)', 'value': 'mz_vacc3'},
            {'label': 'Ranking wojewódzki szczepień (wybór rodzaju gminy) *dane MZ* (mz_vacc4)', 'value': 'mz_vacc4'},
            {'label': 'Ranking powiatowy szczepień (wybór województwa) *dane MZ* (mz_vacc5)', 'value': 'mz_vacc5'},
            {'label': 'Bilans zaszczepienia populacji Polski *dane MZ* (mz_vacc6)', 'value': 'mz_vacc6'},
            {'label': 'Mapa punktów szczepień (wybór województwa) *dane MZ* (mz_psz)', 'value': 'mz_psz'},
            {'label': 'Mapa powiatowych wskaźników szczepień *dane MZ* (mz_psz2)', 'value': 'mz_psz2'},
            {'label': 'Mapa powiatowych szczepień w aptekach (mz_psz5)', 'value': 'mz_psz5'},
            {'label': 'Mapa gmin techniczna (mz_psz2t)', 'value': 'mz_psz2t'},
            {'label': 'Mapa gminnych wskaźników szczepień *dane MZ* (mz_psz3)', 'value': 'mz_psz3'},
            {'label': 'NOP, dawki utracone *MZ* (mz_nop)', 'value': 'mz_nop'},
        ]
    elif n == 'HIT':
        ret_val = [
            {'label': '--wybierz prezentację--'},
            {'label': 'Powiatowa mapa immunizacji (hit1)', 'value': 'hit1'},
            {'label': 'Mapa wojewódzka immunizacji (hit2)', 'value': 'hit2'},
            {'label': 'Korelacja w województwach (hit3)', 'value': 'hit3'},
            {'label': 'Korelacja w powiatach (hit4)', 'value': 'hit4'},
            {'label': 'Przyrost/utrata odporności (hit5)', 'value': 'hit5'},
        ]
    elif n == 'szczepienia ECDC':
        ret_val = [
            {'label': '--wybierz prezentację--'},
            {'label': 'Bilans narastająco (wybór: grupa wiekowa, rodzaj) *ECDC* (ecdc_vacc0)', 'value': 'ecdc_vacc0'},
            {'label': 'Sumy tygodniowe (wybór: lokalizacja, grupa, rodzaj, dawka) *dane ECDC* (ecdc_vacc)', 'value': 'ecdc_vacc'},
            {'label': 'Spadek efektywnego zaszczepienia populacji Polski *dane ECDC* (ecdc_vacc1)', 'value': 'ecdc_vacc1'},
            {'label': 'Efektywne zaszczepienie populacji Polski *dane ECDC* (ecdc_vacc2)', 'value': 'ecdc_vacc2'},
            {'label': 'Liczba zaszczepionych w grupach wiekowych (wybór: dawka)*dane ECDC* (ecdc_vacc3)', 'value': 'ecdc_vacc3'},
            {'label': 'Ranking szczepień w krajach Europy *dane ECDC* (ecdc_vacc5)', 'value': 'ecdc_vacc5'},
            {'label': 'Bilans zaszczepienia populacji Polski *dane ECDC* (ecdc_vacc6)', 'value': 'ecdc_vacc6'},
            {'label': 'Procent zaszczepienia w województwach *dane ECDC* (ecdc_vacc7)', 'value': 'ecdc_vacc7'},
            {'label': 'Udział grup wiekowych w szczepieniach tygodniowych *dane ECDC* (ecdc_vacc8)', 'value': 'ecdc_vacc8'},
            {'label': 'Udział rodzajów szczepionek w szczepieniach tygodniowych *dane ECDC* (ecdc_vacc9)', 'value': 'ecdc_vacc9'},
            {'label': 'Szczepienia w Poslce - tygodniowo *dane ECDC* (ecdc_vacc11)', 'value': 'ecdc_vacc11'},
        ]
    elif n == 'nadmiarowe':
        ret_val = [
            {'label': '--wybierz prezentację--'},
            {'label': 'Analiza zgonów nadmiarowych i zgonów Covid (wplz)', 'value': 'wplz'},
            {'label': 'Porównanie zgonów nadmiarowych i zgonów Covid (wplz0)', 'value': 'wplz0'},
            {'label': 'Roczne krzywe zgonów (wplz1)', 'value': 'wplz1'},
            {'label': 'Roczne krzywe % zgonów nadmiarowych (wplz2)', 'value': 'wplz2'},
            {'label': 'Udział % zgonów Covid-19 w zgonach nadmiarowych (wplz3)', 'value': 'wplz3'},
            {'label': 'Porównanie fali jesiennej i wiosennej w Polsce (cmpf)', 'value': 'cmpf'},
            {'label': 'Porównanie fali 2020 i 2021 w Polsce (cmpf_2)', 'value': 'cmpf_2'},
            {'label': 'Nadmiarowe zgony w województwach (nzwall)', 'value': 'nzwall'},
            {'label': 'Heatmap - nadmiarowe zgony (nz_heatmap)', 'value': 'nz_heatmap'},
            {'label': 'Mapa. Poównanie zgonów nadmiarowych w województwach za okres (nzwcmp)', 'value': 'nzwcmp'},
            {'label': 'Mapa. Poównanie zgonów nadmiarowych 2020-2021 ze średnią (nzwcmp_1)', 'value': 'nzwcmp_1'},
            {'label': 'Mapa. Poównanie umieralności 2016-2019 i 2020-2022 (nzwcmp_2)', 'value': 'nzwcmp_2'},
        ]
    # elif n == 'Apple':
    #     ret_val = [
    #         {'label': '--wybierz prezentację--'},
    #         {'label': 'Raport mobilności Apple w wybranym województwie (rmaw)', 'value': 'rmaw'},
    #         {'label': 'Porównanie mobilności Apple w województwach i miastach (rmaw2)', 'value': 'rmaw2'},
    #         {'label': 'Porównanie mobilności Apple w województwach z R (rmaw3)', 'value': 'rmaw3'},
    #     ]
    elif n == 'inne':
        ret_val = [
            {'label': '--wybierz prezentację--'},
            {'label': 'Współczynnik reprodukcji (R) w województwach (wrrw)', 'value': 'wrrw'},
            {'label': 'Mapa szkół w Irlandii (ireland)', 'value': 'ireland'},
        ]
    elif n == 'BASIW':
        ret_val = [
            {'label': '--wybierz prezentację--'},
            {'label': 'Statystyka zgonów w grupach wiekowych (mzage1)', 'value': 'mzage1'},
            {'label': 'Heatmap struktury wiekowej zgonów (mzage2)', 'value': 'mzage2'},
            {'label': 'Dzienne zgony w grupie wiekowej zaszczepieni / niezaszczepieni (mzage3)', 'value': 'mzage3'},
            {'label': 'Dzienne zgony w grupach wiekowych zaszczepieni / niezaszczepieni (mzage4)', 'value': 'mzage4'},
            {'label': 'CFR w falach pandemii (cfr1)', 'value': 'cfr1'},
            {'label': 'CFR vs status zaszczepienia (cfr2)', 'value': 'cfr2'},
            {'label': 'CFR vs. rodzaj szczepionki (cfr3)', 'value': 'cfr3'},
        ]
    elif n == 'cases BASIW':
        ret_val = [
            {'label': '--wybierz prezentację--'},
            {'label': 'Udział grup wiekowych w dziennej liczbie przypadków (mz_cases1)', 'value': 'mz_cases1'},
            {'label': 'Statystyka zakażeń w grupach wiekowych (mz_cases2)', 'value': 'mz_cases2'},
            {'label': 'Liczba przypadków według płci (mz_cases3)', 'value': 'mz_cases3'},
            {'label': 'Porównanie zakażeń MZ i BASiW (mz_cases4)', 'value': 'mz_cases4'},
            {'label': 'Piramida wieku zachorowań i zgonów (mz_cases5)', 'value': 'mz_cases5'},
            {'label': 'Średnia wieku zachorowań (mz_cases6)', 'value': 'mz_cases6'},
            {'label': 'Udział grup wiekowych w dziennej liczbie przypadków (mz_cases7)', 'value': 'mz_cases7'},
            {'label': 'Heatmap liczby przypadków w grupach wiekowych (mz_cases8)', 'value': 'mz_cases8'},
            {'label': 'Dzienne infekcje w grupach wiekowych (mz_cases10)', 'value': 'mz_cases10'},
            {'label': 'Dzienne infekcje w grupach zaszczepionych i niezaszczepionych (mz_cases11)', 'value': 'mz_cases11'},
            {'label': 'Statystyka wg statusu odporności (mz_cases12)', 'value': 'mz_cases12'},
            {'label': 'Reinfekcje vs infekcje wg rodzaju szczepionki (mz_cases13)', 'value': 'mz_cases13'},
            {'label': 'Reinfekcje vs infekcje wg statusu szczepienia (mz_cases14)', 'value': 'mz_cases14'},
            {'label': 'Reinfekcje i infekcje ilościowo (mz_cases15)', 'value': 'mz_cases15'},
        ]
    elif n == 'LC, PIMS':
        ret_val = [
            {'label': '--wybierz prezentację--'},
            {'label': 'Long Covid (lc1)', 'value': 'lc1'},
            {'label': 'PIMS (lc2)', 'value': 'lc2'},
        ]
    elif n == 'reinf MZ':
        ret_val = [
            {'label': '--wybierz prezentację--'},
            {'label': 'Statystyki reinfekcji (mz_reinf)', 'value': 'mz_reinf'},
            {'label': 'Udział % pozytywnych testów w kolejnych tygodniach po zaszczepieniu 2. dawką (mz_reinf2)',
             'value': 'mz_reinf2'},
            {'label': 'Udział % pozytywnych testów w grupie zaszczepionych i niezaszczepionych (mz_reinf3)',
             'value': 'mz_reinf3'},
            {'label': 'Proporcja udziału % pozytywnych testów w grupie zaszczepionych i niezaszczepionych (mz_reinf4)',
             'value': 'mz_reinf4'},
            {'label': 'Wykrywalność po zaszczepieniu w funkcji liczby tygodni od 2 dawki (mz_reinf5)',
             'value': 'mz_reinf5'},
        ]
    elif n == 'pozostałe':
        ret_val = [
            {'label': '--wybierz prezentację--'},
        ]
    return [ret_val, '--wybierz prezentację--']

###########################################
# IN - wybrany rodzaj prezentacji
# OUT - parametry konkretnej prezentacji
###########################################
@app.callback(
    [
        Output("more_params_1_id", "children"),
        Output("more_params_2_id", "children"),
        Output("more_params_3_id", "children"),
    ],
    [
        Input("more_type_id", "value")
    ],
)
def choose_more_type(n):
    ret_val = ['', '', '']
    if n == 'wplz':  # analiza zgonów nadmiarowych
        ret_val = [
            html.Div([
                badge('Lokalizacja', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(
                               list(DF['mort']['location'].unique())
                               ),
                           value='Polska',
                           )
                ]
            ),
            html.Div([
                    badge('Rok', 'skalowaniey'),
                    dbc.Select(id='more_2_id',
                               options=list_to_options(DF['mort']['year'].unique()),
                               value=2015,
                               ),
                ]
            ),
            html.Div([
                badge('Wiek', 'skalowaniey'),
                dbc.Select(id='more_3_id',
                           options=list_to_options(['UNK', 'Y20-39', 'Y40-59', 'TOTAL', 'Y_GE80', 'Y_LT20', 'Y60-79']),
                           value='TOTAL',
                           ),
            ]
            ),
        ]
    if n == 'wplz0':  # porównanie liczby zgonów nadmiarowych i zgonów Covid
        ret_val = [
            html.Div([
                badge('Lokalizacja', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(
                               list(DF['mort']['location'].unique())
                               ),
                           value='Polska',
                           )
                ]
            ),
            html.Div([
                    badge('Wiek', 'skalowaniey'),
                    dbc.Select(id='more_2_id',
                               options=list_to_options(['UNK', 'Y20-39', 'Y40-59', 'TOTAL', 'Y_GE80', 'Y_LT20', 'Y60-79']),
                               value='TOTAL',
                               ),
                ]
            ),
            html.Div([
                    badge('Rok', 'skalowaniey'),
                    dbc.Select(id='more_3_id',
                               options=list_to_options(DF['mort']['year'].unique()),
                               value=2015,
                               ),
                ]
            ),
        ]
    if n == 'wplz1':  # roczne krzywe zgonów
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    if n == 'wplz2':  # roczne krzywe % zgonów nadmiarowych
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    if n == 'wplz3':  # Udział % zgonów Covid-19 w zgonach nadmiarowych
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'tlzp':  # Tygodniowe liczby zgonów w państwach
        ret_val = [
            html.Div([
                badge('Grupa wiekowa', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(
                               ['Y_LT20', 'Y20-39', 'Y40-59', 'Y60-79', 'Y_GE80', 'UNK', 'TOTAL']),
                           value='TOTAL',
                           )
                ]
            ),
            html.Div([
                badge('Średnia od roku:', 'skalowaniey'),
                dbc.Select(id='more_2_id',
                           options=list_to_options(list(range(2015, 2020))),
                           value=2019
                           ),
                ]
            ),
            html.Div(id='more_3_id'),
        ]
    elif n == 'cmpf':  # porównanie fal epidemii
        ret_val = [
            html.Div([
                badge('Lokalizacja', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(
                               sorted(list(constants.wojew_cap.values()))),
                           value='Polska',
                           )
                ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'cmpf_2':  # porównanie fal epidemii 2020 2021
        ret_val = [
            html.Div([
                badge('Lokalizacja', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(
                               sorted(list(constants.wojew_cap.values()))),
                           value='Polska',
                           )
                ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'nzwall':  # % Nadmiarowych zgonów w województwach
        ret_val = [
            html.Div([
                badge('Lokalizacja', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(
                               list(DF['mort']['location'].unique())),
                           value='Polska',
                           )
                ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'nz_heatmap':  # % Zgony nadmiarowe - heatmap
        ret_val = [
            html.Div([
                badge('Lokalizacja', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(
                               list(DF['mort']['location'].unique())
                               ),
                           value='Polska',
                           )
                ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'nzwcmp':  # Poównanie zgonów nadwmiarowych w województwach za okres
        ret_val = [
            html.Div([
                badge('Od daty', 'oddaty'),
                html.Br(),
                dcc.DatePickerSingle(id='more_1_id',
                                     display_format='DD-MM-Y',
                                     date=datetime.datetime.today(),
                                     ),
            ]
            ),
            html.Div([
                badge('Do daty', 'oddaty'),
                html.Br(),
                dcc.DatePickerSingle(id='more_2_id',
                                     display_format='DD-MM-Y',
                                     date=datetime.datetime.today(),
                                     ),
            ]
            ),
            html.Div(id='more_3_id'),
        ]
    elif n == 'nzwcmp_1':  # Poównanie zgonów nadwmiarowych 2020-2021 w województwach za okres
        ret_val = [
            html.Div([
                badge('Grupa wiekowa', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(
                               ['Y_LT20', 'Y20-39', 'Y40-59', 'TOTAL', 'Y_GE80', 'Y60-79']),
                           value='TOTAL',
                           )
            ]
            ),
            html.Div([
                badge('Lata', 'skalowaniey'),
                dbc.Select(id='more_2_id',
                           options=list_to_options(
                               ['2020-2022', '2021-2022']),
                           value='2020-2022',
                           )
            ]
            ),
            html.Div([
                badge('Skalowanie', 'skalowaniey'),
                dbc.Select(id='more_3_id',
                           options=list_to_options(
                               ['brutto', 'na 100k']),
                           value='brutto',
                           )
            ]
            ),
        ]
    elif n == 'nzwcmp_2':  # Poównanie umieralności w wojwództwach
        ret_val = [
            html.Div([
                badge('Grupa wiekowa', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(
                               ['Y_LT20', 'Y20-39', 'Y40-59', 'TOTAL', 'Y_GE80', 'Y60-79']),
                           value='TOTAL',
                           )
            ]
            ),
            html.Div([
                badge('Lata', 'skalowaniey'),
                dbc.Select(id='more_2_id',
                           options=list_to_options(
                               ['2020-2020', '2021-2021', '2022-2022', '2020-2022', '2020-2021']),
                           value='2020-2022',
                           )
            ]
            ),
            html.Div([
                badge('Skalowanie', 'skalowaniey'),
                dbc.Select(id='more_3_id',
                           options=list_to_options(
                               ['nadmiarowe', 'porównanie']),
                           value='nadmiarowe',
                           )
            ]
            ),
        ]
    elif n == 'wrrw':  # R w województwach
        ret_val = [
            html.Div([
                badge('Uśrednianie (dni)', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(
                               ['2', '7', '14', '21']),
                           value='7',
                           )
                ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_reinf':  # statystyki reinfekcji
        options = ['producent, kategoria_zakażenia', 'producent, płeć', 'producent_grupa wiekowa']
        ret_val = [
            html.Div([
                badge('Od m-ca (2021)', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(options),
                           value='producent, kategoria_zakażenia',
                           )
            ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_reinf2':  # statystyki reinfekcji 2
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_reinf3':  # statystyki reinfekcji 3
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_reinf4':  # statystyki reinfekcji 4
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_reinf5':  # statystyki reinfekcji 5
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_cases1':
        options = ['mężczyźni', 'kobiety', 'wszyscy']
        options2 = ['b.d.', '0-11', '12-19', '20-39', '40-59', '60-69', '70+']
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_cases2':

        options = ['brutto', 'na 100k']
        options2 = ['ALL', '0-9', '10-14', '15-17', '18-24', '25-49', '50-59', '60-69', '70-79', '80+']
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_cases3':
        options = ['wszyscy', 'b.d.', '<12', '12-19', '20-39', '40-59', '60-69', '70+']
        ret_val = [
            html.Div([
                badge('Płeć', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(options),
                           value='wszyscy',
                           )
            ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_cases4':

        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_cases5':
        options = ['populacja', 'infekcje', 'zgony', 'populacja, infekcje', 'populacja, zgony', 'populacja, infekcje, zgony', 'infekcje, zgony']
        ret_val = [
            html.Div([
                badge('Rodzaj', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(options),
                           value='infekcje, zgony',
                           )
            ]
            ),
            html.Div([
                badge('Grupa', 'skalowaniey'),
                dbc.Select(id='more_2_id',
                           options=list_to_options(['w pełni zaszczepieni', 'niezaszczepieni', 'wszyscy']),
                           value='wszyscy',
                           )
            ]
            ),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_cases6':
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_cases7':
        options = ['mężczyźni', 'kobiety', 'wszyscy']
        ret_val = [
            html.Div([
                badge('Płeć', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(options),
                           value='wszyscy',
                           )
            ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_cases8':
        options = ['mężczyźni', 'kobiety', 'wszyscy']
        ret_val = [
            html.Div([
                badge('Od m-ca (2021)', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(options),
                           value='wszyscy',
                           )
            ]
            ),
            html.Div([
                badge('Rodzaj', 'skalowaniey'),
                dbc.Select(id='more_2_id',
                           options=list_to_options(['brutto', 'na 100k grupy']),
                           value='brutto',
                           )
            ]
            ),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_cases10':

        options = ['brutto', 'na 100k']
        options2 = ['wszyscy', '12-19', '20-39', '40-59', '60-69', '70+']
        ret_val = [
            html.Div([
                badge('Rodzaj wykresu', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(options),
                           value='brutto',
                           )
            ]
            ),
            html.Div([
                badge('Grupa wiekowa', 'skalowaniey'),
                dbc.Select(id='more_2_id',
                           options=list_to_options(options2),
                           value='wszyscy',
                           )
            ]
            ),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_cases11':

        options = ['0-4', '5-9', '10-14', '15-17', '18-24', '25-49', '50-59',
                   '60-69', '70-79', '80+']
        ret_val = [
            html.Div([
                badge('Grupa wiekowa', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(options),
                           value='80+',
                           )
            ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_cases12':

        ret_val = [
            html.Div([
                badge('Od daty', 'oddaty'),
                html.Br(),
                dcc.DatePickerSingle(id='more_1_id',
                                     display_format='DD-MM-Y',
                                     date=datetime.datetime.today() - timedelta(days=120),
                                     ),
            ]
            ),
            html.Div([
                badge('Mnożnik', 'skalowaniey'),
                dbc.Select(id='more_2_id',
                           options=list_to_options(['1', '3', '5', '7', '9']),
                           value='5',
                           )
            ]
            ),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_cases13':

        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_cases14':

        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_cases15':

        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'lc1':

        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'lc2':

        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == '2axes':
        ret_val = [
            html.Div([
                badge('Lewa oś', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=[{"label": "<brak>", "value": '<brak>'}] + fields_to_options(),
                           value='<brak>',
                           )
            ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'map_discrete_woj':
        options = ['oddzielne skale', 'wspólna skala']
        ret_val = [
            html.Div([
                badge('Interwał', 'skalowaniey'),
                html.Br(),
                dcc.Input(
                    id="more_1_id",
                    type='number',
                    value=7,
                )
            ]
            ),
            html.Div([
                badge('Skalowanie', 'skalowaniey'),
                dbc.Select(id='more_2_id',
                           options=list_to_options(options),
                           value='wspólna skala',
                           persistence=False,
                           style={'cursor': 'pointer'},
                           ),
            ]),
            html.Div([
                badge('N', 'skalowaniey'),
                html.Br(),
                dcc.Input(
                    id="more_3_id",
                    type='number',
                    value=2,
                )
            ]
            ),
        ]
    elif n == 'map_discrete_pow':
        options = ['oddzielne skale', 'wspólna skala']
        ret_val = [
            html.Div([
                badge('Interwał', 'skalowaniey'),
                html.Br(),
                dcc.Input(
                    id="more_1_id",
                    type='number',
                    value=7,
                )
            ]
            ),
            html.Div([
                badge('Skalowanie', 'skalowaniey'),
                dbc.Select(id='more_2_id',
                           options=list_to_options(options),
                           value='wspólna skala',
                           persistence=False,
                           style={'cursor': 'pointer'},
                           ),
            ]),
            html.Div([
                badge('N', 'skalowaniey'),
                html.Br(),
                dcc.Input(
                    id="more_3_id",
                    type='number',
                    value=2,
                )
            ]
            ),
        ]
    elif n == 'histogram':
        ret_val = [
            badge('Rodzaj agregacji', 'rodzajwykresu'),
            dbc.Select(id='more_2_id',
                       options=[{'label': 'średnia', 'value': 'mean'},
                                {'label': 'suma', 'value': 'sum'},
                                {'label': 'maksimum', 'value': 'max'},
                                {'label': 'minimum', 'value': 'min'},
                                {'label': 'mediana', 'value': 'median'},
                                {'label': 'jeden dzień', 'value': 'oneday'}
                                ],
                       value='mean',
                       persistence=False,
                       style={'cursor': 'pointer'},
                       ),
            badge('Do daty', 'oddaty'),
            dcc.DatePickerSingle(id='more_2_id',
                                 display_format='DD-MM-Y',
                                 date=datetime.datetime.today(),
                                 persistence=False),
            html.Div(id='more_3_id'),
        ]
    elif n == 'multimap_woj':
        ret_val = [
            html.Div([
                badge('Data', 'oddaty'),
                html.Br(),
                dcc.DatePickerSingle(id='more_1_id',
                                     display_format='DD-MM-Y',
                                     date=datetime.datetime.today(),
                                     ),
            ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]

    elif n == 'chronomap_woj':
        options = ['oddzielne skale', 'wspólna skala']
        ret_val = [
            html.Div([
                badge('Interwał', 'skalowaniey'),
                html.Br(),
                dcc.Input(
                    id="more_1_id",
                    type='number',
                    value=7,
                )
            ]
            ),
            html.Div([
                badge('Skalowanie', 'skalowaniey'),
                dbc.Select(id='more_2_id',
                           options=list_to_options(options),
                           value='wspólna skala',
                           persistence=False,
                           style={'cursor': 'pointer'},
                           ),
            ]),
            html.Div([
                badge('N', 'skalowaniey'),
                html.Br(),
                dcc.Input(
                    id="more_3_id",
                    type='number',
                    value=2,
                )
            ]
            ),
        ]
    elif n == 'chronomap_pow':
        options = ['oddzielne skale', 'wspólna skala']
        ret_val = [
            html.Div([
                badge('Interwał', 'skalowaniey'),
                html.Br(),
                dcc.Input(
                    id="more_1_id",
                    type='number',
                    value=7,
                )
            ]
            ),
            html.Div([
                badge('Skalowanie', 'skalowaniey'),
                dbc.Select(id='more_2_id',
                           options=list_to_options(options),
                           value='wspólna skala',
                           persistence=False,
                           style={'cursor': 'pointer'},
                           ),
            ]),
            html.Div([
                badge('N', 'skalowaniey'),
                html.Br(),
                dcc.Input(
                    id="more_3_id",
                    type='number',
                    value=2,
                )
            ]
            ),
        ]
    elif n == 'multimap_pow':
        ret_val = [
            html.Div([
                badge('Data', 'oddaty'),
                html.Br(),
                dcc.DatePickerSingle(id='more_1_id',
                                     display_format='DD-MM-Y',
                                     date=datetime.datetime.today(),
                                     ),
            ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mzage2':  # EWP age6 heatmap
        ret_val = [
            html.Div([
                badge('Od daty', 'oddaty'),
                html.Br(),
                dcc.DatePickerSingle(id='more_1_id',
                                     display_format='DD-MM-Y',
                                     date=datetime.datetime.today() - timedelta(days=30),
                                     ),
            ]
            ),
            html.Div([
                badge('Rodzaj', 'skalowaniey'),
                dbc.Select(id='more_2_id',
                           options=list_to_options(
                               ['brutto', 'udział']),
                           value='brutto',
                           )
            ]
            ),
            html.Div(id='more_3_id'),
        ]
    elif n == 'cfr1':  # CFR fale
        ret_val = [
            html.Div([
                badge('Od daty', 'oddaty'),
                html.Br(),
                dcc.DatePickerSingle(id='more_1_id',
                                     display_format='DD-MM-Y',
                                     date=datetime.datetime.today() - timedelta(days=30),
                                     ),
            ]
            ),
            html.Div([
                badge('Rodzaj', 'skalowaniey'),
                dbc.Select(id='more_2_id',
                           options=list_to_options(
                               ['wszyscy', 'zaszczepieni', 'niezaszczepieni']),
                           value='wszyscy',
                           )
            ]
            ),
            html.Div(id='more_3_id'),
        ]
    elif n == 'cfr2':  # CFR a status zaszczepienia
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'cfr3':  # CFR vs. rodzaj szczepionki
        ret_val = [
            html.Div([
                badge('Od daty', 'oddaty'),
                html.Br(),
                dcc.DatePickerSingle(id='more_1_id',
                                     display_format='DD-MM-Y',
                                     date=datetime.datetime.today() - timedelta(days=30),
                                     ),
            ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mzage3':

        options = ['brutto', 'na 100k']
        options2 = ['ALL', '0-9', '10-14', '15-17', '18-24', '25-49', '50-59', '60-69', '70-79', '80+']
        ret_val = [
            html.Div([
                badge('Od daty', 'oddaty'),
                html.Br(),
                dcc.DatePickerSingle(id='more_1_id',
                                     display_format='DD-MM-Y',
                                     date=datetime.datetime.today() - timedelta(days=120),
                                     ),
            ]
            ),
            html.Div([
                badge('Do daty', 'oddaty'),
                html.Br(),
                dcc.DatePickerSingle(id='more_2_id',
                                     display_format='DD-MM-Y',
                                     date=datetime.datetime.today(),
                                     ),
            ]
            ),
            html.Div([
                badge('Grupa wiekowa', 'skalowaniey'),
                dbc.Select(id='more_3_id',
                           options=list_to_options(options2),
                           value='80+',
                           )
            ]
            ),
        ]
    elif n == 'mzage1':

        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mzage4':  # MZ age4
        options = ['brutto', 'na 100k', 'proporcja']
        ret_val = [
            html.Div([
                badge('Rodzaj wykresu', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(options),
                           value='brutto',
                           )
            ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_nop':  # MZ  NOP, dawki utracone
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_vacc1':  # MZ  vacc bilans
        ret_val = [
            html.Div([
                badge('Rodzaj wykresu', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(['Trendy', 'Wiek narastająco', 'Płeć narastająco',
                                                    'Wiek przyrosty dzienne', 'Płeć przyrosty dzienne',
                                                    'Bilans magazynu', 'Bilans punktów']),
                           value='Trendy',
                           )
            ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_vacc2':  # MZ vacc udział grup wiekowych w tygodniowych szczepieniach
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_vacc2a':  # MZ vacc udział grup wiekowych w dziennych szczepieniach
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_vacc2b':  # MZ Dynamika szczepień w grupach wiekowych
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_vacc3':  # MZ bilans szczepień w powiatach
        ret_val = [
            html.Div([
                badge('Lokalizacja', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(['--wszystkie', '--miasta', '--powiaty ziemskie', '--miasta wojewódzkie'] +
                                                   constants.wojewodztwa_list),
                           value='--miasta wojewódzkie',
                           )
            ]
            ),
            html.Div([
                badge('Kolejność', 'skalowaniey'),
                dbc.Select(id='more_2_id',
                           options=list_to_options(['alfabetycznie', 'rosnąco', 'malejąco']),
                           value='malejąco',
                           )
            ]
            ),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_vacc4':  # MZ bilans szczepień w gminach
        opcje = ['wszystkie gminy', 'do 20 tys.', '20-50 tys.', '50-100 tys', '> 100 tys.']
        ret_val = [
            html.Div([
                badge('Lokalizacja', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(opcje),
                           value='wszystkie gminy',
                           )
            ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_vacc5':  # MZ ranking w grupach wiekowych w powiatach województwa
        ret_val = [
            html.Div([
                badge('Lokalizacja', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(wojewodztwa_list),
                           value='mazowieckie',
                           )
            ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_vacc6':  # MZ bilans populacyjny szczepień
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_psz':  # MZ punkty szczepień
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_psz2':  # MZ mapa
        opcje_1 = ['szczepienia na punkt', 'szczepienia na 100k', '% zaszczepienia', 'szczepienia suma',
                   'sloty na punkt', 'sloty na 100k', 'sloty suma']
        opcje_2 = ['12-19 lat', '20-39 lat', '40-59 lat', '60-69 lat', '70+ lat',
                   'total_1d', 'total_full']
        ret_val = [
            html.Div([
                badge('Lokalizacja', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(opcje_1),
                           value='szczepienia na punkt',
                           )
            ]
            ),
            html.Div([
                badge('Lokalizacja', 'skalowaniey'),
                dbc.Select(id='more_2_id',
                           options=list_to_options(opcje_2),
                           value='total_1d',
                           )
            ]
            ),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_psz5':  # MZ mapa szczepień w aptekach
        opcje_1 = ['suma', 'na 100k']
        ret_val = [
            html.Div([
                badge('Lokalizacja', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(opcje_1),
                           value='suma',
                           )
            ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_psz2t':  # MZ mapa gmin techniczna
        opcje_1 = ['bialskopodlaskie', 'białostockie', 'bielskie', 'bydgoskie', 'chełmskie', 'ciechanowskie',
        'częstochowskie', 'elbląskie', 'gdańskie', 'gorzowskie', 'jeleniogórskie', 'kaliskie', 'katowickie',
        'kieleckie', 'konińskie', 'koszalińskie', '(miejskie) krakowskie', 'krośnieńskie', 'legnickie',
        'leszczyńskie', 'lubelskie', 'łomżyńskie', '(miejskie) łódzkie', 'nowosądeckie', 'olsztyńskie',
        'opolskie', 'ostrołęckie', 'pilskie', 'piotrkowskie', 'płockie', 'poznańskie', 'przemyskie',
        'radomskie', 'rzeszowskie', 'siedleckie', 'sieradzkie', 'skierniewickie', 'słupskie', 'suwalskie',
        'szczecińskie', 'tarnobrzeskie', 'tarnowskie', 'toruńskie', 'wałbrzyskie', '(stołeczne) warszawskie',
        'włocławskie', 'wrocławskie', 'zamojskie', 'zielonogórskie']
        ret_val = [
            html.Div([
                badge('Województwo', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(opcje_1),
                           value='bialskopodlaskie',
                           )
            ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'mz_psz3':  # MZ mapa gmin
        opcje_1 = ['12-19 lat', '20-39 lat', '40-59 lat', '60-69 lat', '70plus lat',
                   'total_1d', 'total_full']
        ret_val = [
            html.Div([
                badge('Wiek, rodzaj', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(opcje_1),
                           value='total_1d',
                           )
            ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'hit1':  # MZ hit powiatowa
        opcje_1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
        ret_val = [
            html.Div([
                badge('Mnożnik', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(opcje_1),
                           value='5',
                           )
            ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id')
        ]
    elif n == 'hit2':  # MZ hit wojewódzka
        opcje_1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
        ret_val = [
            html.Div([
                badge('Mnożnik', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(opcje_1),
                           value='5',
                           )
            ]
            ),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id')
        ]
    elif n == 'hit3':  # korelacja województwa
        ret_val = [
            html.Div([
                badge('Wskaźnik L', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(['Immunizacja', 'Pełne zaszczepienie', 'Zaszczepienie 1. dawką']),
                           value='Pełne zaszczepienie',
                           )
            ]
            ),
            html.Div([
                badge('Od daty', 'oddaty'),
                html.Br(),
                dcc.DatePickerSingle(id='more_2_id',
                                     display_format='DD-MM-Y',
                                     date=datetime.datetime.today() - timedelta(days=90),
                                     ),
            ]
            ),
            html.Div([
                badge('Wskaźnik P', 'skalowaniey'),
                dbc.Select(id='more_3_id',
                           options=list_to_options(['Suma infekcji/100k', 'Suma zgonów/100k',
                                                    'Infekcje/100k dzienne', 'Zgony/100k dzienne',
                                                    'Zajęte łóżka/100k', 'Zajęte respiratory/100k']),
                           value='Suma infekcji/100k',
                           )
            ]
            ),
        ]
    elif n == 'analiza_poznan':  # analiza - Poznań

        opcje_1 = [str(x) for x in list(range(-20, 20))]
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id')
        ]
    elif n == 'analiza_dni_tygodnia':  # analiza dni tygodnia

        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id')
        ]
    elif n == 'analiza_2':  # analiza - synchroniczne porównanie przebiegów

        opcje_1 = [str(x) for x in list(range(-20, 20))]
        ret_val = [
            html.Div([
                badge('Przesunięcie', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(opcje_1),
                           value='0',
                           ),
            ]
            ),
            html.Div([
                badge('Data', 'oddaty'),
                html.Br(),
                dcc.DatePickerSingle(id='more_2_id',
                                     display_format='DD-MM-Y',
                                     date=datetime.datetime.today() - timedelta(days=30),
                                     ),
            ]
            ),
            html.Div(id='more_3_id')
        ]
    elif n == 'hit4':  # MZ hit powiatowa vs. infekcje/100k
        ret_val = [
            html.Div([
                badge('Wskaźnik L', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(['Immunizacja', 'Pełne zaszczepienie', 'Zaszczepienie 1. dawką']),
                           value='Pełne zaszczepienie',
                           )
            ]
            ),
            html.Div([
                badge('Od daty', 'oddaty'),
                html.Br(),
                dcc.DatePickerSingle(id='more_2_id',
                                     display_format='DD-MM-Y',
                                     date=datetime.datetime.today() - timedelta(days=90),
                                     ),
            ]
            ),
            html.Div([
                badge('Wskaźnik P', 'skalowaniey'),
                dbc.Select(id='more_3_id',
                           options=list_to_options(['Suma infekcji/100k', 'Suma zgonów/100k', 'Poparcie dla Dudy 2020']),
                           value='Suma infekcji/100k',
                           )
            ]
            ),
        ]
    elif n == 'hit5':  # przyrost/utrata odporności
        opcje_1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
        ret_val = [
            html.Div([
                badge('Mnożnik', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(opcje_1),
                           value='5',
                           ),
            ]
            ),
            html.Div([
                badge('N choroba', 'skalowaniey'),
                html.Br(),
                dcc.Input(
                    id="more_2_id",
                    type='number',
                    value=40,
                )
            ]
            ),

            html.Div([
                badge('N szczepienie', 'skalowaniey'),
                html.Br(),
                dcc.Input(
                    id="more_3_id",
                    type='number',
                    value=40,
                )
            ]
            ),
        ]
    elif n == 'ireland':  # mapa szkół w Irlandii
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id')
        ]
    elif n == 'heatmap':  # heatmap
        ret_val = [
            html.Div([
                badge('Sortowanie', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(['alfabetycznie', 'max', 'min', 'ostatni']),
                           value='alfabetycznie',
                           )
            ]
            ),
            html.Div([
                badge('Kolejność', 'skalowaniey'),
                dbc.Select(id='more_2_id',
                           options=list_to_options(['rosnąca', 'malejąca']),
                           value='rosnąca',
                           )
            ]
            ),
            dbc.Col([
                badge('Liczba pozycji:', 'pozostale'),
                dbc.Select(id='more_3_id',
                           options=[{'label': 'Wszystkie', 'value': '0'}] +
                                   [{'label': str(i), 'value': str(i)} for i in range(1, 20)],
                           value='19')
            ]),
        ]
    elif n == 'ecdc_vacc':  # ECDC vacc liczba szczepień
        ret_val = [
            html.Div([
                badge('Grupa wiekowa', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(['ALL', 'Age10_14', 'Age15_17', 'Age18_24', 'Age25_49', 'Age50_59', 'Age60_69', 'Age70_79',
                                                    'Age80+', 'AgeUNK', 'HCW']),
                           value='ALL',
                           )
            ]
            ),
            html.Div([
                badge('Szczepionka', 'skalowaniey'),
                dbc.Select(id='more_2_id',
                           options=list_to_options(['MOD', 'COM', 'AZ', 'JANSS', 'NVXD', 'ALL']),
                           value='ALL',
                           )
            ]
            ),
            html.Div([
                badge('Dawka', 'skalowaniey'),
                dbc.Select(id='more_3_id',
                           options=list_to_options(['dawka_1', 'dawka_2', 'dawka_3', 'dawka_4', 'dawka_12', 'dawka_34', 'dawka_1234']),
                           value='dawka_4',
                           ),
            ]),
        ]
    elif n == 'ecdc_vacc0':  # ECDC vacc bilans razem
        ret_val = [
            html.Div([
                badge('Grupa wiekowa', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(['ALL', 'Age10_14', 'Age15_17', 'Age18_24', 'Age25_49', 'Age50_59', 'Age60_69', 'Age70_79',
                                                    'Age80+', 'AgeUNK', 'HCW']),
                           value='ALL',
                           )
            ]
            ),
            html.Div([
                badge('Szczepionka', 'skalowaniey'),
                dbc.Select(id='more_2_id',
                           options=list_to_options(['MOD', 'COM', 'AZ', 'JANSS', "NVXD', "'ALL']),
                           value='ALL',
                           )
            ]
            ),
            html.Div(id='more_3_id'),
        ]
    elif n == 'ecdc_vacc3':  # ECDC vacc liczba wyszczepionych w grupach wiekowych
        ret_val = [
            html.Div([
                badge('Dawka', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(['dawka_1', 'dawka_2', 'dawka_3', 'dawka_4']),
                           value='dawka_1',
                           ),
            ]),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'ecdc_vacc5':  # Ranking szczepień w krajach Europy
        ret_val = [
            html.Div([
                badge('Dawka', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(['dostawa', 'dawka_1', 'dawka_3', 'dawki', 'zapas', 'wpelni']),
                           value='dawka_1',
                           ),
            ]),
            html.Div([
                badge('Grupa wiekowa', 'skalowaniey'),
                dbc.Select(id='more_2_id',
                           options=list_to_options(['ALL', 'Age10_14', 'Age15_17', 'Age18_24', 'Age25_49', 'Age50_59', 'Age60_69',  'Age70_79',
                                                    'Age80+']),
                           value='ALL',
                           ),
            ]),
            html.Div(id='more_3_id'),
        ]
    elif n == 'ecdc_vacc7':  # Procent zaszczepienia w województwach
        ret_val = [
            html.Div([
                badge('Grupa wiekowa', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(['ALL', 'Age10_14', 'Age15_17', 'Age18_24', 'Age25_49', 'Age50_59', 'Age60_69',  'Age70_79',
                                                    'Age80PLUS']),
                           value='ALL',
                           ),
            ]),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'ecdc_vacc8':  # Udział grup wiekowych w szczepieniach
        ret_val = [
            html.Div([
                badge('Dawka', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(['dawka_1', 'dawka_2', 'dawka_3', 'dawka_123']),
                           value='dawka_12',
                           ),
            ]),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'ecdc_vacc9':  # Udział rodzajów szczepionek w szczepieniach tygodniowych
        ret_val = [
            html.Div([
                badge('Grupa wiekowa', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(['ALL', 'Age10_14', 'Age15_17', 'Age18_24', 'Age25_49', 'Age50_59', 'Age60_69',  'Age70_79',
                                                    'Age80+']),
                           value='ALL',
                           ),
            ]),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'ecdc_vacc11':  # Szczepienia w Polsce - tygodniowo
        ret_val = [
            html.Div([
                badge('Grupa wiekowa', 'skalowaniey'),
                dbc.Select(id='more_1_id',
                           options=list_to_options(['ALL', 'Age10_14', 'Age15_17', 'Age18_24', 'Age25_49', 'Age50_59', 'Age60_69', 'Age70_79',
                                                    'Age80PLUS']),
                           value='ALL',
                           ),
            ]),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'ecdc_vacc6':  # Bilans zaszczepienia populacji Polski
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'ecdc_vacc1':  # Bilans efektywnego zaszczepienia populacji Polski
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]
    elif n == 'ecdc_vacc2':  # Bilans efektywnego zaszczepienia populacji Polski
        ret_val = [
            html.Div(id='more_1_id'),
            html.Div(id='more_2_id'),
            html.Div(id='more_3_id'),
        ]

    return ret_val


###########################################
# IN - wybrany rodzaj prezentacji
# OUT - parametry konkretnej prezentacji
###########################################
@app.callback(
    [
        Output("overview_params_1_id", "children"),
        Output("overview_params_2_id", "children"),
        Output("overview_params_3_id", "children"),
    ],
    [
        Input("overview_type_id", "value")
    ],
)
def choose_overview_type(n):
    ret_val = ['' * 3]
    if n == 'wdzp':  # wykresy danych źródłowych w państwach
        ret_val = [
            html.Div([
                badge('Lokalizacja', 'skalowaniey'),
                dbc.Select(id='overview_1_id',
                           options=list_to_options(
                               list(DF['world']['location'].unique())),
                           value='Polska',
                           )
                ]
            ),
            html.Div(id='overview_2_id'),
            html.Div(id='overview_3_id'),
        ]
    elif n == 'wdzw':  # wykresy danych źródłowych w województwach
        ret_val = [
            html.Div([
                badge('Lokalizacja', 'skalowaniey'),
                dbc.Select(id='overview_1_id',
                           options=list_to_options(
                               list(DF['poland']['location'].unique())),
                           value='Mazowieckie',
                           )
                ]
            ),
            html.Div(id='overview_2_id'),
            html.Div(id='overview_3_id'),
        ]
    elif n == 'wdzc':  # wykresy danych źródłowych w powiatach
        ret_val = [
            html.Div([
                badge('Lokalizacja', 'skalowaniey'),
                dbc.Select(id='overview_1_id',
                           options=list_to_options(
                               list(DF['cities']['location'].unique())),
                           value='Warszawa',
                           )
                ]
            ),
            html.Div(id='overview_2_id'),
            html.Div(id='overview_3_id'),
        ]
    elif n == 'wdzc':  # wykresy danych źródłowych w powiatach/miastach
        ret_val = [
            html.Div([
                badge('Lokalizacja', 'skalowaniey'),
                dbc.Select(id='overview_1_id',
                           options=list_to_options(
                               list(DF['poland']['location'].unique())),
                           value='Warszawa',
                           )
                ]
            ),
            html.Div(id='overview_2_id'),
            html.Div(id='overview_3_id'),
        ]
    elif n == 'tdzp':  # tabela danych źródłowych w państwach
        ret_val = [
            html.Div([
                badge('Lokalizacja', 'skalowaniey'),
                dbc.Select(id='overview_1_id',
                           options=list_to_options(
                               list(DF['world']['location'].unique())),
                           value='Polska',
                           )
                ]
            ),
            html.Div(id='overview_2_id'),
            html.Div(id='overview_3_id'),
        ]
    elif n == 'tdzw':  # tabela danych źródłowych w województwach
        ret_val = [
            html.Div([
                badge('Lokalizacja', 'skalowaniey'),
                dbc.Select(id='overview_1_id',
                           options=list_to_options(
                               list(DF['poland']['location'].unique())),
                           value='Mazowieckie',
                           )
                ]
            ),
            html.Div(id='overview_2_id'),
            html.Div(id='overview_3_id'),
        ]
    elif n == 'wz':  # wykorzystanie zasobów
        ret_val = [
            html.Div([
                badge('Lokalizacja', 'skalowaniey'),
                dbc.Select(id='overview_1_id',
                           options=list_to_options(
                               list(DF['poland']['location'].unique())),
                           value='Mazowieckie',
                           )
                ]
            ),
            html.Div(id='overview_2_id'),
            html.Div(id='overview_3_id'),
        ]
    return ret_val


@app.callback(
    [
        Output("more_id", "children"),
        Output('loading2-output_id', 'children'),
    ],
    [
        Input('more_1_id', "value"),
        Input('more_2_id', "value"),
        Input('more_3_id', "value"),
        Input('go1', "value"),  # parametry
        Input('go3', "value"),  # kolory
        Input('go', "value"),  # paleta kolorów
        Input("more_1_id", "date"),
        Input("more_2_id", "date"),
    ],
    [
        State("more_type_id", "value"),
    ]
)
@logit
def go_more_figure(p1, p2, p3, go1, go3, go, p1a, p2a, type):
    if type == 'a':
        return
    ctx = dash.callback_context.triggered[0]['prop_id']
    # if ctx == 'go.value':
    #     session['template_change'] = True
    # else:
    #     session['template_change'] = False
    ret_val = dash.no_update
    if type == 'wrrw':
        ret_val = layout_more_wrrw(DF, p1, p2, p3)
    elif type == 'wplz':
        ret_val = [layout_more_wplz(DF, p1, p2, p3)]
    elif type == 'wplz0':
        ret_val = [layout_more_wplz0(DF, p1, p2, p3)]
    elif type == 'wplz1':
        ret_val = [layout_more_wplz1(DF, p1, p2, p3)]
    elif type == 'wplz2':
        ret_val = [layout_more_wplz2(DF, p1, p2, p3)]
    elif type == 'wplz3':
        ret_val = [layout_more_wplz3(DF, p1, p2, p3)]
    elif type == 'tlzgwp':
        ret_val = layout_more_tlzgw(DF, p1, p2, 'C')
    elif type == 'tlzgww':
        ret_val = layout_more_tlzgw(DF, p1, p2, 'W')
    elif type == 'tlzp':
        ret_val = layout_more_tlzp(DF, p1, p2, 'C')
    elif type == 'cmpf':
        ret_val = layout_more_cmpf(DF, p1, p2, p3)
    elif type == 'cmpf_2':
        ret_val = layout_more_cmpf_2(DF, p1, p2, p3)
    elif type == 'nzwall':
        ret_val = layout_more_nzwall(DF, p1, p2, p3)
    elif type == 'nz_heatmap':
        ret_val = layout_more_nz_heatmap(DF, p1, p2, p3)
    elif type == 'nzwcmp':
        ret_val = layout_more_nzwcmp(DF, p1, p2, p3)
    elif type == 'nzwcmp_1':
        ret_val = layout_more_nzwcmp_1(DF, p1, p2, p3)
    elif type == 'nzwcmp_2':
        ret_val = layout_more_nzwcmp_2(DF, p1, p2, p3)
    # elif type == 'rmaw':
    #     ret_val = layout_more_rmaw(DF, p1, p2, p3)
    # elif type == 'rmaw2':
    #     ret_val = layout_more_rmaw2(DF, p1a, p2, p3)
    # elif type == 'rmaw3':
    #     ret_val = layout_more_rmaw3(DF, p1, p2, p3)
    elif type == 'mz_reinf':
        ret_val = layout_more_mz_reinf(DF, p1, p2, p3)
    elif type == 'mz_reinf2':
        ret_val = layout_more_mz_reinf2(DF, p1, p2, p3)
    elif type == 'mz_reinf3':
        ret_val = layout_more_mz_reinf3(DF, p1, p2, p3)
    elif type == 'mz_reinf4':
        ret_val = layout_more_mz_reinf4(DF, p1, p2, p3)
    elif type == 'mz_reinf5':
        ret_val = layout_more_mz_reinf5(DF, p1, p2, p3)
    elif type == 'mz_cases1':
        ret_val = layout_more_mz_cases1(DF, p1, p2, p3)
    elif type == 'mz_cases2':
        ret_val = layout_more_mz_cases2(DF, p1a, p2a, p3)
    elif type == 'mz_cases3':
        ret_val = layout_more_mz_cases3(DF, p1, p2, p3)
    elif type == 'mz_cases4':
        ret_val = layout_more_mz_cases4(DF, p1, p2a, p3)
    elif type == 'mz_cases5':
        ret_val = layout_more_mz_cases5(DF, p1, p2, p3)
    elif type == 'mz_cases6':
        ret_val = layout_more_mz_cases6(DF, p1, p2, p3)
    elif type == 'mz_cases7':
        ret_val = layout_more_mz_cases7(DF, p1, p2, p3)
    elif type == 'mz_cases8':
        ret_val = layout_more_mz_cases8(DF, p1, p2, p3)
    elif type == 'mz_cases10':
        ret_val = layout_more_mz_cases10(DF, p1, p2, p3)
    elif type == 'mz_cases11':
        ret_val = layout_more_mz_cases11(DF, p1, p2, p3)
    elif type == 'mz_cases12':
        ret_val = layout_more_mz_cases12(DF, p1a, p2, p3)
    elif type == 'mz_cases13':
        ret_val = layout_more_mz_cases13(DF, p1a, p2, p3)
    elif type == 'mz_cases14':
        ret_val = layout_more_mz_cases14(DF, p1a, p2, p3)
    elif type == 'mz_cases15':
        ret_val = layout_more_mz_cases15(DF, p1a, p2, p3)
    elif type == 'lc1':
        ret_val = layout_more_lc1(DF, p1a, p2, p3)
    elif type == 'lc2':
        ret_val = layout_more_lc2(DF, p1a, p2, p3)
    elif type == '2axes':
        ret_val = layout_more_2axes(DF, p1, p2, p3)
    elif type == 'map_discrete_woj':
        ret_val = layout_more_map_discrete_woj(DF, p1, p2, p3)
    elif type == 'map_discrete_pow':
        ret_val = layout_more_map_discrete_pow(DF, p1, p2, p3)
    elif type == 'histogram':
        ret_val = layout_more_histogram(DF, p1, p2, p3)
    elif type == 'multimap_woj':
        ret_val = layout_more_multimap_woj(DF, p1a, p2, p3)
    elif type == 'chronomap_woj':
        ret_val = layout_more_chronomap_woj(DF, p1, p2, p3)
    elif type == 'chronomap_pow':
        ret_val = layout_more_chronomap_pow(DF, p1, p2, p3)
    elif type == 'multimap_pow':
        ret_val = layout_more_multimap_pow(DF, p1a, p2, p3)
    elif type == 'mzage2':
        ret_val = layout_more_mzage2(DF, p1a, p2, p3)
    elif type == 'mzage3':
        ret_val = layout_more_mzage3(DF, p1a, p2a, p3)
    elif type == 'mzage1':
        ret_val = layout_more_mzage1(DF, p1a, p2a, p3)
    elif type == 'mzage4':
        ret_val = layout_more_mzage4(DF, p1, p2, p3)
    elif type == 'cfr1':
        ret_val = layout_more_cfr1(DF, p1a, p2, p3)
    elif type == 'cfr2':
        ret_val = layout_more_cfr2(DF, p1a, p2, p3)
    elif type == 'cfr3':
        ret_val = layout_more_cfr3(DF, p1a, p2, p3)
    elif type == 'mz_nop':
        ret_val = layout_more_mz_nop(DF, p1, p2, p3)
    elif type == 'heatmap':
        ret_val = layout_more_heatmap(DF, p1, p2, p3)
    elif type == 'ireland':
        ret_val = layout_more_ireland(DF, p1, p2, p3)
    elif type == 'mz_vacc1':
        ret_val = layout_more_mz_vacc1(DF, p1, p2, p3)
    elif type == 'mz_vacc2':
        ret_val = layout_more_mz_vacc2(DF, p1, p2, p3)
    elif type == 'mz_vacc2a':
        ret_val = layout_more_mz_vacc2a(DF, p1, p2, p3)
    elif type == 'mz_vacc2b':
        ret_val = layout_more_mz_vacc2b(DF, p1, p2, p3)
    elif type == 'mz_vacc3':
        ret_val = layout_more_mz_vacc3(DF, p1, p2, p3)
    elif type == 'mz_vacc4':
        ret_val = layout_more_mz_vacc4(DF, p1, p2, p3)
    elif type == 'mz_vacc5':
        ret_val = layout_more_mz_vacc5(DF, p1, p2, p3)
    elif type == 'mz_vacc6':
        ret_val = layout_more_mz_vacc6(DF, p1, p2, p3)
    elif type == 'mz_psz':
        ret_val = layout_more_mz_psz(DF, p1, p2, p3)
    elif type == 'mz_psz2':
        ret_val = layout_more_mz_psz2(DF, p1, p2, p3)
    elif type == 'mz_psz5':
        ret_val = layout_more_mz_psz5(DF, p1, p2, p3)
    elif type == 'mz_psz2t':
        ret_val = layout_more_mz_psz2t(DF, p1, p2, p3)
    elif type == 'mz_psz3':
        ret_val = layout_more_mz_psz3(DF, p1, p2, p3)
    elif type == 'hit1':
        ret_val = layout_more_hit1(DF, p1, p2, p3)
    elif type == 'hit2':
        ret_val = layout_more_hit2(DF, p1, p2, p3)
    elif type == 'hit3':
        ret_val = layout_more_hit3(DF, p1, p2a, p3)
    elif type == 'hit4':
        ret_val = layout_more_hit4(DF, p1, p2a, p3)
    elif type == 'hit5':
        ret_val = layout_more_hit5(DF, p1, p2, p3)
    elif type == 'analiza_poznan':
        ret_val = layout_more_analiza_poznan(DF, p1, p2a, p3)
    elif type == 'analiza_dni_tygodnia':
        ret_val = layout_more_analiza_dni_tygodnia(DF, p1, p2a, p3)
    elif type == 'analiza_2':
        ret_val = layout_more_analiza_2(DF, p1, p2a, p3)
    elif type == 'ecdc_vacc':
        ret_val = layout_more_ecdc_vacc(DF, p1, p2, p3)
    elif type == 'ecdc_vacc0':
        ret_val = layout_more_ecdc_vacc0(DF, p1, p2, p3)
    elif type == 'ecdc_vacc3':
        ret_val = layout_more_ecdc_vacc3(DF, p1, p2, p3)
    elif type == 'ecdc_vacc5':
        ret_val = layout_more_ecdc_vacc5(DF, p1, p2, p3)
    elif type == 'ecdc_vacc6':
        ret_val = layout_more_ecdc_vacc6(DF, p1, p2, p3)
    elif type == 'ecdc_vacc7':
        ret_val = layout_more_ecdc_vacc7(DF, p1, p2, p3)
    elif type == 'ecdc_vacc8':
        ret_val = layout_more_ecdc_vacc8(DF, p1, p2, p3)
    elif type == 'ecdc_vacc9':
        ret_val = layout_more_ecdc_vacc9(DF, p1, p2, p3)
    elif type == 'ecdc_vacc11':
        ret_val = layout_more_ecdc_vacc11(DF, p1, p2, p3)
    elif type == 'ecdc_vacc1':
        ret_val = layout_more_ecdc_vacc1(DF, p1, p2, p3)
    elif type == 'ecdc_vacc2':
        ret_val = layout_more_ecdc_vacc2(DF, p1, p2, p3)
    return [ret_val, dash.no_update]


@app.callback(
    Output("overview_id", "children"),
    [
        Input("overview_1_id", "value"),
        Input("overview_2_id", "value"),
        Input("overview_3_id", "value"),
        Input("go1", "value"),
        Input("go3", "value"),
        Input("go", "value"),
    ],
    [
        State("overview_type_id", "value")
    ]
)
@logit
def go_overview_figure(p1, p2, p3, _go1, _go3, _go, type):
    if type == 'wdzp':
        return layout_overview_wdzp(DF, p1, p2, p3)
    elif type == 'wdzw':
        return layout_overview_wdzw(DF, p1, p2, p3)
    elif type == 'wdzc':
        return layout_overview_wdzc(DF, p1, p2, p3)
    elif type == 'tdzp':
        return layout_overview_tdzp(DF, p1, p2, p3)
    elif type == 'tdzw':
        return layout_overview_tdzw(DF, p1, p2, p3)
    elif type == 'wz':
        return layout_overview_wz(DF, p1, p2, p3)


# test kliknięcia na mapę
@app.callback(
    Output('figure_mapa', 'figure'),
    [Input('figure_mapa', 'clickData')])
def update_figure(clickData):
    if clickData is not None:
        location = clickData['points'][0]['location']
        name = clickData['points'][0]['text'][3:].split(sep='<br>')[0]

        with open("log/stare_woj.csv", "a") as file:
            file.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+','+
                       session.get('stare_wojew')+','+'JPT_'+location+','+name)
            file.write('\n')

    return dash.no_update

app.layout = layout
app.css.config.serve_locally = False
app.config.suppress_callback_exceptions = True


if __name__ == "__main__":
    server = FastAPI()

    @server.get("/download/{name_file}")
    def download_file(name_file: str):
        return FileResponse(path=os.getcwd() + "/data/" + name_file, media_type='application/octet-stream', filename=name_file)

    server.add_middleware(SessionMiddleware, secret_key="SECRET_KEY")
    server.mount("/", WSGIMiddleware(app.server))
    uvicorn.run(server, host="0.0.0.0", port=8000)
    # uvicorn.run("index:app", host='127.0.0.1', port=8000, reload=True)