import os

import flask
from dash import html
import dash_bootstrap_components as dbc
from flask import request
import json
import numpy as np
from datetime import datetime as dt, timedelta
from scipy import signal
import constants
from datetime import datetime
import datetime
import pandas as pd
from functools import wraps
from time import time


from config import session

profiler = 1

development = True if os.path.isfile('deploy') else False
production = True if os.getcwd() == '/home/docent/dash' else False

print('Working directory:', os.getcwd, production)

def logit(func):
    @wraps(func)
    def with_logging(*args, **kwargs):
        DEBUG_LEVEL = 1
        if DEBUG_LEVEL > 0:
            print(constants.bcolors.FAIL + "{0}".format(dt.now().strftime('%m-%d-%y %H:%M:%S')),
                  constants.bcolors.OKBLUE + '[' + func.__name__ + ']' + constants.bcolors.ENDC)
            if DEBUG_LEVEL > 1:
                print(args)
                if DEBUG_LEVEL > 2:
                    print(kwargs)
        return func(*args, **kwargs)
    return with_logging

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        if profiler == 1:
            print('func:%r time: %2.4f sec' % (f.__name__, te-ts))
        elif profiler == 2:
            print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        return result
    return wrap

# from apps.covid_2 import db_connection

trace_names_data = {x[0]: x[1] for x in constants.trace_names_list['data']}
trace_names_calculated = {x[0]: x[1] for x in constants.trace_names_list['calculated']}
trace_names = {**trace_names_data, **trace_names_calculated}
resource_names = {
    'hosp_patients': 'Hospitalizacje razem',
    'new_hosp': 'Hospitalizacje nowe',
    'total_beds': 'Łóżka razem',
    'used_beds': 'Lózka w użyciu (%)',
    'total_in_resp': 'Respiratory zajęte razem',
    'new_in_resp': 'Respiratory zajęte nowe',
    'total_resp': 'Respiratory razem',
    'used_resp': 'Respiratory w użyciu (%)'
}
rt_names = {
    'rt': 'Wskaźnik reprodukcji R(t)',
}


# @logit
def controls2session(ctx):
    for x in constants.settings_names:
        control_id = x+'_id'
        if control_id in constants.input_controls:
            control_val = ctx.inputs[constants.input_controls[control_id]['id']]
            if x in ['date_picker']:
                control_val = control_val[:10]
            session[x] = control_val
        else:
            if not 'color_' in x:
                session[x] = constants.get_defa()[x]
    session['chart_type'] = session['chart_type_calculated'] + session['chart_type_data']


def get_template():
    t = constants.default_template
    template = session.get('template')
    template_change = session.get('template_change')
    if template_change:
        t.layout.title.font.size = constants.user_templates[template].get('font_size_title')
        t.layout.title.font.color = constants.user_templates[template].get('color_7')
        t.layout.title.x = constants.user_templates[template].get('titlexpos')
        t.layout.title.y = constants.user_templates[template].get('titleypos')
        t.layout.xaxis.linecolor = constants.user_templates[template].get('color_2')
        t.layout.xaxis.tickfont.size = constants.user_templates[template].get('font_size_xy')
        t.layout.xaxis.tickfont.color = constants.user_templates[template].get('color_3')
        t.layout.xaxis.title.font.color = constants.user_templates[template].get('color_2')
        t.layout.yaxis.linecolor = constants.user_templates[template].get('color_2')
        t.layout.yaxis.tickfont.size = constants.user_templates[template].get('font_size_xy')
        t.layout.yaxis.tickfont.color = constants.user_templates[template].get('color_3')
        t.layout.yaxis.title.font.color = constants.user_templates[template].get('color_2')
        t.layout.paper_bgcolor = constants.user_templates[template].get('color_1')
        t.layout.plot_bgcolor = constants.user_templates[template].get('color_1')
        t.layout.legend.bgcolor = constants.user_templates[template].get('color_6')
        t.layout.legend.font.size = constants.user_templates[template].get('font_size_legend')
        t.layout.legend.font.color = constants.user_templates[template].get('color_10')
        t.layout.colorway = constants.color_scales.get(constants.user_templates[template].get('color_order'))
        session['template_change'] = False
    else:
        t.layout.title.font.size = session.get('font_size_title')
        t.layout.title.font.color = session.get('color_7')
        t.layout.title.x = session.get('titlexpos')
        t.layout.title.y = session.get('titleypos')
        t.layout.xaxis.linecolor = session.get('color_2')
        t.layout.xaxis.tickfont.size = session.get('font_size_xy')
        t.layout.xaxis.tickfont.color = session.get('color_3')
        t.layout.xaxis.title.font.color = session.get('color_2')
        t.layout.yaxis.linecolor = session.get('color_2')
        t.layout.yaxis.tickfont.size = session.get('font_size_xy')
        t.layout.yaxis.tickfont.color = session.get('color_3')
        t.layout.yaxis.title.font.color = session.get('color_2')
        t.layout.paper_bgcolor = session.get('color_1')
        t.layout.plot_bgcolor = session.get('color_1')
        t.layout.legend.bgcolor = session.get('color_6')
        t.layout.legend.font.size = session.get('font_size_legend')
        t.layout.legend.font.color = session.get('color_10')
        t.layout.colorway = constants.color_scales.get(session.get('color_order'))
    return t


# @logit
def defaults2session(all=True):
    # użycie: reset ustawień użytkownika (on_moje)
    for x in constants.settings_names:
        if all or constants.settings_props[x]['save']:
            if session.get(x) != constants.get_defa()[x]:
                session[x] = constants.get_defa()[x]


# @logit
def settings2session(settings):
    # użycie: wczytanie ustawień użytkownika (on_moje)
    if settings is None:
        return
    for x in constants.settings_names:
        if settings.get(x):
            if constants.settings_props[x]['save']:
                session[x] = settings.get(x)
        else:
            print('settings2session: brak w settings: ', x)


# @logit
def session2settings(all=True):
    # użycie: w każdym callbacku
    settings = {}
    for x in constants.settings_names:
        if all or constants.settings_props[x]['save']:
            settings[x] = session.get(x)
            if settings[x] is None:
                session[x] = constants.get_defa()[x]
    return settings


def settings2json(data=None):
    import json

    if data is None:
        data = session2settings()
    if data == '':
        json_string = 'Brak'
    else:
        json_string = json.dumps(data, ensure_ascii=False, indent=4)
    return json_string


# @logit
def read_user_settings(fn):
    if is_logged_in():
        fname = 'users/' + session.get('user_email') + '/' + fn
        if file_exists(fname):
            with open(fname) as infile:
                settings2session(json.load(infile))
            return True
        else:
            log.error('read_user_settings: brak pliku', fname)
            return False
    else:
        return False


# @logit
def write_user_settings(fn):
    if is_logged_in():
        fname = 'users/' + session.get('user_email') + '/' + fn
        with open(fname, 'w') as outfile:
            json.dump(session2settings(all=False), outfile)


def span(txt='', value='white', attr='color'):
    return '<span style="' + attr + ':' + value + '">' + txt + '</span>'


def alert(msg):
    if msg:
        return [msg, {'display': 'block', 'visibility': 'visible', 'background-color': 'orange', 'color': 'black'}]
    else:
        return [[''], {'visibility': 'hidden'}]


def show_info(msg, duration=2000):
    return \
        [dbc.Alert(
            msg,
            id="alert-auto",
            is_open=True,
            fade=True,
            color='success',
            duration_d=duration,
        )]


def heading(text):
    return html.H3(text, style={'text-align': 'center'})


def tail(f, lines=1, _buffer=4098):
    # thanks to glenbot from stackoverflow
    """Tail a file and get X lines from the end"""
    lines_found = []
    block_counter = -1
    while len(lines_found) < lines:
        try:
            f.seek(block_counter * _buffer, os.SEEK_END)
        except IOError:
            f.seek(0)
            lines_found = f.readlines()
            break
        lines_found = f.readlines()
        block_counter -= 1
    return lines_found[-lines:]


def badge(txt, _id='none', color='Gainsboro'):
    return html.Div([dbc.Badge(txt,
                               pill=True,
                               color="dark",
                               className='ml-1 p-0 text-wrap',
                               style={'color': color, 'background': 'transparent'}
                               ),
                     ],
                    className='mt-0 mb-0',
                    style={'display': 'inline-block', 'font-size': '18px'}
                    )


def subtitle(txt):
    return html.Div(html.P(txt, style={'background-color': '#2c2626', 'padding-left': '10px'}),
                    className='ml-1 mt-0 mb-0')


def get_settings(name):
    try:
        ret_val = session.get(name)
    except:
        ret_val = constants.get_defa()[name]
    return ret_val


# @logit
def reset_colors():
    session['color_1'] = get_default_color(1)
    session['color_2'] = get_default_color(2)
    session['color_3'] = get_default_color(3)
    session['color_4'] = get_default_color(4)
    session['color_5'] = get_default_color(5)
    session['color_6'] = get_default_color(6)
    session['color_7'] = get_default_color(7)
    session['color_8'] = get_default_color(8)
    session['color_9'] = get_default_color(9)
    session['color_10'] = get_default_color(10)
    session['color_11'] = get_default_color(11)
    session['color_12'] = get_default_color(12)
    session['color_13'] = get_default_color(13)
    session['color_14'] = get_default_color(14)


def get_session_color(n):
    if not session.get('color_' + str(n)):
        color = constants.get_defa()['color_' + str(n)]
    else:
        color = session.get('color_' + str(n))
    return color


def get_default_color(n):
    ret_val = constants.get_defa()['color_' + str(n)]
    return ret_val


def get_trace_name(trace):
    if trace in ['new_hosp', 'total_beds', 'used_beds', 'total_in_resp',
                 'new_in_resp', 'total_resp', 'used_resp']:
        return resource_names[trace]
    elif trace in ['rt']:
        return rt_names[trace]
    else:
        return trace_names[trace]


def get_title(settings, trace, date='', short=False):
    name = trace_names[trace]
    if settings['dzielna'] != '<brak>':
        name = trace_names[trace] + ' / ' + trace_names[settings['dzielna']].lower()
    mean = constants.table_mean[settings['table_mean']]
    data_od = settings['from_date']
    if date == '':
        data_do = settings['to_date']
    else:
        data_do = date
    if settings['table_mean'] in ['mean', 'sum']:
        dates = ' (od ' + data_od + ' do ' + data_do + ')'
    else:
        dates = ''
    if constants.trace_props[trace]['category'] == 'data':
        tail = {1: '',
                2: ' - na 100 000 osób</sup>',
                3: ' - na 1000 km2</sup>'}[settings['data_modifier']]
    else:
        tail = ''
    if short:
        title = tail + '<br><sup>' + mean + dates
    else:
        title = name + tail + '<br><sup>' + mean + dates
    return title


def list_to_options(lista):
    ret_val = [{'label': str(i), 'value': i} for i in lista]
    return ret_val


def dict_to_options(dict):
    ret_val = [{'label': dict[i], 'value': i} for i in dict]
    return ret_val


def fields_to_options():
    ret_val = [{'label': constants.trace_props[x]['title'], 'value': x}
               for x in constants.trace_props]

    return ret_val


def fields_to_options_scope(scope):
    if scope == 'poland':
        disabled = 'disable_pl'
    elif scope == 'cities':
        disabled = 'disable_cities'
    else:
        disabled = 'disable_world'
    ret_val = [{'label': constants.trace_props[x]['title'], 'value': x}
               for x in constants.trace_props if not constants.trace_props[x][disabled]]
    return ret_val


def trace_options(scope, category):
    if scope == 'poland':
        return [{'label': x[1], 'value': x[0], 'disabled': y} for x, y in zip(constants.trace_names_list[category],
                                                                              constants.trace_disable_pl[category])]
    if scope == 'cities':
        return [{'label': x[1], 'value': x[0], 'disabled': y} for x, y in zip(constants.trace_names_list[category],
                                                                              constants.trace_disable_cities[category])]
    else:
        return [{'label': x[1], 'value': x[0], 'disabled': y} for x, y in zip(constants.trace_names_list[category],

                                                                              constants.trace_disable_world[category])]


def trace_list_options(traces):
    return [{'label': constants.trace_props[x]['title'], 'value': x} for x in traces]


def modal_message(txt):
    ret_val = \
        dbc.Modal(
            [
                dbc.ModalHeader('Komunikat'),
                dbc.ModalBody(txt),
                dbc.ModalFooter(
                    dbc.Button(
                        "Zamknij", id="modal_close", className="ml-auto"
                    )
                ),
            ],
            backdrop='static',
            is_open=True,
            id="modal_body",
        ),
    return ret_val


def xlate(word):
    ans = constants.country_dict.get(word)
    if ans is None:
        return word
    else:
        return ans


def xlate_array(arr):
    ans = np.array([xlate(x) for x in arr])
    return ans


def read_kolory():
    with open('data/dict/kolory.txt', encoding='utf-8') as f:
        lines = f.readlines()
    colors = {}
    new_color = ''
    new_options = []
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            if new_color:
                colors[new_color] = new_options
                new_options = []
            new_color = line[1:]
        else:
            temp = line.split()
            if len(temp) == 2:
                option = {'label': new_color + ' ' + temp[0], 'value': temp[1]}
                new_options.append(option)
    colors[new_color] = new_options
    return colors


def get_ticks(val):
    tick_bounds = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
    tick_vals = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    low_index = len([i for i in tick_bounds if i <= max(0.01, val.min())])
    high_index = len([i for i in tick_bounds if i <= max(0.01, val.max())])
    tickvals = tick_vals[low_index:high_index+1]
    ticktext = [str(i) for i in tick_bounds[low_index:high_index+1]]
    return tickvals, ticktext

def get_nticks(l):

    l0 = min(l)
    ln = max(l)
    l_bound = int(str(l0)[0])*10**(len(str(int(l0)))-1)
    u_bound = (int(str(ln)[0])+1)*10**(len(str(int(ln)))-1)
    n = 5
    dtick = (u_bound - l_bound) / 5
    tickvals = [l_bound + x * dtick for x in range(0, n + 1)]
    ticktext = [str(x) for x in tickvals]
    return tickvals, ticktext



def session_or_die(key, default=True):
    try:
        session[key]
    except:
        return default
    else:
        ret_val = session.get(key)
        if ret_val is None:
            return default
        return ret_val


def divide_ranges(inp, ranges=10):
    s = str(inp)
    if type(inp) == float:
        n = int(inp * 10 ** (len(s) - s.index('.') - 1))
        power = -(len(s) - s.index('.') - 1)
    else:
        n = inp
        power = 1
    s = str(n)
    if s[0] == '1':
        ul = int(str('2' + '0' * (len(s) - 1)))
    elif s[0] in ['2', '3', '4']:
        ul = int(str('5' + '0' * (len(s) - 1)))
    else:
        ul = int(str('1' + '0' * (len(s))))
    ran = [(ul // ranges * i) * 10 ** (power-1) for i in range(ranges+1)]
    return ran


def is_logged_in():
    return True


def get_help(_id):
    with open('apps/help/'+_id+'.md', encoding='utf-8') as f:
        text = _id+'\n![](/assets/tux.png)\n' + f.read()
    return text


def yesterday():
    return (dt.now() - timedelta(days=1)).strftime('%Y-%m-%d')


############ TOOLS ############


def get_modification_date(filename):
    ret_val = '%s' % datetime.datetime.fromtimestamp(os.path.getmtime(filename))
    return ret_val[:19]


def file_exists(fname):
    return os.path.exists(fname)


def mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def dir_exists(dirname):
    return os.path.exists(dirname)


def trace(traced, text=''):
    print(constants.bcolors.FAIL + 'TRACE (' + text  + ') ' + str(traced) + constants.bcolors.ENDC)


def debug(*msg):
    print(constants.bcolors.FAIL + "{0}".format(dt.now().strftime('%m-%d-%y %H:%M:%S')),
          constants.bcolors.OKBLUE + '[', msg, ']' + constants.bcolors.ENDC)


def trim_leading_nulls(df, column):
    nulls = df[df[column].notnull()]
    if len(nulls) == 0:
        return nulls
    else:
        first_valid = df[df[column].notnull()].index[0]
        return df.iloc[first_valid:].copy()


##########################################################
# Filtrowanie aktualnej bazy danych - jedna lokalizacja  #
##########################################################

@timing
def filter_data(settings, DF,
                loc='',
                trace='',
                scope='',
                one_day=False,
                ):
    data_modifier = settings['data_modifier']
    dzielna = settings['dzielna']
    table_mean = settings['table_mean']
    from_date = settings['from_date']
    to_date = settings['to_date']
    duration_d = settings['duration_d']
    duration_r = settings['duration_r']
    timeline_opt = settings['timeline_opt']
    smooth = settings['smooth']
    radio_flow = settings['radio_flow']

    df = DF[scope]
    filtered_df = df[df.location.str.lower() == loc.lower()].copy()
    filtered_df.reset_index(drop=True, inplace=True)

    filtered_df = trim_leading_nulls(filtered_df, trace)

    filtered_df = recalc(filtered_df, trace,
                         transform_after=table_mean,
                         timeline_opt=timeline_opt,
                         smooth=smooth,
                         divideby=dzielna,
                         duration_d=duration_d,
                         data_modifier=data_modifier,
                         parms=dict(duration_d=duration_d, duration_r=duration_r))

    # od daty

    if from_date and not one_day:
        filtered_df = filtered_df[filtered_df['date'] >= from_date[:10]].copy()

    # do daty

    if to_date and not one_day:
        filtered_df = filtered_df[filtered_df['date'] <= to_date[:10]].copy()

    # jeden dzień

    if one_day:
        if table_mean == 'daily':
            filtered_df = filtered_df[filtered_df['date'] == to_date[:10]].copy()
        else:
            filtered_df = filtered_df.tail(1).copy()

    # wykres proporcjonalny

    if radio_flow == 'proportional':
        vmax = max(filtered_df['data'])
        vmin = min(filtered_df['data'])
        filtered_df['data'] = (filtered_df['data'] - vmin) / (vmax - vmin)

    if len(filtered_df) > 0:
        if filtered_df.iloc[-1]['data'] == 0:
            filtered_df.at[filtered_df.index[-1], 'data'] = np.nan
            filtered_df.at[filtered_df.index[-1], 'data_points'] = np.nan

    return filtered_df.copy()

@timing
def prepare_data_new(DF,
                 locations=[],
                 scope='',
                 data_modifier=1,
                 dzielna='<brak>',
                 from_date='',
                 to_date='',
                 duration_d=1,
                 duration_r=1,
                 table_mean='daily',
                 ):
    filtered_df = pd.DataFrame()
    for loc in locations:
        filtered_df = filtered_df.append(
            filter_data(DF,
                        loc=loc,
                        trace=trace,
                        total_min=1,
                        scope=scope,
                        data_modifier=data_modifier,
                        dzielna=dzielna,
                        table_mean=table_mean,
                        from_date=from_date,
                        to_date=to_date,
                        duration_d=duration_d,
                        duration_r=duration_r,
                        ))
    return filtered_df



#####################################################
# Podstawowe filtrowanie danych - stara wersja      #
#####################################################

# scope, locations, chart_types, date, all_columns
@timing
def prepare_data(settings, DF,
                     locations=[],
                     chart_types=[],
                     scope='',
                     date='',
                     all_columns=False
                     ):
    data_modifier = settings['data_modifier']
    table_mean = settings['table_mean']
    from_date = settings['from_date']
    to_date = settings['to_date']
    dzielna = settings['dzielna']
    duration_d = settings['duration_d']
    duration_r = settings['duration_r']

    df = DF[scope]
    if scope == 'poland':
        filtered_df = df[df['location'].isin(locations)].copy()
        group = 'location'
    elif scope == 'cities':
        filtered_df = df[df['location'].isin(locations)].copy()
        group = ['location', 'wojew']
    else:
        filtered_df = df[df['location'].isin(locations)].copy()
        group = ['location', 'iso_code']
    filtered_df.reset_index(drop=True, inplace=True)

    if 'smiertelnosc' in chart_types:
        shift = filtered_df['total_cases']
        x = pd.concat([pd.Series([0] * duration_d), shift])
        lenn = len(x)
        xx = x.iloc[:lenn - duration_d]
        xx.reset_index(drop=True, inplace=True)
        zz = round(filtered_df['total_deaths'] / xx * 100, 3)
        zz.fillna(0, inplace=True)
        zz.replace(np.inf, 0, inplace=True)
        filtered_df['smiertelnosc'] = zz
    if 'wyzdrawialnosc' in chart_types:
        shift = filtered_df['total_cases']
        x = pd.concat([pd.Series([0] * duration_r), shift])
        lenn = len(x)
        xx = x.iloc[:lenn - duration_r]
        xx.reset_index(drop=True, inplace=True)
        zz = round(filtered_df['total_recoveries'] / xx * 100, 3)
        zz.fillna(0, inplace=True)
        zz.replace(np.inf, 0, inplace=True)
        filtered_df['wyzdrawialnosc'] = zz
        # filtered_df = filtered_df.iloc[duration_r:].copy()
    if 'dynamikaI' in chart_types:
        shift = filtered_df['new_cases']
        x = pd.concat([pd.Series([0] * duration_d), shift]).rolling(session['average_days'], min_periods=1).mean()
        lenn = len(x)
        xx = x.iloc[:lenn - duration_d]
        xx.reset_index(drop=True, inplace=True)
        zz = (filtered_df['new_cases'].rolling(session['average_days'], min_periods=1).mean() - xx) / xx + 1
        zz.fillna(0, inplace=True)
        zz.replace(np.inf, 0, inplace=True)
        filtered_df['dynamikaI'] = zz
    if 'dynamikaD' in chart_types:
        shift = filtered_df['new_deaths']
        x = pd.concat([pd.Series([0] * duration_d), shift]).rolling(session['average_days'], min_periods=1).mean()
        lenn = len(x)
        xx = x.iloc[:lenn - duration_d]
        xx.reset_index(drop=True, inplace=True)
        zz = (filtered_df['new_deaths'].rolling(session['average_days'], min_periods=1).mean() - xx) / xx + 1
        zz.fillna(0, inplace=True)
        zz.replace(np.inf, 0, inplace=True)
        filtered_df['dynamikaD'] = zz

    rounding = 3

    for trace in chart_types:
        if len(trace) > 0:
            if trace in constants.trace_props.keys():
                rounding = constants.trace_props[trace]['round']
            else:
                rounding = 2
        if dzielna == '<brak>':
            filtered_df[trace] = round(filtered_df[trace], rounding)
        else:
            filtered_df[trace] = filtered_df[trace]

        first_idx = filtered_df[trace].first_valid_index()
        last_idx = filtered_df[trace].last_valid_index()
        filtered_df = filtered_df.loc[first_idx:last_idx].copy()

    if from_date:
        filtered_df = filtered_df[filtered_df['date'] >= from_date[:10]].copy()

    if date != '':
        filtered_df = filtered_df[filtered_df['date'] <= date[:10]].copy()
    else:
        filtered_df = filtered_df[filtered_df['date'] <= to_date[:10]].copy()

    for trace in chart_types:

        if data_modifier == 2 and constants.trace_props[trace]['category'] == 'data':
            filtered_df[trace] = filtered_df[trace] / filtered_df['population'] * 100000
        elif data_modifier == 3 and constants.trace_props[trace]['category'] == 'data':
            filtered_df[trace] = filtered_df[trace] / filtered_df['area'] * 1000
        if dzielna != '<brak>' and constants.trace_props[trace]['category'] == 'data':
            filtered_df[trace] = (filtered_df[trace] / filtered_df[dzielna]).replace([np.inf, -np.inf], 0)

        # Dodanie pól _points

        filtered_df[trace + '_points'] = filtered_df[trace]

        # Przeliczenie danych

        if table_mean == 'monthly_mean':
            filtered_df.index = filtered_df['date'].astype('datetime64[ns]')
            filtered_df = filtered_df.groupby(group).resample('M').mean().reset_index()
            filtered_df['date'] = filtered_df['date'].dt.strftime('%Y-%m-%d')
        if table_mean == 'mean':
            filtered_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            filtered_df.dropna(subset=[trace], inplace=True)
            filtered_df[trace] = filtered_df.groupby(group)[trace].transform('mean')
            filtered_df = filtered_df.drop_duplicates(subset=group).copy()
        if table_mean == 'sum':
            filtered_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            filtered_df.dropna(subset=[trace], inplace=True)
            filtered_df[trace] = filtered_df.groupby(group)[trace].transform('sum')
            filtered_df = filtered_df.drop_duplicates(subset=group).copy()
        elif table_mean == 'monthly_sum':
            filtered_df.index = filtered_df['date'].astype('datetime64[ns]')
            filtered_df = filtered_df.groupby(group).resample('M').sum().reset_index()
            filtered_df['date'] = filtered_df['date'].dt.strftime('%Y-%m-%d')
        elif table_mean == 'weekly_mean':
            filtered_df.index = filtered_df['date'].astype('datetime64[ns]')
            filtered_df = filtered_df.groupby(group).resample('W').mean().reset_index()
            filtered_df['date'] = filtered_df['date'].dt.strftime('%Y-%m-%d')
        elif table_mean == 'biweekly_mean':
            filtered_df.index = filtered_df['date'].astype('datetime64[ns]')
            filtered_df = filtered_df.groupby(group).resample('2W').mean().reset_index()
            filtered_df['date'] = filtered_df['date'].dt.strftime('%Y-%m-%d')
        elif table_mean == 'weekly_sum':
            filtered_df.index = filtered_df['date'].astype('datetime64[ns]')
            filtered_df = filtered_df.groupby(group).resample('W').sum().reset_index()
            filtered_df['date'] = filtered_df['date'].dt.strftime('%Y-%m-%d')
        elif table_mean == 'biweekly_sum':
            filtered_df.index = filtered_df['date'].astype('datetime64[ns]')
            filtered_df = filtered_df.groupby(group).resample('2W').sum().reset_index()
            filtered_df['date'] = filtered_df['date'].dt.strftime('%Y-%m-%d')
        elif table_mean == 'daily_diff':
            xx = filtered_df.groupby('location')[trace].diff().reset_index()
            filtered_df[trace] = xx[trace].values
        elif table_mean == 'daily7_diff':
            xx = filtered_df.groupby('location')[trace].diff().reset_index()
            filtered_df[trace] = xx[trace].values
            filtered_df[trace] = filtered_df[trace].rolling(session['average_days'], min_periods=1).mean()
        elif table_mean == 'daily7':
            filtered_df[trace] = filtered_df[trace].rolling(session['average_days'], min_periods=1).mean()
        elif table_mean == 'dynamics':
            shift = filtered_df[trace]
            x = pd.concat([pd.Series([0] * 7), shift]).rolling(session['average_days'], min_periods=1).mean()
            lenn = len(x)
            xx = x.iloc[:lenn - 7]
            xx.reset_index(drop=True, inplace=True)
            filtered_df.reset_index(drop=True, inplace=True)
            zz = (filtered_df[trace].rolling(session['average_days'], min_periods=1).mean() - xx) / xx + 1
            zz.fillna(0, inplace=True)
            zz.replace(np.inf, 0, inplace=True)
            filtered_df[trace] = zz

        # filtered_df[trace] = filtered_df[trace].rolling(session['average_days'], min_periods=1).mean()

    if date != '':
        if table_mean in ['daily', 'daily7']:
            filtered_df = filtered_df[filtered_df['date'] == date[:10]].copy()
        else:
            filtered_df = filtered_df.groupby(group).tail(1).copy()


    fnames_points = [ct + '_points' for ct in chart_types]

    if all_columns:
        return filtered_df.copy()
    else:
        if scope == 'cities':
            return filtered_df[['date', 'location', 'wojew'] + chart_types + fnames_points].copy()
        else:
            return filtered_df[['date', 'location'] + chart_types + fnames_points].copy()


def create_html(fig, title):
    repo_dir = 'templates/'
    file = title + '_102.html'
    try:
        fig.write_html(repo_dir + file, auto_play=False)
    except:
        return 'Błąd I/O'
    return flask.request.referrer + 'go/' + file


def save_github(file, title):
    fn = file.split(sep='/')[-1]
    dst = '/home/docent/git_upload/'+fn
    from shutil import copyfile
    copyfile(file, dst)


def recalc(df, trace,
           transform_after='none',
           timeline_opt=[],
           data_modifier=1,
           smooth=3,
           divideby='<brak>',
           duration_d=7,
           parms={}):

    win_type = 'równe wagi'

    # pole kalkulowane

    if trace == 'smiertelnosc':
        shift = df['total_cases']
        x = pd.concat([pd.Series([0] * parms['duration_d']), shift])
        lenn = len(x)
        xx = x.iloc[:lenn - parms['duration_d']]
        xx.reset_index(drop=True, inplace=True)
        zz = round(df['total_deaths'] / xx * 100, 3)
        zz.fillna(0, inplace=True)
        zz.replace(np.inf, 0, inplace=True)
        df[trace] = zz
        df = df.iloc[parms['duration_d']:].copy()
    elif trace == 'wyzdrawialnosc':
        shift = df['total_cases']
        x = pd.concat([pd.Series([0] * parms['duration_r']), shift])
        lenn = len(x)
        xx = x.iloc[:lenn - parms['duration_r']]
        xx.reset_index(drop=True, inplace=True)
        zz = round(df['total_recoveries'] / xx * 100, 3)
        zz.fillna(0, inplace=True)
        zz.replace(np.inf, 0, inplace=True)
        df[trace] = zz
        df = df.iloc[parms['duration_r']:].copy()
    elif trace == 'dynamikaI':
        shift = df['new_cases']
        x = pd.concat([pd.Series([0] * parms['duration_d']), shift]).\
            rolling(session['average_days'], min_periods=1).mean()
        lenn = len(x)
        xx = x.iloc[:lenn - parms['duration_d']]
        xx.reset_index(drop=True, inplace=True)
        zz = (df['new_cases'].rolling(session['average_days'], min_periods=1).mean() - xx) / xx + 1
        zz.fillna(0, inplace=True)
        zz.replace(np.inf, 0, inplace=True)
        df[trace] = zz
        df = df.iloc[parms['duration_d']:].copy()
    elif trace == 'dynamikaD':
        shift = df['new_deaths']
        x = pd.concat([pd.Series([0] * parms['duration_d']), shift]).\
            rolling(session['average_days'], min_periods=1).mean()
        lenn = len(x)
        xx = x.iloc[:lenn - parms['duration_d']]
        xx.reset_index(drop=True, inplace=True)
        zz = (df['new_deaths'].rolling(session['average_days'], min_periods=1).mean() - xx) / xx + 1
        zz.fillna(0, inplace=True)
        zz.replace(np.inf, 0, inplace=True)
        df[trace] = zz
        df = df.iloc[parms['duration_d']:].copy()

    df['data'] = df[trace]

    if data_modifier == 2 and constants.trace_props[trace]['category'] == 'data':
        df['data'] = round(df['data'] / df['population'] * 100000, 3)
    elif data_modifier == 3 and constants.trace_props[trace]['category'] == 'data':
        df['data'] = round(df['data'] / df['area'] * 1000, 3)

    if divideby != '<brak>':
        df['data'] = df['data'].ffill()
        df.dropna(axis='rows', subset=['data'], inplace=True)
        df['data'] = df['data'].rolling(7, min_periods=7).mean()
        df[divideby] = df[divideby].rolling(7, min_periods=7).mean()
        if data_modifier == 2 and constants.trace_props[trace]['category'] == 'data':
            df[divideby] = round(df[divideby] / df['population'] * 100000, 3)
        elif data_modifier == 3 and constants.trace_props[trace]['category'] == 'data':
            df[divideby] = round(df[divideby] / df['area'] * 1000, 3)
        df['data'] = df['data'] / df[divideby]
        df['data'] = df['data'].replace(np.inf, 0)
        df['data'] = df['data'].replace(np.nan, 0)

    # przetwarzanie końcowe

    if transform_after == 'monthly_mean':
        df.index = df['date'].astype('datetime64[ns]')
        df = df.groupby('location').resample('M').mean().reset_index()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    elif transform_after == 'mean':
        df['data'] = df.groupby('location')[trace].mean()
        df = df.drop_duplicates(subset=['location']).copy()
    elif transform_after == 'sum':
        df['data'] = df.groupby('location')[trace].sum()
        df = df.drop_duplicates(subset=['location']).copy()
    elif transform_after == 'monthly_sum':
        df.index = df['date'].astype('datetime64[ns]')
        df = df.groupby('location').resample('M').sum().reset_index()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    elif transform_after == 'weekly_mean':
        df.index = df['date'].astype('datetime64[ns]')
        df = df.groupby('location').resample('W').mean().reset_index()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    elif transform_after == 'biweekly_mean':
        df.index = df['date'].astype('datetime64[ns]')
        df = df.groupby('location').resample('2W').mean().reset_index()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    if transform_after == 'weekly_sum':
        df.index = df['date'].astype('datetime64[ns]')
        df = df.groupby('location').resample('W').sum().reset_index()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    elif transform_after == 'biweekly_sum':
        df.index = df['date'].astype('datetime64[ns]')
        df = df.groupby('location').resample('2W').sum().reset_index()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    elif transform_after == 'daily_diff':
        xx = df.groupby('location')['data'].diff().reset_index()
        df['data'] = xx['data'].values
    elif transform_after == 'daily7_diff':
        xx = df.groupby('location')['data'].rolling(session['average_days'], min_periods=1).mean().diff().reset_index()
        df['data'] = xx['data'].values
        # df['data'] = df['data'].rolling(session['average_days'], min_periods=1).mean()
    elif transform_after == 'daily7':
        df['data'] = df['data'].rolling(session['average_days'], min_periods=1).mean()
    elif transform_after == 'dynamics':
        shift = df['data']
        x = pd.concat([pd.Series([0] * 7), shift]).rolling(session['average_days'], min_periods=1).mean()
        lenn = len(x)
        xx = x.iloc[:lenn - 7]
        xx.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        zz = (df['data'].rolling(session['average_days'], min_periods=1).mean() - xx) / xx + 1
        zz.fillna(0, inplace=True)
        zz.replace(np.inf, 0, inplace=True)
        df['data'] = zz

    df['data_points'] = df['data']

    # opcje wygładzania wyniku

    if 'usrednianie' in timeline_opt:
        if win_type == 'równe wagi':
            df['data'] = df['data'].rolling(session['average_days'], min_periods=1).mean()
        elif win_type == 'gaussian':
            df['data'] = df['data'].rolling(session['average_days'], min_periods=1, win_type='gaussian').mean(std=0.5)
        else:
            df['data'] = df['data'].rolling(session['average_days'], min_periods=1, win_type=win_type).mean()
        df['data'].fillna(0, inplace=True)
    if 'usrednianieo' in timeline_opt:
        df.index = df['date'].astype('datetime64[ns]')
        df['data'] = round(df['data'].resample(str(session['average_days'])+'D').mean(), 3)
        df.dropna(axis='rows', subset=['data'], inplace=True)
    if 'wygladzanie' in timeline_opt:
        if session['smooth_method'] == 'wielomiany':
            xvals = np.array(pd.to_datetime(df['date']).astype(int))
            yvals = df['data']
            z = np.polyfit(xvals, yvals, smooth)
            f = np.poly1d(z)
            yvals = f(xvals)
        else:
            yvals = df['data']
            try:
                tmpyvals = signal.savgol_filter(yvals, 53, smooth)
            except Exception as e:
                print('Wystapil wyjatek : ', str(e))
            else:
                yvals = tmpyvals
        df['data'] = [round(yvals[i], 3) for i in range(len(yvals))]

        df['data'] = df['data'].round(constants.trace_props[trace]['round'])

    return df


def fn(x, digits=2):
    x1 = str(x)
    if '.' in x1:
        n = round(float(x), digits)
    else:
        n = int(x)
    ret_val = '{:,}'.format(n).replace(',', '  ')
    return ret_val

def add_copyright(obj, settings, x=0, y=-0):
    if settings['copyrightxpos'] > -0.20:
        obj.add_annotation(text='@docent_ws #TAN',
                              xref="paper", yref="paper",
                              align='left',
                              bgcolor=settings['color_5'],
                              font=dict(color=settings['color_7'], size=settings['font_size_legend']),
                              x=settings['copyrightxpos'] + x, y=settings['copyrightypos'] + y, showarrow=False)
    return obj


def get_bins(series, nbins, map_cut='własne'):
    # map_cut: równe, kwantyle, własne
    if map_cut == 'równe':
        from operator import attrgetter
        s, bins = pd.cut(series, nbins, retbins=True, duplicates='drop')
        bin_labels = [str(i) for i in bins[:-1]]
        ret_val = s.map(attrgetter('left')).astype(float)
    elif map_cut == 'kwantyle':
        from operator import attrgetter
        s, bins = pd.qcut(series, nbins, retbins=True, duplicates='drop')
        bin_labels = [str(i) for i in bins[:-1]]
        ret_val = s.map(attrgetter('left')).astype(float)
    else:
        xmax = max(series)
        maxes = [10000, 5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 5, 2, 1, 0.1, 0.01, 0.001, 0.0001]
        x = 0
        for n in maxes:
            if xmax > n:
                x = n
                break
        if x == 0:
            return [], 'Błąd: xmax spoza zakresu '
        bins = [0., .1*x, .2*x, .5*x, x, 10000000]
        bin_labels = [str(i) for i in bins[:-1]]
        s = pd.cut(series, bins=bins, labels=bin_labels).astype(float)
        s.fillna(0, inplace=True)
        ret_val = s

    return ret_val, bins, bin_labels

def discrete_colorscale(bvals, colors):
    """
    bvals - list of values bounding intervals/ranges of interest
    colors - list of rgb or hex colorcodes for values in [bvals[k], bvals[k+1]],0<=k < len(bvals)-1
    returns the plotly  discrete colorscale
    """
    if len(bvals) != len(colors) + 1:
        raise ValueError('len(boundary values) should be equal to  len(colors)+1')
    bvals = sorted(bvals)
    nvals = [(v - bvals[0]) / (bvals[-1] - bvals[0]) for v in bvals]  # normalized values

    dcolorscale = []  # discrete colorscale
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k + 1], colors[k]]])
    return dcolorscale