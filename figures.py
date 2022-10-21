from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
# from flask import session
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import constants
from constants import trace_props, mapbox_styles, image_logo_timeline, image_logo_dynamics, color_scales, \
    table_mean
from util import get_session_color, get_trace_name, get_ticks, span, session2settings, is_logged_in, \
    filter_data, prepare_data, get_template, add_copyright, timing, get_title
from layouts import layout_title, layout_legend
from plotly.subplots import make_subplots
import datetime
import json
import math
import dash_tabulator
import random
from config import session


#######################################
# Redraw single chart - timeline types
#######################################

@timing
def update_figures_timeline_types(DF):
    template = get_template()
    settings = session2settings()
    chart_type = settings['chart_type_data'] + settings['chart_type_calculated']

    def calculate_figure_timeline_types():

        filtered_df = filter_data(settings, DF,
                                  loc=location,
                                  trace=trace,
                                  scope=settings['scope'],
                                  )
        if settings['radio_scale'] == 'log':
            filtered_df['data'] = filtered_df['data'].ffill()
            filtered_df.dropna(axis='rows', subset=['data'], inplace=True)
            filtered_df['data'] = filtered_df['data'].replace(0, 0.00001)
        _fig_data_scatter = {}
        if ('usrednianie' in settings['timeline_opt'] or 'wygladzanie' in settings['timeline_opt']) \
                and 'points' in settings['options_yesno']:
            _fig_data_scatter = go.Scatter(
                x=list(filtered_df['date']),
                y=list(filtered_df['data_points']),
                name=location,
                mode='markers',
                # mode='lines+markers+text',
                line=dict(width=settings['linewidth_thin']),
                marker=dict(symbol='circle-open', size=6),
                opacity=0.5
            )

        if settings['line_dash'] == 'solid':
            dash = 'solid'
        else:
            dash = constants.dashes[random.randint(0, len(constants.dashes) - 1)]
        line = dict(width=settings['linewidth_basic'], dash=dash)
        marker = dict(line=dict(color='black', width=float(settings['bar_frame'])),
                                pattern=dict(shape=settings['bar_fill']))
        opacity = 1
        fill = 'none'
        if settings['timeline_highlight'] != 'Brak':
            if settings['radio_type'] == 'bar':
                opacity = 0.2
            if location == settings['timeline_highlight']:
                line = dict(width=settings['linewidth_thick'], color=get_session_color(9))
                fill = 'tozeroy'
                opacity = 1

        if settings['radio_type'] == 'scatter':
            _fig_data = go.Scatter(
                x=list(filtered_df['date'].astype('datetime64[ns]')),
                y=list(filtered_df['data']),
                line=line,
                fill=fill,
                type=settings['radio_type'],
                name=location,
                mode=session.get('linedraw'),
                opacity=opacity
            )
        else:
            _fig_data = go.Bar(
                x=list(filtered_df['date'].astype('datetime64[ns]')),
                y=list(filtered_df['data']),
                type=settings['radio_type'],
                marker=marker,
                name=location,
                opacity=opacity
            )
        return [_fig_data, _fig_data_scatter]

    traces = {key: [] for key in chart_type}
    children = []
    height = 660
    wid = 12
    if len(chart_type) == 1:
        rows = 1
        cols = 1
        wid = 12
        height = 660
    elif len(chart_type) == 2:
        rows = 1
        cols = 2
        wid = 6
        height = 660
    elif len(chart_type) <= 4:
        rows = 2
        cols = 2
        wid = 6
        height = 330
    elif len(chart_type) <= 6:
        rows = 2
        cols = 3
        wid = 4
        height = 330
    else:
        rows = 3
        cols = 3
        wid = 4
        height = 220
    height = float(settings['plot_height']) * height
    xaxis_text = ''
    if 'usrednianie' in settings['timeline_opt']:
        xaxis_text += 'uśrednianie (' + str(settings['average_days']) + ' dni) '
    if 'usrednianieo' in settings['timeline_opt']:
        xaxis_text += 'średnia okresowa (' + str(settings['average_days']) + ' dni) '
    if 'wygladzanie' in settings['timeline_opt']:
        xaxis_text += 'wygładzanie (' + str(settings['smooth']) + ' stopnia) '
    dtick = ''
    for trace in chart_type:
        xanno = []
        yanno = []
        txt = []
        for location in settings['locations']:
            fig_data, fig_data_scatter = calculate_figure_timeline_types()
            if len(fig_data['y']) == 0:
                continue
            if 'max' in settings['annotations']:
                max_y = max(fig_data['y'])
                index_max = fig_data['y'].index(max_y)
                yanno.append(max_y)
                xanno.append(fig_data['x'][index_max])
                txt.append(location)
            if 'last' in settings['annotations']:
                yanno.append(fig_data['y'][-1])
                xanno.append(fig_data['x'][-1])
                txt.append(location)
            if settings['annotations'] == 'all':
                yanno.extend([fig_data['y'][i] for i in range(len(fig_data['y']))])
                xanno.extend([fig_data['x'][i] for i in range(len(fig_data['x']))])
                txt.extend(['' for i in range(len(fig_data['x']))])
            traces[trace].append(fig_data)
            if fig_data_scatter:
                traces[trace].append(fig_data_scatter)
        for i in range(len(yanno)):
            yanno[i] = round(yanno[i], 2)
        if trace in ['dynamikaI', 'reproduction_rate']:
            fig_data_1 = go.Scatter(
                x=fig_data['x'],
                y=[1 for i in range(len(fig_data['x']))],
                line=dict(color='yellow', width=0.5),
                name='1',
                opacity=0.7
            )
            traces[trace].append(fig_data_1)
        legend_y = 1.
        tickvals = pd.DatetimeIndex(fig_data['x']).month.unique()
        ticktext = pd.DatetimeIndex(fig_data['x']).month.unique()
        if settings['dzielna'] != '<brak>' and trace_props[trace]['category'] == 'data':
            trace_name = get_trace_name(trace) + ' / ' + get_trace_name(settings['dzielna']).lower()
        def anno(i):
            prefix = ''
            if len(settings['timeline_opt']) > 0:
                prefix = ''
                # prefix = '~'
            if settings['anno_form'] == 'num':
                ret_val = '<b>' + prefix + str(yanno[i])
            elif settings['anno_form'] == 'name':
                ret_val = txt[i]
            elif settings['anno_form'] == 'namenum':
                ret_val = txt[i] + '<br><b>' + prefix + str(yanno[i])
            else:
                ret_val = txt[i] + '<br>' + str(xanno[i])[:10] + '<br><b>' + prefix + str(yanno[i])
            return ret_val
        figure = go.Figure(
            data=traces[trace],
            layout=dict(
                bargap=float(settings['bar_gap']),
                barmode=settings['bar_mode'],
                annotations=[{'x': xanno[i], 'y': yanno[i],
                              'text': anno(i),
                              'font': {'size': int(settings['font_size_anno']), 'color': get_session_color(4)},
                              'arrowcolor': 'red',
                              'bgcolor': get_session_color(5)
                              } for i in range(len(xanno))],
                title=dict(text=get_title(settings, trace)),
                xaxis={
                    'rangeslider': {'visible': ('suwak' in settings['timeline_view']), 'thickness': 0.05},
                    'tickmode': 'array',
                    'nticks': len(tickvals),
                    'ticktext': ticktext,
                    'dtick': dtick,
                    'title': {'text': xaxis_text, 'standoff': 5}
                },
                yaxis={
                    'exponentformat': 'none',
                    'title': {'text': '', 'font': {'size': int(settings['font_size_xy']), 'color': get_session_color(7)}},
                    'type': 'log' if settings['radio_scale'] == 'log' else 'linear',
                },
                height=height,
                margin=dict(l=settings['marginl'] + 50, r=settings['marginr'] + 40, b=settings['marginb'] + 50, t=settings['margint'] + 40,
                            pad=4),
                template=template,
                colorway=color_scales[settings['color_order']],
                showlegend=('legenda' in settings['timeline_view']),
                legend=layout_legend(y=legend_y, legend_place=settings['legend_place'])
            )
        )
        figure = add_copyright(figure, settings, y=-0.12)
        config = {
            'displaylogo': False,
            'responsive': True,
            'locale': 'pl-PL',
            'toImageButtonOptions': {
                'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
                'filename': 'docent_image', 'height': height, 'width': 120 * wid, 'scale': 1}
        }
        f = dbc.Col(dcc.Graph(id=trace,
                              figure=figure,
                              config=config),
                    width=wid,
                    className=constants.nomargins)
        children.append(f)

    ret_val = []
    for i in range(rows):
        row = []
        for j in range(cols):
            nelem = i * cols + j
            if nelem < len(children):
                row.append(children[nelem])
        ret_val.append(dbc.Row(row))

    return dbc.Col(ret_val)


#######################################
# Redraw single chart - timeline places
#######################################
@timing
def update_figures_timeline_places(DF):
    template = get_template()
    settings = session2settings()
    chart_type = settings['chart_type_data'] + settings['chart_type_calculated']
    t_start = datetime.datetime.now()

    def calculate_figure_timeline_places():
        filtered_df = filter_data(settings, DF,
                                  loc=location,
                                  trace=trace,
                                  scope=settings['scope'],
                                  )
        if settings['radio_scale'] == 'log':
            filtered_df['data'] = filtered_df['data'].ffill()
            filtered_df.dropna(axis='rows', subset=['data'], inplace=True)
            filtered_df['data'] = filtered_df['data'].replace(0, 0.00001)
        _fig_data_scatter = {}
        if settings['dzielna'] != '<brak>' and trace_props[trace]['category'] == 'data':
            trace_name = get_trace_name(trace) + ' / ' + get_trace_name(settings['dzielna']).lower()
        else:
            trace_name = get_trace_name(trace)
        if ('usrednianie' in settings['timeline_opt'] or 'wygladzanie' in settings['timeline_opt']) \
                and 'points' in settings['options_yesno']:
            _fig_data_scatter = go.Scatter(
                x=list(filtered_df['date']),
                y=list(filtered_df['data_points']),
                name=trace_name,
                mode='lines+markers+text',
                line=dict(width=settings['linewidth_thin']),
                marker=dict(symbol='circle', size=3),
                opacity=0.5
            )

        # konstrukcja nitki wykresu

        if settings['line_dash'] == 'solid':
            dash = 'solid'
        else:
            dash = constants.dashes[random.randint(0, len(constants.dashes) - 1)]
        line = dict(width=settings['linewidth_basic'], dash=dash)
        marker = dict(line=dict(color='black', width=float(settings['bar_frame'])),
                      pattern=dict(shape=settings['bar_fill']))
        opacity = 1
        if settings['timeline_highlight'] != 'Brak':
            opacity = 0.6
            if location == settings['timeline_highlight']:
                line = dict(width=settings['linewidth_thick'], color=get_session_color(9))
                opacity = 1
        if settings['radio_type'] == 'scatter':
            _fig_data = go.Scatter(
                x=list(filtered_df['date'].astype('datetime64[ns]')),
                y=list(filtered_df['data']),
                line=line,
                type=settings['radio_type'],
                name=trace_name,
                mode=session.get('linedraw'),
                opacity=opacity
            )
        else:
            _fig_data = go.Bar(
                x=list(filtered_df['date'].astype('datetime64[ns]')),
                y=list(filtered_df['data']),
                type=settings['radio_type'],
                marker=marker,
                name=trace_name,
                opacity=opacity
            )
        return [_fig_data, _fig_data_scatter]

    traces = {key: [] for key in settings['locations']}
    children = []
    height = 660
    wid = 12
    if len(settings['locations']) == 1:
        rows = 1
        cols = 1
        wid = 12
        height = 660
    elif len(settings['locations']) == 2:
        rows = 1
        cols = 2
        wid = 6
        height = 660
    elif len(settings['locations']) <= 4:
        rows = 2
        cols = 2
        wid = 6
        height = 330
    elif len(settings['locations']) <= 6:
        rows = 2
        cols = 3
        wid = 4
        height = 330
    elif len(settings['locations']) <= 9:
        rows = 3
        cols = 3
        wid = 4
        height = 220
    elif len(settings['locations']) <= 12:
        rows = 3
        cols = 4
        wid = 3
        height = 220
    else:
        rows = 4
        cols = 4
        wid = 3
        height = 165
    height = float(settings['plot_height']) * height
    xaxis_text = ''
    if 'usrednianie' in settings['timeline_opt']:
        xaxis_text += 'uśrednianie (' + str(settings['average_days']) + ' dni) '
    if 'usrednianieo' in settings['timeline_opt']:
        xaxis_text += 'średnia okresowa (' + str(settings['average_days']) + ' dni) '
    if 'wygladzanie' in settings['timeline_opt']:
        xaxis_text += 'wygładzanie (' + str(settings['smooth']) + ' stopnia) '
    dtick = ''
    for location in settings['locations']:
        xanno = []
        yanno = []
        txt = []
        tail = ''
        for trace in chart_type:
            fig_data, fig_data_scatter = calculate_figure_timeline_places()
            if len(fig_data['y']) > 0:
                if 'max' in settings['annotations']:
                    max_y = max(fig_data['y'])
                    index_max = fig_data['y'].index(max_y)
                    yanno.append(max_y)
                    xanno.append(fig_data['x'][index_max])
                    txt.append(get_trace_name(trace))
                if 'last' in settings['annotations']:
                    yanno.append(fig_data['y'][-1])
                    xanno.append(fig_data['x'][-1])
                    txt.append(get_trace_name(trace))
                if settings['annotations'] == 'all':
                    yanno.extend([fig_data['y'][i] for i in range(len(fig_data['y']))])
                    xanno.extend([fig_data['x'][i] for i in range(len(fig_data['x']))])
                    txt.extend(['' for i in range(len(fig_data['x']))])
                traces[location].append(fig_data)
                if fig_data_scatter:
                    traces[location].append(fig_data_scatter)
        for i in range(len(yanno)):
            yanno[i] = round(yanno[i], 2)
        legend_y = 1.
        tickvals = pd.DatetimeIndex(fig_data['x']).month.unique()
        ticktext = pd.DatetimeIndex(fig_data['x']).month.unique()
        print('places', settings['color_order'])

        def anno(i):
            prefix = ''
            if len(settings['timeline_opt']) > 0:
                prefix = ''
                # prefix = '~'
            if settings['anno_form'] == 'num':
                ret_val = '<b>' + prefix + str(yanno[i])
            elif settings['anno_form'] == 'name':
                ret_val = txt[i]
            elif settings['anno_form'] == 'namenum':
                ret_val = txt[i] + '<br><b>' + prefix + str(yanno[i])
            else:
                ret_val = txt[i] + '<br>' + str(xanno[i])[:10] + '<br><b>' + prefix + str(yanno[i])
            return ret_val

        figure = go.Figure(
            data=traces[location],
            layout=dict(
                bargap=float(settings['bar_gap']),
                barmode=settings['bar_mode'],
                template=template,
                annotations=[{'x': xanno[i], 'y': yanno[i],
                              'text': anno(i),
                              'font': {'size': int(settings['font_size_anno']), 'color': get_session_color(4)},
                              'arrowcolor': 'red',
                              'bgcolor': get_session_color(5)
                              } for i in range(len(xanno))],
                title=dict(text=location + get_title(settings, trace, short=True)),
                xaxis={
                    'rangeslider': {'visible': ('suwak' in settings['timeline_view']), 'thickness': 0.05},
                    'tickmode': 'array',
                    'nticks': len(tickvals),
                    'ticks': 'outside',
                    'ticktext': ticktext,
                    'dtick': dtick,
                    'tickfont': {'size': int(settings['font_size_xy']), 'color': get_session_color(3)},
                    'title': {'text': xaxis_text, 'standoff': 25}
                },
                yaxis={
                    'title': {'text': ''},
                    'exponentformat': 'none',
                    'separatethousands': False,
                    'type': 'log' if settings['radio_scale'] == 'log' else 'linear',
                    # 'type': 'log' if trace_props[trace]['log_scale'] and settings['radio_scale'] == 'log' else 'linear',
                    # 'type': 'linear',
                },
                height=height,
                margin=dict(l=settings['marginl'] + 25,
                            r=settings['marginr'] + 40,
                            b=settings['marginb'] + 20,
                            t=settings['margint'] + 20,
                            pad=4),
                colorway=color_scales[settings['color_order']],
                showlegend=('legenda' in settings['timeline_view']),
                legend=layout_legend(y=legend_y,
                                     legend_place=settings['legend_place'])
            )
        )
        figure = add_copyright(figure, settings, y=-0.12)
        config = {
            'displaylogo': False,
            'responsive': True,
            'locale': 'pl-PL',
            'toImageButtonOptions': {
                'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
                'filename': 'docent_image', 'height': height, 'width': 120 * wid, 'scale': 1}
        }
        f = dbc.Col(dcc.Graph(id=location,
                              figure=figure,
                              config=config),
                    width=wid,
                    style={'background': constants.get_defa()['color_1']},
                    className=constants.nomargins)
        children.append(f)

    ret_val = []
    for i in range(rows):
        row = []
        for j in range(cols):
            nelem = i * cols + j
            if nelem < len(children):
                row.append(children[nelem])
        ret_val.append(dbc.Row(row))
    return dbc.Col(ret_val)

###################
# update rankings
###################

@timing
def update_figures_rankings(DF):
    template = get_template()
    settings = session2settings()
    chart_type = settings['chart_type_data'] + settings['chart_type_calculated']
    t_start = datetime.datetime.now()

    def data_for_rankings():
        if type(settings['locations']) == str:
            locations = [settings['locations']]
        else:
            locations = settings['locations']

        filtered_df = pd.DataFrame()
        for loc in settings['locations']:
            filtered_df = filtered_df.append(
                filter_data(settings, DF,
                            loc=loc,
                            trace=trace,
                            scope=settings['scope'],
                            one_day=True,
                            ))
        filtered_df = filtered_df.sort_values(by=['data'], ascending=False)
        _xvals = list(filtered_df['location'])
        _yvals = list(round(filtered_df['data'], 3))

        # brak danych dla danego dnia

        if len(_xvals) == 0:
            return _xvals, _yvals

        # ograniczenie liczby widocznych pozycji, pozostałe jako "pozostałe"

        if settings['max_cut'] != '0':
            if len(locations) > int(settings['max_cut']):
                pos = min(int(settings['max_cut']), len(locations) - 1)
                _xvals = _xvals[:pos].copy()
                _yvals = _yvals[:pos].copy()
                if 'rest' in settings['options_rank']:
                    _xvals.append('pozostałe')
                    _yvals.append(sum(list(filtered_df['data'])[pos:]))

        return _xvals, _yvals

    children = []
    wid = 12 if len(chart_type) < 2 else 6
    # figure = {}
    for trace in chart_type:
        xvals, yvals = data_for_rankings()
        xvals.reverse()
        yvals.reverse()
        height = settings['margint'] + 75 + settings['marginb'] + 22 + 25 * float(settings['plot_height']) * len(xvals)
        colors = [get_session_color(8)] * len(xvals)
        if settings['rankings_highlight'] in xvals:
            colors[xvals.index(settings['rankings_highlight'])] = get_session_color(9)
        fig_data = go.Bar(
            x=yvals,
            y=xvals,
            text=yvals,
            textposition='auto',
            orientation='h',
            hoverinfo='x+y',
            name=trace,
            marker={'color': colors},
            # margin=dict(l=50, r=20, b=50, t=20, pad=4),
        )
        trace_name = get_trace_name(trace)
        if settings['dzielna'] != '<brak>':
            trace_name = get_trace_name(trace) + ' / ' + get_trace_name(settings['dzielna']).lower()
        figure = go.Figure(
            data=[fig_data],
            layout=dict(
                template=template,
                xaxis=dict(side='top',
                           automargin=True),
                yaxis=dict(automargin=True),
                height=height,
                margin=dict(l=settings['marginl'], r=settings['marginr'], b=settings['marginb'],
                            t=settings['margint'] + 75, pad=0),
                title=dict(text=get_title(settings, trace) + ' (' + settings['date_picker'][:10] + ')'),
                font=dict(size=16, color="white"),
            )
        )
        graph_config = {
            'displaylogo': False,
            'responsive': True,
            'locale': 'pl-PL',
            'toImageButtonOptions': {
                'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
                'filename': 'docent_image', 'width': 120 * wid, 'scale': 0.7}
        }
        if len(xvals) == 0:
            children.append('-- brak danych dla ' + trace + ' (' + settings['date_picker'][:10] + ')')
        else:
            children.append(
                dbc.Col(dcc.Graph(id=trace, figure=figure, config=graph_config), width=wid, className='mt-4'))

    return children


###################
# Update dynamics
###################

@timing
def update_figures_dynamics(DF):
    template = get_template()
    settings = session2settings()
    chart_type = settings['chart_type_data'] + settings['chart_type_calculated']
    t_start = datetime.datetime.now()

    # margint = session.get('margint') + 100

    def calculate_figure_dynamics():
        filtered = filter_data(settings, DF,
                               loc=location,
                               trace=chart,
                               scope=settings['scope'],
                               )
        filtered.sort_values(by=['location', 'date'], inplace=True)
        filtered.reset_index(drop=True, inplace=True)
        xvals = list(filtered['date'])

        # Obliczenie uśredniedniania

        # yvals = list(filtered['data'].rolling(session['average_days'], min_periods=1).mean())
        yvals = list(filtered['data'])

        yvals = [0 if math.isnan(i) else i for i in yvals]
        fig_data = go.Scatter(x=xvals,
                        y=yvals,
                        line=dict(color='brown'),
                        name=location)

        # obliczenie wskaźnika koloru wykresu
        # - zielony: min < 50% max, ost < min + 20% * (max-min)
        # - czerwony: min >= 50% max, ost >= min + 20% (max-min)
        # - pomarańczowy - pozostałe
        if 'new_' in chart and settings['dzielna'] == '<brak':
            y_max = max(yvals)
            i_max = yvals.index(y_max)
            y_min = min(yvals[i_max:])
            y_last = yvals[-1]
            if (y_min < 0.5 * y_max) and (y_last < (0.2 * (y_max-y_min))):
                color = 'green'
            elif (y_min >= 0.5 * y_max) and (y_last >= (0.2 * (y_max-y_min))):
                color = 'red'
            else:
                color = 'orange'
            fig_data.line['color'] = color
        if len(xvals) > 0:
            return fig_data, min(xvals), max(xvals), min(yvals), max(yvals)
        else:
            return fig_data, 0, 0, 0, 0

    children = []
    for chart in chart_type:
        rows = int((len(settings['locations']) - 1) / settings['columns']) + 1
        figure = make_subplots(rows=rows,
                               cols=settings['columns'],
                               subplot_titles=settings['locations'],
                               print_grid=False,
                               )
        row = 1
        col = 0
        date_min = '2099-12-31'
        date_max = '1899-01-01'
        value_min = 999999999999999
        value_max = -999999999999999
        for location in settings['locations']:
            col += 1
            if col > settings['columns']:
                row += 1
                col = 1
            f, x_min, x_max, y_min, y_max = calculate_figure_dynamics()
            date_min = min(date_min, x_min)
            date_max = max(date_max, x_max)
            value_min = min(value_min, y_min)
            value_max = max(value_max, y_max)
            figure.add_trace(f, row=row, col=col)

        if settings['dynamics_scaling'] == 'Zachowanie proporcji':
            figure.update_yaxes(range=[value_min, value_max])
            figure.update_xaxes(range=[date_min, date_max], showticklabels=True),
        height = settings['margint'] + 100 + int(settings['dynamics_chart_height']) * row
        trace_name = get_trace_name(chart)
        if settings['dzielna'] != '<brak>':
            trace_name = get_trace_name(chart) + ' / ' + get_trace_name(settings['dzielna']).lower()
        figure.update_layout(height=height,
                             images=image_logo_dynamics,
                             title=dict(text=get_title(settings, chart)),
                             margin=dict(l=settings['marginl'], r=settings['marginr'], b=settings['marginb'],
                                         t=settings['margint'] + 100, pad=0),
                             template=template,
                             showlegend=False),
        figure = add_copyright(figure, settings, y=-0.12)
        graph_config = {
            'displaylogo': False,
            'responsive': True,
            'locale': 'pl-PL',
            'toImageButtonOptions': {
                'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
                'filename': 'docent_image',
                'height': int(settings['dynamics_chart_height']) * row,
                'width': 120 * 12, 'scale': 1}
        }
        fig = dcc.Graph(id=location, figure=figure, config=graph_config)
        children.append(dbc.Col(fig, width=12))
    return children


############################################
# Korelator
############################################
def update_figures_core(DF):
    from sklearn.linear_model import LinearRegression
    template = get_template()
    settings = session2settings()
    from_date = settings['from_date']
    locations = settings['locations']
    core_opt = settings['core_opt']
    core_view = settings['core_view']
    chart_types = settings['chart_type']
    core_highlight = settings['core_highlight']
    if settings['from_date'] is None:
        return
    if settings['to_date'] is None:
        return
    agg_opis = {'mean': 'średnia',
               'sum': 'suma',
               'max': 'maksimum',
               'min': 'minimum',
               'median': 'mediana',
               'oneday': 'jeden dzień'}

    anno_text = 'Rodzaj agregacji X: ' + agg_opis[settings['core_agg_x']] + '<br>' + \
                'Rodzaj agregacji Y: ' + agg_opis[settings['core_agg_y']] + '<br>' + \
                'Wyróżnienie: ' + settings['core_highlight'] + '<br>' + \
                ''
    if len(locations) == 0:
        return 'Wybierz 2 wielkości'
    locations.sort()
    if len(chart_types) != 2:
        return 'Wybierz 2 wielkości'
    if 'flipaxes' in core_opt:
        _fieldx = chart_types[0]
        _fieldy = chart_types[1]
    else:
        _fieldx = chart_types[1]
        _fieldy = chart_types[0]
    tailx = ''
    taily = ''
    if trace_props[_fieldx]['category'] == 'data':
        tailx = {1: '',
            2: ' - na 100 000 osób',
            3: ' - na 1000 km2'}[settings['data_modifier']]
    if trace_props[_fieldy]['category'] == 'data':
        taily = {1: '',
            2: ' - na 100 000 osób',
            3: ' - na 1000 km2'}[settings['data_modifier']]
    name_x = constants.trace_props[_fieldx]['title'] + tailx
    name_y = constants.trace_props[_fieldy]['title'] + taily
    core_agg_x = session.get('core_agg_x')
    core_agg_y = session.get('core_agg_y')
    core_date = session.get('core_date')[:10]
    if core_agg_x == 'oneday' or core_agg_y == 'oneday':
        from_date = ''
    else:
        core_date = ''
    dfs = []
    for trace in chart_types:
        df_trace = pd.DataFrame()
        for location in locations:
            df_loc = filter_data(settings, DF,
                                  loc=location,
                                  trace=trace,
                                  scope=settings['scope'],
                                  )[['date', 'location', 'data']].copy()
            df_loc.rename(columns={'data': trace}, inplace=True)
            df_trace = df_trace.append(df_loc, ignore_index=True).copy()
        dfs.append(df_trace)
    df = pd.merge(dfs[0], dfs[1], how='left',
                  left_on=['date', 'location'],
                  right_on=['date', 'location'])
    if len(df) == 0:
        return 'Brak rekordów spełniających podane warunki'
    if core_agg_x == 'sum':
        seriesx = df.groupby('location')[_fieldx].sum()
    elif core_agg_x == 'max':
        seriesx = df.groupby('location')[_fieldx].max()
    elif core_agg_x == 'min':
        seriesx = df.groupby('location')[_fieldx].min()
    elif core_agg_x == 'median':
        seriesx = df.groupby('location')[_fieldx].median()
    elif core_agg_x == 'mean':
        seriesx = df.groupby('location')[_fieldx].mean()
    else:  # oneday
        seriesx = df.groupby('location')[_fieldx].mean()

    if core_agg_y == 'sum':
        seriesy = df.groupby('location')[_fieldy].sum()
    elif core_agg_y == 'max':
        seriesy = df.groupby('location')[_fieldy].max()
    elif core_agg_y == 'min':
        seriesy = df.groupby('location')[_fieldy].min()
    elif core_agg_y == 'median':
        seriesy = df.groupby('location')[_fieldy].median()
    elif core_agg_y == 'mean':
        seriesy = df.groupby('location')[_fieldy].mean()
    else:  # oneday
        seriesy = df.groupby('location')[_fieldy].mean()
    seriesx.fillna(0, inplace=True)
    seriesy.fillna(0, inplace=True)
    xvals = list(seriesx)
    yvals = list(seriesy)
    figure = go.Figure()
    fig_data1 = go.Scatter(x=xvals,
                           y=yvals,
                           mode='lines+markers+text',
                           text=seriesx.index,
                           textfont=dict(color=get_session_color(4)),
                           line=dict(color='yellow', width=0),
                           marker=dict(symbol='diamond',
                                       size=20,
                                       opacity=0.5,
                                       color='green'),
                           name='x',
                           opacity=1,
                           )
    figure.add_trace(fig_data1)
    if core_highlight != 'Brak':
        sx = seriesx[{core_highlight: seriesx[core_highlight]}].copy()
        sy = seriesy[{core_highlight: seriesy[core_highlight]}].copy()
        xvals4 = list(sx)
        yvals4 = list(sy)
        fig_data4 = go.Scatter(x=xvals4,
                               y=yvals4,
                               mode='lines+markers+text',
                               text=sx.index,
                               line=dict(color='red', width=0),
                               marker={'size': 20, 'color': 'red'},
                               name='xx',
                               opacity=1,
                               )
        figure.add_trace(fig_data4)

    reg_txt = ''
    if 'regresja' in core_view:

        # linia regresji liniowej

        xvals = seriesx.values.reshape(-1, 1)
        yvals = seriesy.values.reshape(-1, 1)
        linear_regressor = LinearRegression()
        linear_regressor.fit(xvals, yvals)
        score = linear_regressor.score(xvals, yvals)
        yvals = linear_regressor.predict(xvals)
        xvals = list(xvals.flatten())
        y0 = list(linear_regressor.predict(pd.Series([0]).values.reshape(-1, 1)).flatten())[0]
        anno_text += 'y(0) = ' + str(y0)
        yvals = list(yvals.flatten())
        yorg = list(seriesy)
        reg_txt = '<br>Współczynnik ' + str(round(score, 3))
        err = [yvals[i] - yorg[i] for i in range(len(yvals))]

        if 'errors' in core_view:
            error_y = dict(type='data',
                           symmetric=False,
                           array=err,
                           visible=True,
                           color='yellow',
                           thickness=0.5,
                           width=5)
        else:
            error_y = {}
        fig_data2 = go.Scatter(x=xvals,
                              y=yvals,
                              mode='lines',
                              line=dict(color='green', width=0.3),
                              error_y=error_y)
        figure.add_trace(fig_data2)

    tozerox = 'normal'
    tozeroy = 'normal'
    if 'tozerox' in core_opt:
        tozerox = 'tozero'
    if 'tozeroy' in core_opt:
        tozeroy = 'tozero'
    figure.update_layout(height=750,
                         template=template,
                         annotations=[
                             go.layout.Annotation(
                                 text=anno_text,
                                 font=dict(color='yellow', size=14),
                                 align='left',
                                 showarrow=False,
                                 xref='paper',
                                 yref='paper',
                                 x=0.02,
                                 y=1.0,
                                 bordercolor='black',
                                 borderwidth=0
                             )
                         ],
                         title=layout_title(
                             text='Korelacja: ' + name_x + ' vs. ' + name_y + \
                                  '<br>Zakres danych: od ' + from_date + ' do ' + session.get('core_date')[:10] + reg_txt,
                             font_size=settings['font_size_title'],
                             color=get_session_color(7),
                             posx=settings['titlexpos'],
                             posy=settings['titleypos'],
                         ),
                         xaxis=dict(
                             title=dict(text=name_x,
                                        font=dict(color=get_session_color(4))
                                        ),
                             rangemode=tozerox,
                             rangeslider={'visible': ('suwak' in core_view), 'thickness': 0.15}
                         ),
                         yaxis=dict(
                             title=dict(text=name_y,
                                        font=dict(color=get_session_color(4))),
                             rangemode=tozeroy,
                         ),
                         showlegend=False),
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'docent_image',
            'width': 1200, 'height': 750, 'scale': 1}
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = dbc.Col(fig, width=12, className='mb-2 mt-2')
    return ret_val


#######################################
# Redraw map Mapbox
#######################################
@timing
def update_figures_map_mapbox(DF):
    settings = session2settings()
    json_file = ''
    if settings['map_date'] is None:
        return
    scope = settings['scope']
    date = settings['map_date'][:10]
    from_date = settings['from_date']
    to_date = settings['to_date']
    locations = settings['locations']
    chart_types = settings['chart_type']
    if len(locations) == 0:
        return
    locations.sort()
    if len(chart_types) != 1:
        return
    chart_type = chart_types[0]
    tail0 = table_mean[settings['table_mean']]
    if settings['table_mean'] in ['sum', 'mean']:
        tail0 += ' (od ' + str(from_date)[:10] + ' do ' + str(to_date)[:10] + ')'
    df = prepare_data(settings, DF,
                      scope=scope,
                      locations=locations,
                      date=date,
                      chart_types=chart_types,
                      all_columns=True)
    if len(df) == 0:
        return 'Brak rekordów spełniających podane warunki'
    df.dropna(subset=['Long', 'Lat'], inplace=True)

    if scope == 'poland':
        df['Long'] = df['location'].apply(lambda x: constants.wojew_mid[x][1])
        df['Lat'] = df['location'].apply(lambda x: constants.wojew_mid[x][0])

    tail_color = '<br><sup>' + tail0 + \
                 {1: '',
                  2: ' - na 100 000 osób</sup>',
                  3: ' - na 1000 km2</sup>'}[settings['data_modifier']]

    mapbox_access_token = open("data/mapbox.token").read()
    featuredikey = 'properties.adm0_a3'
    ikey = 'adm0_a3'
    locations = df['iso_code']
    if settings['scope'] == 'poland':
        json_file = r'data/geojson/woj.min.geo.json'
        mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=19.18, lat=52.11), zoom=5.5)
        df['location'] = df['location'].str.lower()
        locations = df['location']
        featuredikey = 'properties.nazwa'
        ikey = 'nazwa'
    if settings['scope'] == 'cities':
        json_file = r'data/geojson/powiaty-min-ws.geojson'
        mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=19.18, lat=52.11), zoom=5.5)
        locations = df['location']
        featuredikey = 'properties.nazwa'
        ikey = 'nazwa'
    if settings['scope'] == 'world':
        if 'quality' in settings['map_options']:
            json_file = r'data/geojson/world.geo.json'
        else:
            json_file = r'data/geojson/world-small.geo.json'
        mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=19.18, lat=52.11), zoom=1)
    elif settings['scope'] == 'Europa':
        json_file = r'data/geojson/europe.geo.json'
        mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=17.1, lat=57.1), zoom=3)
    elif settings['scope'] == 'Azja':
        json_file = r'data/geojson/asia.geo.json'
        mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=85, lat=28), zoom=2.5)
    elif settings['scope'] == 'Afryka':
        json_file = r'data/geojson/africa.geo.json'
        mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=18.5, lat=6.1), zoom=2.5)
    elif settings['scope'] == 'Oceania':
        json_file = r'data/geojson/oceania.geo.json'
        mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=133.7, lat=-23.1), zoom=3)
    elif settings['scope'] == 'Ameryka Północna':
        json_file = r'data/geojson/north_america.geo.json'
        mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=-102.4, lat=52.5), zoom=2.2)
    elif settings['scope'] == 'Ameryka Południowa':
        json_file = r'data/geojson/south_america.geo.json'
        mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=-60.1, lat=-19.4), zoom=2.2)
    with open(json_file, encoding="utf-8") as f:
        geojson = json.load(f)
    round_digits = 3
    if trace_props[chart_type]['postfix'] == ' %':
        round_digits = 1


    def map_choropleth():
        hover_text = '<b>' + span(attr='font-size', txt=df['location'], value='18px') + '</b><br>' + settings[
                                                                                                         'map_date'][
                                                                                                     :10] + '<br><br>' + \
                     span(txt=trace_props['total_cases']['title'], value='blue') + ': ' + \
                     '<b>' + df['total_cases'].round(round_digits).astype(str) + trace_props['total_cases'][
                         'postfix'] + '</b><br>' + \
                     span(txt=trace_props['total_deaths']['title'], value='red') + ': ' + \
                     '<b>' + df['total_deaths'].round(round_digits).astype(str) + trace_props['total_deaths'][
                         'postfix'] + '</b><br>' + \
                     span(txt=trace_props['total_recoveries']['title'], value='green') + ': ' + \
                     '<b>' + df['total_recoveries'].round(round_digits).astype(str) + trace_props['total_recoveries'][
                         'postfix'] + '</b><br>' + \
                     span(txt=trace_props['total_active']['title'], value='brown') + ': ' + \
                     '<b>' + df['total_active'].round(round_digits).astype(str) + trace_props['total_active'][
                         'postfix'] + '</b><br>' + \
                     span(txt=trace_props['new_cases']['title'], value='blue') + ': ' + \
                     '<b>' + df['new_cases'].round(round_digits).astype(str) + trace_props['new_cases'][
                         'postfix'] + '</b><br>' + \
                     span(txt=trace_props['new_deaths']['title'], value='red') + ': ' + \
                     '<b>' + df['new_deaths'].round(round_digits).astype(str) + trace_props['new_deaths'][
                         'postfix'] + '</b><br>' + \
                     span(txt=trace_props['new_recoveries']['title'], value='green') + ': ' + \
                     '<b>' + df['new_recoveries'].round(round_digits).astype(str) + trace_props['new_recoveries'][
                         'postfix'] + '</b><br>' + \
                     trace_props['zapadalnosc']['title'] + ': ' + \
                     '<b>' + df['zapadalnosc'].round(round_digits).astype(str) + trace_props['zapadalnosc'][
                         'postfix'] + '</b><br>' + \
                     trace_props['umieralnosc']['title'] + ': ' + \
                     '<b>' + df['umieralnosc'].round(round_digits).astype(str) + trace_props['umieralnosc'][
                         'postfix'] + '</b><br>' + \
                     trace_props['smiertelnosc']['title'] + ': ' + \
                     '<b>' + df['smiertelnosc'].round(round_digits).astype(str) + trace_props['smiertelnosc'][
                         'postfix'] + '</b><br>'
        tickvals, ticktext = get_ticks(df[chart_type])
        fig = go.Figure()
        if settings['map_opt'] == 'log':
            z = np.log10(df[chart_type] + 0.001)
        else:
            z = df[chart_type]
        fig.add_trace(go.Choroplethmapbox(
            geojson=geojson, featureidkey=featuredikey, locations=locations,
            z=z,
            text=hover_text,
            marker=dict(line=dict(color='black', width=0.5 if 'drawborder' in settings['map_options'] else 0.)),
            hoverinfo='text',
            hoverlabel=dict(bgcolor='rgb(237,207,154)', bordercolor='purple'),
            colorscale=settings['map_color_scale'],
            reversescale=True if 'reversescale' in settings['map_options'] else False,
            marker_opacity=settings['map_opacity'],
            colorbar=dict(
                len=0.5,
                bgcolor=get_session_color(1),
                tickfont=dict(
                    size=settings['font_size_xy'],
                    color=get_session_color(3),
                ),
                tickmode="array" if settings['map_opt'] == 'log' else 'auto',
                tickvals=tickvals,
                ticktext=ticktext,
                ticks="outside",
            ),
        ))
        if 'annotations' in settings['map_options']:
            if '0c' in settings['map_options']:
                acc = 0
            elif '1c' in settings['map_options']:
                acc = 1
            elif '2c' in settings['map_options']:
                acc = 2
            else:
                acc = 3
            dfx = df.copy()
            dfx[chart_type] = dfx[chart_type].round(acc).astype(str)
            if constants.trace_props[chart_type]['category'] == 'data':
                dfx['color2'] = dfx.groupby('Long')[chart_type].transform(lambda x: ', '.join(x))
                dfx['location2'] = dfx.groupby('Long')['location'].transform(lambda x: ', '.join(x))
                dfx[['location2']].drop_duplicates()
            else:
                dfx['color2'] = dfx[chart_type]
                dfx['location2'] = dfx['location']
            if 'number' in settings['map_options']:
                anno_text = dfx['color2'] + trace_props[chart_type]['postfix']
            else:
                anno_text = dfx['location2'] + '<br>' + dfx['color2'] + trace_props[chart_type]['postfix']
            fig.add_trace(
                go.Scattermapbox(
                    lat=dfx.Lat, lon=dfx.Long,
                    mode='text',
                    hoverinfo='none',
                    below="''",
                    marker=dict(allowoverlap=True),
                    text=anno_text,
                    textfont=dict(family='Bebas Neue', size=settings['font_size_anno'], color=get_session_color(4)),
                ))
        return fig


    mapbox['style'] = mapbox_styles[settings['map_mapbox']]

    figure = map_choropleth()
    figure.update_layout(
        title=dict(
            text='<b>' + trace_props[chart_type]['title'] + '</b>' +
                 tail_color + '<br><sup>' + settings['map_date'][:10],
            x=settings['titlexpos'], y=settings['titleypos'] - 0.04,
            font=dict(size=settings['font_size_title'], color=get_session_color(7)
                      )
        ),
        images=image_logo_timeline,
        hovermode='x',
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Rockwell"),
        height=int(settings['map_height']),
        margin=dict(l=settings['marginl'], r=settings['marginr'], b=settings['marginb'], t=settings['margint'],
                    pad=0),
        paper_bgcolor=get_session_color(1),
        plot_bgcolor=get_session_color(1),
        mapbox=mapbox,
    )
    figure = add_copyright(figure, settings, x=0.04, y=0)
    config = {
        'displaylogo': False,
        'responsive': True,
        'toImageButtonOptions': {
            'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
            'filename': 'docent_image', 'height': 700, 'width': 1200}
    }
    ret_val = dbc.Col(dcc.Graph(id='figure_mapa', figure=figure, config=config), className='mb-2 mt-2')
    return ret_val


#######################################
# Redraw single chart - table
#######################################
@timing
def update_figures_table(DF):
    settings = session2settings()

    scope = settings['scope']
    locations = settings['locations']
    chart_types = settings['chart_type']
    if len(locations) == 0:
        return
    locations.sort()
    filtered_df = prepare_data(settings, DF,
                      scope=scope,
                      locations=locations,
                      chart_types=chart_types,
                      all_columns=True)
    okres = len(filtered_df['date'].unique())

    filtered_df['lp'] = np.arange(len(filtered_df)) + 1
    filtered_df.reset_index(drop=True, inplace=True)

    height = 700
    wid = 12
    tail0 = {1: '',
             2: '/100000',
             3: '/1000km2'}[settings['data_modifier']]
    columns = [
            {'id': 'lp', 'name': 'Lp'},
            {'id': 'date', 'name': 'Data'},
            {'id': 'location', 'name': 'Lokalizacja'},
        ] + [{"name": [trace_props[i]['title'] + tail0], "id": i} for i in chart_types]
    style_data_conditional = []
    if scope == 'cities':
        filtered_df = filtered_df[['lp', 'date', 'location', 'wojew']+chart_types]
    elif scope == 'world':
        filtered_df = filtered_df[['lp', 'date', 'location', 'iso_code']+chart_types]
    else:
        filtered_df = filtered_df[['lp', 'date', 'location']+chart_types]

    # dane do heatmap.................................
    if scope == 'cities':
        idx = ['location', 'wojew']
    else:
        idx = 'location'
    field_h = chart_types[0]
    fname = 'data/' + field_h + '_heatmap.csv'
    d = filtered_df.copy()
    d = d[d['date'] >= '2021.01.03'].copy()
    d['date'] = d['date'].str.slice(5, 10)
    del d['lp']
    d.sort_values(by=idx, inplace=True)
    d[field_h] = d[field_h].round(2)
    d[field_h] = d[field_h].astype(str)
    d[field_h] = d[field_h].str.replace('.', ',')
    dd = d.pivot(index=idx, columns='date')
    dd.to_csv(fname)

    # różnice

    if scope == 'cities':
        idx = ['location', 'wojew']
    else:
        idx = 'location'
    fname = 'data/' + field_h + '_diff_heatmap.csv'
    d = filtered_df.copy()
    d = d[d['date'] >= '2021.01.03'].copy()
    d['date'] = d['date'].str.slice(5, 10)
    del d['lp']
    d.sort_values(by=idx, inplace=True)
    d['diff'] = d[field_h].diff()
    d['diff'] = d['diff'].round(2)
    d['diff'] = d['diff'].astype(str)
    d['diff'] = d['diff'].str.replace('.', ',')
    del d[field_h]
    dd = d.pivot(index=idx, columns='date')
    dd.to_csv(fname)

    # różnice

    fname = 'tmp/table_data.csv'
    filtered_df.to_csv(fname)
    df_pivot = filtered_df.pivot(index='location', columns='date')[chart_types[0]]
    fname_pivot = 'tmp/table_data_pivot.csv'
    df_pivot.to_csv(fname_pivot)
    if scope == 'world':
        tytul = 'Świat'
    elif scope == 'poland':
        tytul = 'Polska - województwa'
    elif scope == 'cities':
        tytul = 'Polska - miasta i powiaty'
    else:
        tytul = scope
    tail = {1: '', 2: ' (na 100 000 osób)'}[settings['data_modifier']]
    tytul += tail
    tytul2 = ''
    if settings['table_mean'] != 'daily':
        if settings['table_mean'] == 'weekly_mean':
            tytul2 += '  średnie tygodniowe'
        elif settings['table_mean'] == 'biweekly_mean':
            tytul2 += '  średnie dwutygodniowe'
        elif settings['table_mean'] == 'mean':
            tytul2 += '  średnia za okres'
        elif settings['table_mean'] == 'weekly_sum':
            tytul2 += '  sumy tygodniowe'
        elif settings['table_mean'] == 'biweekly_sum':
            tytul2 += '  sumy dwutygodniowe'
        elif settings['table_mean'] == 'mean':
            tytul2 += '  średnia dla okresu (' + str(okres) + 'dni)'
        elif settings['table_mean'] == 'median':
            tytul2 += ' mediana dla okresu (' + str(okres) + 'dni)'
        elif settings['table_mean'] == 'sum':
            tytul2 += ' suma dla okresu (' + str(okres) + 'dni)'
    if tytul2 != '':
        tytul2 = ' ' + tytul2

    data = filtered_df.to_dict('records')

    columns = [
        {"title": "Data",
         "field": "date",
         "hozAlign": "left"},
        {"title": "Lokalizacja",
         "field": "location",
         'headerFilter': 'select',
         'headerFilterPlaceholder': "Wybierz lokalizację...",
         'headerFilterParams': {'values': True}},
    ] + [{"title": [trace_props[i]['title'] + tail0], "field": i} for i in chart_types]
    options = dict(
        height='600px',
            initialSort=[
            {'column': 'date', 'dir': 'desc'},
            {'column': 'location', 'dir': 'asc'},
        ],
        selectable=1
    )
    figure = html.Div([
        dash_tabulator.DashTabulator(
            id='tabulator',
            columns=columns,
            data=data,
            options=options,
        ),
    ], className='mb-8', id='example-table-theme')

    ret_val = [
        html.H4(tytul + tytul2),
        dbc.Col(figure, width=wid, className='mb-2 mt-2')
    ]
    return ret_val
