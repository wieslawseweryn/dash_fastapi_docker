import base64

from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
import dash_table
from flask import session
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from apps.proc.constants import trace_props, image_logo_timeline, color_scales, table_mean
from apps.proc.util import get_session_color, get_trace_name, session2settings, is_logged_in, \
    filter_data
from apps.layouts import layout_title, layout_legend
import datetime

#######################################
# Redraw single chart - timeline types
#######################################

def update_figures_timeline_types(DF):
    settings = session2settings()
    chart_type = settings['chart_type_data'] + settings['chart_type_calculated']
    t_start = datetime.datetime.now()

    def calculate_figure_timeline_types():

        filtered_df = filter_data(DF,
                                  loc=location,
                                  trace=trace,
                                  total_min=settings['total_min'],
                                  scope=settings['scope'],
                                  data_modifier=settings['data_modifier'],
                                  dzielna=settings['dzielna'],
                                  table_mean=settings['table_mean'],
                                  from_date=settings['from_date'],
                                  duration_d=settings['duration_d'],
                                  duration_r=settings['duration_r'],
                                  timeline_opt=settings['timeline_opt'],
                                  smooth=settings['smooth'],
                                  flow=settings['flow'],
                                  win_type=settings['win_type']
                                  )
        _fig_data_scatter = {}
        if 'usrednianie' in settings['timeline_opt'] and 'points' in settings['options_yesno']:
            _fig_data_scatter = dict(
                x=list(filtered_df['date']),
                y=list(filtered_df['data_points']),
                name=location,
                mode='lines+markers+text',
                line=dict(width=settings['linewidth_thin']),
                marker=dict(symbol='circle', size=3),
                opacity='1'
            )
        if 'wygladzanie' in settings['timeline_opt'] and 'points' in settings['options_yesno']:
            _fig_data_scatter = dict(
                x=list(filtered_df['date']),
                y=list(filtered_df['data_points']),
                name=location,
                mode='lines+markers+text',
                line=dict(width=settings['linewidth_thin']),
                marker=dict(symbol='cross', size=5),
                opacity='0.5'
            )

        # konstrukcja nitki wykresu

        line = dict(width=settings['linewidth_basic'])
        opacity = 1
        fill = ''
        if settings['timeline_highlight'] != 'Brak':
            if settings['radio_type'] == 'bar':
                opacity = 0.2
            if location == settings['timeline_highlight']:
                line = dict(width=settings['linewidth_thick'], color=get_session_color(9))
                fill = 'tozeroy'
                opacity = 1
        _fig_data = dict(
            x=list(filtered_df['date'].astype('datetime64[ns]')),
            y=list(filtered_df['data']),
            line=line,
            # fill='tozeroy',
            fill=fill,
            type=settings['radio_type'],
            name=location,
            mode=session.get('linedraw'),
            opacity=opacity
        )
        return [_fig_data, _fig_data_scatter]

    traces = {key: [] for key in chart_type}
    children = []
    height = 660
    wid = 12
    if settings['arrangement'] == 'smart':
        if len(chart_type) > 2:
            wid = 6
            height = 330
        elif len(chart_type) == 2:
            wid = 6
            height = 660
    height = float(settings['plot_height']) * height
    xaxis_text = ''
    if 'usrednianie' in settings['timeline_opt']:
        xaxis_text += 'uśrednianie (' + str(settings['average_days']) + ' dni) '
    if 'usrednianieo' in settings['timeline_opt']:
        xaxis_text += 'średnia okresowa (' + str(settings['average_days']) + ' dni) '
    if 'wygladzanie' in settings['timeline_opt']:
        xaxis_text += 'wygładzanie (' + str(settings['smooth']) + ' stopnia) '
    for trace in chart_type:
        xanno = []
        yanno = []
        txt = []
        tail = table_mean[settings['table_mean']]
        if trace_props[trace]['category'] == 'data':
            tail += {1: '', 2: ' - na 100 000 osób'}[settings['data_modifier']]
        for location in settings['locations']:
            fig_data, fig_data_scatter = calculate_figure_timeline_types()
            if len(fig_data['y']) > 0:
                if settings['annotations'] == 'max':
                    max_y = max(fig_data['y'])
                    index_max = fig_data['y'].index(max_y)
                    yanno.append(max_y)
                    xanno.append(fig_data['x'][index_max])
                    txt.append(location)
                elif settings['annotations'] == 'last':
                    yanno.append(fig_data['y'][-1])
                    xanno.append(fig_data['x'][-1])
                    txt.append(location)
                elif settings['annotations'] == 'all':
                    yanno.extend([fig_data['y'][i] for i in range(len(fig_data['y']))])
                    xanno.extend([fig_data['x'][i] for i in range(len(fig_data['x']))])
                    txt.extend(['' for i in range(len(fig_data['x']))])

                traces[trace].append(fig_data)
                if fig_data_scatter:
                    traces[trace].append(fig_data_scatter)
        legend_y = 1.
        tickvals = pd.DatetimeIndex(fig_data['x']).month.unique()
        ticktext = pd.DatetimeIndex(fig_data['x']).month.unique()

        def anno(i):
            prefix = ''
            if len(settings['timeline_opt']) > 0:
                prefix = '~'
            if settings['anno_form'] == 'num':
                ret_val = '<b>' + prefix + str(yanno[i])
            elif settings['anno_form'] == 'name':
                ret_val = txt[i]
            elif settings['anno_form'] == 'namenum':
                ret_val = txt[i] + '<br><b>' + prefix + str(yanno[i])
            else:
                ret_val = txt[i] + '<br>' + str(xanno[i])[:10] + '<br><b>' + prefix + str(yanno[i])
            return ret_val
        trace_name = get_trace_name(trace)

        if settings['dzielna'] != '<brak>' and trace_props[trace]['category'] == 'data':
            trace_name = get_trace_name(trace) + ' / ' + get_trace_name(settings['dzielna']).lower()
        figure = {
            'data': traces[trace],
            'layout': dict(
                images=image_logo_timeline,
                annotations=[{'x': xanno[i], 'y': yanno[i],
                              'text': anno(i),
                              'font': {'size': settings['font_size_anno'], 'color': get_session_color(4)},
                              'opacity': 1,
                              'arrowcolor': 'red',
                              'arrowwidth': 1.5,
                              'bgcolor': get_session_color(5)
                              } for i in range(len(xanno))],
                title=layout_title(text=trace_name + '<br><sup>' + tail,
                                   font_size=settings['font_size_title'],
                                   color=get_session_color(7),
                                   posx=settings['titlexpos'],
                                   posy=settings['titleypos'],
                                   ),
                xaxis={
                    'rangeslider': {'visible': ('suwak' in settings['timeline_view']), 'thickness': 0.05},
                    'linecolor': get_session_color(2),
                    # 'tickformat': "%m-%Y",
                    'hoverformat': "%d-%m-%Y",
                    'tickmode': 'array',
                    'nticks': len(tickvals),
                    'ticks': 'outside',
                    # 'tickvals': tickvals,
                    'ticktext': ticktext,
                    'showticklabels': 'true',
                    'tickfont': {'size': settings['font_size_xy'], 'color': get_session_color(3)},
                    'title': {'text': xaxis_text, 'font': {'size': '12'}, 'standoff': '5'}
                },
                yaxis={
                    'linecolor': get_session_color(2),
                    'tickfont': {'size': settings['font_size_xy'], 'color': get_session_color(3)},
                    'exponentformat': 'none',
                    'separatethousands': False,
                    'title': {'text': '', 'font': {'size': settings['font_size_xy'], 'color': get_session_color(7)}},
                    # 'title': {'text': get_trace_name(trace), 'font': {'size': settings['font_size_xy'], 'color': get_session_color(7)}},
                    'type': 'log' if trace_props[trace]['log_scale'] and settings['radio_scale'] == 'log' else 'linear',
                },
                height=height,
                margin=dict(l=settings['marginl'] + 50, r=settings['marginr'], b=settings['marginb'] + 50, t=settings['margint'] + 50,
                            pad=4),
                template='plotly_dark',
                paper_bgcolor=get_session_color(1),
                plot_bgcolor=get_session_color(1),
                colorway=color_scales[settings['color_order']],
                showlegend=('legenda' in settings['timeline_view']),
                legend=layout_legend(y=legend_y,
                                     legend_place=settings['legend_place'])
            )
        }
        config = {
            'displaylogo': False,
            'responsive': True,
            'locale': 'pl-PL',
            'toImageButtonOptions': {
                'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
                'filename': 'docent_image', 'height': height, 'width': 120 * wid, 'scale': 1}
        }
        children.append(dbc.Col(dcc.Graph(id=trace, figure=figure, config=config), width=wid, className='mb-2 mt-2'))

    try:
        return children
    except:
        return html.Div('Wystąpił błąd. Spróbuj jeszcze raz.')

