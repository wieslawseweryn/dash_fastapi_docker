import json
from random import randint, random

import pandas as pd
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from datetime import datetime as dt, timedelta
import plotly.graph_objects as go
import numpy as np

from layouts import layout_legend
from util import read_kolory, logit, get_template, session2settings, prepare_data, timing, \
    get_session_color, create_html, fn, get_trace_name, filter_data, get_ticks, add_copyright, get_nticks, get_bins, \
    get_title, discrete_colorscale
from plotly.subplots import make_subplots
import constants as constants
import math
from flask import session
from flask import request
from scipy import signal
start_server_date = str(dt.today())[:16]

kolory = read_kolory()

nomargins = 'ml-0 pl-0'


def zoom_center(lons: tuple = None, lats: tuple = None, lonlats: tuple = None,
                format: str = 'lonlat', projection: str = 'mercator',
                width_to_height: float = 2.0) -> (float, dict):
    """Finds optimal zoom and centering for a plotly mapbox.
    Must be passed (lons & lats) or lonlats.
    Temporary solution awaiting official implementation, see:
    https://github.com/plotly/plotly.js/issues/3434

    Parameters
    --------
    lons: tuple, optional, longitude component of each location
    lats: tuple, optional, latitude component of each location
    lonlats: tuple, optional, gps locations
    format: str, specifying the order of longitud and latitude dimensions,
        expected values: 'lonlat' or 'latlon', only used if passed lonlats
    projection: str, only accepting 'mercator' at the moment,
        raises `NotImplementedError` if other is passed
    width_to_height: float, expected ratio of final graph's with to height,
        used to select the constrained axis.

    Returns
    --------
    zoom: float, from 1 to 20
    center: dict, gps position with 'lon' and 'lat' keys

    >>> print(zoom_center((-109.031387, -103.385460), (25.587101, 31.784620)))
    (5.75, {'lon': -106.208423, 'lat': 28.685861})
    """
    if lons is None and lats is None:
        if isinstance(lonlats, tuple):
            lons, lats = zip(*lonlats)
        else:
            raise ValueError(
                'Must pass lons & lats or lonlats'
            )

    maxlon, minlon = max(lons), min(lons)
    maxlat, minlat = max(lats), min(lats)
    center = {
        'lon': round((maxlon + minlon) / 2, 6),
        'lat': round((maxlat + minlat) / 2, 6)
    }

    # longitudinal range by zoom level (20 to 1)
    # in degrees, if centered at equator
    lon_zoom_range = np.array([
        0.0007, 0.0014, 0.003, 0.006, 0.012, 0.024, 0.048, 0.096,
        0.192, 0.3712, 0.768, 1.536, 3.072, 6.144, 11.8784, 23.7568,
        47.5136, 98.304, 190.0544, 360.0
    ])

    if projection == 'mercator':
        margin = 1.2
        height = (maxlat - minlat) * margin * width_to_height
        width = (maxlon - minlon) * margin
        lon_zoom = np.interp(width, lon_zoom_range, range(20, 0, -1))
        lat_zoom = np.interp(height, lon_zoom_range, range(20, 0, -1))
        zoom = round(min(lon_zoom, lat_zoom), 2)
    else:
        raise NotImplementedError(
            f'{projection} projection is not implemented'
        )

    return zoom, center


def layout_timeline_two_axes(df, x, y1, y2, t1=[], t2=[],
                             color1='red', color2='green',
                             height=300):
    template = get_template()
    settings = session2settings()
    rounding = settings['rounding']
    className = 'mb-0 mt-0 ml-0 mr-0 pb-0 pt-0 pl-0 pr-0'


    def get_anno(seria):
        anno = settings['annotations']
        # seria = [round(x, 2) for x in seria]
        if 'x01' in anno:
            xanno = 1
        elif 'x02' in anno:
            xanno = 2
        elif 'x05' in anno:
            xanno = 5
        elif 'x10' in anno:
            xanno = 10
        elif 'x20' in anno:
            xanno = 20
        else:
            xanno = 10000
        if anno in ['x01', 'x02', 'x05', 'x10', 'x20']:
            text = [str(seria[i]) if i % xanno == 0 else ' ' for i in range(len(seria))]
            # text = [str(yval2[i]) if i % xanno == 0 else ' ' for i in range(len(seria))]
            text[-1] = seria[-1]
        else:
            text = [' ' for i in range(len(seria))]
            if 'max last' in anno:
                mx = max(seria)
                idx = seria.index(mx)
                text[idx] = mx
                text[-1] = seria[-1]
            elif 'last' in anno:
                text[-1] = seria[-1]
            elif 'max' in anno:
                mx = max(seria)
                idx = seria.index(mx)
                text[idx] = mx
        return text


    if len(t1) == 0:
        t1 = ['scatter'] * len(y1)
    if len(t2) == 0:
        t2 = ['scatter'] * len(y2)
    if len(t2) < len(y2):
        t2.append('scatter')

    if len(settings['locations']) == 1:
        wid = 12
        height = 660
    elif len(settings['locations']) == 2:
        wid = 6
        height = 660
    else:
        wid = 6
        height = 330

    children = []

    xval = list(df[x].unique())

    for location in settings['locations']:
        dfx = df[df['location'] == location].copy()
        figure = go.Figure()

        # przebiegi do lewej osi

        for i in range(len(y1)):
            if y1[i] == '<brak>':
                break

            if rounding == 'int':
                dfx[y1[i]] = dfx[y1[i]].astype(int)
                dfx[y1[i] + '_points'] = dfx[y1[i] + '_points'].astype(int)
            elif rounding in ['0', '1', '2', '3', '4', '5']:
                dfx[y1[i]] = round(dfx[y1[i]], int(rounding))
                dfx[y1[i] + '_points'] = round(dfx[y1[i] + '_points'], int(rounding))

            if y1[i].isnumeric():
                yval1 = [float(y1[i]) for k in range(len(xval))]
                yval2 = [float(y1[i] + '_points') for k in range(len(xval))]
                name = y1[i]
            else:
                yval1 = list(dfx[y1[i]])
                yval2 = list(dfx[y1[i] + '_points'])
                name = constants.trace_props[y1[i]]['title']
            if t1[i] == 'scatter':
                if 'points' not in settings['options_yesno']:
                    text = get_anno(yval1)
                    text[-1] = yval1[-1]
                else:
                    text = None
                figure.add_trace(go.Scatter(
                    x=xval,
                    y=yval1,
                    text=text,
                    textposition='top center',
                    textfont=dict(color=settings['color_7'], size=settings['font_size_legend']),
                    mode='lines+text',
                    line=dict(
                        width=settings['linewidth_basic'],
                        color=color1,
                        dash=constants.dashes[i]),
                    name=name,
                    opacity=1,
                ))
                if 'points' in settings['options_yesno']:
                    text = get_anno(yval2)
                    text[-1] = yval2[-1]
                    figure.add_trace(go.Scatter(
                        x=xval,
                        y=yval2,
                        text=text,
                        textposition='top center',
                        textfont=dict(color=settings['color_7'], size=settings['font_size_legend']),
                        mode='lines+markers+text',
                        line=dict(
                            width=settings['linewidth_thin'],
                            color=color1,
                            dash='dash'),
                        name=name,
                        # yaxis="y2",
                        marker=dict(symbol='circle-open', size=3),
                        opacity=1,
                    ))
            else:
                figure.add_trace(go.Bar(
                    x=xval,
                    y=yval1,
                    marker=dict(color=color1),
                    name=name,
                    opacity=1,
                ))

        # przebiegi do prawej osi

        def layout_legend(y=0, legend_place='lewo'):
            if legend_place == 'lewo':
                orientation = 'v'
                x = 0
            elif legend_place == 'prawo':
                orientation = 'v'
                x = 1.05
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

        for i in range(len(y2)):
            if rounding == 'int':
                dfx[y2[i]] = dfx[y2[i]].astype(int)
                dfx[y2[i] + '_points'] = dfx[y2[i] + '_points'].astype(int)
            elif rounding in ['0', '1', '2', '3', '4', '5']:
                dfx[y2[i]] = round(dfx[y2[i]], int(rounding))
                dfx[y2[i] + '_points'] = round(dfx[y2[i] + '_points'], int(rounding))

            if y2[i].isnumeric():
                yval1 = [float(y2[i]) for k in range(len(xval))]
                yval2 = [float(y2[i] + '_points') for k in range(len(xval))]
                name = y2[i]
            else:
                yval1 = list(dfx[y2[i]])
                yval2 = list(dfx[y2[i] + '_points'])
                name = constants.trace_props[y2[i]]['title']
            if t2[i] == 'scatter':
                if 'points' not in settings['options_yesno']:
                    text = get_anno(yval1)
                    text[-1] = yval1[-1]
                else:
                    text = None
                figure.add_trace(go.Scatter(
                    x=xval,
                    y=yval1,
                    text=text,
                    textposition='top center',
                    textfont=dict(color=settings['color_7'], size=settings['font_size_legend']),
                    mode='lines+text',
                    line=dict(
                        width=settings['linewidth_basic'],
                        color=color2,
                        dash=constants.dashes[i]),
                    name=name,
                    yaxis="y2"
                ))
                if 'points' in settings['options_yesno']:
                    text = get_anno(yval2)
                    text[-1] = yval2[-1]
                    figure.add_trace(go.Scatter(
                        x=xval,
                        y=yval2,
                        text=text,
                        textposition='top center',
                        textfont=dict(color=settings['color_7'], size=settings['font_size_legend']),
                        mode='lines+markers+text',
                        line=dict(
                            width=settings['linewidth_thin'],
                            color=color2,
                            dash='dash'),
                        name=name,
                        yaxis="y2",
                        marker=dict(symbol='circle-open', size=3),
                        opacity=1,
                    ))
            else:
                figure.add_trace(go.Bar(
                    x=xval,
                    y=yval2,
                    marker=dict(color=color2),
                    name=name,
                    opacity=1,
                ))
        figure.update_layout(
            title_text=location,
            legend=layout_legend(y=1, legend_place=settings['legend_place']),
        # legend=dict(orientation='h', y=1),

            # yaxis={'type': 'log' if settings['radio_scale'] == 'log' else 'linear'},

            template=template,
            yaxis=dict(
                title='',
                type='log' if settings['radio_scale'] == 'log' else 'linear',
                titlefont=dict(color=color1),
                tickfont=dict(color=color1)
            ),
            yaxis2=dict(
                title='',
                type='log' if settings['radio_scale'] == 'log' else 'linear',
                anchor="x",
                overlaying="y",
                side="right",
                titlefont=dict(color=color2),
                tickfont=dict(color=color2)
            ),
            height=height,
            margin=dict(l=settings['marginl'] + 50, r=settings['marginr'] + 50, b=settings['marginb'] + 30,
                        t=settings['margint'], pad=0),
        )
        figure = add_copyright(figure, settings)
        config = {
            'responsive': True,
            'displayModeBar': True,
            'locale': 'pl-PL',
        }
        children.append(dbc.Col(dcc.Graph(id='twoaxes'+location,
                                          figure=figure,
                                          config=config,
                                          className=className), width=wid))

    return children

@timing
def heatmap(_df, yaxis, xaxis, chart_type, settings, template, mapa={}, title='Heatmap', gapx_0=5, gapy_0=5):

    # map_palette: ciągły, dyskretny
    # map_cut: równe, kwantyle, własne
    map_cut = settings['map_cut']
    map_palette = settings['map_palette']

    colors = constants.color_scales[settings['color_order']]
    # s, bins, bin_labels = get_bins(dfxs[chart_type], len(colors), map_cut)

    df = _df[[xaxis, yaxis, chart_type]].copy()
    gapx = gapx_0
    gapy = gapy_0
    if len(df) > 50:
        gapx = 0
    bins = df[yaxis].unique()
    if len(mapa) == 0:
        ticktext = [x for x in bins]
    else:
        ticktext = [mapa[x] for x in bins]
    rounding = settings['rounding']
    annotations = settings['annotations']
    zarray = []
    yvalues = pd.Series()
    date_min = min(df[xaxis])
    date_max = max(df[xaxis])
    x = sorted(list(df[xaxis].unique()))
    height = float(settings['plot_height']) * 40 * len(bins) + 80
    for bin in bins:
        dfx = df[df[yaxis] == bin].copy()
        if date_min not in list(dfx[xaxis]):
            dfx.loc[len(dfx)] = [date_min, bin, 0]
        if date_max not in list(dfx[xaxis]):
            dfx.loc[len(dfx)] = [date_max, bin, 0]
        dfx['idx'] = pd.to_datetime(dfx[xaxis])
        dfx.set_index(dfx['idx'], inplace=True)
        dfx = dfx.resample('D').sum().fillna(0)
        # dfx = dfx.resample('D').sum().fillna(method='ffill')
        dfx['bin'] = bin
        if rounding == 'int':
            dfx[chart_type] = dfx[chart_type].astype(int)
        elif rounding in ['0', '1', '2', '3', '4', '5']:
            dfx[chart_type] = round(dfx[chart_type], int(rounding))
        y = list(dfx[chart_type])
        yvalues = pd.concat([yvalues,pd.Series(y)])
        zarray.append(y)
    z = np.array(zarray)
    if annotations == 'anno_heatmap':
        anno_text = zarray
        texttemplate = "%{text}"
    else:
        anno_text = None
        texttemplate = None
    if map_palette == 'ciągły':
        colorscale = constants.color_scales[settings['color_order']]
        colorbar = dict(title=dict(text='',
                                   side='top',
                                   font=dict(color='white')
                                   ),
                        tickfont=dict(color=get_session_color(3))
                        )
    else:
        s, xbins, bin_labels = get_bins(yvalues, len(colors), map_cut)
        colors = colors[:len(xbins) - 1]
        dcolorsc = discrete_colorscale(xbins, colors)
        colorscale = dcolorsc
        if rounding == 'int':
            xbins = [int(x) for x in xbins]
        elif rounding in ['0', '1', '2', '3', '4', '5']:
            xbins = [round(x, int(rounding)) for x in xbins]
        bvals = np.array(xbins)
        xtickvals = [np.mean(bvals[k:k + 2]) for k in
                    range(len(bvals) - 1)]  # position with respect to bvals where ticktext is displayed
        xticktext = [f'<{bvals[1]}'] + [f'{bvals[k]}-{bvals[k + 1]}' for k in range(1, len(bvals) - 2)] + [
            f'>{bvals[-2]}']
        colorbar = dict(title=dict(text='',
                                   side='top',
                                   font=dict(color='white')
                                   ),
                        tickfont=dict(color=get_session_color(3)),
                        tickvals=xtickvals,
                        ticktext=xticktext
                        )

    trace = go.Heatmap(
        x=x,
        y=ticktext,
        z=zarray,
        xgap=gapx,
        ygap=gapy,
        text=anno_text,
        texttemplate=texttemplate,
        textfont={'size': int(settings['font_size_anno'])},
        name='Udział',
        colorscale=colorscale,
        colorbar=colorbar
    )
    figure = go.Figure(
        data=[trace],
        layout=dict(
            template=template,
            autosize=True,
            title=dict(text=title),
            margin=dict(l=220),
            height=height,
            showlegend=False,
        )
    )
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
        'toImageButtonOptions': {
            'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
            'filename': 'docent_image', 'height': height, 'scale': 1}
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    return fig


def heatmap_week(_df, yaxis, xaxis, chart_type, settings, template, mapa={}, title='Heatmap'):
    df = _df[[xaxis, yaxis, chart_type]].copy()
    bins = df[yaxis].unique()
    if len(mapa) == 0:
        ticktext = [x for x in bins]
    else:
        ticktext = [mapa[x] for x in bins]
    zarray = []
    date_min = min(df[xaxis])
    date_max = max(df[xaxis])
    x = sorted(list(df[xaxis].unique()))
    height = float(settings['plot_height']) * 40 * len(bins) + 80
    for bin in bins:
        dfx = df[df[yaxis] == bin].copy()
        if date_min not in list(dfx[xaxis]):
            dfx.loc[len(dfx)] = [date_min, bin, 0]
        if date_max not in list(dfx[xaxis]):
            dfx.loc[len(dfx)] = [date_max, bin, 0]
        dfx['idx'] = pd.to_datetime(dfx[xaxis])
        dfx.set_index(dfx['idx'], inplace=True)
        dfx = dfx.resample('W-MON').sum().fillna(0)
        # dfx = dfx.resample('D').sum().fillna(method='ffill')
        dfx['bin'] = bin
        y = list(dfx[chart_type])
        zarray.append(y)
    z = np.array(zarray)
    trace = go.Heatmap(
        x=x,
        y=ticktext,
        z=zarray,
        # zmid=0.2,
        name='Udział',
        colorscale=constants.color_scales[settings['color_order']],
        colorbar=dict(title=dict(text='',
                                 side='top',
                                 font=dict(color='white')
                                 ),
                      tickfont=dict(color='white')
    )
    )
    figure = go.Figure(
        data=[trace],
        layout=dict(
            template=template,
            autosize=True,
            title=dict(text=title),
            margin=dict(l=220),
            height=height,
            showlegend=False,
        )
    )
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
        'toImageButtonOptions': {
            'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
            'filename': 'docent_image', 'height': height, 'scale': 1}
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    return fig


def heatmap_bis(_df, yaxis, xaxis, chart_type, settings, template, mapa={}, title='Heatmap'):
    df = _df[[xaxis, yaxis, chart_type]].copy()
    df = df.replace(0, np.nan)
    bins = df[yaxis].unique()
    if len(mapa) == 0:
        ticktext = [x for x in bins]
    else:
        ticktext = [mapa[x] for x in bins]
    zarray = []
    date_min = min(df[xaxis])
    date_max = max(df[xaxis])
    x = sorted(list(df[xaxis].unique()))
    height = float(settings['plot_height']) * 33 * len(bins) + 80
    for bin in bins:
        dfx = df[df[yaxis] == bin].copy()
        dfx['bin'] = bin
        y = list(dfx[chart_type])
        zarray.append(y)
    z = np.array(zarray)
    trace = go.Heatmap(
        x=x,
        y=ticktext,
        z=zarray,
        # zmid=0.2,
        name='Udział',
        colorscale=constants.color_scales[settings['color_order']],
        colorbar=dict(title=dict(text='',
                                 side='top',
                                 font=dict(color='white')
                                 ),
                      tickfont=dict(color='white')
    )
    )
    figure = go.Figure(
        data=[trace],
        layout=dict(
            template=template,
            autosize=True,
            title=dict(text=title),
            margin=dict(l=220),
            height=height,
            showlegend=False,
        )
    )
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
        'toImageButtonOptions': {
            'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
            'filename': 'docent_image', 'height': height, 'scale': 1}
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    return fig


############################################
# Korelator
############################################
def layout_korelacja(_names, series_x, series_y, name_x, name_y, title_date=str(dt.today())[:10]):
    from sklearn.linear_model import LinearRegression
    from numpy import cov
    from scipy.stats import pearsonr, spearmanr

    template = get_template()
    settings = session2settings()
    from_date = settings['from_date']
    core_view = settings['core_view']

    names = list(_names)
    xvals = list(series_x)
    yvals = list(series_y)
    corr, p_v_p = pearsonr(xvals, yvals)
    corrs, p_v_s = spearmanr(xvals, yvals)
    figure = go.Figure()
    fig_data1 = go.Scatter(x=xvals,
                           y=yvals,
                           mode='lines+markers+text',
                           text=names,
                           textfont=dict(size=settings['font_size_anno'], color=get_session_color(4)),
                           line=dict(color='yellow', width=0),
                           marker=dict(symbol='diamond',
                                       size=15,
                                       opacity=0.5,
                                       color='red'),
                           name='x',
                           opacity=1,
                           )
    figure.add_trace(fig_data1)

    reg_txt = ''

    # linia regresji liniowej

    xvals = series_x.values.reshape(-1, 1)
    yvals = series_y.values.reshape(-1, 1)
    linear_regressor = LinearRegression()
    linear_regressor.fit(xvals, yvals)
    score = linear_regressor.score(xvals, yvals)
    yvals = linear_regressor.predict(xvals)
    xvals = list(xvals.flatten())
    yvals = list(yvals.flatten())
    yorg = list(series_y)
    reg_txt = '<br>Współczynnik determinacji: ' + str(round(score, 3)) + \
              ', współczynnik korelacji Pearsona: ' + str(round(corr, 3)) + \
              ', współczynnik korelacji Spearmana: ' + str(round(corrs, 3))
    err = [yvals[i] - yorg[i] for i in range(len(yvals))]

    error_y = {}
    fig_data2 = go.Scatter(x=xvals,
                          y=yvals,
                          mode='lines',
                          line=dict(color='green', width=1),
                          error_y=error_y)
    figure.add_trace(fig_data2)
    figure.update_layout(height=float(settings['plot_height']) * 660,
                         template=template,
                         title=layout_title(
                             text=name_x + ' vs. ' + name_y + \
                                  '<br><sub>'+ title_date + reg_txt,
                             font_size=settings['font_size_title'],
                             color=get_session_color(7),
                             posx=settings['titlexpos'],
                             posy=settings['titleypos'],
                         ),
                         xaxis=dict(
                             title=dict(text=name_x,
                                        font=dict(color=get_session_color(4))
                                        ),
                             # rangemode=tozerox,
                             rangeslider={'visible': ('suwak' in core_view), 'thickness': 0.15}
                         ),
                         yaxis=dict(
                             title=dict(text=name_y,
                                        font=dict(color=get_session_color(4))),
                             # rangemode=tozeroy,
                         ),
                         showlegend=False),
    figure = add_copyright(figure, settings)
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
    return fig


def layout_title(text, font_size, color, posx=0.5, posy=0.0, pad_t=20, pad_r=0, pad_b=50, pad_l=0, xanchor='center'):
    ret_val = dict(
        x=posx,
        y=posy,
        xanchor=xanchor,
        yanchor='top',
        pad={'t': pad_t, 'b': pad_b, 'l': pad_l, 'r': pad_r},
        text=text,
        font=dict(size=font_size, color=color),
    )
    return ret_val


####################################
# Analiza zgonów nadmiarowych
####################################

@logit
def layout_more_wplz(DF, location, year0, _age):

    template = get_template()
    settings = session2settings()
    traces = []
    oddaty = '2020-09-01'
    df = pd.read_csv(constants.data_files['mortality_eurostat']['data_fn'])
    df_p = DF['poland']
    df = df[(df['age'] == _age) & (df['location'] == location)].copy()

    # zgony w latach year0 - 2019

    df = df[(df['year'] >= int(year0)) & (df['total'] != 0.)].copy()

    mean_deaths = df[df['year'] < 2020]['total'].mean()
    min_deaths = df['total'].min()
    max_deaths = df['total'].max()

    df = df[(df['year'] >= 2020) & (df['total'] != 0.)].copy()

    df['location_lower'] = df['location'].str.lower()
    df_p['location_lower'] = df_p['location'].str.lower()

    # uzupełnienie dat
    df_lastd = list(df.date.unique())[-1]
    df_p_lastd = list(df_p.date.unique())[-1]
    date_list = [str(i)[:10] for i in pd.date_range(df_lastd, periods=10, freq="7D")][1:]
    for date in date_list:
        if date <= df_p_lastd:
            row = {'date': date, 'year': 2022, 'total': np.NaN, 'age': _age, 'short': 'PL', 'location': location, 'location_lower': location.lower()}
            df = df.append(row, ignore_index=True).copy()

    filtered_df = pd.merge(df, df_p[['total_deaths', 'total_cases', 'positive_rate', 'location_lower', 'date']],
                           how='left',
                           left_on=['location_lower', 'date'],
                           right_on=['location_lower', 'date']).copy()

    filtered_df = filtered_df[filtered_df['date'] > oddaty].copy()

    filtered_df.sort_values(by=['location_lower', 'date'], inplace=True)
    filtered_df['total_excess'] = filtered_df['total'] - mean_deaths
    filtered_df['deaths_week'] = filtered_df.groupby(['location_lower'])['total_deaths'].diff()
    filtered_df['deaths_week'].fillna(0, inplace=True)
    filtered_df['cases_week'] = filtered_df.groupby(['location_lower'])['total_cases'].diff()
    filtered_df['cases_week'].fillna(0, inplace=True)

    ymin = min_deaths
    ymax = max_deaths

    # przeskalowanie cases_week

    cwmin = filtered_df['cases_week'].min()
    cwmax = filtered_df['cases_week'].max()
    filtered_df['cw'] = (filtered_df['cases_week'] - cwmin) / (cwmax-cwmin) * (ymax - ymin) + ymin

    # przeskalowanie positive rate

    prmin = filtered_df['positive_rate'].min()
    prmax = filtered_df['positive_rate'].max()
    filtered_df['pr'] = (filtered_df['positive_rate'] - prmin) / (prmax-prmin) * (ymax - ymin) + ymin

    ile_2020_c = filtered_df[filtered_df.year == 2020]['deaths_week'].sum()
    ile_2020_n = filtered_df[filtered_df.year == 2020]['total_excess'].sum()
    prop_2020 = round(ile_2020_c / ile_2020_n * 100, 1)
    ile_2021_c = filtered_df[filtered_df.year == 2021]['deaths_week'].sum()
    ile_2021_n = filtered_df[filtered_df.year == 2021]['total_excess'].sum()
    prop_2021 = round(ile_2021_c / ile_2021_n * 100, 1)

    anno_text = 'Średnia liczba zgonów tygodniowo w latach ' + str(year0) + '- 2019: ' + str(int(mean_deaths)) + \
                '<br><br>Liczba zgonów Covid w 2020: ' + str(int(ile_2020_c)) + \
                '<br>Liczba zgonów nadmiarowych w 2020: ' + str(int(ile_2020_n)) + \
                '<br>Udział zgonów Covid w zgonach nadmiarowych w 2020: ' + str(prop_2020) + '%' + \
                '<br><br>Liczba zgonów Covid w 2021: ' + str(int(ile_2021_c)) + \
                '<br>Liczba zgonów nadmiarowych w 2021 (do ' + df_lastd + '): ' + str(int(ile_2021_n)) + \
                '<br><br><sub>* Wykresy wykrywalności i infekcji są skalowane proporcjonalnie ' \
                'i służą do orientacyjnego porównania przebiegów'

    # zgony razem

    fig_data = go.Scatter(
        x=list(filtered_df['date'].astype('datetime64[ns]')),
        y=list(filtered_df['total']),
        fill='tozeroy',
        line=dict(color='orange', width=0),
        name='liczba zgonów ogółem',
        opacity=1
    )
    traces.append(fig_data)

    # zgony Covid

    fig_data = go.Scatter(
        x=list(filtered_df['date'].astype('datetime64[ns]')),
        y=list(filtered_df['deaths_week'] + mean_deaths),
        fill='tozeroy',
        line=dict(color='red', width=0),
        name='zgony Covid (+ średnia)',
        opacity=1
    )
    traces.append(fig_data)

    # cases week

    fig_data = go.Scatter(
        x=list(filtered_df['date'].astype('datetime64[ns]')),
        y=list(filtered_df['cw']),
        line=dict(color='yellow', width=settings['linewidth_basic'], dash='longdash'),
        name='infekcje',
        opacity=1
    )
    traces.append(fig_data)

    # positive rate

    fig_data = go.Scatter(
        x=list(filtered_df['date'].astype('datetime64[ns]')),
        y=list(filtered_df['pr']),
        line=dict(color='aqua', width=settings['linewidth_basic'], dash='dashdot'),
        name='wykrywalność',
        opacity=1
    )
    traces.append(fig_data)

    # linia średniej rocznej zgonów

    fig_data = go.Scatter(
        x=list(filtered_df['date'].astype('datetime64[ns]')),
        y=[mean_deaths for k in range(len(filtered_df))],
        line=dict(width=2, color='black'),
        # line=dict(width=settings['linewidth_basic'], color='lightblue'),
        fill='tozeroy',
        name='średnia roczna zgonów ' + str(year0) + '-19 (' + str(int(mean_deaths)) + ')',
        opacity=1
    )
    traces.append(fig_data)

    figure = go.Figure(
        data=traces,
        layout=dict(
            margin=dict(l=80, r=50, b=50, t=130, pad=2),
            template=template,
            height=float(settings['plot_height']) * 720,
            yaxis=dict(
                range=[0, max_deaths],
                # range=[min_deaths, max_deaths],
            ),
            title=dict(text=location.upper() + '. Analiza zgonów Covid-19 i zgonów nadmiarowych w porównaniu do infekcji i wykrywalności.' +
                            '<br><sub> (sumy tygodniowe)<br>Średnia z lat ' + str(year0) + '-2019. Dane: Eurostat, MZ'),
            legend=dict(x=0.5, y=-0.1, orientation='h', xanchor='center')
            # legend=dict(x=1.02, y=1., orientation='v')
        )
    )
    figure.add_annotation(text=anno_text,
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=1., y=1.05,
                          # x=0.01, y=0.8,
                          showarrow=False)
    figure = add_copyright(figure, settings)
    config = {
        'displaylogo': False,
        'responsive': True,
        'locale': 'pl-PL',
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'mortality_image', 'width': 1200, 'scale': 1}
    }
    children = dbc.Col(dcc.Graph(id=location, config=config, figure=figure))

    return children


####################################################
# Porównanie zgonów nadmiarowych i zgonów Covid-19
####################################################

@logit
def layout_more_wplz0(DF, location, _age, year0):

    template = get_template()
    settings = session2settings()
    traces = []
    oddaty = '2020-09-01'

    # dane Eurostat

    df = pd.read_csv(constants.data_files['mortality_eurostat']['data_fn'])
    df = df[(df['age'] == _age) & (df['location'] == location)].copy()
    df = df[(df['year'] >= int(year0)) & (df['total'] != 0.)].copy()

    # obliczenie średnich tygodniowych

    df_mean = df[df['year'] < 2020].copy()
    df_mean['mean'] = df_mean.groupby(['week'])['total'].transform('mean')
    df_mean = df_mean.drop_duplicates(subset=['week']).copy()
    df = pd.merge(df, df_mean[['week', 'mean']], how='left',
                  left_on=['week'], right_on=['week']).copy()

    mean_deaths = df[df['year'] < 2020]['total'].mean()
    min_deaths = df['total'].min()
    max_deaths = df['total'].max()
    df = df[(df['year'] >= 2020) & (df['total'] != 0.)].copy()
    df['location_lower'] = df['location'].str.lower()

    # dane Polska

    df_p = DF['poland']
    df_p['location_lower'] = df_p['location'].str.lower()

    # uzupełnienie dat

    df_lastd = sorted(list(df.date.unique()))[-1]
    df_p_lastd = list(df_p.date.unique())[-1]
    date_list = [str(i)[:10] for i in pd.date_range(df_lastd, periods=10, freq="7D")][1:]
    for date in date_list:
        if date <= df_p_lastd:
            row = {'date': date, 'year': 2021, 'total': np.NaN, 'age': 'TOTAL', 'short': 'PL', 'location': location, 'location_lower': location.lower()}
            df = df.append(row, ignore_index=True).copy()

    # agregacja

    filtered_df = pd.merge(df, df_p[['total_deaths', 'total_cases', 'positive_rate', 'location_lower', 'date']],
                           how='left',
                           left_on=['location_lower', 'date'],
                           right_on=['location_lower', 'date']).copy()

    filtered_df = filtered_df[filtered_df['date'] > oddaty].copy()
    filtered_df.sort_values(by=['location_lower', 'date'], inplace=True)
    filtered_df['total_excess'] = filtered_df['total'] - mean_deaths
    filtered_df['deaths_week'] = filtered_df.groupby(['location_lower'])['total_deaths'].diff()
    filtered_df['deaths_week'].fillna(0, inplace=True)
    filtered_df['cases_week'] = filtered_df.groupby(['location_lower'])['total_cases'].diff()
    filtered_df['cases_week'].fillna(0, inplace=True)

    ile_2020_c = filtered_df[filtered_df.year == 2020]['deaths_week'].sum()
    ile_2020_n = filtered_df[filtered_df.year == 2020]['total_excess'].sum()
    prop_2020 = round(ile_2020_c / ile_2020_n * 100, 1)
    ile_2021_c = filtered_df[filtered_df.year == 2021]['deaths_week'].sum()
    ile_2021_n = filtered_df[filtered_df.year == 2021]['total_excess'].sum()
    prop_2021 = round(ile_2021_c / ile_2021_n * 100, 1)
    ile_2022_c = filtered_df[filtered_df.year == 2022]['deaths_week'].sum()
    ile_2022_n = filtered_df[filtered_df.year == 2022]['total_excess'].sum()

    filtered_df['deaths_covid_all'] = filtered_df['total'] - filtered_df['deaths_week']
    anno_text = 'Średnia liczba zgonów tygodniowo w latach ' + str(year0) + '- 2019: ' + str(int(mean_deaths)) + \
                '<br>Liczba zgonów Covid w 2020: ' + str(int(ile_2020_c)) + \
                '<br>Liczba zgonów nadmiarowych w 2020: ' + str(int(ile_2020_n)) + \
                '<br>Udział zgonów Covid w zgonach nadmiarowych w 2020: ' + str(prop_2020) + '%' + \
                '<br>Liczba zgonów Covid w 2021: ' + str(int(ile_2021_c)) + \
                '<br>Liczba zgonów nadmiarowych w 2021: ' + str(int(ile_2021_n)) + \
                '<br>Udział zgonów Covid w zgonach nadmiarowych w 2021: ' + str(prop_2021) + '%' + \
                '<br>Liczba zgonów Covid w 2022: ' + str(int(ile_2022_c)) + \
                '<br>Liczba zgonów nadmiarowych w 2022 (do ' + df_lastd + '): ' + str(int(ile_2022_n))

        # zgony razem

    fig_data = go.Scatter(
        x=list(filtered_df['date'].astype('datetime64[ns]')),
        y=list(filtered_df['total']),
        fill='tonexty',
        line=dict(color='orange', width=0),
        name='liczba wszystkich zgonów bez Covid-19',
        opacity=1
    )
    traces.append(fig_data)

    # zgony Covid

    fig_data = go.Scatter(
        x=list(filtered_df['date'].astype('datetime64[ns]')),
        y=list(filtered_df['deaths_covid_all']),
        fill='tonexty',
        line=dict(color='red', width=0),
        name='zgony Covid-19',
        opacity=1
    )
    traces.append(fig_data)

    # cases week

    fig_data = go.Scatter(
        x=list(filtered_df['date'].astype('datetime64[ns]')),
        y=list(filtered_df['cases_week']),
        line=dict(color='blue', width=0.6, dash='longdash'),
        name='infekcje',
        yaxis='y2',
        opacity=1
    )
    traces.append(fig_data)

    # średnie zgonów

    fig_data = go.Scatter(
        x=list(filtered_df['date'].astype('datetime64[ns]')),
        y=list(filtered_df['mean']),
        line=dict(width=settings['linewidth_basic'], color='green'),
        fill='tozeroy',
        name='średnie tygodniowe liczby zgonów z lat ' + str(year0) + '-19',
        opacity=1
    )
    traces.append(fig_data)

    # linia średniej rocznej zgonów

    fig_data = go.Scatter(
        x=list(filtered_df['date'].astype('datetime64[ns]')),
        y=[mean_deaths for k in range(len(filtered_df))],
        line=dict(width=3, color='lightblue'),
        name='średnia roczna zgonów ' + str(year0) + '-19 (' + str(int(mean_deaths)) + ')',
        opacity=1
    )
    traces.append(fig_data)

    figure = go.Figure(
        data=traces,
        layout=dict(
            margin=dict(l=80, r=50, b=50, t=130, pad=2),
            template=template,
            height=float(settings['plot_height']) * 720,
            yaxis=dict(
                title='Zgony tygodniowo',
                titlefont=dict(color='red'),
                tickfont=dict(color='red'),
                range=[0, max_deaths],
            ),
            title=dict(text=location.upper() + '<br><b>Porównanie zgonów Covid-19 i zgonów nadmiarowych.</b>' + \
                            '<br>' + _age + ' (sumy tygodniowe)<br>Średnia z lat ' + \
                            str(year0) + '-2019. Dane: Eurostat, GUS, MZ'),
            legend=dict(x=0.5, y=-0.1, orientation='h', xanchor='center')
        )
    )
    figure.update_layout(
        yaxis2=dict(
            title='infekcje tygodniowo',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue'),
            anchor="free",
            overlaying="y",
            side="right",
            position=1.
        ),
    )
    figure.add_annotation(text=anno_text,
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.99, y=0.97,
                          # x=0.01, y=0.8,
                          showarrow=False)
    figure = add_copyright(figure, settings)
    config = {
        'displaylogo': False,
        'responsive': True,
        'locale': 'pl-PL',
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'mortality_image', 'width': 1200, 'scale': 1}
    }
    children = dbc.Col(dcc.Graph(id=location, config=config, figure=figure))

    return children


####################################################
# Roczne krzywe zgonów 2016-2022
####################################################

@logit
def layout_more_wplz1(DF, _1, _2, _3):

    template = get_template()
    settings = session2settings()
    traces = []

    # dane USC

    df = pd.read_csv('data/last/last_zgony_usc.csv')

    # obliczenie średnich tygodniowych 2016-2019

    df['mean1619'] = df[['2016', '2017', '2018', '2019']].mean(axis=1)

    last22 = df[df['2022'].isna()].iloc[0]['Nr tygodnia']-1

    fig_data = go.Scatter(
        x=list(df['Nr tygodnia']),
        y=list(df['mean1619']),
        line=dict(width=settings['linewidth_thick']),
        name='średnia 2016-2019',
        opacity=1
    )
    traces.append(fig_data)

    fig_data = go.Scatter(
        x=list(df['Nr tygodnia']),
        y=list(df['2020']),
        line=dict(width=settings['linewidth_thin']),
        name='2020',
        opacity=1
    )
    traces.append(fig_data)
    fig_data = go.Scatter(
        x=list(df['Nr tygodnia']),
        y=list(df['2021']),
        line=dict(width=settings['linewidth_thin']),
        name='2021',
        opacity=1
    )
    traces.append(fig_data)
    fig_data = go.Scatter(
        x=list(df['Nr tygodnia']),
        y=list(df[df['Nr tygodnia'] <= (last22-2)]['2022']),
        # y=list(df['2022']),
        line=dict(width=settings['linewidth_basic']),
        name='2022',
        opacity=1
    )
    traces.append(fig_data)

    fig_data = go.Scatter(
        # x=list(df['Nr tygodnia']),
        x=list(df[df['Nr tygodnia'].between((last22-2), last22)]['Nr tygodnia']),
        y=list(df[df['Nr tygodnia'].between((last22-2), last22)]['2022']),
        # y=list(df['2022']),
        line=dict(width=settings['linewidth_basic'], dash='dot'),
        name='2022 niekompletne dane',
        opacity=1
    )
    traces.append(fig_data)

    figure = go.Figure(
        data=traces,
        layout=dict(
            margin=dict(l=80, r=50, b=50, t=130, pad=2),
            template=template,
            height=float(settings['plot_height']) * 720,
            title=dict(text='Roczne krzywe zgonów w latach 2016-2019 i 2020-2022<br>' + \
                            '<sub>(wartości tygodniowe, źródło danych: USC)<br><br>'),
        )
    )
    figure = add_copyright(figure, settings)
    config = {
        'displaylogo': False,
        'responsive': True,
        'locale': 'pl-PL',
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'mortality_image', 'width': 1200, 'scale': 1}
    }
    children = dbc.Col(dcc.Graph(id='krzyweroczne', config=config, figure=figure))

    return children


########################
# Zgony nadmiarowe (%)
########################

@logit
def layout_more_wplz2(DF, _1, _2, _3):

    template = get_template()
    settings = session2settings()
    traces = []

    # dane USC

    df = pd.read_csv('data/last/last_zgony_usc.csv')

    # obliczenie średnich tygodniowych 2016-2019

    df['mean1619'] = df[['2016', '2017', '2018', '2019']].mean(axis=1)
    df['ex2020'] = (df['2020'] / df['mean1619'] -1) * 100
    df['ex2021'] = (df['2021'] / df['mean1619'] -1) * 100
    df['ex2022'] = (df['2022'] / df['mean1619'] -1) * 100

    last22 = df[df['2022'].isna()].iloc[0]['Nr tygodnia']-1

    anno_text = 'Średni procent zgonów nadmiarowych w latach:' + \
        '<br><br> 2020: ' + str(round(df['ex2020'].mean(), 2)) + '%' + \
        '<br> 2021: ' + str(round(df['ex2021'].mean(), 2)) + '%' + \
        '<br> 2022: ' + str(round(df['ex2022'].mean(), 2)) + '%'

    fig_data = go.Scatter(
        x=list(df['Nr tygodnia']),
        y=list(df['ex2020']),
        line=dict(width=settings['linewidth_thin']),
        name='zgony nadmiarowe 2020 (%)',
        opacity=1
    )
    traces.append(fig_data)
    fig_data = go.Scatter(
        x=list(df['Nr tygodnia']),
        y=list(df['ex2021']),
        line=dict(width=settings['linewidth_thin']),
        name='zgony nadmiarowe 2021 (%)',
        opacity=1
    )
    traces.append(fig_data)
    fig_data = go.Scatter(
        x=list(df['Nr tygodnia']),
        y=list(df[df['Nr tygodnia'] <= (last22-2)]['ex2022']),
        line=dict(width=settings['linewidth_basic']),
        name='zgony nadmiarowe 2022 (%)',
        opacity=1
    )
    traces.append(fig_data)

    fig_data = go.Scatter(
        x=list(df[df['Nr tygodnia'].between((last22-2), last22)]['Nr tygodnia']),
        y=list(df[df['Nr tygodnia'].between((last22-2), last22)]['ex2022']),
        # y=list(df['2022']),
        line=dict(width=settings['linewidth_basic'], dash='dot'),
        name='zgony nadmiarowe 2022 (%) niekompletne dane',
        opacity=1
    )
    traces.append(fig_data)

    figure = go.Figure(
        data=traces,
        layout=dict(
            margin=dict(l=80, r=50, b=50, t=130, pad=2),
            template=template,
            height=float(settings['plot_height']) * 720,
            title=dict(text='Roczne krzywe procentu zgonów nadmiarowych w latach 2020-2022<br>' + \
                            '<sub>(w odniesieniu do średniej 2016-2019, tygodniami, źródło danych: USC)<br><br>'),
        )
    )
    figure.add_annotation(text=anno_text,
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.5, y=0.93,
                          showarrow=False)

    figure = add_copyright(figure, settings)
    config = {
        'displaylogo': False,
        'responsive': True,
        'locale': 'pl-PL',
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'mortality_image', 'width': 1200, 'scale': 1}
    }
    children = dbc.Col(dcc.Graph(id='krzyweroczne', config=config, figure=figure))

    return children


###################################################
# Udział % zgonów Covid-19 w zgonach nadmiarowych
###################################################

@logit
def layout_more_wplz3(DF, _1, _2, _3):

    template = get_template()
    settings = session2settings()
    traces = []

    df = pd.read_csv(constants.data_files['mortality_eurostat']['data_fn'])
    df = df[df['age'] == 'TOTAL'].copy()
    df = df[df['location'] == 'Polska'].copy()
    df = df[['total', 'year', 'week']].copy()
    df['mean'] = df[df['year'].between(2016, 2019)]['total'].mean()
    # df['mean'] = df[df['year'].between(2016, 2019)].groupby(['week'])['total'].transform('mean')
    # df_mean = df[df['year'] == 2019][['week', 'mean']]
    # del df['mean']
    # df = pd.merge(df, df_mean, how='left', left_on=['week'], right_on=['week']).copy()

    df_p = DF['poland']
    df_p = df_p[df_p['location'] == 'Polska'].copy()
    df_p['formatted_date'] = pd.to_datetime(df_p['date'])
    df_p['week'] = df_p.formatted_date.apply(lambda x: x.weekofyear)
    df_p['year'] = df_p.formatted_date.apply(lambda x: x.year)
    df_p = df_p[['new_deaths', 'year', 'week']].copy()
    df_p['week_deaths'] = df_p.groupby(['year', 'week'])['new_deaths'].transform('sum')
    df_p = df_p.drop_duplicates(subset=['year', 'week']).copy()

    dfx = pd.merge(df, df_p, how='left', left_on=['year', 'week'], right_on=['year', 'week']).copy()
    dfx = dfx[['year', 'week', 'mean', 'total', 'week_deaths']].copy()
    dfx.sort_values(by=['year', 'week'], inplace=True)
    dfx['share'] = dfx['week_deaths'] / (dfx['total'] - dfx['mean'])

    years = [2020, 2021, 2022]
    for year in years:
        dfy = dfx[dfx['year'] == year].copy()
        fig_data = go.Scatter(
            x=list(dfy['week']),
            y=list(dfy['share']),
            line=dict(width=settings['linewidth_thin']),
            name=str(year),
            opacity=1
        )
        traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            margin=dict(l=80, r=50, b=50, t=130, pad=2),
            template=template,
            height=float(settings['plot_height']) * 720,
            title=dict(text='Udział % zgonów Covid-19 w zgonach nadmiarowych<br>' + \
                            '<sub>(w odniesieniu do średniej tygodniowej 2016-2019)<br><br>'),
        )
    )
    figure = add_copyright(figure, settings)
    config = {
        'displaylogo': False,
        'responsive': True,
        'locale': 'pl-PL',
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'mortality_image', 'width': 1200, 'scale': 1}
    }
    children = dbc.Col(dcc.Graph(id='krzyweroczne', config=config, figure=figure))

    return children


#########################################################
# Tygodniowe liczby zgonów w grupach wiekowych
#########################################################

def layout_more_tlzgw(DF, location, year0, typ):
    template = get_template()
    settings = session2settings()
    df = pd.read_csv(constants.data_files['mortality_eurostat']['data_fn'])
    if typ != '':
        df = df[df['typ'] == typ].copy()
    year = int(year0)
    ages = list(df['age'].unique())
    ages.sort()
    means = {}

    for i in ages:
        means[i] = df[(df['age'] == i) & (df['year'] < 2020) & (df['year'] >= year) & (df['location'] == location)][
            'total'].mean()
    df = df[(df['year'].isin([2020, 2021])) & (df['location'] == location) & (df['total'] > 0)]
    cols = 5
    row_height = 250
    rows = int((len(ages) - 1) / cols) + 1
    figure = make_subplots(rows=rows,
                           cols=cols,
                           subplot_titles=ages,
                           print_grid=False,
                           )
    row = 1
    col = 0
    for age in ages:
        filtered = df[(df['age'] == age)]
        value_min = min(0, means[age])
        value_max = max(filtered['total'].max(), means[age])
        filtered.sort_values(by=['date'], inplace=True)
        filtered.reset_index(drop=True, inplace=True)
        xvals = list(filtered['date'])
        yvals = list(filtered['total'])
        yvals = [0 if math.isnan(i) else i for i in yvals]
        fig_data = go.Scatter(x=xvals,
                              y=yvals,
                              line=dict(color='yellow', width=0.5),
                              name=age)
        col += 1
        if col > cols:
            row += 1
            col = 1
        figure.add_trace(fig_data, row=row, col=col)
        # średnia
        xvals = list(filtered['date'])
        yvals = [means[age] for i in xvals]
        fig_data = go.Scatter(x=xvals,
                              y=yvals,
                              line=dict(color='red', width=1),
                              name=age)
        figure.add_trace(fig_data, row=row, col=col)
        figure.update_yaxes(range=[value_min, value_max], row=row, col=col)

    height = 100 + row_height * row
    figure.update_layout(height=height,
                         template=template,
                         title=dict(text='Tygodniowe liczby zgonów w grupach wiekowych w porównaniu do średniej z ' +
                                         str(year) + '-2019' + '<br>(' + location + ')'),
                         showlegend=False),
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displayModeBar': False,
        'locale': 'pl-PL',
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'docent_image',
            'width': 120 * 12, 'scale': 1}
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)

    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


###############################################
# Tygodniowe liczby zgonów w państwach
###############################################

def layout_more_tlzp(DF, age0, year0, typ):
    template = get_template()
    settings = session2settings()
    df = pd.read_csv(constants.data_files['mortality_eurostat']['data_fn'])
    if typ != '':
        df = df[df['typ'] == typ].copy()
    year = int(year0)
    age = age0
    locations = list(df['location'].unique())
    locations.sort()
    means = {}

    for i in locations:
        means[i] = df[(df['location'] == i) & (df['year'] < 2020) & (df['year'] >= year) & (df['age'] == age)][
            'total'].mean()
    df = df[(df['year'] == 2020) & (df['age'] == age) & (df['total'] > 0)]
    cols = 9
    row_height = 200
    rows = int((len(locations) - 1) / cols) + 1
    figure = make_subplots(rows=rows,
                           cols=cols,
                           subplot_titles=locations,
                           print_grid=False,
                           )
    row = 1
    col = 0
    for location in locations:
        filtered = df[(df['location'] == location)]
        value_min = min(0, means[location])
        value_max = max(filtered['total'].max(), means[location])
        filtered.sort_values(by=['date'], inplace=True)
        filtered.reset_index(drop=True, inplace=True)
        xvals = list(filtered['date'])
        yvals = list(filtered['total'])
        yvals = [0 if math.isnan(i) else i for i in yvals]
        fig_data = go.Scatter(x=xvals,
                              y=yvals,
                              line=dict(color='yellow', width=0.5),
                              name=location)
        col += 1
        if col > cols:
            row += 1
            col = 1
        figure.add_trace(fig_data, row=row, col=col)
        # średnia
        xvals = list(filtered['date'])
        yvals = [means[location] for i in xvals]
        fig_data = go.Scatter(x=xvals,
                              y=yvals,
                              line=dict(color='red', width=1),
                              name=location)
        figure.add_trace(fig_data, row=row, col=col)
        figure.update_yaxes(range=[value_min, value_max], row=row, col=col)

    height = 100 + row_height * row
    figure.update_layout(height=height,
                         template=template,
                         title=dict(text='Tygodniowe liczby zgonów w państwach w porównaniu do średniej z ' +
                                         str(year) + '-2019' + '<br>(' + age + ')'),
                         showlegend=False),
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        # 'displayModeBar': False,
        'locale': 'pl-PL',
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'docent_image',
            # 'height': int(settings['dynamics_chart_height']) * row,
            'width': 120 * 12, 'scale': 1}
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)

    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


###############################################
#  Współczynnik reprodukcji (R) w województwach
###############################################

def layout_more_wrrw(DF, days, _sex0, _3):
    # session['template'] = 'white'
    # session['template_change'] = True
    template = get_template()
    settings = session2settings()
    df = pd.read_csv(constants.data_files['woj_rt']['data_fn'])
    df = df[df['date'] >= '2021-01-01'].copy()
    locations = list(df['location'].unique())
    locations.sort()

    cols = 6
    row_height = 230
    rows = int((len(locations) - 1) / cols) + 1
    figure = make_subplots(rows=rows,
                           cols=cols,
                           subplot_titles=locations,
                           print_grid=False,
                           )
    row = 1
    col = 0
    for location in locations:
        filtered = df[(df['location'] == location)]
        value_min = 0
        value_max = min(df['reproduction_rate'].max(), 3)
        filtered.sort_values(by=['date'], inplace=True)
        filtered.reset_index(drop=True, inplace=True)

        # województwo

        xvals = list(filtered['date'])
        yvals = round(filtered['reproduction_rate'], 5)
        # yvals = round(filtered['reproduction_rate'].rolling(int(days), min_periods=1).mean(), 5)
        yvals = [0 if math.isnan(i) else i for i in yvals]
        fig_data = go.Scatter(x=xvals,
                              y=yvals,
                              line=dict(color='purple', width=2),
                              name=location)
        col += 1
        if col > cols:
            row += 1
            col = 1
        figure.add_trace(fig_data, row=row, col=col)
        xvals = list(filtered['date'])
        yvals = [1 for i in xvals]
        fig_data = go.Scatter(x=xvals,
                              y=yvals,
                              line=dict(color='orange', width=2),
                              name=location)
        figure.add_trace(fig_data, row=row, col=col)
        figure.update_yaxes(range=[value_min, value_max], row=row, col=col)
    height = 100 + row_height * row
    figure.update_layout(height=height,
                         margin=dict(l=50, r=50, b=70, t=100, pad=2),
                         template=template,
                         title=dict(text='<b>Współczynnik reprodukcji R(t) w województwach'),
                         showlegend=False),
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'docent_image',
            'width': 1200, 'height': height, 'scale': 1}
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)

    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


##############################################################
#  MZ Szczepienia - NOP, dawki utracone
##############################################################


def layout_more_mz_nop(_0, _1, _2, _3):
    template = get_template()
    settings = session2settings()

    df = pd.read_csv(constants.data_files['mz_api_age_v']['data_fn'])

    df.sort_values(by=['date'], inplace=True)
    traces = []
    x = list(df['date'])
    y = list(df['ODCZYNY_NIEPOZADANE'])
    fig_data = go.Bar(
        x=x,
        y=y,
        textposition='outside',
        name='Odczyny niepożądane (NOP)'
    )
    traces.append(fig_data)
    y = list(df['DAWKI_UTRACONE'])
    fig_data = go.Bar(
        x=x,
        y=y,
        textposition='outside',
        name='Dawki utracone'
    )
    traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            # barmode='stack',
            title=dict(text='Odczyny niepożądane (NOP) i dawki utracone'),
            height=float(settings['plot_height']) * 660,
            legend=dict(x=0, y=1.03, orientation='h'),
        )
    )
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='nop', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


def ie_schools(fn):
    import geocoder
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="docent.space")

    def f(r):
        phrase = r['School Name'] + ' ' + r['Address'] + ' Ireland'
        location = geolocator.geocode(phrase)
        if location is not None:
            r['location'] = location
            r['Long'] = location.longitude
            r['Lat'] = location.latitude
            r['type'] = 'SA'
            print('++++ JEST geolocator 1 School Name+Address')
        else:
            phrase = r['School Name'] + ' Ireland'
            location = geolocator.geocode(phrase)
            print('geolocator 2 phrase=' + phrase)
            if location is not None:
                r['location'] = location
                r['Long'] = location.longitude
                r['Lat'] = location.latitude
                r['type'] = 'S'
                print('++++ JEST geolocator 2 School Name')
            else:
                phrase = r['Address'] + ' ' + r['County'] + ' Ireland'
                print('geocoder phrase=' + phrase)
                g = geocoder.osm(phrase)
                r['location'] = str(g)
                r['Long'] = g.x
                r['Lat'] = g.y
                r['type'] = 'AC'
                print(r['Long'], r['Lat'])
                print('JEST geocoderAddress+County')
                # print('############ nie ma ############')
        return r

    df = pd.read_csv('/home/docent/ireland/' + fn + '.csv')
    df = df.apply(f, axis=1)
    df.to_csv('/home/docent/ireland/' + fn + '_coord.csv', index=False)
    print('wczytano')
    return df

##############################################################
#  Animowana mapa szkół w Irlandii
##############################################################
def layout_more_ireland(_0, _1, _2, _3):
    settings = session2settings()

    df5 = pd.read_csv('/home/docent/ireland/25.10-01.11.csv', dtype=object)
    df6 = pd.read_csv('/home/docent/ireland/01-08.11.csv', dtype=object)
    df7 = pd.read_csv('/home/docent/ireland/08-15.11.csv', dtype=object)
    df5['coordinates'] = df5['coordinates'].str.replace(', ', ',')
    df6['coordinates'] = df6['coordinates'].str.replace(', ', ',')
    df7['coordinates'] = df7['coordinates'].str.replace(', ', ',')
    df5['coordinates'] = df5['coordinates'].str.replace(' ', ',')
    df6['coordinates'] = df6['coordinates'].str.replace(' ', ',')
    df7['coordinates'] = df7['coordinates'].str.replace(' ', ',')
    r_date = '2021-11-01'
    df = df5.copy()
    df = df.append(df6, ignore_index=True)
    df = df.append(df7, ignore_index=True)

    def f(r):
        d = r['Report Date'].replace('/', '.').split(sep='.')
        print(d)
        if len(d) == 3:
            if len(d[0]) == 1:
                d[0] = '0' + d[0]
            r['date'] = d[2] + '-' + d[1] + '-' + d[0]
            if len(r['date']) != 10:
                r['date'] = '2021-11-01'
        x = r['coordinates']
        if type(x) == str:
            print(x)
            if ',' in x:
                r['Long'] = float(x.split(sep=',')[1])
                r['Lat'] = float(x.split(sep=',')[0])
        return r
    df = df.apply(f, axis=1)
    df.sort_values(by=['School Name'], inplace=True)
    df.dropna(axis='rows', subset=['Lat', 'Long'], inplace=True)
    df['Lat'] = df['Lat']
    df['Long'] = df['Long']
    df['School Name'] = df['School Name'].str.replace("'", '')

    df = df[df.date >= r_date].copy()
    df['Day cases'] = df['Day cases'].astype(float)
    days = sorted(list(df['date'].unique()))

    ################
    longs = df['Long'].unique()
    db = pd.DataFrame()
    for long in longs:
        df_skel = pd.DataFrame({'date': days})
        print(long, 'skel:', len(df_skel))
        df_skel['Long'] = str(long)
        # df_skel['Long'] = long.astype(str)
        df_one = df.loc[df['Long'] == long].copy()
        df_one['Long'] = str(long)
        # df_one['Long'] = long.astype(str)
        df_part = pd.merge(df_skel, df_one,
                           how='left',
                           left_on=['date', 'Long'],
                           right_on=['date', 'Long']).copy()
        df_part['Day cases'].fillna(0, inplace=True)
        df_part.loc[:, ].ffill(inplace=True)
        df_part.sort_values(by=['date', 'Long'], inplace=True)

        df_part['cumsum'] = df_part['Day cases'].cumsum()
        print(long, len(df_part))
        db = db.append(df_part, ignore_index=True)
    df = db.copy()
    del db
    df.dropna(axis='rows', subset=['cumsum'], inplace=True)
    df.sort_values(by=['date', 'Long'], inplace=True)
    df['size'] = df['cumsum']
    df.reset_index(drop=True, inplace=True)
    df.index = df['date']
    frames = [{
        'name': 'frame_{}'.format(day),
        'layout': {'showlegend': False},
        'data': [
            {
                'type': 'scattermapbox',
                'lat': pd.Series(df.xs(day)['Lat']),
                'lon': pd.Series(df.xs(day)['Long']),
                'marker': go.scattermapbox.Marker(
                    size=10,
                    color='blue',
                    symbol='toilet',
                    showscale=False,
                    allowoverlap=True,
                    opacity=0.9,
                ),
            },
            {
                'type': 'scattermapbox',
                'lat': pd.Series(df.xs(day)['Lat']),
                'lon': pd.Series(df.xs(day)['Long']),
                'marker': go.scattermapbox.Marker(
                    size=pd.Series(df.xs(day)['size']) * 3,
                    opacity=0.4,
                    showscale=False,
                    colorscale=settings['map_color_scale'],
                ),
                'customdata': np.stack((df.xs(day)['School Name'], df.xs(day)['Address']), axis=-1),
                'hovertemplate': "%{customdata[0]}<em>🚨  %{customdata[1]}",
            },
        ],
    } for day in days]

    sliders = [{
        'transition': {'duration': 0},
        'x': 0.08,
        'len': 0.88,
        'currentvalue': {'font': {'size': 16}, 'prefix': '📅 ', 'visible': True, 'xanchor': 'center'},
        'steps': [
            {
                'label': day,
                'method': 'animate',
                'args': [
                    ['frame_{}'.format(day)],
                    {'mode': 'immediate', 'frame': {'duration': 100, 'redraw': True}, 'transition': {'duration': 50}}
                ],
            } for day in days]
    }]

    play_button = [{
        'type': 'buttons',
        'showactive': True,
        'x': 0.045, 'y': -0.08,
        'buttons': [{
            'label': '🎬',  # Play
            'method': 'animate',
            'args': [
                None,
                {
                    'frame': {'duration': 1000, 'redraw': True},
                    # 'transition': {'duration': 200, 'easing': 'cubic-in-out'},
                    'fromcurrent': True,
                    'label': 'Odtwórz',
                    'mode': 'immediate',
                }
            ]
        }]
    }]

    data = frames[0]['data']

    # Adding all sliders and play button to the layout
    layout = go.Layout(
        sliders=sliders,
        updatemenus=play_button,
        showlegend=False,
        height=800,
        title=dict(text='Covid-19 in Irish Schools (since ' + r_date + ')'),
        mapbox={
            'accesstoken': open("data/mapbox.token").read(),
            'center': {"lat": 53, "lon": -7.5},
            'zoom': 6,
            'style': 'outdoors',
        }
    )
    config = {
        'displaylogo': False,
    }
    fig = go.Figure(data=data, layout=layout, frames=frames)
    create_html(fig, 'Ireland_ALL')
    ret_val = [
        dbc.Col(dcc.Graph(id='mapax', figure=fig, config=config), width=12, className='mb-0 mt-0')
    ]
    return ret_val


##############################################################
#  Heatmap
##############################################################

def layout_more_heatmap(DF, _sort, _order, _n):
    template = get_template()
    settings = session2settings()

    n = int(_n)

    # ['alfabetycznie', 'max', 'min', 'ostatni']
    # ['rosnąca', 'malejąca']

    scope = settings['scope']
    locations = settings['locations']
    chart_types = settings['chart_type']
    if len(locations) == 0:
        return
    locations.sort()
    if len(chart_types) != 1:
        return
    chart_type = chart_types[0]
    df = prepare_data(settings, DF,
                      scope=scope,
                      locations=locations,
                      chart_types=chart_types,
                      date='',
                      all_columns=True)

    if len(df) == 0:
        return 'Brak rekordów spełniających podane warunki'

    if _sort == 'alfabetycznie':
        if _order == 'rosnąca':
            df.sort_values(by=['location', 'date'], ascending=False, inplace=True)
        else:
            df.sort_values(by=['location', 'date'], inplace=True)
    elif _sort == 'max':
        df['max'] = df.groupby(['location'])[chart_type].transform('max')
        if _order == 'rosnąca':
            df.sort_values(by=['max', 'date'], ascending=False, inplace=True)
        else:
            df.sort_values(by=['max', 'date'], inplace=True)
    elif _sort == 'min':
        df['max'] = df.groupby(['location'])[chart_type].transform('min')
        if _order == 'rosnąca':
            df.sort_values(by=['min', 'date'], ascending=False, inplace=True)
        else:
            df.sort_values(by=['min', 'date'], inplace=True)
    elif _sort == 'ostatni':
        dd = df[['location', 'date', chart_type]].copy()
        dd = dd.groupby(['location']).tail(1).copy()
        dd.rename(columns={chart_type: 'last'}, inplace=True)
        df = pd.merge(df, dd[['location', 'last']], how='left', left_on=['location'], right_on=['location']).copy()
        if _order == 'rosnąca':
            df.sort_values(by=['last', 'date'], ascending=False, inplace=True)
        else:
            df.sort_values(by=['last', 'date'], inplace=True)

    l = list(df.location.unique())
    if _order == 'rosnąca':
        df = df[df['location'].isin(l[-n:])].copy()
    else:
        df = df[df['location'].isin(l[-n:])].copy()
    df.reset_index(drop=True, inplace=True)
    df = df[['date', 'location', chart_type]].copy()

    df.index = df['date']

    title = get_trace_name(chart_type)
    if settings['dzielna'] != '<brak>' and constants.trace_props[chart_type]['category'] == 'data':
        title = get_trace_name(chart_type) + ' / ' + get_trace_name(settings['dzielna']).lower()
    tail = constants.table_mean[settings['table_mean']]
    if constants.trace_props[chart_type]['category'] == 'data':
        tail += {1: '',
                 2: ' - na 100 000 osób',
                 3: ' - na 1000 km2'}[settings['data_modifier']]
    fig = heatmap(df, 'location', 'date', chart_type, settings, template, mapa=[], title='<b>' + title + '</b><br><sub>' + tail)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


##############################################################
#  MZ Szczepienia - podstawowy
##############################################################

def layout_more_mz_vacc1(_0, variant, _2, _3):
    settings = session2settings()
    template = get_template()

    # ['Trendy', 'Wiek narastająco', 'Płeć narastająco', 'Wiek przyrosty dzienne', 'Płeć przyrosty dzienne',
    #  'Bilans magazynu', 'Bilans punktów']
    # variant:
    # age
    # sex
    # balance
    df = pd.read_csv(constants.data_files['mz_api_age_v']['data_fn'])
    df['SZCZEPIENIA_SUMA'] = df['SZCZEPIENIA_SUMA'].replace('None', np.nan, regex=False).astype(float)
    df['SZCZEPIENIA_SUMA'] = df['SZCZEPIENIA_SUMA'].fillna(method='ffill')
    df.sort_values(by=['date'], inplace=True)
    if variant == 'Wiek narastająco':
        y_anno = 0.75
        tytul = 'Liczba podanych dawek według wieku - narastająco. Porównanie do progów populacyjnych'
        text = 'Liczba wykonanych szczepień: ' + str(list(df['SZCZEPIENIA_SUMA'])[-1])
        cols = {'SZCZEPIENIA0_17': '0-17 lat',
                'SZCZEPIENIA18_30': '18-30 lat',
                'SZCZEPIENIA31_40': '31-40 lat',
                'SZCZEPIENIA41_50': '41-50 lat',
                'SZCZEPIENIA51_60': '51-60 lat',
                'SZCZEPIENIA61_70': '61-70 lat',
                'SZCZEPIENIA71_75': '71-75 lat',
                'SZCZEPIENIA75_': '> 75 lat',
                'SZCZEPIENIA_WIEK_NIEUSTALONO': 'wiek nieustalony'}
    elif variant == 'Trendy':
        y_anno = 1.05
        tytul = 'Szczepienia w Polsce - dziennie (średnia krocząca 7-dniowa)'
        text = 'Liczby na dziś:<br> '
        df = df.iloc[:-1].copy()
        df['SZCZEPIENIA_SUMA_DIFF'] = df['SZCZEPIENIA_SUMA'].diff()
        df['SZCZEPIENIA_SUMA_DIFF'].fillna(0, inplace=True)
        df['SZCZEPIENIA_SUMA_DIFF'] = df['SZCZEPIENIA_SUMA_DIFF'].rolling(7, min_periods=1).mean()

        df['DAWKA_1_SUMA_DIFF'] = df['DAWKA_1_SUMA'].diff()
        df['DAWKA_1_SUMA_DIFF'].fillna(0, inplace=True)
        df['DAWKA_1_SUMA_DIFF'] = df['DAWKA_1_SUMA_DIFF'].rolling(7, min_periods=1).mean()

        df['DAWKA_2_SUMA_DIFF'] = df['DAWKA_2_SUMA'].diff()
        df['DAWKA_2_SUMA_DIFF'].fillna(0, inplace=True)
        df['DAWKA_2_SUMA_DIFF'] = df['DAWKA_2_SUMA_DIFF'].rolling(7, min_periods=1).mean()

        df['DAWKA_12_SUMA_DIFF'] = df['DAWKA_1_SUMA_DIFF'] + df['DAWKA_2_SUMA_DIFF']

        # cols = {'SZCZEPIENIA_SUMA_DIFF': 'Obie dawki',
        cols = {
            # 'SZCZEPIENIA_SUMA_DIFF': 'Wszystkie dawki',
            'DAWKA_12_SUMA_DIFF': 'Obie dawki podstawowe',
            'DAWKA_1_SUMA_DIFF': 'Dawka 1',
            'DAWKA_2_SUMA_DIFF': 'Dawka 2'}
        df[['date'] + list(cols.keys())].to_csv('/home/docent/violet_ace.csv')
    elif variant == 'Wiek przyrosty dzienne':
        y_anno = 1.05
        tytul = 'Liczba podanych dawek według wieku - przyrosty dzienne.'
        text = 'Liczba dotychczas wykonanych szczepień: ' + str(list(df['SZCZEPIENIA_SUMA'])[-1]) + \
            '<br><br>w tym w ciągu ostatniej doby:'
        df['_0_17_DIFF'] = df['SZCZEPIENIA0_17'].diff()
        df['_18_30_DIFF'] = df['SZCZEPIENIA18_30'].diff()
        df['_31_40_DIFF'] = df['SZCZEPIENIA31_40'].diff()
        df['_41_50_DIFF'] = df['SZCZEPIENIA41_50'].diff()
        df['_51_60_DIFF'] = df['SZCZEPIENIA51_60'].diff()
        df['_61_70_DIFF'] = df['SZCZEPIENIA61_70'].diff()
        df['_71_75_DIFF'] = df['SZCZEPIENIA71_75'].diff()
        df['_75_DIFF'] = df['SZCZEPIENIA75_'].diff()
        # df['_75_DIFF'] = df['SZCZEPIENIA75_'].rolling(7, min_periods=7).mean().diff()
        df['NIEUSTALONO_DIFF'] = df['SZCZEPIENIA_WIEK_NIEUSTALONO'].rolling(7, min_periods=7).mean().diff()
        cols = {'_0_17_DIFF': '0-17 lat',
                '_18_30_DIFF': '18-30 lat',
                '_31_40_DIFF': '31-40 lat',
                '_41_50_DIFF': '41-50 lat',
                '_51_60_DIFF': '51-60 lat',
                '_61_70_DIFF': '61-70 lat',
                '_71_75_DIFF': '71-75 lat',
                '_75_DIFF': '> 75 lat',
                'NIEUSTALONO_DIFF': 'wiek nieustalony'}
    elif variant == 'Płeć narastająco':
        y_anno = 0.75
        tytul = 'Liczba podanych dawek według płci - narastająco. Porównanie do progów populacyjnych'
        text = 'Liczba dotychczas wykonanych szczepień: ' + str(list(df['SZCZEPIENIA_SUMA'])[-1])
        cols = {'SZCZEPIENIA_PLEC_NIEUSTALONO': 'nie ustalono',
                'SZCZEPIENIA_KOBIETY': 'Kobiety',
                'SZCZEPIENIA_MEZCZYZNI': 'Mężczyźni'}
    elif variant == 'Płeć przyrosty dzienne':
        y_anno = 1.05
        tytul = 'Liczba podanych dawek według płci - przyrosty dzienne.'
        text = 'Liczba dotychczas wykonanych szczepień: ' + str(list(df['SZCZEPIENIA_SUMA'])[-1]) + \
            '<br><br>w tym w ciągu ostatniej doby:'
        df['PLEC_NIEUSTALONO_DIFF'] = df['SZCZEPIENIA_PLEC_NIEUSTALONO'].rolling(7, min_periods=7).mean().diff()
        df['KOBIETY_DIFF'] = df['SZCZEPIENIA_KOBIETY'].diff()
        df['MEZCZYZNI_DIFF'] = df['SZCZEPIENIA_MEZCZYZNI'].diff()
        cols = {'PLEC_NIEUSTALONO_DIFF': 'nie ustalono',
                'KOBIETY_DIFF': 'Kobiety',
                'MEZCZYZNI_DIFF': 'Mężczyźni'}
    elif variant == 'Bilans magazynu':
        y_anno = 0.75
        cols = {
            'STAN_MAGAZYN': 'Stan magazynu (składnica RARS)',
            'LICZBA_DAWEK_PUNKTY': 'Liczba dawek dostarczonych do punktów',
            'zamowienia_realizacja': 'Zamówienia w trakcie realizacji',
            'rozn_mag': 'Różnica magazynowa (do wyjaśnienia)'
        }
        tytul = 'Bilans szczepień - magazyn (składnica RARS)'
        text = 'Liczba dawek dostarczonych do Polski: ' + str(list(df['SUMA_DAWEK_POLSKA'])[-1])
        df['STAN_MAGAZYN'] = df['STAN_MAGAZYN'].replace('None', '0').astype(np.float64)
        df['LICZBA_DAWEK_PUNKTY'] = df['LICZBA_DAWEK_PUNKTY'].replace('None', '0').astype(np.float64)
        df['SUMA_DAWEK_POLSKA'] = df['SUMA_DAWEK_POLSKA'].replace('None', '0').astype(np.float64)
        df['zamowienia_realizacja'] = df['zamowienia_realizacja'].replace('None', '0').astype(np.float64)
        df['rozn_mag'] = df['SUMA_DAWEK_POLSKA'] - df['LICZBA_DAWEK_PUNKTY'] - df['STAN_MAGAZYN'] - \
                         df['zamowienia_realizacja']
    elif variant == 'Bilans punktów':
        y_anno = 0.75
        cols = {
            'DAWKA_1_SUMA': 'Liczba szczepień 1. dawką',
            'DAWKA_2_SUMA': 'Liczba szczepień 2. dawką',
            'DAWKI_UTRACONE': 'Dawki utracone',
            'rozn_punkty': 'Różnica między dostawami a zużyciem w punktach'
        }
        tytul = 'Bilans szczepień - punkty'
        text = 'Liczba dawek dostarczonych do punktów: ' + str(list(df['LICZBA_DAWEK_PUNKTY'])[-1])
        df['LICZBA_DAWEK_PUNKTY'] = df['LICZBA_DAWEK_PUNKTY'].replace('None', '0').astype(np.float64)
        df['DAWKA_1_SUMA'] = df['DAWKA_1_SUMA'].replace('None', '0').astype(np.float64)
        df['DAWKA_2_SUMA'] = df['DAWKA_2_SUMA'].replace('None', '0').astype(np.float64)
        df['DAWKI_UTRACONE'] = df['DAWKI_UTRACONE'].replace('None', '0').astype(np.float64)
        df['rozn_punkty'] = df['LICZBA_DAWEK_PUNKTY'] - df['DAWKA_1_SUMA'] - df['DAWKA_2_SUMA'] - df['DAWKI_UTRACONE']
    else:
        return 'Nieprawidłowa opcja: ' + str(variant)
    tytul += '<br><sub>Źródło: MZ, stan na ' + list(df['date'])[-1]
    traces = []
    if variant == 'Trendy':
        stackgroup = ''
    else:
        stackgroup = 'one'
    for col in cols.keys():
        x = list(df['date'])
        y = list(df[col])
        fig_data = go.Scatter(
            x=x,
            y=y,
            stackgroup=stackgroup,
            line=dict(width=2),
            name=cols[col]
        )
        traces.append(fig_data)
        text += '<br>' + cols[col] + ': ' + str(int(y[-1]))

    if variant in ['Wiek narastająco', 'Płeć narastająco', 'Bilans magazynu', 'Bilans punktów']:

        # populacja 31.12.2020

        pop = 38265013
        pop_80perc = int(0.8 * pop)
        pop_ge_18 = 31311374
        pop_ge_15 = 32386679
        pop_ge_12 = 33604057
        y = [pop_80perc for i in range(len(df['date']))]
        fig_data = go.Scatter(
            x=x,
            y=y,
            line=dict(dash='dash'),
            fill='tozeroy',
            fillcolor='rgba(144, 47, 168, 0.15)',
            name='80% populacji: ' + str(pop_80perc)
        )
        traces.append(fig_data)

        y = [pop_ge_18 for i in range(len(df['date']))]
        fig_data = go.Scatter(
            x=x,
            y=y,
            name='osoby w wieku >= 18 lat: ' + str(pop_ge_18)
        )
        traces.append(fig_data)

        y = [pop_ge_15 for i in range(len(df['date']))]
        fig_data = go.Scatter(
            x=x,
            y=y,
            name='osoby w wieku >= 15 lat: ' + str(pop_ge_15)
        )
        traces.append(fig_data)

        y = [pop_ge_12 for i in range(len(df['date']))]
        fig_data = go.Scatter(
            x=x,
            y=y,
            name='osoby w wieku >= 12 lat: ' + str(pop_ge_12)
        )
        traces.append(fig_data)

        y = [pop for i in range(len(df['date']))]
        fig_data = go.Scatter(
            x=x,
            y=y,
            name='cała populacja Polski: ' + str(pop)
        )
        traces.append(fig_data)

    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            autosize=True,
            # barmode=barmode,
            title=dict(text=tytul),
            margin=dict(pad=5),
            height=float(settings['plot_height']) * 660,
            legend=dict(x=1.03, y=1., orientation='v')
        )
    )
    # if variant != 'Trendy':
    figure.add_annotation(text=text,
                          xref="paper",
                          yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.05,
                          y=y_anno,
                          showarrow=False)
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
        'toImageButtonOptions': {
            'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
            'filename': 'docent_image', 'height': float(settings['plot_height']) * 660, 'scale': 1}
    }
    fig = dcc.Graph(id='xxxx'+variant, figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


###########################################################
#  Udział grup wiekowych w tygodniowych szczepieniach
###########################################################

def layout_more_mz_vacc2(_0, _1, _2, _3):
    settings = session2settings()
    template = get_template()
    ages = {'SZCZEPIENIA0_17': '0-17 lat',
            'SZCZEPIENIA18_30': '18-30 lat',
            'SZCZEPIENIA31_40': '31-40 lat',
            'SZCZEPIENIA41_50': '41-50 lat',
            'SZCZEPIENIA51_60': '51-60 lat',
            'SZCZEPIENIA61_70': '61-70 lat',
            'SZCZEPIENIA71_75': '71-75 lat',
            'SZCZEPIENIA75_': '> 75 lat',
            'SZCZEPIENIA_WIEK_NIEUSTALONO': 'wiek nieustalony'}
    df = pd.read_csv(constants.data_files['mz_api_age_v']['data_fn'])
    df = df[['SZCZEPIENIA_SUMA', 'SZCZEPIENIA_DZIENNIE', 'SZCZEPIENIA0_17', 'SZCZEPIENIA18_30',
             'SZCZEPIENIA31_40', 'SZCZEPIENIA41_50', 'SZCZEPIENIA51_60', 'SZCZEPIENIA61_70',
             'SZCZEPIENIA71_75', 'SZCZEPIENIA75_', 'SZCZEPIENIA_WIEK_NIEUSTALONO', 'date']].copy()
    df.sort_values(by=['date'], inplace=True)

    traces = []
    for age in ages.keys():
        df[age + '_diff'] = df[age].diff()
    df['SUMA_diff'] = df['SZCZEPIENIA0_17_diff'] + df['SZCZEPIENIA18_30_diff'] + df['SZCZEPIENIA31_40_diff'] + \
                      df['SZCZEPIENIA41_50_diff'] + df['SZCZEPIENIA51_60_diff'] + df['SZCZEPIENIA61_70_diff'] + \
                      df['SZCZEPIENIA71_75_diff'] + df['SZCZEPIENIA75__diff'] + df['SZCZEPIENIA_WIEK_NIEUSTALONO_diff']

    # resample week

    df.index = pd.to_datetime(df['date'])
    df = df.resample('W').sum()
    df['date'] = df.index

    for age in ages.keys():
        x = list(df['date'])
        y = list(df[age + '_diff'] / df['SUMA_diff'] * 100)
        fig_data = go.Bar(
            x=x,
            y=y,
            text=[round(x, 2) for x in y],
            textposition='auto',
            insidetextfont=dict(color='black'),
            outsidetextfont=dict(color='white'),
            name=ages[age]
        )
        traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            barmode='stack',
            title=dict(text='Udział procentowy grup wiekowych w szczepieniach (podane dawki - sumy tygodniowe)' +
                            '<br>(dane MZ z dnia ' + str(dt.today())[:10] + ')'),
            height=float(settings['plot_height']) * 660,
            paper_bgcolor=get_session_color(1),
            plot_bgcolor=get_session_color(1),
            legend=dict(x=1, y=1., orientation='v')
        )
    )
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


###########################################################
#  Udział grup wiekowych w dziennych szczepieniach
###########################################################

def layout_more_mz_vacc2a(_0, _1, _2, _3):
    template = get_template()
    settings = session2settings()
    ages = {'SZCZEPIENIA0_17': '0-17 lat',
            'SZCZEPIENIA18_30': '18-30 lat',
            'SZCZEPIENIA31_40': '31-40 lat',
            'SZCZEPIENIA41_50': '41-50 lat',
            'SZCZEPIENIA51_60': '51-60 lat',
            'SZCZEPIENIA61_70': '61-70 lat',
            'SZCZEPIENIA71_75': '71-75 lat',
            'SZCZEPIENIA75_': '> 75 lat',
            'SZCZEPIENIA_WIEK_NIEUSTALONO': 'wiek nieustalony'}
    df = pd.read_csv(constants.data_files['mz_api_age_v']['data_fn'])
    df = df[['SZCZEPIENIA_SUMA', 'SZCZEPIENIA_DZIENNIE', 'SZCZEPIENIA0_17', 'SZCZEPIENIA18_30',
             'SZCZEPIENIA31_40', 'SZCZEPIENIA41_50', 'SZCZEPIENIA51_60', 'SZCZEPIENIA61_70',
             'SZCZEPIENIA71_75', 'SZCZEPIENIA75_', 'SZCZEPIENIA_WIEK_NIEUSTALONO', 'date']].copy()
    df.sort_values(by=['date'], inplace=True)

    traces = []
    for age in ages.keys():
        df[age + '_diff'] = df[age].diff()
    df['SUMA_diff'] = df['SZCZEPIENIA0_17_diff'] + df['SZCZEPIENIA18_30_diff'] + df['SZCZEPIENIA31_40_diff'] + \
        df['SZCZEPIENIA41_50_diff'] + df['SZCZEPIENIA51_60_diff'] + df['SZCZEPIENIA61_70_diff'] + \
             df['SZCZEPIENIA71_75_diff'] + df['SZCZEPIENIA75__diff'] + df['SZCZEPIENIA_WIEK_NIEUSTALONO_diff']
    df = df.iloc[3:]
    for age in ages.keys():
        x = list(df['date'])
        y = list(df[age + '_diff'] / df['SUMA_diff'] * 100)
        fig_data = go.Bar(
            x=x,
            y=y,
            text=[round(x, 2) for x in y],
            textposition='auto',
            insidetextfont=dict(color='black'),
            outsidetextfont=dict(color='white'),
            name=ages[age]
        )
        traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            barmode='stack',
            title=dict(text='Udział procentowy grup wiekowych w szczepieniach (podane dawki - sumy dzienne)' +
                            '<br>(dane MZ z dnia ' + str(dt.today())[:10] + ')'),
            height=float(settings['plot_height']) * 660,
            legend=dict(x=1, y=1., orientation='v')
        )
    )
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


###########################################################
#  Dynamika szczepień w grupach wiekowych
###########################################################

def layout_more_mz_vacc2b(_0, _1, _2, _3):
    # session['template'] = 'white'
    template = get_template()
    settings = session2settings()
    from_date = settings['from_date']
    to_date = settings['to_date']
    ages = {'SZCZEPIENIA18_30': '18-30 lat',
            'SZCZEPIENIA31_40': '31-40 lat',
            'SZCZEPIENIA41_50': '41-50 lat',
            'SZCZEPIENIA51_60': '51-60 lat',
            'SZCZEPIENIA61_70': '61-70 lat',
            'SZCZEPIENIA71_75': '71-75 lat',
            'SZCZEPIENIA75_': '> 75 lat'}
    df = pd.read_csv(constants.data_files['mz_api_age_v']['data_fn'])
    df = df[['SZCZEPIENIA_SUMA', 'SZCZEPIENIA_DZIENNIE', 'SZCZEPIENIA0_17', 'SZCZEPIENIA18_30',
             'SZCZEPIENIA31_40', 'SZCZEPIENIA41_50', 'SZCZEPIENIA51_60', 'SZCZEPIENIA61_70',
             'SZCZEPIENIA71_75', 'SZCZEPIENIA75_', 'date']].copy()
    df = df[df['date'].between(from_date, to_date)].copy()
    df.sort_values(by=['date'], inplace=True)

    traces = []
    for age in ages.keys():
        df[age + '_diff'] = df[age].diff()
    df['SUMA_diff'] = df['SZCZEPIENIA18_30_diff'] + \
                      df['SZCZEPIENIA31_40_diff'] + \
                      df['SZCZEPIENIA41_50_diff'] + \
                      df['SZCZEPIENIA51_60_diff'] + \
                      df['SZCZEPIENIA61_70_diff'] + \
                      df['SZCZEPIENIA71_75_diff'] + \
                      df['SZCZEPIENIA75__diff']
    # df = df.iloc[3:]
    anno_text1 = 'Liczba dawek podanych od ' + from_date + ':<br>'
    anno_text2 = ' <br>'
    suma = 0
    for age in ages.keys():
        print(age)
        anno_text1 += '<br>' + ages[age]
        df[age + '_diff'].fillna(0, inplace=True)
        x = list(df['date'])
        y = list(df[age + '_diff'])
        suma0 = sum(y)
        anno_text2 += ' <br>' + str(int(suma0))
        suma += suma0
        fig_data = go.Scatter(
            x=x,
            y=y,
            text=[str(int(x)) for x in y],
            textposition='top left',
            textfont=dict(color=settings['color_7'], size=settings['font_size_legend']),
            mode='lines+text',
            line=dict(
                width=settings['linewidth_basic']),
            stackgroup='one',
            name=ages[age]
        )
        traces.append(fig_data)
    anno_text1 += '<br><br>RAZEM'
    anno_text2 += '<br><br>' + str(int(suma))
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            barmode='stack',
            title=dict(text='Dynamika szczepień w grupach wiekowych<br><sub>dzienne liczby podanych dawek ' +
                            '(dane MZ z dnia ' + str(dt.today())[:10] + ')'),
            height=float(settings['plot_height']) * 660,
            legend=dict(x=0, y=1.02, orientation='h')
        )
    )
    figure.add_annotation(text=anno_text1,
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.01, y=0.90,
                          showarrow=False)
    figure.add_annotation(text=anno_text2,
                          xref="paper", yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.07, y=0.90,
                          showarrow=False)
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


#########################################
#  Procent zaszczepienia w powiatach
#########################################

def layout_more_mz_vacc3(_0, _location, _order, _3):
    settings = session2settings()
    template = get_template()
    df = pd.read_csv(constants.data_files['mz_api_vacc_powiaty']['data_fn'], sep=',')

    def f(r):
        if r['powiat'][0].islower():
            r['typ'] = 'ziemski'
        elif r['powiat'] in constants.wojewodzkie_list:
            r['typ'] = 'wojewódzkie'
        else:
            r['typ'] = 'grodzki'
        return r
    df = df.apply(f, axis=1)

    loc_txt = ''
    if _location == '--miasta':
        df = df[df['typ'] == 'grodzki']
        loc_txt = 'miasta na prawach powiatu'
    elif _location == '--powiaty ziemskie':
        df = df[df['typ'] == 'ziemski']
        loc_txt = 'powiaty ziemskie'
    elif _location == '--miasta wojewódzkie':
        df = df[df['typ'] == 'wojewódzkie']
        loc_txt = 'miasta wojewódzkie'
    elif _location != '--wszystkie':
        df = df[df['wojew'] == _location]
        loc_txt = 'województwo ' + _location

    traces = []
    df['x1'] = (df['total_vacc'] - df['total_vacc_2']) / df['population'] * 100
    df['x2'] = df['total_vacc_2'] / df['population'] * 100
    df['sloty'] = df['slots_30'] / df['population'] * 100
    df['x12'] = df['total_vacc'] / df['population'] * 100

    if _order == 'malejąco':
        df.sort_values(by=['x12'], ascending=True, inplace=True)
    elif _order == 'rosnąco':
        df.sort_values(by=['x12'], ascending=False, inplace=True)
    else:
        df.sort_values(by=['powiat'], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    y = list(df['powiat'])
    x1 = list(df['x1'])
    x2 = list(df['x2'])
    x12 = list(df['x12'])
    sloty = list(df['sloty'])
    fig_data = go.Bar(
        y=y,
        x=x1,
        orientation='h',
        base=0,
        offsetgroup=0,
        text=[str(round(x, 2))+'%' for x in x1],
        textposition='auto',
        insidetextfont=dict(color='black', size=14),
        outsidetextfont=dict(color='white', size=14),
        name='I dawka'
    )
    traces.append(fig_data)
    fig_data = go.Bar(
        y=y,
        x=x2,
        orientation='h',
        offsetgroup=0,
        base=x1,
        text=[str(round(x, 2))+'%' for x in x12],
        textposition='outside',
        insidetextfont=dict(color='black', size=14),
        outsidetextfont=dict(color='white', size=14),
        name='II dawka'
    )
    traces.append(fig_data)
    fig_data = go.Bar(
        y=y,
        x=sloty,
        orientation='h',
        offsetgroup=0,
        # base=x2,
        text=[str(round(x, 2))+'%' for x in sloty],
        marker=dict(opacity=0.3, color='rgba(0,0,0,0)', line=dict(color='white', width=4)),
        textposition='outside',
        insidetextfont=dict(color='black', size=14),
        outsidetextfont=dict(color='white', size=14),
        name='wolne sloty 30 dni'
    )
    traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            xaxis=dict(side='top',
                       automargin=True),
            yaxis=dict(automargin=True),
            barmode='stack',
            margin=dict(l=settings['marginl'], r=settings['marginr'], b=settings['marginb'],
                        t=settings['margint'] + 100,
                        pad=4),
            title=dict(text='Proporcja (%) liczby wykonanych szczepień do liczby ludności w powiatach (dane MZ z ' +
                            str(dt.now())[:10] + ')' +
                            '<br>' + loc_txt),
            height=37*len(df),
            paper_bgcolor=get_session_color(1),
            colorway=['green', 'yellow', 'red'],
            plot_bgcolor=get_session_color(1),
            legend=dict(x=0.8, y=0.9,
                        orientation='h')
                        # font=dict(size=16, color='white')),
        )
    )
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
        'toImageButtonOptions': {
            'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
            'filename': 'docent_image', 'scale': 1, 'width': 1200, 'height': 150 + 45*len(df)},
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val

#########################################
#  Ranking wojewódzki szczepień
#########################################


def layout_more_mz_vacc4(_0, _typ, _1, _3):
    template = get_template()
    settings = session2settings()
    # ['wszystkie gminy', 'gminy miejskie', 'gminy miejsko-wiejskie', 'miasta', 'do 20 tys.', '20-50 tys.',
    #  '50-100 tys', '> 100 tys.']
    # ['powiat', '70+ lat', '12-19 lat', '20-39 lat', '40-59 lat', '60-69 lat', 'population',
    #  'total_full', 'percent_full', 'total_1d', 'nazwa', 'wojew']
    df = pd.read_csv(constants.data_files['mz_api_vacc_gminy']['data_fn'])

    if _typ == 'do 20 tys.':
        df = df[df['typ_p'] == '<20k'].copy()
    elif _typ == '20-50 tys.':
        df = df[df['typ_p'] == '20k-50k'].copy()
    elif _typ == '50-100 tys.':
        df = df[df['typ_p'] == '50k-100k'].copy()
    elif _typ == '> 100 tys.':
        df = df[df['typ_p'] == '>100k'].copy()

    df.sort_values(by=['wojew', 'nazwa'], inplace=True)

    df['total_w_1'] = df.groupby(['wojew'])['total_1d'].transform('sum')
    df['total_w_full'] = df.groupby(['wojew'])['total_full'].transform('sum')
    df['total_w_population'] = df.groupby(['wojew'])['population'].transform('sum')
    df = df.drop_duplicates(subset=['wojew']).copy()
    df['total_1_percent'] = df['total_w_1'] / df['total_w_population'] * 100
    df['total_full_percent'] = df['total_w_full'] / df['total_w_population'] * 100
    df.sort_values(by=['total_1_percent'], inplace=True, ascending=False)
    x = list(df['wojew'].unique())
    y_total_1 = df['total_1_percent']
    y_total_full = df['total_full_percent']
    y_brak = 100 - y_total_1
    y_total_1_t = round(y_total_1, 2)
    y_total_full_t = round(y_total_full, 2)
    y_brak_t = round(y_brak, 2)
    traces = []
    fig_data = go.Bar(
        x=x,
        y=[100] * len(x),
        base=0,
        offsetgroup=0,
        text=[str(x) for x in y_brak_t],
        textposition='auto',
        insidetextfont=dict(size=settings['font_size_anno'], color='black'),
        outsidetextfont=dict(color='green', size=10),
        name='niezaszczepieni'
    )
    traces.append(fig_data)
    fig_data = go.Bar(
        x=x,
        y=list(y_total_1),
        offsetgroup=0,
        base=0,
        text=[str(x) for x in list(y_total_1_t)],
        textposition='outside',
        insidetextfont=dict(size=settings['font_size_anno'], color='black'),
        outsidetextfont=dict(color='black', size=settings['font_size_anno']),
        name='co najmniej I dawka'
    )
    traces.append(fig_data)
    fig_data = go.Bar(
        x=x,
        y=y_total_full,
        base=0,
        offsetgroup=0,
        text=[str(x) for x in y_total_full_t],
        textposition='auto',
        insidetextfont=dict(size=settings['font_size_anno'], color='black'),
        outsidetextfont=dict(color='green', size=10),
        name='w pełni zaszczepieni'
    )
    traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            margin=dict(l=80, r=50, b=170, t=120, pad=2),
            title=dict(text='Ranking wojewódzki szczepień w grupie "' + _typ +
                            '" (dane MZ z dnia ' + str(dt.now())[:10] + ')' +
                            '<br><sub>w odniesieniu do całości populacji<br><br>'),
            height=float(settings['plot_height']) * 760,
            paper_bgcolor=get_session_color(1),
            colorway=['red', 'yellow', 'green'],
            plot_bgcolor=get_session_color(1),
            legend=dict(x=0.5, y=1.03, orientation='h', xanchor='center')
        )
    )
    figure = add_copyright(figure, settings, y=-0.3)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


################################################
#  Ranking powiatowy szczepień w województwach
################################################

def layout_more_mz_vacc5(_0, _wojew, _2, _3):
    template = get_template()
    settings = session2settings()
    # ['powiat', '70+ lat', '12-19 lat', '20-39 lat', '40-59 lat', '60-69 lat', 'population',
    #  'total_full', 'percent_full', 'total_1d', 'nazwa', 'wojew']
    df = pd.read_csv(constants.data_files['mz_api_vacc_gminy']['data_fn'])
    df = df[df['wojew'] == _wojew].copy()
    df.sort_values(by=['nazwa'], inplace=True)

    df['total_p_1'] = df.groupby(['powiat'])['total_1d'].transform('sum')
    df['total_p_full'] = df.groupby(['powiat'])['total_full'].transform('sum')
    df['total_p_population'] = df.groupby(['powiat'])['population'].transform('sum')
    df = df.drop_duplicates(subset=['powiat']).copy()
    df['total_1_percent'] = df['total_p_1'] / df['total_p_population'] * 100
    df['total_full_percent'] = df['total_p_full'] / df['total_p_population'] * 100
    df.sort_values(by=['total_1_percent'], inplace=True, ascending=False)
    x = list(df['powiat'].unique())
    y_total_1 = df['total_1_percent']
    y_total_full = df['total_full_percent']
    y_brak = 100 - y_total_1
    y_total_1_t = round(y_total_1, 2)
    y_total_full_t = round(y_total_full, 2)
    y_brak_t = round(y_brak, 2)
    traces = []
    fig_data = go.Bar(
        x=x,
        y=[100] * len(x),
        base=0,
        offsetgroup=0,
        text=[str(x) for x in y_brak_t],
        textposition='auto',
        insidetextfont=dict(size=settings['font_size_anno'], color='black'),
        # outsidetextfont=dict(color='green', size=10),
        name='niezaszczepieni'
    )
    traces.append(fig_data)
    fig_data = go.Bar(
        x=x,
        y=list(y_total_1),
        offsetgroup=0,
        base=0,
        text=[str(x) for x in list(y_total_1_t)],
        textposition='outside',
        insidetextfont=dict(size=settings['font_size_anno'], color='black'),
        outsidetextfont=dict(color='black', size=settings['font_size_anno']),
        name='co najmniej I dawka'
    )
    traces.append(fig_data)
    fig_data = go.Bar(
        x=x,
        y=y_total_full,
        base=0,
        offsetgroup=0,
        text=[str(x) for x in y_total_full_t],
        textposition='auto',
        insidetextfont=dict(size=settings['font_size_anno'], color='black'),
        # outsidetextfont=dict(color='green', size=12),
        name='w pełni zaszczepieni'
    )
    traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            # barmode='stack',
            margin=dict(l=80, r=50, b=170, t=120, pad=2),
            title=dict(text='Ranking powiatowy szczepień, województwo ' + _wojew +
                            ' (dane MZ z dnia ' + str(dt.now())[:10] + ')' + \
                            '<br><sub>w odniesieniu do całości populacji<br><br>'),
            height=float(settings['plot_height']) * 760,
            paper_bgcolor=get_session_color(1),
            colorway=['red', 'yellow', 'green'],
            plot_bgcolor=get_session_color(1),
            legend=dict(x=0.5, y=1.03, orientation='h', xanchor='center')
        )
    )
    figure = add_copyright(figure, settings, y=-0.3)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


###############################################
#  Bilans zaszczepienia populacji Polski MZ
###############################################

def layout_more_mz_vacc6(_0, _1, _2, _3):
    template = get_template()
    settings = session2settings()

    # ['powiat', '70+ lat', '12-19 lat', '20-39 lat', '40-59 lat', '60-69 lat', 'population',
    #  'total_full', 'percent_full', 'total_1d', 'nazwa', 'wojew']

    # populacja 31.12.2020

    total_population = 38265013
    total_uprawnieni = 33604057
    ages = {
           '12-19 lat': dict(name='12-19 lat', population=3011688, population_u=3011688),
           '20-39 lat': dict(name='20-39 lat', population=10418299, population_u=10418299),
           '40-59 lat': dict(name='40-59 lat', population=10373837, population_u=10373837),
           '60-69 lat': dict(name='60-69 lat', population=5185843, population_u=5185843),
           '70+ lat': dict(name='70+ lat', population=4614390, population_u=4614390),
    }
    df = pd.read_csv(constants.data_files['mz_api_vacc_gminy']['data_fn'])
    pola = ['12-19 lat', '20-39 lat', '40-59 lat', '60-69 lat', '70+ lat']
    df = df[pola + ['total_1d', 'total_full']].copy()
    total_1d = df['total_1d'].sum()
    total_full = df['total_full'].sum()
    sums = []
    for pole in pola:
        sums.append(dict(wiek=ages[pole]['name'],
                         suma=df[pole].sum(),
                         populacja=ages[pole]['population'],
                         populacja_u=ages[pole]['population_u'],
                         )
                    )
    df = pd.DataFrame(sums)
    df['procent'] = df['suma'] / df['populacja']

    text_w = 'Wiek<br><br>12-19<br>20-39<br>40-59<br>60-69<br>70+'
    text1 = 'Populacja<br><br>'
    text2 = 'I dawka<br><br>'
    text3 = '%<br><br>'

    # '{:,}'.format(1234567890.001).replace(',', ' ')

    for index, row in df.iterrows():
        text1 += fn(row['populacja']) + '<br>'
        text2 += fn(row['suma']) + '<br>'
        text3 += fn(row['procent'] * 100, 2) + '%<br>'

    text_w += '<br><br>Osoby zaszczepione 1. dawką: ' + fn(total_1d)
    text_w += '<br>- procent populacji (' + fn(total_population) + '): ' + \
              fn(total_1d / total_population * 100) + '%'
    text_w += '<br>- procent osób kwalifikujących się (' + fn(total_uprawnieni) + '): ' + \
              fn(total_1d / total_uprawnieni * 100) + '%'
    text_w += '<br><br>Osoby zaszczepione w pełni: ' + fn(total_full)
    text_w += '<br>- procent populacji (' + fn(total_population) + '): ' + \
              fn(total_full / total_population * 100) + '%'
    text_w += '<br>- procent osób kwalifikujących się (' + fn(total_uprawnieni) + '): ' + \
              fn(total_full / total_uprawnieni * 100) + '%'
    traces = []

    x = list(df['wiek'])
    y_populacja = list(df['populacja'])
    fig_data = go.Bar(
        x=x,
        y=y_populacja,
        marker=dict(line=dict(width=1, color='black')),
        text=['populacja<br><b>'+fn(x) for x in y_populacja],
        textposition='inside',
        base=0,
        offsetgroup=0,
        insidetextfont=dict(size=14),
        name='populacja grupy wiekowej'
    )
    traces.append(fig_data)

    y_total_1 = list(df['suma'])
    fig_data = go.Bar(
        x=x,
        y=y_total_1,
        marker=dict(line=dict(width=1, color='black')),
        text=['1. dawka<br><b>'+fn(x) for x in y_total_1],
        textposition='inside',
        base=0,
        offsetgroup=0,
        insidetextfont=dict(size=14),
        name='zaszczepieni co najmniej 1. dawką'
    )
    traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            # barmode='stack',
            title=dict(text='Bilans populacyjny szczepień w grupach wiekowych (dane MZ z dnia ' +
                            str(dt.now())[:10] + ')<br>'),
            height=float(settings['plot_height']) * 760,
            paper_bgcolor=get_session_color(1),
            plot_bgcolor=get_session_color(1),
            legend=dict(x=0., y=1.03, orientation='h')
        )
    )
    figure.add_annotation(text=text_w,
                          xref="paper",
                          yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.90,
                          y=1.05,
                          showarrow=False)
    figure.add_annotation(text=text1,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.75,
                          y=1.05,
                          showarrow=False)
    figure.add_annotation(text=text2,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.85,
                          y=1.05,
                          showarrow=False)
    figure.add_annotation(text=text3,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.95,
                          y=1.05,
                          showarrow=False)
    figure = add_copyright(figure, settings, y=-0.1)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


################################################
#  Punkty szczepień
################################################

def layout_more_mz_psz(_0, _wojew, _typ, _3):
    settings = session2settings()

    # ['id', 'ordinalNumber', 'facilityName', 'terc', 'address', 'zipCode',
    #  'voivodeship', 'county', 'community', 'place', 'lon', 'lat',
    #  'facilityType']

    df = pd.read_csv(constants.data_files['mz_psz']['data_fn'])
    df = df[df['facilityType'] == 4].copy()

    mapbox_access_token = open("data/mapbox.token").read()
    mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=19.18, lat=52.11), zoom=5.5)
    mapbox['style'] = constants.mapbox_styles[settings['map_mapbox']]

    figure = go.Figure()
    figure.add_trace(go.Scattermapbox(
        lat=df['lat'],
        lon=df['lon'],
        text='<b>'+df['facilityName']+'</b><br>' + df['zipCode']+' '+df['address'] + '<br>' +
             df['place'] + '<br>województwo ' + df['wojew'] + '<br>powiat ' + df['powiat'],
        mode='markers',
        marker=go.scattermapbox.Marker(size=9),
    ))

    figure.update_layout(
        autosize=True,
        hovermode='closest',
        title=dict(
            text='Mapa punktów szczepień',
            x=settings['titlexpos'], y=settings['titleypos'] - 0.04,
            font=dict(size=settings['font_size_title'], color=get_session_color(7))
        ),
        height=int(settings['map_height']),
        margin=dict(l=settings['marginl'], r=settings['marginr'], b=settings['marginb'], t=settings['margint'],
                    pad=0),
        mapbox=mapbox,
    )
    figure = add_copyright(figure, settings)
    config = {
        'displaylogo': False,
        'responsive': True,
        'toImageButtonOptions': {
            'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
            'filename': 'docent_image', 'height': 700, 'width': 1200}
    }
    link = create_html(figure, 'punkty_szczepien')
    ret_val = dbc.Col(dcc.Graph(id='mapa', figure=figure, config=config), className='mb-2 mt-2')

    return ret_val


################################################
#  Mapa powiatowych wskaźników szczepień
################################################

def layout_more_mz_psz2(_0, _typ, _field, _3):
    settings = session2settings()

    # ['id', 'ordinalNumber', 'facilityName', 'terc', 'address', 'zipCode',
    #  'voivodeship', 'county', 'community', 'place', 'lon', 'lat',
    #  'facilityType']

    # ['percent full', '12-19 lat', '20-39 lat', '40-59 lat', '60-69 lat', '70+ lat',
    #  'gmina', 'powiat', 'grupa_wiekowa', 'population', 'Województwo',
    #  'total_1d', 'total_full', 'typ_a', 'typ_p', 'nazwa', 'wojew']
    if _field == '70plus lat':
        _field = '70+ lat'

    # udział grupy wiekowej w populacji
    field_dict = {'20-39 lat': ['I dawka, wiek 20-39 lat', '_20_39'],
                  '40-59 lat': ['I dawka, wiek 40-59 lat', '_40_59'],
                  '60-69 lat': ['I dawka, wiek 60-69 lat', '_60_69'],
                  '70+ lat':   ['I dawka, wiek 70+ lat', '_70+'],
                  'total_1d':  ['I dawka, wszystkie osoby', 1.],
                  'total_full': ['w pełni zaszczepieni, wszystkie osoby', 1.]
    }

    # punkty szczepień

    df_psz = pd.read_csv(constants.data_files['mz_psz']['data_fn'])
    df_psz['ilep'] = df_psz.groupby(['powiat', 'wojew'])['id'].transform('count')
    df_psz = df_psz.drop_duplicates(subset=['powiat', 'wojew']).copy()
    df_psz = df_psz[['powiat', 'wojew', 'ilep']].copy()

    postfix = ''
    ticksuffix = ''
    if 'szczepienia' in _typ:

        postfix = '_'+_field

        # szczepienia w powiatach

        df_gm = pd.read_csv(constants.data_files['mz_api_vacc_gminy']['data_fn'])
        df_gm.sort_values(by=['powiat', 'wojew'], inplace=True)
        df_gm['field_p'] = df_gm.groupby(['powiat', 'wojew'])[_field].transform('sum')
        df_gm['population_p'] = df_gm.groupby(['powiat', 'wojew'])['population'].transform('sum')
        df_gm = df_gm.drop_duplicates(subset=['powiat', 'wojew']).copy()
        df_gm = pd.merge(df_gm, df_psz, how='left', left_on=['powiat', 'wojew'], right_on=['powiat', 'wojew'])
        if 'na 100k' in _typ:
            df_gm['ile'] = round(df_gm['field_p'] / df_gm['population_p'] * 100000, 1)
            tytul = 'Liczba osób zaszczepionych w powiatach na 100 000 mieszkańców: ' + field_dict[_field][0]
        if '% ' in _typ:
            if _field in ['total_1d', 'total_full']:
                df_gm['ile'] = round(df_gm['field_p'] / df_gm['population_p'] * 100, 1)
            else:
                df_bdl = pd.read_csv('data/dict/bdl_last.csv')
                df_gm = pd.merge(df_gm, df_bdl, how='left', left_on=['powiat', 'wojew'], right_on=['nazwa', 'wojew'])
                df_gm['dzielnik'] = df_gm[field_dict[_field][1]]
                df_gm['ile'] = round(df_gm['field_p'] / df_gm['dzielnik'] * 100, 1)
                df_gm['population_p'] = df_gm['dzielnik']
            ticksuffix = '%'
            tytul = 'Procent zaszczepienia grupy wiekowej w powiatach: ' + field_dict[_field][0]
        elif 'na punkt' in _typ:
            tytul = 'Liczba osób zaszczepionych w powiatach na punkt: ' + field_dict[_field][0]
            df_gm['ile'] = round(df_gm['field_p'] / df_gm['ilep'], 1)
        elif 'suma' in _typ:
            df_gm['ile'] = round(df_gm['field_p'], 0)
            tytul = 'Liczba osób zaszczepionych I dawką w powiatach: ' + field_dict[_field][0]
        hover_text = '<b>' + df_gm['powiat'] + '</b><br>populacja: ' + df_gm['population_p'].astype(str) + \
                     '<br>liczba punktów: ' + df_gm['ilep'].astype(str) + '<br>liczba szczepień (' + \
                     field_dict[_field][0] + '): ' + df_gm['field_p'].astype(str) + \
                     '<br>wartość wskaźnika: ' + df_gm['ile'].astype(str)
    elif 'sloty' in _typ:

        # sloty_30

        df_gm = pd.read_csv(constants.data_files['mz_api_vacc_powiaty']['data_fn'])
        df_gm = pd.merge(df_gm, df_psz,  how='left', left_on=['powiat', 'wojew'], right_on=['powiat', 'wojew']).copy()
        if 'na punkt' in _typ:
            tytul = 'Liczba zaplanowanych szczepień (30 dni) w powiatach na jeden punkt'
            df_gm['ile'] = round(df_gm['slots_30'] / df_gm['ilep'], 1)
        elif 'na 100k' in _typ:
            tytul = 'Liczba zaplanowanych szczepień (30 dni) w powiatach na 100 000 mieszkańców'
            df_gm['ile'] = round(df_gm['slots_30'] / df_gm['population'] * 100000, 1)
        elif 'suma' in _typ:
            tytul = 'Liczba zaplanowanych szczepień (30 dni) w powiatach'
            df_gm['ile'] = round(df_gm['slots_30'], 0)
        hover_text = '<b>' + df_gm['powiat']+'</b><br>populacja: ' + df_gm['population'].astype(str) + \
                     '<br>liczba punktów: ' + \
                     df_gm['ilep'].astype(str)+'<br>liczba slotów: ' + df_gm['slots_30'].astype(str) + \
                     '<br>wartość wskaźnika: ' + df_gm['ile'].astype(str)

    df_rank = df_gm.copy()
    if '% ' in _typ:
        suff = '%'
    else:
        suff = ''
    df_rank.sort_values(by=['ile'], ascending=True, inplace=True)
    anno_text = '<b>Powiaty z najniższym wskaźnikiem:</b>'
    for index, row in df_rank.head(10).iterrows():
        anno_text += '<br>' + row['powiat'] + ': ' + str(row['ile']) + suff
    anno_text += '<br><br><b>Powiaty z najwyższym wskaźnikiem:</b>'
    df_rank.sort_values(by=['ile'], ascending=False, inplace=True)
    for index, row in df_rank.head(11).iterrows():
        if row['powiat'] != 'karkonoski':
            anno_text += '<br>' + row['powiat'] + ': ' + str(row['ile']) + suff

    mapbox_access_token = open("data/mapbox.token").read()
    mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=19.18, lat=52.11), zoom=5.5)
    mapbox['style'] = constants.mapbox_styles[settings['map_mapbox']]

    json_file = r'data/geojson/powiaty-min-ws.geojson'
    locations = df_gm['powiat']
    featuredikey = 'properties.nazwa'
    with open(json_file, encoding="utf-8") as f:
        geojson = json.load(f)

    figure = go.Figure()
    figure.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        featureidkey=featuredikey,
        locations=locations,
        z=df_gm['ile'],
        text=hover_text,
        marker=dict(line=dict(color='black', width=0.3 if 'drawborder' in settings['map_options'] else 0.)),
        hoverinfo='text',
        colorscale=settings['map_color_scale'],
        reversescale=False if 'reversescale' in settings['map_options'] else True,
        marker_opacity=settings['map_opacity'],
        colorbar=dict(
            bgcolor='white',
            # bgcolor=get_session_color(1),
            title=dict(text='wskaźnik'),
            ticksuffix=ticksuffix,
        ),
    ))
    figure.add_annotation(text=str(dt.now())[:10],
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'], size=24),
                          x=0.01, y=1,
                          showarrow=False)
    figure.add_annotation(text=anno_text,
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.01, y=0.85,
                          showarrow=False)

    figure.update_layout(
        # images=constants.image_logo_map,
        autosize=True,
        hovermode='closest',
        title=dict(
            text='<b>' + tytul + '</b>',
            x=settings['titlexpos'], y=settings['titleypos'] - 0.04,
            font=dict(size=settings['font_size_title'], color=get_session_color(7))
        ),
        height=int(settings['map_height']),
        margin=dict(l=settings['marginl'], r=settings['marginr'], b=settings['marginb'], t=settings['margint'],
                    pad=0),
        paper_bgcolor=get_session_color(1),
        plot_bgcolor=get_session_color(1),
        # mapbox=None,
        mapbox=mapbox,
    )
    figure = add_copyright(figure, settings)
    config = {
        'displaylogo': False,
        'responsive': True,
        'toImageButtonOptions': {
            'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
            'filename': 'docent_image', 'height': 700, 'width': 1200}
    }
    # create_html(figure, 'mz_psz2_'+_typ+postfix)
    ret_val = dbc.Col(dcc.Graph(id='mapa', figure=figure, config=config), className='mb-2 mt-2')

    return ret_val


################################################
#  Mapa wskaźników szczepień w aptekach
################################################

def layout_more_mz_psz5(DF, _typ, _2, _3):
    settings = session2settings()
    scope = settings['scope']
    locations = settings['locations']
    from_date = settings['from_date']
    to_date = settings['to_date'][:10]

    # punkty szczepień

    df_psz = pd.read_csv(constants.data_files['mz_psz']['data_fn'])
    df_psz = df_psz[df_psz['facilityType'] == 4].copy()

    # powiaty

    df_i = DF['cities'][['location', 'wojew', 'Lat', 'Long', 'population']].copy()
    df_i.drop_duplicates(subset=['location', 'wojew'], inplace=True)

    # szczepienia w aptekach

    df = pd.read_csv('data/sources/woj_pow_szczep_apteki.csv', sep=',')
    df.columns = ['date', 'teryt', 'woj', 'pow', 'count']
    df['teryt'] = df['teryt'].astype(int)
    # df['woj'] = df['woj'].str.replace("'", "")
    # df['pow'] = df['pow'].str.replace("'", "")
    # df['wojpow'] = (df['woj'] + df['pow']).astype(int)
    # df.columns = ['woj', 'pow', 'date', 'count', 'wojpow']
    # df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y').astype(str)
    df = df[df['date'].between(from_date, to_date)].copy()

    df_teryt = pd.read_csv('data/dict/teryt.csv')
    df_teryt = df_teryt[df_teryt['NAZWA_DOD'].isin(['powiat', 'miasto na prawach powiatu',
                                                    'miasto stołeczne, na prawach powiatu'])].copy()
    df_teryt['teryt_woj'] = df_teryt['teryt_woj'].astype(int)
    df_teryt['teryt_pow'] = df_teryt['teryt_pow'].astype(int)
    df_teryt = df_teryt[['teryt_pow', 'powiat', 'wojew']]
    df = pd.merge(df, df_teryt, how='left', left_on=['teryt'], right_on=['teryt_pow']).copy()
    for p in constants.powiat_translation_table:
        df.loc[(df.powiat == p['location']) & (df.wojew == p['wojew']), 'powiat'] = p['new_location']
    df['ile_sz'] = df.groupby(['powiat'])['count'].transform('sum')
    df.drop_duplicates(subset=['powiat'], inplace=True)

    df = pd.merge(df, df_i, how='left', left_on=['wojew', 'powiat'], right_on=['wojew', 'location']).copy()

    if _typ == 'suma':
        df['ile'] = df['ile_sz']
    else:
        df['ile'] = df['ile_sz'] / df['population'] * 100000
    if scope == 'cities' and len(locations) > 0:
        df = df[df['powiat'].isin(locations)].copy()


    df_rank = df.copy()
    df_rank.sort_values(by=['ile'], ascending=True, inplace=True)
    anno_text = '<b>Powiaty z najwyższym wskaźnikiem:</b>'
    df_rank.sort_values(by=['ile'], ascending=False, inplace=True)
    for index, row in df_rank.head(25).iterrows():
        if row['powiat'] != 'karkonoski':
            anno_text += '<br>' + row['powiat'] + ': ' + str(int(row['ile']))

    df['ile'] = df['ile'].replace(0, np.nan)

    hover_text = '<b>' + df['powiat'] + '</b><br>populacja: ' + df['population'].astype(str) + \
                 '<br>liczba szczepień: ' + df['ile_sz'].astype(str) + '<br>' + \
                 '<br>wartość wskaźnika: ' + round(df['ile'], 1).astype(str)

    mapbox_access_token = open("data/mapbox.token").read()
    mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=19.18, lat=52.11), zoom=5.5)
    mapbox['style'] = constants.mapbox_styles[settings['map_mapbox']]

    json_file = r'data/geojson/powiaty-min-ws.geojson'
    locations = df['powiat']
    featuredikey = 'properties.nazwa'
    with open(json_file, encoding="utf-8") as f:
        geojson = json.load(f)

    tytul = 'Liczba zaszczepionych osób (' + _typ + ') w powiatach w aptecznych punktach szczepień' + \
        '<br><sub> od ' + from_date + ' do ' + to_date

    figure = go.Figure()
    figure.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        featureidkey=featuredikey,
        locations=locations,
        z=df['ile'],
        text=hover_text,
        marker=dict(line=dict(color='black', width=0.3 if 'drawborder' in settings['map_options'] else 0.)),
        hoverinfo='text',
        colorscale=settings['map_color_scale'],
        reversescale=False if 'reversescale' in settings['map_options'] else True,
        marker_opacity=settings['map_opacity'],
        colorbar=dict(
            bgcolor='white',
            # bgcolor=get_session_color(1),
            title=dict(text='wskaźnik'),
        ),
    ))
    figure.add_trace(go.Scattermapbox(
        lat=df_psz['lat'],
        lon=df_psz['lon'],
        text=df_psz['zipCode']+' '+df_psz['address'] + '<br>' +
             df_psz['place'] + '<br>województwo ' + df_psz['wojew'] + '<br>powiat ' + df_psz['powiat'],
        mode='markers',
        marker=go.scattermapbox.Marker(size=5),
    ))
    figure.add_annotation(text=str(dt.now())[:10],
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'], size=24),
                          x=0.01, y=1,
                          showarrow=False)
    figure.add_annotation(text=anno_text,
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.01, y=0.85,
                          showarrow=False)
    figure.update_layout(
        # images=constants.image_logo_map,
        autosize=True,
        hovermode='closest',
        title=dict(
            text='<b>' + tytul + '</b>',
            x=settings['titlexpos'], y=settings['titleypos'] - 0.04,
            font=dict(size=settings['font_size_title'], color=get_session_color(7))
        ),
        height=int(settings['map_height']),
        margin=dict(l=settings['marginl'], r=settings['marginr'], b=settings['marginb'], t=settings['margint'],
                    pad=0),
        paper_bgcolor=get_session_color(1),
        plot_bgcolor=get_session_color(1),
        # mapbox=None,
        mapbox=mapbox,
    )
    figure = add_copyright(figure, settings)
    config = {
        'displaylogo': False,
        'responsive': True,
        'toImageButtonOptions': {
            'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
            'filename': 'docent_image', 'height': 700, 'width': 1200}
    }
    create_html(figure, 'mz_psz5_'+_typ)
    ret_val = dbc.Col(dcc.Graph(id='mapa', figure=figure, config=config), className='mb-2 mt-2')

    return ret_val


################################################
#  Mapa gmin - techniczna
################################################

def layout_more_mz_psz2t(_0, _wojew, _2, _3):
    session['template'] = 'white'
    session['template_change'] = True
    settings = session2settings()
    session['stare_wojew'] = _wojew

    # przypisanie gmin do starych województw

    df_stw = pd.read_csv('log/stare_woj.csv', header=None)
    df_stw.columns = ['date', 'st_wojew', 'JPT', 'nazwa']
    df_stw['JPT'] = df_stw['JPT'].str.slice(4, 20)
    df_stw = df_stw.drop_duplicates(subset=['JPT'], keep='last').copy()
    df_stw.to_csv('data/dict/gminy_stare_woj.csv', index=False)

    # punkty szczepień

    df_gm = pd.read_csv(constants.data_files['mz_api_vacc_gminy']['data_fn'])
    df_gm['JPT'] = df_gm['JPT'].str.slice(4, 20)
    df_gm = pd.merge(df_gm, df_stw[['JPT', 'st_wojew']], how='left', left_on=['JPT'], right_on=['JPT'])
    df_gm = df_gm.drop_duplicates(subset=['JPT']).copy()
    df_gm['st_wojew'].fillna('brak', inplace=True)
    stare_woj = ['brak'] + constants.stare_woj
    stare_woj_colors = ['white'] + ['rgb(' +
                                    str(randint(0, 255)) + ',' +
                                    str(randint(0, 255)) + ',' +
                                    str(randint(0, 255)) + ')' for x in range(49)]
    df_gm.sort_values(by=['JPT'], inplace=True)
    df_gm['kolor'] = df_gm['st_wojew'].apply(lambda x: stare_woj.index(x))

    hover_text = '<b>'+df_gm['nazwa'] + '<br>"stare" województwo: ' + df_gm['st_wojew'] + \
        '<br>powiat: ' + df_gm['powiat'] + \
                 '<br>liczba mieszkańców: ' + df_gm['population'].astype(str) + \
                 '<br>liczba w pełni zaszczepionych: ' + df_gm['total_full'].astype(str) + \
                 '<br>pełne zaszczepienie: ' + df_gm['percent full'].astype(str) + '%'

    df_gm['st_wojew'] = df_gm['st_wojew'].str.replace('(miejskie) ', '', regex=False)
    df_gm['st_wojew'] = df_gm['st_wojew'].str.replace('(stołeczne) ', '', regex=False)

    # czołówka gmin

    df_top = df_gm.sort_values(by=['percent full'], ascending=False).groupby('st_wojew').head(1)
    df_top = df_top.sort_values(by=['st_wojew', 'percent full'], ascending=True)
    df_top['rank'] = 1
    df_gm = pd.merge(df_gm, df_top[['JPT', 'rank']], how='left', left_on=['JPT'], right_on=['JPT'])
    df_gm['rank'].fillna(0.3, inplace=True)

    anno_text1 = '<b>Najlepsze gminy w "starych" województwach:</b><br>'
    anno_text2 = '&#8205;<br>'
    for index, row in df_top[:25].iterrows():
        if float(row['percent full'].replace(',', '.')) >= 50.:
            sp = '<span style="color: red"><b>' + str(row['percent full']) + '%</b></span>'
        else:
            sp = '<span style="color: blue">' + str(row['percent full']) + '%</span>'

        anno_text1 += '<br>' + row['st_wojew'] + '-<b>' + row['nazwa'] + '</b>: ' + sp
    for index, row in df_top[25:].iterrows():
        if float(row['percent full'].replace(',', '.')) >= 50.:
            sp = '<span style="color: red"><b>' + str(row['percent full']) + '%</b></span>'
        else:
            sp = '<span style="color: blue">' + str(row['percent full']) + '%</span>'
        anno_text2 += '<br>' + row['st_wojew'] + '-<b>' + row['nazwa'] + '</b>: ' + sp

    mapbox_access_token = open("data/mapbox.token").read()
    mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=14.7, lat=52.11), zoom=5.7)
    mapbox['style'] = constants.mapbox_styles['Szary - bez etykiet']

    json_file = r'data/geojson/Gminy_5_WS.json'
    locations = df_gm['JPT']
    featuredikey = 'properties.JPT'
    with open(json_file, encoding="utf-8") as f:
        geojson = json.load(f)

    figure = go.Figure()
    figure.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        featureidkey=featuredikey,
        locations=df_top['JPT'],
        z=[0 for k in range(len(df_top))],
        text=hover_text,
        marker=dict(line=dict(color='black', width=2), opacity=1),
        colorscale=list(stare_woj_colors),
        showscale=False,
    ))
    figure.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        featureidkey=featuredikey,
        locations=locations,
        z=df_gm['kolor'],
        text=hover_text,
        marker=dict(line=dict(color='black', width=0.2), opacity=0.5),
        hoverinfo='text',
        colorscale=list(stare_woj_colors),
        showscale=False,
        colorbar=dict(
            tickmode='array',
            ticktext=constants.stare_woj,
            tickvals=list(range(49)),
            bgcolor=get_session_color(1),
            title=dict(text='województwa'),
            ticksuffix='',
        ),
    ))

    figure.add_annotation(text=str(dt.now())[:10] + '<br><sup>Odsetek mieszkańców gmin w pełni zaszczepionych',
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'], size=36),
                          x=0.01, y=1,
                          showarrow=False)
    figure.add_annotation(text='Twitter: @docent_ws #TAN<br><sub>dane: Ministerstwo Zdrowia, GUS, internet',
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'], size=20),
                          x=0.03, y=0.023,
                          showarrow=False)
    figure.add_annotation(text=anno_text1,
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'], size=17),
                          x=0.01, y=0.86,
                          showarrow=False)
    figure.add_annotation(text=anno_text2,
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'], size=17),
                          x=0.27, y=0.86,
                          showarrow=False)
    figure.update_layout(
        # images=constants.image_logo_map,
        # images=constants.image_stare_woj,
        autosize=True,
        hovermode='closest',
        height=830,
        margin=dict(l=settings['marginl'], r=settings['marginr'], b=settings['marginb'], t=settings['margint'],
                    pad=0),
        mapbox=mapbox,
    )
    figure = add_copyright(figure, settings)
    config = {
        'displaylogo': False,
        'responsive': True,
        'toImageButtonOptions': {
            'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
            'filename': 'docent_image', 'height': 700, 'width': 1200}
    }
    create_html(figure, 'ranking_gmin')
    ret_val = dbc.Col(dcc.Graph(id='mapa', figure=figure, config=config), className='mb-2 mt-2')

    return ret_val


################################################
#  Mapa gminnych wskaźników szczepień
################################################

def layout_more_mz_psz3(_DF, _field, _2, _3):
    settings = session2settings()
    if _field == '70plus lat':
        _field = '70+ lat'
    field_dict = {'12-19 lat': ['I dawka, wiek 12-19 lat', 0.778],
                  '20-39 lat': ['I dawka, wiek 20-39 lat', 0.2752],
                  '40-59 lat': ['I dawka, wiek 40-59 lat', 0.2696],
                  '60-69 lat': ['I dawka, wiek 60-69 lat', 0.1358],
                  '70+ lat':   ['I dawka, wiek 70+ lat', 0.1192],
                  'total_1d':  ['I dawka, wszystkie osoby', 1],
                  'total_full': ['w pełni zaszczepieni', 1]}

    # 12 - 19    20 - 39    40 - 59    60 - 69    70
    # 2984051    10553652   10340096   5206931    4571373
    # 7,78%      27,52%     26,96%     13,58% 11, 92%

    # szczepienia w gminach

    df_gm = pd.read_csv(constants.data_files['mz_api_vacc_gminy']['data_fn'])
    df_gm.sort_values(by=['nazwa', 'powiat', 'wojew'], inplace=True)
    df_gm.dropna(axis='rows', subset=['JPT'], inplace=True)
    df_gm['JPT'] = df_gm['JPT'].str.slice(4, 20)
    df_gm['ile'] = round(df_gm[_field] / df_gm['population'] * 100000, 1)
    tytul = '<b>Liczba zaszczepionych osób na 100000 mieszkańców gminy<br>' + field_dict[_field][0] + '</b>'
    hover_text = '<b>'+df_gm['nazwa'] + '</b><br>populacja: ' + df_gm['population'].astype(str) + \
                 '<br>liczba szczepień: ' + df_gm[_field].astype(str) + \
                 '<br>wartość wskaźnika: ' + df_gm['ile'].astype(str)

    df_rank = df_gm.copy()
    df_rank.sort_values(by=['ile'], ascending=True, inplace=True)
    anno_text1 = '<b>Gminy z najniższym wskaźnikiem:</b>'
    for index, row in df_rank.head(30).iterrows():
        anno_text1 += '<br>' + row['nazwa'] + ' (' + row['wojew'] + '): ' + str(row['ile'])
    anno_text2 = '<b>Gminy z najwyższym wskaźnikiem:</b>'
    df_rank.sort_values(by=['ile'], ascending=False, inplace=True)
    for index, row in df_rank.head(30).iterrows():
        anno_text2 += '<br>' + row['nazwa'] + ' (' + row['wojew'] + '): ' + str(row['ile'])

    mapbox_access_token = open("data/mapbox.token").read()
    mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=19.18, lat=52.11), zoom=5.5)
    # mapbox['style'] = constants.mapbox_styles[settings['map_mapbox']]
    mapbox['style'] = constants.mapbox_styles['Szary - bez etykiet']

    json_file = r'data/geojson/Gminy_5_WS.json'
    featuredikey = 'properties.JPT'
    with open(json_file, encoding="utf-8") as f:
        geojson = json.load(f)

    figure = go.Figure()
    figure.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        featureidkey=featuredikey,
        locations=df_gm['JPT'],
        z=df_gm['ile'],
        text=hover_text,
        marker=dict(line=dict(color='black', width=0.5 if 'drawborder' in settings['map_options'] else 0.)),
        hoverinfo='text',
        colorscale=settings['map_color_scale'],
        reversescale=False,
        marker_opacity=settings['map_opacity'],
        colorbar=dict(
            bgcolor=get_session_color(1),
            title=dict(text='wskaźnik'),
        ),
    ))
    figure.add_annotation(text=str(dt.now())[:10],
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'], size=24),
                          x=0.01, y=1,
                          showarrow=False)
    figure.add_annotation(text=anno_text1,
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.01, y=0.9,
                          showarrow=False)
    figure.add_annotation(text=anno_text2,
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.27, y=0.9,
                          showarrow=False)

    figure.update_layout(
        images=constants.image_logo_map,
        autosize=True,
        hovermode='closest',
        title=dict(
            text=tytul,
            x=settings['titlexpos'], y=settings['titleypos'] - 0.04,
            font=dict(size=settings['font_size_title'], color=get_session_color(7))
        ),
        height=int(settings['map_height']),
        margin=dict(l=settings['marginl'], r=settings['marginr'], b=settings['marginb'], t=settings['margint'],
                    pad=0),
        paper_bgcolor=get_session_color(1),
        plot_bgcolor=get_session_color(1),
        mapbox=mapbox,
    )
    figure = add_copyright(figure, settings)
    config = {
        'displaylogo': False,
        'responsive': True,
        'toImageButtonOptions': {
            'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
            'filename': 'docent_image', 'height': 700, 'width': 1200}
    }
    create_html(figure, 'mz_psz3_'+'_'+_field)
    ret_val = dbc.Col(dcc.Graph(id='mapa', figure=figure, config=config), className='mb-2 mt-2')

    return ret_val


################################################
#  Mapa powiatowa odporności
################################################

def layout_more_hit1(DF, _n, _fully, _3):
    settings = session2settings()

    # zachorowania w powiatach

    df_i = DF['cities'].copy()
    start_date = '2021-04-28'
    last_date = max(df_i['date'])
    df_i['new_cases'] = df_i['new_cases'].rolling(7, min_periods=7).mean()
    df_i['total_cases_6m'] = df_i[df_i['date'] >= start_date].groupby(['location', 'wojew'])['new_cases'].transform('sum')
    df_i = df_i[df_i.date == last_date].copy()
    df_i = df_i[['location', 'wojew', 'total_cases', 'total_cases_6m']].copy()
    df_i.columns = ['powiat', 'wojew', 'total_cases', 'total_cases_6m']

    # szczepienia w powiatach

    df_gm = pd.read_csv(constants.data_files['mz_api_vacc_gminy']['data_fn'])
    df_gm.sort_values(by=['powiat', 'wojew'], inplace=True)

    # połączenie dwóch przedziałów

    df_gm['1-19 lat'] = df_gm['12-19 lat']

    # sumy powiatowe szczepień

    df_gm['vacc_p'] = df_gm.groupby(['powiat', 'wojew'])['total_full'].transform('sum')

    # populacje dla powiatów

    df_gm['population_p'] = df_gm.groupby(['powiat', 'wojew'])['population'].transform('sum')
    df_gm = df_gm.drop_duplicates(subset=['powiat', 'wojew']).copy()
    df_gm = pd.merge(df_gm, df_i, how='left', left_on=['powiat', 'wojew'], right_on=['powiat', 'wojew'])

    # obliczenie wskaźnika

    df_gm['natural_p'] = df_gm['total_cases_6m'] * int(_n)
    wsk_szczepienia = round(df_gm['vacc_p'].sum() / df_gm['population_p'].sum(), 3)
    wsk_choroba = round(df_gm['natural_p'].sum() / df_gm['population_p'].sum(), 3)
    df_gm['n_imm_p'] = df_gm['vacc_p'] + df_gm['natural_p'] * (1 - wsk_szczepienia)
    df_gm['ile'] = round(df_gm['n_imm_p'] / df_gm['population_p'] * 100, 1)
    # df_gm['ile'] = round((df_gm['vacc_p'] + df_gm['natural_p'] * (1 - wsk_szczepienia)) /
    #                      df_gm['population_p'] * 100, 1)

    df_gm.sort_values(by=['ile'], inplace=True)
    anno_text = '<b>Powiaty z najniższym wskaźnikiem odporności:</b>'
    for index, row in df_gm.head(10).iterrows():
        anno_text += '<br>' + row['powiat'] + ': ' + str(row['ile']) + '%'
    anno_text += '<br><br><b>Powiaty z najwyższym wskaźnikiem odporności:</b>'
    df_gm.sort_values(by=['ile'], ascending=False, inplace=True)
    for index, row in df_gm.head(11).iterrows():
        if row['powiat'] != 'karkonoski':
            anno_text += '<br>' + row['powiat'] + ': ' + str(row['ile']) + '%'

    tytul = '<b>Powiatowa mapa immunizacji</b>' + \
            '<br><sub>mnożnik dla liczby potwierdzonych infekcji: ' + _n + ' ' + \
            '<br>osób w pełni zaszczepionych: ' + str(round(wsk_szczepienia * 100, 2)) + '%' + \
            ', osób z odpornością w wyniku przechorowania (od 28.04.2021): ' + str(round(wsk_choroba * 100, 2)) + '%'
    hover_text = '<b>'+df_gm['powiat']+'</b><br>populacja: '+round(df_gm['population_p'], 0).astype(str) + \
                 '<br>liczba szczepień:'+df_gm['vacc_p'].astype(str)+'<br>liczba infekcji: ' + \
                 (round(df_gm['total_cases'], 0)).astype(str) + \
                 '<br>wartość wskaźnika: '+df_gm['ile'].astype(str)

    mapbox_access_token = open("data/mapbox.token").read()
    mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=19.18, lat=52.11), zoom=5.5)
    # mapbox['style'] = constants.mapbox_styles['Szary - bez etykiet']
    # mapbox['style'] = constants.mapbox_styles[settings['map_mapbox']]
    mapbox['style'] = constants.mapbox_styles['Przezroczysty - bez etykiet']

    json_file = r'data/geojson/powiaty-min-ws.geojson'
    locations = df_gm['powiat']
    featuredikey = 'properties.nazwa'
    with open(json_file, encoding="utf-8") as f:
        geojson = json.load(f)

    figure = go.Figure()
    figure.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        featureidkey=featuredikey,
        locations=locations,
        z=df_gm['ile'],
        text=hover_text,
        marker=dict(line=dict(color='black', width=0.5 if 'drawborder' in settings['map_options'] else 0.)),
        hoverinfo='text',
        colorscale=settings['map_color_scale'],
        reversescale=False,
        # reversescale=True if 'reversescale' in settings['map_options'] else False,
        marker_opacity=settings['map_opacity'],
        colorbar=dict(
            len=0.5,
            bgcolor=get_session_color(1),
            tickcolor='red',
            tickfont=dict(color=settings['color_10']),
            title=dict(text=''),
            ticksuffix='%',
        ),
    ))
    figure.add_annotation(text=str(dt.now())[:10],
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'], size=24),
                          x=0.01, y=1,
                          showarrow=False)
    figure.add_annotation(text=anno_text,
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.01, y=0.8,
                          showarrow=False)

    figure.update_layout(
        # images=constants.image_logo_map,
        autosize=True,
        hovermode='closest',
        title=dict(
            text=tytul,
            x=settings['titlexpos'], y=settings['titleypos'] - 0.04,
            font=dict(size=settings['font_size_title'], color=get_session_color(7)
                      )
        ),
        height=int(settings['map_height']),
        margin=dict(l=settings['marginl'], r=settings['marginr'], b=settings['marginb'], t=settings['margint'],
                    pad=0),
        paper_bgcolor=get_session_color(1),
        plot_bgcolor=get_session_color(1),
        mapbox=mapbox,
    )
    figure = add_copyright(figure, settings)
    config = {
        'displaylogo': False,
        'responsive': True,
        'toImageButtonOptions': {
            'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
            'filename': 'docent_image', 'height': 700, 'width': 1200}
    }
    # create_html(figure, 'hit_')
    ret_val = dbc.Col(dcc.Graph(id='mapa', figure=figure, config=config), className='mb-2 mt-2')

    return ret_val


###################################
#  Mapa wojewódzka odporności
###################################

def layout_more_hit2(DF, _n, _fully, _3):
    settings = session2settings()

    # zachorowania w powiatach

    df_i = DF['poland'].copy()
    start_date = '2021-09-01'
    last_date = max(df_i['date'])
    df_i = df_i[df_i['location'] != 'Polska'].copy()
    df_i['new_cases'] = df_i['new_cases'].rolling(7, min_periods=7).mean()
    df_i['total_cases_6m'] = df_i[df_i['date'] >= start_date].groupby(['wojew'])['new_cases'].transform('sum')
    df_i = df_i[df_i.date == last_date].copy()
    df_i = df_i[['wojew', 'total_cases', 'total_cases_6m', 'Lat', 'Long']].copy()
    df_i.columns = ['wojew', 'total_cases', 'total_cases_6m', 'Lat', 'Long']

    # szczepienia w województwach

    df = pd.read_csv(constants.data_files['ecdc_vacc']['data_fn'])
    df = df[['date', 'location', 'dawka_1', 'dawka_2', 'dawka_3', 'grupa', 'denom', 'marka']]
    df = df[df['location'] != 'Polska'].copy()
    df = df[df['grupa'] != 'ALL'].copy()
    df['wojew'] = df['location'].str.lower()
    df.sort_values(by=['date', 'wojew', 'marka'], inplace=True)
    df['dawka_1_marka_total'] = df.groupby(['wojew', 'marka'])['dawka_1'].transform('sum')
    df['dawka_2_marka_total'] = df.groupby(['wojew', 'marka'])['dawka_2'].transform('sum')
    df['dawka_3_marka_total'] = df.groupby(['wojew', 'marka'])['dawka_3'].transform('sum')
    df = df.drop_duplicates(subset=['wojew', 'marka']).copy()
    df = pd.merge(df, df_i, how='left', left_on=['wojew'], right_on=['wojew']).copy()
    df['dawka_full_marka_total'] = df['dawka_2_marka_total']
    df.loc[df.marka == 'JANSS', 'dawka_full_marka_total'] = df['dawka_1_marka_total']
    df['dawka_full_total'] = df.groupby(['wojew'])['dawka_full_marka_total'].transform('sum')
    df['dawka_3_total'] = df.groupby(['wojew'])['dawka_3_marka_total'].transform('sum')
    df = df.drop_duplicates(subset=['wojew']).copy()
    df['population'] = df['wojew'].map(constants.wojew_pop)
    df['ile_vacc'] = (df['dawka_3_total']) / df['population'] * 100
    df['ile_ch'] = df['total_cases_6m'] * 5 / df['population'] * 100
    df['ile'] = df['ile_ch'] + df['ile_vacc']

    wsk_szczepienia = round(df['dawka_3_total'].sum() / df['population'].sum(), 2)
    wsk_choroba = round(df['total_cases_6m'].sum() * 5 / df['population'].sum(), 2)
    anno_text = '<b>Wskaźniki odporności:</b>'
    for index, row in df.iterrows():
        anno_text += '<br>' + row['wojew'] + ': ' + str(round(row['ile'], 2)) + '%'

    df['Long'] = df['wojew'].apply(lambda x: constants.wojew_mid_lower[x][1])
    df['Lat'] = df['wojew'].apply(lambda x: constants.wojew_mid_lower[x][0])

    tytul = '<b>Wojewódzka mapa immunizacji</b>' + \
            '<br><sub>mnożnik dla potwierdzonych infekcji: ' + _n + ', ' + \
            'zaszczepionych 3 dawką: ' + str(round(wsk_szczepienia * 100, 2)) + '%' + \
            ', odporność w wyniku przechorowania (od 1.09.2021): ' + str(round(wsk_choroba * 100, 2)) + '%'
    # hover_text = '<b>'+df_gm['wojew']+'</b><br>populacja: '+round(df_gm['population_w'], 0).astype(str) + \
    #              '<br>liczba szczepień:'+df_gm['vacc_w'].astype(str)+'<br>liczba infekcji: ' + \
    #              (round(df_gm['total_cases'], 0)).astype(str) + \
    #              '<br>wartość wskaźnika: '+df_gm['ile'].astype(str)

    mapbox_access_token = open("data/mapbox.token").read()
    mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=19.18, lat=52.11), zoom=5.5)
    mapbox['style'] = constants.mapbox_styles['Przezroczysty - bez etykiet']

    json_file = r'data/geojson/woj.min.geo.json'
    locations = df['wojew']
    featuredikey = 'properties.nazwa'
    with open(json_file, encoding="utf-8") as f:
        geojson = json.load(f)

    figure = go.Figure()
    figure.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        featureidkey=featuredikey,
        locations=locations,
        z=df['ile'],
        # text=hover_text,
        marker=dict(line=dict(color='black', width=0.5 if 'drawborder' in settings['map_options'] else 0.)),
        hoverinfo='text',
        colorscale=settings['map_color_scale'],
        reversescale=False,
        # reversescale=True if 'reversescale' in settings['map_options'] else False,
        marker_opacity=settings['map_opacity'],
        colorbar=dict(
            len=0.5,
            tickcolor='red',
            tickfont=dict(color=settings['color_10']),
            title=dict(text=''),
            ticksuffix='%',
        ),
    ))
    figure.add_annotation(text=str(dt.now())[:10],
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'], size=24),
                          x=0.01, y=1,
                          showarrow=False)
    figure.add_annotation(text=anno_text,
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.01, y=0.8,
                          showarrow=False)

    if 'annotations' in settings['map_options']:
        anno_text = df['ile'].round(1).astype(str) + '%'
        figure.add_trace(
            go.Scattermapbox(
                lat=df.Lat, lon=df.Long,
                mode='text',
                hoverinfo='none',
                below="''",
                marker=dict(allowoverlap=True),
                text=anno_text,
                textfont=dict(family='Bebas Neue', size=settings['font_size_anno'], color=get_session_color(4)),
            ))

    figure.update_layout(
        autosize=True,
        hovermode='closest',
        title=dict(
            text=tytul,
            x=settings['titlexpos'], y=settings['titleypos'] - 0.04,
            font=dict(size=settings['font_size_title'], color=get_session_color(7)
                      )
        ),
        height=int(settings['map_height']),
        margin=dict(l=settings['marginl'], r=settings['marginr'], b=settings['marginb'], t=settings['margint'],
                    pad=0),
        paper_bgcolor=get_session_color(1),
        plot_bgcolor=get_session_color(1),
        mapbox=mapbox,
    )
    figure = add_copyright(figure, settings)
    config = {
        'displaylogo': False,
        'responsive': True,
        'toImageButtonOptions': {
            'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
            'filename': 'docent_image', 'height': 700, 'width': 1200}
    }
    create_html(figure, 'hit_')
    ret_val = dbc.Col(dcc.Graph(id='mapa', figure=figure, config=config), className='mb-2 mt-2')

    return ret_val


################################################
#  Mapa wojewódzka odporności vs. infekcje
################################################

def layout_more_hit3(DF, _typ_lewy, _hit3_date, _typ_prawy):
    template = get_template()
    settings = session2settings()

    if settings['to_date'] is None:
        return
    to_date = str(settings['to_date'])[:10]
    hit3_date = str(_hit3_date)[:10]

    _n = 5

    # ['Suma infekcji/100k', 'Suma zgonów/100k', 'Zajęte łóżka/100k', 'Zajęte respiratory/100k',
    # 'Infekcje/100k dzienne', 'Zgony/100k dzienne']
    # 'Immunizacja', 'Pełne zaszczepienie', 'Zaszczepienie 1. dawką'

    # zachorowania w województwach

    df_i = DF['poland'].copy()
    date_6m = str(pd.to_datetime(to_date) - timedelta(days=183))[:10]
    # date_6m = '2021-05-01'
    df_i['new_cases_roll'] = df_i['new_cases'].rolling(7, min_periods=7).mean()
    df_i['total_cases_6m'] = df_i[df_i['date'] >= date_6m].groupby(['wojew'])['new_cases'].transform('sum')
    df_i['total_cases_xm'] = df_i[df_i['date'] >= hit3_date].groupby(['wojew'])['new_cases'].transform('sum')
    df_i['total_deaths_xm'] = df_i[df_i['date'] >= hit3_date].groupby(['wojew'])['new_deaths'].transform('sum')
    if 'dzienne' in _typ_prawy:
        df_i = df_i[df_i.date == hit3_date].copy()
        title_date_prefix = 'za dzień '
    else:
        title_date_prefix = 'od '
        df_i = df_i[df_i.date == to_date].copy()
    df_i = df_i[['wojew', 'total_cases', 'total_cases_6m', 'new_cases', 'new_cases_roll', 'new_deaths',
                 'total_cases_xm', 'icu_patients', 'hosp_patients', 'total_deaths_xm', 'Lat', 'Long']].copy()

    # szczepienia w województwach

    df_gm = pd.read_csv(constants.data_files['mz_api_vacc_gminy']['data_fn'])
    df_gm.sort_values(by=['wojew'], inplace=True)

    # połączenie dwóch przedziałów
    df_gm['1-19 lat'] = df_gm['12-19 lat']

    # sumy wojewódzkie szczepień
    df_gm['vacc_w'] = df_gm.groupby(['wojew'])['total_full'].transform('sum')
    df_gm['vacc_1d_w'] = df_gm.groupby(['wojew'])['total_1d'].transform('sum')

    # populacje dla województw
    df_gm['population_w'] = df_gm.groupby(['wojew'])['population'].transform('sum')
    df_gm = df_gm.drop_duplicates(subset=['wojew']).copy()
    df_gm = pd.merge(df_gm, df_i, how='left', left_on=['wojew'], right_on=['wojew']).copy()

    # obliczenie wskaźnika
    df_gm['natural_w'] = df_gm['total_cases_6m'] * int(_n)
    wsk_szczepienia = round(df_gm['vacc_w'].sum() / df_gm['population_w'].sum(), 3)
    wsk_choroba = round(df_gm['natural_w'].sum() / df_gm['population_w'].sum(), 3)
    df_gm['n_imm_w'] = df_gm['vacc_w'] + df_gm['natural_w'] * (1 - wsk_szczepienia)
    df_gm['ile_imm_w'] = round(df_gm['n_imm_w'] / df_gm['population_w'] * 100, 1)

    df_gm['ile_full'] = round(df_gm['vacc_w'] / df_gm['population_w'] * 100, 1)
    df_gm['ile_part'] = round(df_gm['vacc_1d_w'] / df_gm['population_w'] * 100, 1)
    if _typ_lewy == 'Immunizacja':
        df_gm['ile'] = df_gm['ile_imm_w']
    elif _typ_lewy == 'Pełne zaszczepienie':
        df_gm['ile'] = df_gm['ile_full']
    elif _typ_lewy == 'Zaszczepienie 1. dawką':
        df_gm['ile'] = df_gm['ile_part']

    if _typ_prawy == 'Suma infekcji/100k':
        df_gm['ile2'] = round(df_gm['total_cases_xm'] / df_gm['population_w'] * 100000, 3)
    elif _typ_prawy == 'Zajęte łóżka/100k':
        df_gm['ile2'] = round(df_gm['hosp_patients'] / df_gm['population_w'] * 100000, 3)
    elif _typ_prawy == 'Infekcje/100k dzienne':
        df_gm['ile2'] = round(df_gm['new_cases'] / df_gm['population_w'] * 100000, 3)
    elif _typ_prawy == 'Zgony/100k dzienne':
        df_gm['ile2'] = round(df_gm['new_deaths'] / df_gm['population_w'] * 100000, 3)
    elif _typ_prawy == 'Zajęte respiratory/100k':
        df_gm['ile2'] = round(df_gm['icu_patients'] / df_gm['population_w'] * 100000, 3)
    else:
        df_gm['ile2'] = round(df_gm['total_deaths_xm'] / df_gm['population_w'] * 100000, 3)

    df_gm['Long'] = df_gm['wojew'].apply(lambda x: constants.wojew_mid_lower[x][1])
    df_gm['Lat'] = df_gm['wojew'].apply(lambda x: constants.wojew_mid_lower[x][0])

    df_gm.sort_values(by=['ile'], inplace=True)
    mapbox_access_token = open("data/mapbox.token").read()
    mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=19.18, lat=52.11), zoom=5.5)
    mapbox['style'] = constants.mapbox_styles['Przezroczysty - bez etykiet']

    json_file = r'data/geojson/woj.min.geo.json'
    locations = df_gm['wojew']
    featuredikey = 'properties.nazwa'
    with open(json_file, encoding="utf-8") as f:
        geojson = json.load(f)

    # Immunizacja

    figure1 = go.Figure()
    figure1.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        featureidkey=featuredikey,
        locations=locations,
        z=df_gm['ile'],
        marker=dict(line=dict(color='black', width=0.5 if 'drawborder' in settings['map_options'] else 0.)),
        hoverinfo='text',
        showscale=False,
        colorscale='Greys',
        reversescale=True,
        marker_opacity=0.5,
        # marker_opacity=settings['map_opacity'],
        colorbar=dict(
            bgcolor=get_session_color(1),
            title=dict(text=_typ_lewy),
            ticksuffix='%',
        ),
    ))
    if _typ_lewy == 'Immunizacja':
        anno_text = df_gm['ile'].round(1).astype(str) + '%'
        tail = 'pełne zaszczepienie: ' + str(wsk_szczepienia * 100) + '%, przechorowanie (od.' + date_6m + '): ' + str(wsk_choroba * 100) + '%'
    else:
        anno_text = df_gm['ile'].round(1).astype(str) + '%'
        tail = ''
    # anno_text = df_gm['ile'].round(1).astype(str) + '%'
    figure1.add_trace(
        go.Scattermapbox(
            lat=df_gm.Lat, lon=df_gm.Long,
            mode='text',
            hoverinfo='none',
            below="''",
            marker=dict(allowoverlap=True),
            text=anno_text,
            textfont=dict(family='Bebas Neue', size=settings['font_size_anno'], color=get_session_color(4)),
        ))
    figure1.update_layout(
        template=template,
        autosize=True,
        # hovermode='closest',
        title=dict(
            text='<b>' + _typ_lewy + '</b><br><sup>' + tail,
            x=settings['titlexpos'], y=settings['titleypos'] - 0.04,
            font=dict(size=settings['font_size_title'], color=get_session_color(7))
        ),
        height=int(settings['map_height']),
        margin=dict(l=settings['marginl'], r=settings['marginr'] + 10, b=settings['marginb'], t=settings['margint'],
                    pad=0),
        paper_bgcolor=get_session_color(1),
        plot_bgcolor=get_session_color(1),
        mapbox=mapbox,
    )
    figure1 = add_copyright(figure1, settings)
    # Nowe infekcje

    figure2 = go.Figure()
    figure2.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        featureidkey=featuredikey,
        locations=locations,
        z=df_gm['ile2'],
        marker=dict(line=dict(color='black', width=0.5 if 'drawborder' in settings['map_options'] else 0.)),
        hoverinfo='text',
        showscale=False,
        colorscale='Greys',
        reversescale=False,
        marker_opacity=0.5,
        # marker_opacity=settings['map_opacity'],
        colorbar=dict(
            bgcolor=get_session_color(1),
            title=dict(text='infekcje/100k'),
            ticksuffix='',
        ),
    ))
    anno_text = df_gm['ile2'].round(3).astype(str)
    figure2.add_trace(
        go.Scattermapbox(
            lat=df_gm.Lat, lon=df_gm.Long,
            mode='text',
            hoverinfo='none',
            below="''",
            marker=dict(allowoverlap=True),
            text=anno_text,
            textfont=dict(family='Bebas Neue', size=settings['font_size_anno'], color=get_session_color(4)),
        ))
    figure2.update_layout(
        images=constants.image_logo_map,
        template=template,
        autosize=True,
        # hovermode='closest',
        title=dict(
            text='<b>' + _typ_prawy + '</b><br><sup>od ' + hit3_date + ' do ' + to_date,
            x=settings['titlexpos'], y=settings['titleypos'] - 0.04,
            font=dict(size=settings['font_size_title'], color=get_session_color(7)
                      )
        ),
        height=int(settings['map_height']),
        margin=dict(l=settings['marginl'], r=settings['marginr'] + 10, b=settings['marginb'], t=settings['margint'],
                    pad=0),
        paper_bgcolor=get_session_color(1),
        plot_bgcolor=get_session_color(1),
        mapbox=mapbox,
    )
    config = {
        'displaylogo': False,
        'responsive': True,
        'toImageButtonOptions': {
            'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
            'filename': 'docent_image', 'height': 700, 'width': 1200}
    }
    graph_core = layout_korelacja(df_gm['wojew'], df_gm['ile'], df_gm['ile2'], _typ_lewy + ' (%)', _typ_prawy + \
                                  ' (' + title_date_prefix + hit3_date + ')')
    ret_val = [
        dbc.Col(graph_core, width=12, className='pr-0 pl-0 mr-0 ml-0'),
        dbc.Col(dcc.Graph(id='mapa1', figure=figure1, config=config), width=6, className='pr-0 pl-0 mr-0 ml-0'),
        dbc.Col(dcc.Graph(id='mapa2', figure=figure2, config=config), width=6, className='pr-0 pl-0 mr-0 ml-0'),
    ]

    return ret_val


################################################
#  Mapa powiatowa odporności vs. infekcje
################################################

def layout_more_hit4(DF, _typ_lewy, _hit4_date, _typ_prawy):
    template = get_template()
    settings = session2settings()
    scope = settings['scope']
    locations = settings['locations']

    _n = 5

    # ['Suma infekcji/100k', 'Suma zgonów/100k', 'Poparcie dla Dudy 2020]
    # 'Immunizacja', 'Pełne zaszczepienie', 'Zaszczepienie 1. dawką'

    # zachorowania w powiatach

    df_i = DF['cities'].copy()
    start_date = str(_hit4_date)[:10]
    last_date = max(df_i['date'])
    date_6m = str(pd.to_datetime(last_date) - timedelta(days=183))[:10]
    # date_6m = '2021-05-01'
    df_i['new_cases'] = df_i['new_cases'].rolling(7, min_periods=7).mean()
    df_i['total_cases_6m'] = df_i[df_i['date'] >= date_6m].groupby(['location', 'wojew'])['new_cases'].transform('sum')
    df_i['total_cases_xm'] = df_i[df_i['date'] >= start_date].groupby(['location', 'wojew'])['new_cases'].transform('sum')
    df_i['total_deaths_xm'] = df_i[df_i['date'] >= start_date].groupby(['location', 'wojew'])['new_deaths'].transform('sum')
    df_i = df_i[df_i.date == last_date].copy()
    df_i = df_i[['location', 'wojew', 'total_cases', 'total_cases_6m', 'new_cases', 'total_cases_xm', 'total_deaths_xm', 'Lat', 'Long']].copy()

    # szczepienia w powiatach

    df_gm = pd.read_csv(constants.data_files['mz_api_vacc_gminy']['data_fn'])
    df_gm.sort_values(by=['wojew', 'powiat'], inplace=True)

    # wybory

    df_w = pd.read_csv('data/sources/wybory_prezydenckie_2020.csv')
    df_w.rename(columns={'Powiat': 'powiat'}, inplace=True)
    df_w.rename(columns={'Województwo': 'wojew'}, inplace=True)
    df_w['DUDA'] = df_w['DUDA'].str.replace(',', '.').astype(float)
    df_w['TRZASKOWSKI'] = df_w['TRZASKOWSKI'].str.replace(',', '.').astype(float)
    df_w.sort_values(by=['wojew', 'powiat'], inplace=True)
    for p in constants.powiat_translation_table:
        df_w.loc[(df_w.powiat == p['location']) & (df_w['wojew'] == p['wojew']), 'powiat'] = p['new_location']


    # połączenie dwóch przedziałów

    df_gm['1-19 lat'] = df_gm['12-19 lat']

    # sumy powiatowe szczepień

    df_gm['vacc_p'] = df_gm.groupby(['wojew', 'powiat'])['total_full'].transform('sum')
    df_gm['vacc_1d_p'] = df_gm.groupby(['wojew', 'powiat'])['total_1d'].transform('sum')

    # populacje dla powiatów

    df_gm['population_p'] = df_gm.groupby(['wojew', 'powiat'])['population'].transform('sum')
    df_gm = df_gm.drop_duplicates(subset=['wojew', 'powiat']).copy()
    df_gm = pd.merge(df_gm, df_i, how='left', left_on=['wojew', 'powiat'], right_on=['wojew', 'location'])
    df_gm = pd.merge(df_gm, df_w, how='left', left_on=['wojew', 'powiat'], right_on=['wojew', 'powiat'])

    # obliczenie wskaźnika
    df_gm['natural_p'] = df_gm['total_cases_6m'] * int(_n)
    wsk_szczepienia = round(df_gm['vacc_p'].sum() / df_gm['population_p'].sum(), 3)
    wsk_choroba = round(df_gm['natural_p'].sum() / df_gm['population_p'].sum(), 3)
    df_gm['ile_p'] = df_gm['vacc_p'] + df_gm['natural_p'] * (1 - wsk_szczepienia)
    if _typ_lewy == 'Immunizacja':
        df_gm['ile'] = round(df_gm['ile_p'] / df_gm['population_p'] * 100, 1)
    elif _typ_lewy == 'Pełne zaszczepienie':
        df_gm['ile'] = round(df_gm['vacc_p'] / df_gm['population_p'] * 100, 1)
    elif _typ_lewy == 'Zaszczepienie 1. dawką':
        df_gm['ile'] = round(df_gm['vacc_1d_p'] / df_gm['population_p'] * 100, 1)

    if _typ_prawy == 'Suma infekcji/100k':
        df_gm['ile2'] = round(df_gm['total_cases_xm'] / df_gm['population_p'] * 100000, 3)
    elif _typ_prawy == 'Poparcie dla Dudy 2020':
        df_gm['ile2'] = df_gm['DUDA']
    else:
        df_gm['ile2'] = round(df_gm['total_deaths_xm'] / df_gm['population_p'] * 100000, 3)
    df_gm['ile'].fillna(0, inplace=True)
    df_gm['ile2'].fillna(0, inplace=True)
    df_gm.sort_values(by=['ile'], inplace=True)

    if scope == 'cities' and len(locations) > 0:
        df_gm = df_gm[df_gm['location'].isin(locations)].copy()

    # df_gm = df_gm[df_gm['population_p'] > 100000].copy()

    mapbox_access_token = open("data/mapbox.token").read()
    mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=19.1, lat=52.3), zoom=5.5)
    mapbox['style'] = constants.mapbox_styles['Przezroczysty - bez etykiet']

    json_file = r'data/geojson/powiaty-min-ws.geojson'
    locations = df_gm['powiat']
    featuredikey = 'properties.nazwa'

    with open(json_file, encoding="utf-8") as f:
        geojson = json.load(f)

    # Immunizacja

    figure1 = go.Figure()
    figure1.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        featureidkey=featuredikey,
        locations=locations,
        z=df_gm['ile'],
        marker=dict(line=dict(color='black', width=0.5 if 'drawborder' in settings['map_options'] else 0.)),
        hoverinfo='text',
        # showscale=False,
        colorscale='Greys',
        reversescale=True,
        marker_opacity=settings['map_opacity'],
        colorbar=dict(
            len=0.5,
            bgcolor=get_session_color(1),
            tickcolor='red',
            tickfont=dict(color=settings['color_10']),
        ),
    ))
    figure1.update_layout(
        images=constants.image_logo_map,
        template=template,
        autosize=True,
        title=dict(
            text='<b>' + _typ_lewy + '</b><br><sup>(' + last_date + ')',
            x=settings['titlexpos'], y=settings['titleypos'] - 0.04,
            font=dict(size=settings['font_size_title'], color=get_session_color(7))
        ),
        height=int(settings['map_height']),
        margin=dict(l=settings['marginl'], r=settings['marginr'] + 10, b=settings['marginb'], t=settings['margint'],
                    pad=0),
        paper_bgcolor=get_session_color(1),
        plot_bgcolor=get_session_color(1),
        mapbox=mapbox,
    )
    figure1 = add_copyright(figure1, settings)

    # Nowe infekcje

    figure2 = go.Figure()
    figure2.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        featureidkey=featuredikey,
        locations=locations,
        z=df_gm['ile2'],
        marker=dict(line=dict(color='black', width=0.5 if 'drawborder' in settings['map_options'] else 0.)),
        hoverinfo='text',
        # showscale=False,
        colorscale='Greys',
        reversescale=False,
        marker_opacity=settings['map_opacity'],
        colorbar=dict(
            len=0.5,
            bgcolor=get_session_color(1),
            tickcolor='red',
            tickfont=dict(color=settings['color_10']),
            ticksuffix='',
        ),
    ))
    if _typ_prawy == 'Poparcie dla Dudy 2020':
        title = '<b>' + _typ_prawy + '</b><br><sup>Poparcie procentowe Andrzeja Dudy w II turze wyborów prezydenckich 2020'
        title2 = ''
    else:
        title = '<b>' + _typ_prawy + '</b><br><sup>od ' + start_date + ' do ' + last_date
        title2 = str(_hit4_date)[:10]
    figure2.update_layout(
        images=constants.image_logo_map,
        template=template,
        autosize=True,
        # hovermode='closest',
        title=dict(
            text=title,
            x=settings['titlexpos'], y=settings['titleypos'] - 0.04,
            font=dict(size=settings['font_size_title'], color=get_session_color(7)
                      )
        ),
        height=int(settings['map_height']),
        margin=dict(l=settings['marginl'], r=settings['marginr'] + 10, b=settings['marginb'], t=settings['margint'],
                    pad=0),
        paper_bgcolor=get_session_color(1),
        plot_bgcolor=get_session_color(1),
        mapbox=mapbox,
    )
    config = {
        'displaylogo': False,
        'responsive': True,
        'toImageButtonOptions': {
            'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
            'filename': 'docent_image', 'height': 700, 'width': 1200}
    }
    df_gm = df_gm[df_gm['powiat'] != 'karkonoski'].copy()
    if _typ_prawy == 'Poparcie dla Dudy 2020':
        title_right = _typ_prawy
    else:
        title_right = _typ_prawy + ' (od ' + start_date + ')'
    graph_core = layout_korelacja(df_gm['powiat'], df_gm['ile'], df_gm['ile2'], _typ_lewy + ' (%)', title_right,
                                  title_date=title2)
    ret_val = [
        dbc.Col(dcc.Graph(id='mapa1', figure=figure1, config=config), width=6, className='pr-0 pl-0 mr-0 ml-0'),
        dbc.Col(dcc.Graph(id='mapa2', figure=figure2, config=config), width=6, className='pr-0 pl-0 mr-0 ml-0'),
        dbc.Col(graph_core, width=12, className='pr-0 pl-0 mr-0 ml-0')
    ]

    return ret_val


################################################
#  Przyrosty/utrata odporności
################################################

def layout_more_hit5(DF, _n, n_ozdr, n_szcz):
    template = get_template()
    settings = session2settings()

    locations = settings['locations']
    locations_lower = [x.lower() for x in locations]
    # n_szcz = 40
    # n_ozdr = 40
    # zachorowania PL

    df_i = DF['cities'].copy()
    df_i = df_i[df_i['date'] >= '01-09-2020'].copy()
    df_i = df_i[df_i['location'] == 'Polska'].copy()
    start_date = str(dt.now() - timedelta(days=180))[:10]
    df_i['new_cases'] = df_i['new_cases'].rolling(7, min_periods=7).mean()
    df_i['total_cases_6m'] = df_i[df_i['date'] >= start_date].groupby(['location', 'wojew'])['new_cases'].transform('sum')
    df_i = df_i[['date', 'location', 'new_cases']].copy()
    df_i.index = pd.to_datetime(df_i['date'])
    df_i = df_i.resample('W-MON').sum()
    df_i['date'] = df_i.index
    df_i.reset_index(drop=True, inplace=True)
    df_i['date'] = pd.to_datetime(df_i['date']).dt.date
    df_i['date'] = df_i['date'].astype(str)

    df = pd.read_csv(constants.data_files['ecdc_vacc']['data_fn'])
    df = df[df['location'] == 'Polska'].copy()
    df = df[df['grupa'] != 'ALL'].copy()
    df['full'] = df['dawka_2'] + df['dawka_3']
    df.loc[df.marka == 'JANSS', 'full'] = df['dawka_1'] + df['dawka_3']
    # def f(r):
    #     if r.marka == 'JANSS':
    #         r['full'] = r['dawka_1']
    #     else:
    #         r['full'] = r['dawka_2']
    #     r['full'] += r['dawka_3']
    #     return r
    # df = df.apply(f, axis=1)

    # sumy dzienne pełnych szczepień
    df['vacc'] = df.groupby(['date'])['full'].transform('sum')
    df = df.drop_duplicates(subset=['date']).copy()
    df = df[['date', 'vacc']].copy()

    df = pd.merge(df_i, df, how='left', left_on=['date'], right_on=['date']).copy()

    # obliczenie wskaźnika
    wsp_redukcji = df['vacc'].sum() / 38000000
    df['natural'] = df['new_cases'] * int(_n) * (1-wsp_redukcji)
    df['vacc_minus'] = df['vacc'].shift(n_szcz) * -1
    df['natural_minus'] = df['natural'].shift(n_ozdr) * -1
    df.fillna(0, inplace=True)
    df['bilans'] = df['vacc'] + df['natural'] + (df['vacc_minus'] + df['natural_minus'])

    # sumy kumulowane
    df['vacc_cumsum'] = df['vacc'].cumsum()
    df['natural_cumsum'] = df['natural'].cumsum()
    df['vacc_minus_cumsum'] = df['vacc_minus'].cumsum()
    df['natural_minus_cumsum'] = df['natural_minus'].cumsum()
    df['bilans_cumsum'] = df['vacc_cumsum'] + df['natural_cumsum'] + (df['vacc_minus_cumsum'] + df['natural_minus_cumsum'])

    df = df.iloc[:len(df)-1].copy()

    all_traces = {
        'natural': 'przyrost odporności - ozdrowieńcy',
        'vacc': 'przyrost odporności - zaszczepieni',
        'natural_minus': 'utrata odporności - ozdrowieńcy',
        'vacc_minus': 'utrata odporności - zaszczepieni',
        'bilans': 'bilans'
    }
    traces = []
    i = 0
    for trace in all_traces.keys():
        width = 1
        if trace == 'bilans':
            width = 5
        x = list(df['date'])
        y = list(df[trace])
        fig_data = go.Scatter(
            x=x,
            y=y,
            line=dict(width=width, dash=constants.dashes[i]),
            name=all_traces[trace]
        )
        i += 1
        traces.append(fig_data)

    figure1 = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            # barmode='stack',
            title=dict(text='Symulacja immunizacji - przyrosty tygodniowe <sub>' +
                            '(dane ECDC i MZ [tygodniowe])'),
            height=float(settings['plot_height']) * 760,
            margin=dict(l=80, r=50, b=50, t=140, pad=2),
            legend=dict(x=0, y=1.02, orientation='h')
        )
    )
    anno_text = 'Parametry:' + \
        '<br><br>Utrata odporności po szczepieniu: ' + str(n_szcz) + ' tyg.' + \
        '<br>Utrata odporności po przechorowaniu: ' + str(n_ozdr) + ' tyg.' + \
        '<br>Mnożnik potwierdzonych infekcji: ' + str(_n) + \
        '<br>Odjęta część wspólna: zaszczepieni ozdrowieńcy' + \
        '<br>Odporność po przechorowaniu liczona od daty dodatniego testu' + \
        '<br>Odporność po szczepieniu liczona od daty podania II lub III dawki'
    figure1.add_annotation(text=anno_text,
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.02, y=0.83,
                          showarrow=False)

    # kumulowane

    all_traces = {
        'natural_cumsum': 'przyrost odporności - ozdrowieńcy',
        'vacc_cumsum': 'przyrost odporności - zaszczepieni',
        'natural_minus_cumsum': 'utrata odporności - ozdrowieńcy',
        'vacc_minus_cumsum': 'utrata odporności - zaszczepieni',
        'bilans_cumsum': 'bilans'
    }
    traces = []
    i = 0
    for trace in all_traces.keys():
        width = 2
        if trace == 'bilans_cumsum':
            width = 5
        x = list(df['date'])
        y = list(df[trace])
        fig_data = go.Scatter(
            x=x,
            y=y,
            line=dict(width=width, dash=constants.dashes[i]),
            name=all_traces[trace]
        )
        i += 1
        traces.append(fig_data)

    figure2 = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            # barmode='stack',
            title=dict(text='Symulacja immunizacji - dane kumulowane <sub>' +
                            '(dane ECDC i MZ [tygodniowe])'),
            height=float(settings['plot_height']) * 760,
            margin=dict(l=80, r=50, b=50, t=140, pad=2),
            legend=dict(x=0, y=1.02, orientation='h')
        )
    )
    # figure2.add_annotation(text=anno_text,
    #                       xref="paper", yref="paper",
    #                       align='left',
    #                       font=dict(color=settings['color_4'],
    #                                 size=settings['font_size_anno']),
    #                       x=0.02, y=0.83,
    #                       showarrow=False)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig1 = dcc.Graph(id='xxxx', figure=figure1, config=config)
    fig2 = dcc.Graph(id='xxxx', figure=figure2, config=config)
    ret_val = [
        dbc.Col(fig1, width=6, className='mb-2 mt-2'),
        dbc.Col(fig2, width=6, className='mb-2 mt-2')
    ]
    return ret_val


###########################################################
#  Liczba zaszczepionych ECDC
###########################################################

def layout_more_ecdc_vacc(_0, age, _brand, _dawka):
    template = get_template()
    settings = session2settings()
    from_date = settings['from_date'][:10]
    to_date = settings['to_date'][:10]
    if _dawka == 'dawka_1234':
        dawki = ['dawka_1', 'dawka_2', 'dawka_3', 'dawka_4']
    elif _dawka == 'dawka_12':
        dawki = ['dawka_1', 'dawka_2']
    elif _dawka == 'dawka_34':
        dawki = ['dawka_3', 'dawka_4']
    else:
        dawki = [_dawka]
    sl_dawki = {'dawka_1': 'dawka 1',
             'dawka_2': 'dawka 2',
             'dawka_3': 'dawka przypominająca 1',
             'dawka_4': 'dawka przypominająca 2',
             'dawka_1234': 'wszystkie dawki',
             'dawka_12': 'dawki podstawowe',
             'dawka_34': 'dawki przypominające',
             'ALL': 'wszystkie dawki',
             }
    ages = {'ALL': 'wszystkie grupy wiekowe',
            'Age0_4': 'wiek 0-4',
            'Age5_9': 'wiek 5-9',
            'Age10_14': 'wiek 10-14',
            'Age15_17': 'wiek 15-17',
            'Age18_24': 'wiek 18-24',
            'Age25_49': 'wiek 25-49',
            'Age50_59': 'wiek 50-59',
            'Age60_69': 'wiek 60-69',
            'Age70_79': 'wiek 70-79',
            'Age80+': 'wiek 80+',
            'AgeUNK': 'wiek nieznany',
            'HCW': 'pracownicy ochrony zdrowia'}
    if _brand == 'ALL':
        brands = ['MOD', 'COM', 'AZ', 'JANSS', 'NVXD']
        anno_flag = ''
    else:
        brands = [_brand]
        anno_flag = '+text'
    width = 12
    height = 660
    figures = []
    df0 = pd.read_csv(constants.data_files['ecdc_vacc']['data_fn'])
    df0 = df0[(df0['location'] == 'Polska') & (df0['grupa'] == age)].copy()
    df0 = df0[df0['date'].between(from_date, to_date)].copy()
    df0.sort_values(by=['date', 'location'], inplace=True)
    traces = []
    text = '<b>Podsumowanie okresu ' + from_date + ' - ' + to_date + '</b>' + \
           '<br><br><i>szczepionka - rodzaj dawki: ostatni tydzień (razem)</i><br>'
    df = df0[df0['location'] == 'Polska'].copy()
    ile_all = 0
    ile_week = 0
    for dawka in sorted(dawki):
        for brand in sorted(brands):
            ile = 0
            df_f = df[df['marka'] == brand].copy()
            # df_f['dawka_1234'] = df_f['dawka_1'] + df_f['dawka_2'] + df_f['dawka_3'] + df_f['dawka_4']
            x = list(df_f['date'])
            y = list(df_f[dawka])
            fig_data = go.Scatter(
                x=x,
                y=y,
                mode='lines+markers' + anno_flag,
                text=y,
                textposition='top left',
                textfont=dict(
                    size=settings['font_size_anno'],
                    color=settings['color_4']
                ),
                marker=dict(symbol='circle-dot', size=6),
                stackgroup='one',
                name=brand + ' (' + sl_dawki[dawka] + ')'
            )
            ile0 = int(y[-1])
            ile += sum(y)
            ile_week += ile0
            ile_all += ile
            if ile > 0:
                text += '<br><b>' + brand + ' </b>(' + sl_dawki[dawka] + '): <b>' + str(ile0) + ' ( ' + str(ile) + ')</b>'
                traces.append(fig_data)
    text += '<br><br>Ostatni tydzień (może być niepełny): <b>' + str(ile_week)
    text += '</b><br>RAZEM: <b>' + str(ile_all) + '</b>'
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            barmode='stack',
            title=dict(text='<b>Szczepienia (sumy tygodniowe)</b>' +
                            '<br><sub>' + sl_dawki[_dawka] + '<br>(' + ages[age] +
                            ', szczepionka: ' + _brand + ')'),
            height=float(settings['plot_height']) * height,
            margin=dict(l=settings['marginl'] + 50, r=settings['marginr'] + 50,
                        b=settings['marginb'] + 50, t=settings['margint'] + 50,
                        pad=2),
            legend=dict(x=0.05, y=0.9, orientation='h')
        )
    )
    figure = add_copyright(figure, settings)
    figure.add_annotation(text=text,
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=settings['annoxpos'], y=settings['annoypos'],
                          # x=0.01, y=0.8,
                          showarrow=False)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx' + 'Polska',
                    figure=figure,
                    style={'background': get_session_color(0)},
                    className='mb-0 mt-0 ml-0 mt-0 pl-0 pr-0 pt-0 pb-0',
                    config=config)
    figures.append(fig)
    ret_val = [
                  dbc.Col(figures[i], width=width, className='mb-0 mt-0 ml-0 mt-0 pl-0 pr-0 pt-0 pb-0') for i in
                  range(len(figures))
              ]
    return html.Div(children=dbc.Row(ret_val), style={'width': '99%', 'background-color': get_session_color(1)})
    # return ret_val


###########################################################
#  Szczepienia ECDC bilans razem
###########################################################

def layout_more_ecdc_vacc0(_0,  age, brand, _3):

    settings = session2settings()
    template = get_template()
    locations = settings['locations']
    if len(locations) == 0:
        locations = ['Polska']
    df = pd.read_csv(constants.data_files['ecdc_vacc']['data_fn'])
    df = df[(df.location.isin(locations)) & (df.grupa == age)].copy()
    if brand != 'ALL':
        df = df[df.marka == brand].copy()
    df.sort_values(by=['date'], inplace=True)
    df = df[['date', 'dawka_1', 'dawka_2', 'dawka_3', 'dostawa', 'marka']].copy()
    df['dawka_123'] = df['dawka_1'] + df['dawka_2'] + df['dawka_3']

    df['dawka_1_day'] = df.groupby(['date'])['dawka_1'].transform('sum')
    df['dawka_2_day'] = df.groupby(['date'])['dawka_2'].transform('sum')
    df['dawka_3_day'] = df.groupby(['date'])['dawka_3'].transform('sum')
    df['dawka_123_day'] = df.groupby(['date'])['dawka_123'].transform('sum')
    df['dostawa_day'] = df.groupby(['date'])['dostawa'].transform('sum')
    dj = df.loc[df['marka'] == 'JANSS']['dawka_1'].sum()

    df = df.drop_duplicates(subset=['date']).copy()
    df['dawka_1_cum'] = df['dawka_1_day'].cumsum()
    df['dawka_2_cum'] = df['dawka_2_day'].cumsum()
    df['dawka_3_cum'] = df['dawka_3_day'].cumsum()
    df['dawka_123_cum'] = df['dawka_123_day'].cumsum()
    df['dostawa_cum'] = df['dostawa_day'].cumsum()

    df['magazyn'] = df['dostawa_cum'] - df['dawka_123_cum']
    d1 = df.iloc[-1]['dawka_1_cum']
    d2 = df.iloc[-1]['dawka_2_cum']
    d3 = df.iloc[-1]['dawka_3_cum']
    dm = df.iloc[-1]['magazyn']
    dd = df.iloc[-1]['dostawa_cum']
    text1 = ''
    text1 += '<br>Zaszczepieni 1 dawka: '
    text1 += '<br>Zaszczepieni 2 dawką: '
    text1 += '<br>Zaszczepieni 3 dawką: '
    text1 += '<br>W pełni zaszczepieni: '
    if age == 'ALL':
        text1 += '<br>Stan magazynu i w trakcie dostawy do punktów: '
        text1 += '<br>Suma dostaw od producenta: '
    text2 = ''
    text2 += '<br>' + str(int(d1))
    text2 += '<br>' + str(int(d2))
    text2 += '<br>' + str(int(d3))
    text2 += '<br>' + str(int(d2 + dj))
    if age == 'ALL':
        text2 += '<br>' + str(int(dm))
        text2 += '<br>' + str(int(dd))
    x = list(df['date'])
    y = list(df['dawka_1_cum'])
    figures = []
    traces = []

    # dawka 1

    fig_data = go.Scatter(
        x=x,
        y=y,
        mode='lines+markers+text',
        # text=y,
        textposition='top left',
        textfont=dict(
            size=settings['font_size_anno'],
            color=settings['color_4']
        ),
        marker=dict(symbol='circle-dot', size=12),
        stackgroup='one',
        name='Dawka 1'
    )
    traces.append(fig_data)

    # dawka 2

    y = list(df['dawka_2_cum'])
    fig_data = go.Scatter(
        x=x,
        y=y,
        mode='lines+markers+text',
        # text=y,
        textposition='top left',
        textfont=dict(
            size=settings['font_size_anno'],
            color=settings['color_4']
        ),
        marker=dict(symbol='circle-dot', size=12),
        stackgroup='one',
        name='Dawka 2'
    )
    traces.append(fig_data)

    # dawka 3

    y = list(df['dawka_3_cum'])
    fig_data = go.Scatter(
        x=x,
        y=y,
        mode='lines+markers+text',
        # text=y,
        textposition='top left',
        textfont=dict(
            size=settings['font_size_anno'],
            color=settings['color_4']
        ),
        marker=dict(symbol='circle-dot', size=12),
        stackgroup='one',
        name='Dawka 3'
    )
    traces.append(fig_data)

    # magazyn

    if age == 'ALL':
        y = list(df['magazyn'])
        fig_data = go.Scatter(
            x=x,
            y=y,
            mode='lines+markers+text',
            # text=list(df['dawka_1_cum'].astype(int) + df['dawka_2_cum'].astype(int) + df['dawka_3_cum'].astype(int) + df['magazyn'].astype(int)),
            textposition='top left',
            textfont=dict(
                size=settings['font_size_anno'],
                color=settings['color_4']
            ),
            marker=dict(symbol='circle-dot', size=12),
            stackgroup='one',
            name='Magazyn'
        )
        traces.append(fig_data)

    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            barmode='stack',
            title=dict(text='Bilans szczepionek narastająco<br>'
                            'Grupa wiekowa: ' + age + ', szczepionka: ' + brand +
                            '<br><sub>dane ECDC z dnia ' + str(dt.today())[:10]),
            height=680,
            legend=dict(x=0.05, y=1.03, orientation='h')
        )
    )
    figure.add_annotation(text=text1,
                          xref="paper",
                          yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.05,
                          y=0.85,
                          showarrow=False)
    figure.add_annotation(text=text2,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.35,
                          y=0.85,
                          showarrow=False)
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxxx', figure=figure, config=config,
                    className='mb-0 mt-0 ml-0 mt-0 pl-0 pr-0 pt-0 pb-0')
    figures.append(fig)
    ret_val = [
        dbc.Col(figures, width=12, className='mb-0 mt-0'),
    ]
    return html.Div(children=dbc.Row(ret_val), style={'width': '99%', 'background-color': get_session_color(1)})


###########################################################
#  Liczba zaszczepionych w grupach wiekowych ECDC - 1 dawka
###########################################################

def layout_more_ecdc_vacc3(_0, _dawka, _2, _3):

    template = get_template()
    settings = session2settings()
    locations = settings['locations']
    if len(locations) != 1:
        locations = ['Polska']
    location_txt = locations[0]
    dawki = {
        'dawka_1': 'dawką 1',
        'dawka_2': 'dawką 2',
        'dawka_3': 'dawką 3',
        'dawka_4': 'dawką 4',
    }
    df = pd.read_csv(constants.data_files['ecdc_vacc']['data_fn'])
    df = df[['date', 'location', 'dawka_1', 'dawka_2', 'dawka_3', 'dawka_4', 'grupa', 'denom', 'marka']]
    df = df[(df['location'].isin(locations)) & (df['grupa'].isin(constants.ages_ecdc_order.keys()))].copy()
    df.sort_values(by=['grupa', 'marka'], inplace=True)
    df['total'] = df.groupby(['grupa', 'marka'])[_dawka].transform('sum')
    df['share'] = df['total'] / df['denom']
    df = df.drop_duplicates(subset=['grupa', 'marka']).copy()
    df[['location', 'grupa', 'marka', 'total', 'share']].to_csv('data/out/vacc3_'+location_txt+'.csv', index=False)
    traces = []
    text = '<br>Wiek 18-24: ' + str(int(df[df['grupa'] == 'Age18_24']['total'].sum()))
    text += '<br>Wiek 25-49: ' + str(int(df[df['grupa'] == 'Age25_49']['total'].sum()))
    text += '<br>Wiek 50-59: ' + str(int(df[df['grupa'] == 'Age50_59']['total'].sum()))
    text += '<br>Wiek 60-69: ' + str(int(df[df['grupa'] == 'Age60_69']['total'].sum()))
    text += '<br>Wiek 70-79: ' + str(int(df[df['grupa'] == 'Age70_79']['total'].sum()))
    text += '<br>Wiek 80+: ' + str(int(df[df['grupa'] == 'Age80+']['total'].sum()))

    dfx = pd.read_csv(constants.data_files['ecdc_vacc']['data_fn'])
    dfx = dfx[dfx['location'].isin(locations)].copy()
    dfx.sort_values(by=['grupa', 'marka'], inplace=True)
    dfx['total'] = dfx.groupby(['grupa', 'marka'])[_dawka].transform('sum')
    dfx = dfx.drop_duplicates(subset=['grupa', 'marka']).copy()
    text += '<br>Wiek nieznany: ' + str(int(dfx[dfx['grupa'] == 'AgeUNK']['total'].sum()))
    text += '<br><br>Razem: ' + str(int(dfx[dfx['grupa'] == 'ALL']['total'].sum()))
    text += '<br><br>w tym medycy: ' + str(int(dfx[dfx['grupa'] == 'HCW']['total'].sum()))
    del dfx

    df['order'] = df['grupa'].map(constants.ages_ecdc_order)
    df.sort_values(by=['order', 'grupa'], inplace=True)

    x = list(df[df['marka'] == 'MOD']['grupa'])
    y = list(df[df['marka'] == 'MOD']['total'])
    fig_data = go.Bar(
        x=x,
        y=y,
        marker=dict(line=dict(width=1, color='black')),
        text=[int(x) for x in y],
        textposition='auto',
        insidetextfont=dict(color='black', size=settings['font_size_anno']),
        outsidetextfont=dict(color='black', size=settings['font_size_anno']),
        name='Moderna'
    )
    traces.append(fig_data)
    y = list(df[df['marka'] == 'COM']['total'])
    fig_data = go.Bar(
        x=x,
        y=y,
        marker=dict(line=dict(width=1, color='black')),
        text=[int(x) for x in y],
        textposition='auto',
        insidetextfont=dict(color='black', size=settings['font_size_anno']),
        outsidetextfont=dict(color='black', size=settings['font_size_anno']),
        name='Comirnaty'
    )
    traces.append(fig_data)
    y = list(df[df['marka'] == 'AZ']['total'])
    fig_data = go.Bar(
        x=x,
        y=y,
        marker=dict(line=dict(width=1, color='black')),
        text=[int(x) for x in y],
        textposition='auto',
        insidetextfont=dict(color='black', size=settings['font_size_anno']),
        outsidetextfont=dict(color='black', size=settings['font_size_anno']),
        name='AstraZeneca'
    )
    traces.append(fig_data)
    y = list(df[df['marka'] == 'JANSS']['total'])
    fig_data = go.Bar(
        x=x,
        y=y,
        marker=dict(line=dict(width=1, color='black')),
        text=[int(x) for x in y],
        textposition='auto',
        insidetextfont=dict(color='black', size=settings['font_size_anno']),
        outsidetextfont=dict(color='black', size=settings['font_size_anno']),
        name='Janssen'
    )
    traces.append(fig_data)
    y = list(df[df['marka'] == 'NVXD']['total'])
    fig_data = go.Bar(
        x=x,
        y=y,
        marker=dict(line=dict(width=1, color='black')),
        text=[int(x) for x in y],
        textposition='auto',
        insidetextfont=dict(color='black', size=settings['font_size_anno']),
        outsidetextfont=dict(color='black', size=settings['font_size_anno']),
        name='Novavax'
    )
    traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            barmode='stack',
            title=dict(text='Liczba zaszczepionych ' + dawki[_dawka] +
                            ' w grupach wiekowych (dane ECDC z dnia ' + str(dt.today())[:10] +
                            ')<br>' + location_txt),
            height=float(settings['plot_height']) * 660,
            margin=dict(l=settings['marginl'] + 50, r=settings['marginr'] + 40, b=settings['marginb'] + 50,
                        t=settings['margint'] + 40,
                        pad=4),
            legend=layout_legend(y=1, legend_place=settings['legend_place'])
            # legend=dict(x=0.6, y=1., orientation='h')
        )
    )
    figure.add_annotation(text=text,
                          xref="paper",
                          yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.01,
                          y=0.75,
                          showarrow=False)
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


#########################################
#  Ranking szczepień w krajach Europy
#########################################

def layout_more_ecdc_vacc5(_0, _typ, _grupa, _3):
    # 'dawka_1', 'dawka_2', 'dostawa', 'grupa', 'marka',
    # 'denom', 'location', 'week2', 'year2', 'date'
    # typ na 100k:
    # - 'dostawa'
    # - 'dawka_1'
    # - 'dawki' (dawka_1 + dawka_2)
    # - 'zapas' (dostawa - (dawka_1 + dawka_2))

    template = get_template()
    settings = session2settings()
    ages = {
        'Age10_14': ' ( grupa wiekowa 10-14',
        'Age15_17': ' ( grupa wiekowa 15-17',
        'Age18_24': ' ( grupa wiekowa 18-24 )',
        'Age25_49': ' ( grupa wiekowa 25-49 )',
        'Age50_59': ' ( grupa wiekowa 50-59 )',
        'Age60_69': ' ( grupa wiekowa 60-69 )',
        'Age70_79': ' ( grupa wiekowa 70-79 )',
        'Age80+': ' ( grupa wiekowa 80+ )'}
    tits = {'dawka_1': 'Liczba podanych I dawek',
            'dawki': 'Liczba podanych wszystkich dawek',
            'dostawa': 'Dostawy szczepionek',
            'wpelni': 'Liczba osób w pełni zaszczepionych',
            'zapas': 'Dawki w magazynie lub punktach - oczekujące na zaszczepienie'}
    df0 = pd.read_csv(constants.data_files['ecdc_vacc_eu']['data_fn'])
    df = df0[df0.grupa == _grupa].copy()

    df.sort_values(by=['location', 'date'], inplace=True)
    df['dawka_12'] = df['dawka_1'] + df['dawka_2']
    df['zapas'] = df['dostawa'] - df['dawka_1'] - df['dawka_2']
    df['wpelni'] = df['dawka_2']
    df.loc[df.marka == 'JANSS', 'wpelni'] = df['dawka_1']

    # def f(r):
    #     if r.marka == 'JANSS':
    #         r['wpelni'] = r['dawka_1']
    #     else:
    #         r['wpelni'] = r['dawka_2']
    #     return r
    # df = df.apply(f, axis=1)

    if _typ == 'dawka_1':
        df['data0'] = df.groupby(['location', 'date'])['dawka_1'].transform('sum')
        df.drop_duplicates(subset=['location', 'date'], inplace=True)
        df['data'] = df.groupby(['location'])['data0'].cumsum()
        df['data'] = df['data'] / df['denom'] * 100
    elif _typ == 'dostawa':
        df['data0'] = df.groupby(['location', 'date'])['dostawa'].transform('sum')
        df.drop_duplicates(subset=['location', 'date'], inplace=True)
        df['data'] = df.groupby(['location'])['data0'].cumsum()
        df['data'] = df['data'] / df['denom'] * 100
    elif _typ == 'dawki':
        df['data0'] = df.groupby(['location', 'date'])['dawka_12'].transform('sum')
        df.drop_duplicates(subset=['location', 'date'], inplace=True)
        df['data'] = df.groupby(['location'])['data0'].cumsum()
        df['data'] = df['data'] / df['denom'] * 100
    elif _typ == 'wpelni':
        df['data0'] = df.groupby(['location', 'date'])['wpelni'].transform('sum')
        df.drop_duplicates(subset=['location', 'date'], inplace=True)
        df['data'] = df.groupby(['location'])['data0'].cumsum()
        df['data'] = df['data'] / df['denom'] * 100
    else:    # typ == 'zapas'
        df['data0'] = df.groupby(['location', 'date'])['zapas'].transform('sum')
        df.drop_duplicates(subset=['location', 'date'], inplace=True)
        df['data'] = df.groupby(['location'])['data0'].cumsum()
        df['data'] = df['data'] / df['denom'] * 100
    df = df.groupby('location').tail(1)
    df = df[df['data'] > 0].copy()
    df.sort_values(by=['data'], inplace=True)
    traces = []
    tit_0 = tits[_typ]
    if _grupa != 'ALL':
        tit_0 += ages[_grupa]
    text = '* w danych niektórych krajów (np. Irlandia) liczebność grupy wiekowej 80+'
    text += '<br>odbiega od rzeczywistej wartości'
    x = list(df['location'].unique())
    y = list(round(df['data'], 1))
    colors = [get_session_color(8)] * len(x)
    colors[x.index('Polska')] = get_session_color(9)
    fig_data = go.Bar(
        x=x,
        y=y,
        text=y,
        textposition='inside',
        marker={'color': colors},
        name='ranking'
    )
    traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text=tit_0 + ' w krajach UE<br><sub>(w przeliczeniu na 100 osób kwalifikujących się '
                                    'do szczepienia)<br>Żródło: ECDC z dnia ' + str(dt.today())[:10]),
            height=700,
            # colorway=constants.color_scales[settings['color_order']],
            legend=dict(x=0.05, y=1.03, orientation='h')
        )
    )
    figure.add_annotation(text=text,
                          xref="paper",
                          yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.05,
                          y=0.95,
                          showarrow=False)
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxxx', figure=figure, config=config,
                    className='mb-0 mt-0 ml-0 mt-0 pl-0 pr-0 pt-0 pb-0')
    ret_val = [
        dbc.Col(fig, width=12, className='mb-0 mt-0')
    ]
    return html.Div(children=dbc.Row(ret_val), style={'width': '99%', 'background-color': get_session_color(1)})


###############################################
#  Bilans zaszczepienia populacji Polski ECDC
###############################################

def layout_more_ecdc_vacc6(_0, _location, _2, _3):
    template = get_template()
    settings = session2settings()
    locations = ['Polska']
    location_txt = locations[0]
    data_do = settings['to_date']
    ages = {
        'Age0_4': 'wiek 0-4',
        'Age5_9': 'wiek 5-9',
        'Age10_14': 'wiek 10-14',
        'Age15_17': 'wiek 15-17',
        'Age18_24': 'wiek 18-24',
        'Age25_49': 'wiek 25-49',
        'Age50_59': 'wiek 50-59',
        'Age60_69': 'wiek 60-69',
        'Age70_79': 'wiek 70-75',
        'Age80+': 'wiek 80+'}
    ages_sort = {
        'Age0_4': 0,
        'Age5_9': 1,
        'Age10_14': 2,
        'Age15_17': 3,
        'Age18_24': 4,
        'Age25_49': 5,
        'Age50_59': 6,
        'Age60_69': 7,
        'Age70_79': 8,
        'Age80+': 9}
    df = pd.read_csv(constants.data_files['ecdc_vacc']['data_fn'])
    df = df[['date', 'location', 'dawka_1', 'dawka_2', 'dawka_3', 'grupa', 'denom', 'marka']]
    df = df[(df['location'].isin(locations)) & (df['grupa'].isin(ages.keys()))].copy()
    df = df[df['date'] <= data_do].copy()
    df['grupa_key'] = df['grupa'].map(ages_sort)
    df.sort_values(by=['grupa_key', 'marka'], inplace=True)

    df['dawka_12'] = df['dawka_2']
    df.loc[df.marka == 'JANSS', 'dawka_12'] = df['dawka_1']

    df['total_1'] = df.groupby(['grupa'])['dawka_1'].transform('sum')
    df['total_12'] = df.groupby(['grupa'])['dawka_12'].transform('sum')
    df['total_3'] = df.groupby(['grupa'])['dawka_3'].transform('sum')
    df['share_1'] = df['total_1'] / df['denom']
    df['share_12'] = df['total_12'] / df['denom']
    df['share_3'] = df['total_3'] / df['denom']
    df = df.drop_duplicates(subset=['grupa']).copy()
    df[['location', 'grupa', 'marka', 'total_1', 'total_12', 'share_1', 'share_12', 'share_3', 'total_3']] \
        .to_csv('data/out/vacc6_'+location_txt+'.csv', index=False)
    traces = []
    text_w = 'Wiek<br><br>0-4<br>5-9<br>10-14<br>15-17<br>18-24<br>25-49<br>50-59<br>60-69<br>70-79<br>80+'
    text1 = 'Populacja<br>'
    text1 += '<br>' + str(int(df[df['grupa'] == 'Age0_4'].iloc[0]['denom']))
    text1 += '<br>' + str(int(df[df['grupa'] == 'Age5_9'].iloc[0]['denom']))
    text1 += '<br>' + str(int(df[df['grupa'] == 'Age10_14'].iloc[0]['denom']))
    text1 += '<br>' + str(int(df[df['grupa'] == 'Age15_17'].iloc[0]['denom']))
    text1 += '<br>' + str(int(df[df['grupa'] == 'Age18_24'].iloc[0]['denom']))
    text1 += '<br>' + str(int(df[df['grupa'] == 'Age25_49'].iloc[0]['denom']))
    text1 += '<br>' + str(int(df[df['grupa'] == 'Age50_59'].iloc[0]['denom']))
    text1 += '<br>' + str(int(df[df['grupa'] == 'Age60_69'].iloc[0]['denom']))
    text1 += '<br>' + str(int(df[df['grupa'] == 'Age70_79'].iloc[0]['denom']))
    text1 += '<br>' + str(int(df[df['grupa'] == 'Age80+'].iloc[0]['denom']))

    text2a = 'I dawka<br>'
    text2a += '<br>' + str(int(df[df['grupa'] == 'Age0_4']['total_1'].sum()))
    text2a += '<br>' + str(int(df[df['grupa'] == 'Age5_9']['total_1'].sum()))
    text2a += '<br>' + str(int(df[df['grupa'] == 'Age10_14']['total_1'].sum()))
    text2a += '<br>' + str(int(df[df['grupa'] == 'Age15_17']['total_1'].sum()))
    text2a += '<br>' + str(int(df[df['grupa'] == 'Age18_24']['total_1'].sum()))
    text2a += '<br>' + str(int(df[df['grupa'] == 'Age25_49']['total_1'].sum()))
    text2a += '<br>' + str(int(df[df['grupa'] == 'Age50_59']['total_1'].sum()))
    text2a += '<br>' + str(int(df[df['grupa'] == 'Age60_69']['total_1'].sum()))
    text2a += '<br>' + str(int(df[df['grupa'] == 'Age70_79']['total_1'].sum()))
    text2a += '<br>' + str(int(df[df['grupa'] == 'Age80+']['total_1'].sum()))

    text2b = '%<br>'
    text2b += '<br>' + str(round(df[df['grupa'] == 'Age0_4']['share_1'].sum()*100, 2))+' %'
    text2b += '<br>' + str(round(df[df['grupa'] == 'Age5_9']['share_1'].sum()*100, 2))+' %'
    text2b += '<br>' + str(round(df[df['grupa'] == 'Age10_14']['share_1'].sum()*100, 2))+' %'
    text2b += '<br>' + str(round(df[df['grupa'] == 'Age15_17']['share_1'].sum()*100, 2))+' %'
    text2b += '<br>' + str(round(df[df['grupa'] == 'Age18_24']['share_1'].sum()*100, 2))+' %'
    text2b += '<br>' + str(round(df[df['grupa'] == 'Age25_49']['share_1'].sum()*100, 2))+' %'
    text2b += '<br>' + str(round(df[df['grupa'] == 'Age50_59']['share_1'].sum()*100, 2))+' %'
    text2b += '<br>' + str(round(df[df['grupa'] == 'Age60_69']['share_1'].sum()*100, 2))+' %'
    text2b += '<br>' + str(round(df[df['grupa'] == 'Age70_79']['share_1'].sum()*100, 2))+' %'
    text2b += '<br>' + str(round(df[df['grupa'] == 'Age80+']['share_1'].sum()*100, 2))+' %'

    text3a = 'w pełni<br>'
    text3a += '<br>' + str(int(df[df['grupa'] == 'Age0_4']['total_12'].sum()))
    text3a += '<br>' + str(int(df[df['grupa'] == 'Age5_9']['total_12'].sum()))
    text3a += '<br>' + str(int(df[df['grupa'] == 'Age10_14']['total_12'].sum()))
    text3a += '<br>' + str(int(df[df['grupa'] == 'Age15_17']['total_12'].sum()))
    text3a += '<br>' + str(int(df[df['grupa'] == 'Age18_24']['total_12'].sum()))
    text3a += '<br>' + str(int(df[df['grupa'] == 'Age25_49']['total_12'].sum()))
    text3a += '<br>' + str(int(df[df['grupa'] == 'Age50_59']['total_12'].sum()))
    text3a += '<br>' + str(int(df[df['grupa'] == 'Age60_69']['total_12'].sum()))
    text3a += '<br>' + str(int(df[df['grupa'] == 'Age70_79']['total_12'].sum()))
    text3a += '<br>' + str(int(df[df['grupa'] == 'Age80+']['total_12'].sum()))

    text3b = '%<br>'
    text3b += '<br>' + str(round(df[df['grupa'] == 'Age0_4']['share_12'].sum()*100, 2))+' %'
    text3b += '<br>' + str(round(df[df['grupa'] == 'Age5_9']['share_12'].sum()*100, 2))+' %'
    text3b += '<br>' + str(round(df[df['grupa'] == 'Age10_14']['share_12'].sum()*100, 2))+' %'
    text3b += '<br>' + str(round(df[df['grupa'] == 'Age15_17']['share_12'].sum()*100, 2))+' %'
    text3b += '<br>' + str(round(df[df['grupa'] == 'Age18_24']['share_12'].sum()*100, 2))+' %'
    text3b += '<br>' + str(round(df[df['grupa'] == 'Age25_49']['share_12'].sum()*100, 2))+' %'
    text3b += '<br>' + str(round(df[df['grupa'] == 'Age50_59']['share_12'].sum()*100, 2))+' %'
    text3b += '<br>' + str(round(df[df['grupa'] == 'Age60_69']['share_12'].sum()*100, 2))+' %'
    text3b += '<br>' + str(round(df[df['grupa'] == 'Age70_79']['share_12'].sum()*100, 2))+' %'
    text3b += '<br>' + str(round(df[df['grupa'] == 'Age80+']['share_12'].sum()*100, 2))+' %'

    text4a = 'III dawka<br>'
    text4a += '<br>' + str(int(df[df['grupa'] == 'Age0_4']['total_3'].sum()))
    text4a += '<br>' + str(int(df[df['grupa'] == 'Age5_9']['total_3'].sum()))
    text4a += '<br>' + str(int(df[df['grupa'] == 'Age10_14']['total_3'].sum()))
    text4a += '<br>' + str(int(df[df['grupa'] == 'Age15_17']['total_3'].sum()))
    text4a += '<br>' + str(int(df[df['grupa'] == 'Age18_24']['total_3'].sum()))
    text4a += '<br>' + str(int(df[df['grupa'] == 'Age25_49']['total_3'].sum()))
    text4a += '<br>' + str(int(df[df['grupa'] == 'Age50_59']['total_3'].sum()))
    text4a += '<br>' + str(int(df[df['grupa'] == 'Age60_69']['total_3'].sum()))
    text4a += '<br>' + str(int(df[df['grupa'] == 'Age70_79']['total_3'].sum()))
    text4a += '<br>' + str(int(df[df['grupa'] == 'Age80+']['total_3'].sum()))

    text4b = '%<br>'
    text4b += '<br>' + str(round(df[df['grupa'] == 'Age0_4']['share_3'].sum()*100, 2))+' %'
    text4b += '<br>' + str(round(df[df['grupa'] == 'Age5_9']['share_3'].sum()*100, 2))+' %'
    text4b += '<br>' + str(round(df[df['grupa'] == 'Age10_14']['share_3'].sum()*100, 2))+' %'
    text4b += '<br>' + str(round(df[df['grupa'] == 'Age15_17']['share_3'].sum()*100, 2))+' %'
    text4b += '<br>' + str(round(df[df['grupa'] == 'Age18_24']['share_3'].sum()*100, 2))+' %'
    text4b += '<br>' + str(round(df[df['grupa'] == 'Age25_49']['share_3'].sum()*100, 2))+' %'
    text4b += '<br>' + str(round(df[df['grupa'] == 'Age50_59']['share_3'].sum()*100, 2))+' %'
    text4b += '<br>' + str(round(df[df['grupa'] == 'Age60_69']['share_3'].sum()*100, 2))+' %'
    text4b += '<br>' + str(round(df[df['grupa'] == 'Age70_79']['share_3'].sum()*100, 2))+' %'
    text4b += '<br>' + str(round(df[df['grupa'] == 'Age80+']['share_3'].sum()*100, 2))+' %'
    x = list(ages.keys())
    y_denom = list(df['denom'])
    fig_data = go.Bar(
        x=x,
        y=y_denom,
        marker=dict(line=dict(width=1, color='black')),
        base=0,
        offsetgroup=0,
        name='niezaszczepieni'
    )
    traces.append(fig_data)
    y_total_1 = list(df['total_1'])
    fig_data = go.Bar(
        x=x,
        y=y_total_1,
        marker=dict(line=dict(width=1, color='black')),
        offsetgroup=0,
        name='tylko I dawka'
    )
    traces.append(fig_data)
    y_total_12 = list(df['total_12'])
    fig_data = go.Bar(
        x=x,
        y=y_total_12,
        marker=dict(line=dict(width=1, color='black')),
        base=0,
        offsetgroup=0,
        name='w pełni zaszczepieni'
    )
    traces.append(fig_data)
    y_total_3 = list(df['total_3'])
    fig_data = go.Bar(
        x=x,
        y=y_total_3,
        marker=dict(line=dict(width=1, color='orange')),
        base=0,
        offsetgroup=1,
        name='III dawka'
    )
    traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            # barmode='stack',
            title=dict(text='Bilans populacyjny szczepień w grupach wiekowych (dane ECDC z dnia ' +
                            str(data_do)[:10] + ')<br>' + location_txt),
            height=float(settings['plot_height']) * 760,
            paper_bgcolor=get_session_color(1),
            # colorway=['rgb(255, 128, 0)', 'rgb(255, 191, 0)', 'rgb(191, 255, 0)'],
            plot_bgcolor=get_session_color(1),
            legend=dict(x=0.6, y=1.03, orientation='h')
        )
    )
    figure.add_annotation(text=text_w,
                          xref="paper",
                          yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.01,
                          y=0.85,
                          showarrow=False)
    figure.add_annotation(text=text1,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.07,
                          y=0.85,
                          showarrow=False)
    figure.add_annotation(text=text2a,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.13,
                          y=0.85,
                          showarrow=False)
    figure.add_annotation(text=text2b,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.19,
                          y=0.85,
                          showarrow=False)
    figure.add_annotation(text=text3a,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.25,
                          y=0.85,
                          showarrow=False)
    figure.add_annotation(text=text3b,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.31,
                          y=0.85,
                          showarrow=False)
    figure.add_annotation(text=text4a,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.38,
                          y=0.85,
                          showarrow=False)
    figure.add_annotation(text=text4b,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.44,
                          y=0.85,
                          showarrow=False)
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


#########################################
#  Procent zaszczepienia w województwach
#########################################

def layout_more_ecdc_vacc7(_0, _age, _2, _3):
    template = get_template()
    settings = session2settings()
    dodaty = settings['to_date'][:10]
    ages = {'ALL': 'wszystkie grupy wiekowe',
            'Age0_4': 'wiek 0-4',
            'Age5_9': 'wiek 5-9',
            'Age10_14': 'wiek 10-14',
            'Age15_17': 'wiek 15-17',
            'Age18_24': 'wiek 18-24',
            'Age25_49': 'wiek 25-49',
            'Age50_59': 'wiek 50-59',
            'Age60_69': 'wiek 60-69',
            'Age70_79': 'wiek 70-79',
            'Age80+': 'wiek 80+',
            }
    age = _age
    if _age == 'Age80PLUS':
        age = 'Age80+'
    df = pd.read_csv(constants.data_files['ecdc_vacc']['data_fn'])
    df = df[['date', 'location', 'dawka_1', 'dawka_2', 'grupa', 'denom', 'population', 'marka']]
    df = df[df['grupa'] == age].copy()
    df = df[df['date'] <= dodaty].copy()
    df.sort_values(by=['location', 'date'], inplace=True)
    df['dawka_12'] = df['dawka_2']
    df['full_janss'] = 0
    df.loc[df.marka == 'JANSS', 'dawka_12'] = df['dawka_1']
    df.loc[df.marka == 'JANSS', 'full_janss'] = df['dawka_1']

    df['total_1'] = df.groupby(['location'])['dawka_1'].transform('sum')
    df['total_2'] = df.groupby(['location'])['dawka_2'].transform('sum')
    df['total_12'] = df.groupby(['location'])['dawka_12'].transform('sum')
    df['total_full_janss'] = df.groupby(['location'])['full_janss'].transform('sum')
    df = df.drop_duplicates(subset=['location']).copy()
    traces = []
    df['total_1_percent'] = df['total_1'] / df['denom'] * 100
    df.sort_values(by=['total_1_percent'], inplace=True, ascending=False)
    x = list(df['location'].unique())
    y_total_1 = df['total_1_percent']
    y_total_full = (df['total_full_janss'] + df['total_2']) / df['denom'] * 100
    y_brak = 100 - y_total_1
    y_total_1_t = round(y_total_1, 2)
    y_total_full_t = round(y_total_full, 2)
    y_brak_t = round(y_brak, 2)
    fig_data = go.Bar(
        x=x,
        y=[100] * len(x),
        base=0,
        offsetgroup=0,
        text=[str(x) for x in y_brak_t],
        textposition='auto',
        insidetextfont=dict(size=settings['font_size_anno'], color='black'),
        outsidetextfont=dict(color='green', size=10),
        name='niezaszczepieni'
    )
    traces.append(fig_data)
    fig_data = go.Bar(
        x=x,
        y=list(y_total_1),
        offsetgroup=0,
        base=0,
        text=[str(x) for x in list(y_total_1_t)],
        textposition='outside',
        insidetextfont=dict(size=settings['font_size_anno'], color='black'),
        outsidetextfont=dict(color='black', size=settings['font_size_anno']),
        name='tylko I dawka'
    )
    traces.append(fig_data)
    fig_data = go.Bar(
        x=x,
        y=y_total_full,
        base=0,
        offsetgroup=0,
        text=[str(x) for x in y_total_full_t],
        textposition='auto',
        insidetextfont=dict(size=settings['font_size_anno'], color='black'),
        outsidetextfont=dict(color='green', size=10),
        name='w pełni zaszczepieni'
    )
    traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            margin=dict(l=80, r=50, b=170, t=80, pad=2),
            title=dict(text='Procent zaszczepienia w województwach ' + ages[age] + ' (dane ECDC z dnia ' +
                            str(dodaty) + ')' +
                            '<br><sub>w odniesieniu do osób kwalifikujących się do szczepienia (18+)<br><br>'),
            height=float(settings['plot_height']) * 760,
            paper_bgcolor=get_session_color(1),
            colorway=['red', 'yellow', 'green'],
            plot_bgcolor=get_session_color(1),
            legend=dict(x=0.3, y=1.03, orientation='h')
        )
    )
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


###########################################################
#  Udział grup wiekowych w tygodniowych szczepieniach
###########################################################

def layout_more_ecdc_vacc8(_0, _dawka, _2, _3):
    template = get_template()
    settings = session2settings()
    locations = settings['locations']
    if len(locations) != 1:
        return
    location_txt = locations[0]
    ages = {
        'Age0_4': 'wiek 0-4',
        'Age5_9': 'wiek 5-9',
        'Age10_14': 'wiek 10-14',
        'Age15_17': 'wiek 15-17',
        'Age18_24': 'wiek 18-24',
        'Age25_49': 'wiek 25-49',
        'Age50_59': 'wiek 50-59',
        'Age60_69': 'wiek 60-69',
        'Age70_79': 'wiek 70-79',
        'Age80+': 'wiek 80+'}
    dawki = {
        'dawka_1': 'dawką 1',
        'dawka_2': 'dawką 2',
        'dawka_3': 'dawką 3',
        'dawka_123': 'dawkami 1,2,3',
    }
    df = pd.read_csv(constants.data_files['ecdc_vacc']['data_fn'])
    df = df[['date', 'location', 'dawka_1', 'dawka_2', 'dawka_3', 'grupa', 'denom', 'marka']]
    df = df[(df['location'].isin(locations)) & (df['grupa'].isin(ages.keys()))].copy()
    df.sort_values(by=['date', 'grupa'], inplace=True)

    df['dawka_123'] = df['dawka_1'] + df['dawka_2'] + df['dawka_3']
    df['total_da1'] = df.groupby(['date', 'grupa'])['dawka_1'].transform('sum')
    df['total_da2'] = df.groupby(['date', 'grupa'])['dawka_2'].transform('sum')
    df['total_da3'] = df.groupby(['date', 'grupa'])['dawka_3'].transform('sum')
    df['total_da123'] = df.groupby(['date', 'grupa'])['dawka_123'].transform('sum')
    df['total_d1'] = df.groupby(['date'])['dawka_1'].transform('sum')
    df['total_d2'] = df.groupby(['date'])['dawka_2'].transform('sum')
    df['total_d3'] = df.groupby(['date'])['dawka_3'].transform('sum')
    df['total_d123'] = df.groupby(['date'])['dawka_123'].transform('sum')
    if _dawka == 'dawka_1':
        df['total_da'] = df['total_da1']
        df['total_d'] = df['total_d1']
    elif _dawka == 'dawka_2':
        df['total_da'] = df['total_da2']
        df['total_d'] = df['total_d2']
    elif _dawka == 'dawka_3':
        df['total_da'] = df['total_da3']
        df['total_d'] = df['total_d3']
    else:
        df['total_da'] = df['total_da123']
        df['total_d'] = df['total_d123']
    df = df.drop_duplicates(subset=['total_da1']).copy()
    traces = []
    for age in ages.keys():
        dfx = df[df['grupa'] == age].copy()
        x = list(dfx['date'])
        y = list(dfx['total_da'] / dfx['total_d'] * 100)
        fig_data = go.Bar(
            x=x,
            y=y,
            text=[round(x, 2) for x in y],
            textposition='auto',
            insidetextfont=dict(color='black'),
            outsidetextfont=dict(color='white'),
            name=ages[age]
        )
        base = y
        traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            barmode='stack',
            title=dict(text='Udział procentowy grup wiekowych w tygodniowych szczepieniach ' + dawki[_dawka] +
                            '<br>' + location_txt +
                            '<br><sub>(dane ECDC z dnia ' + str(dt.today())[:10] + ')'),
            height=float(settings['plot_height']) * 660,
            paper_bgcolor=get_session_color(1),
            plot_bgcolor=get_session_color(1),
            legend=dict(x=1, y=1., orientation='v')
        )
    )
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


###########################################################
#  Udział rodzajów szczepionek w tygodniowych szczepieniach
###########################################################

def layout_more_ecdc_vacc9(_0, _age, _2, _3):
    template = get_template()
    settings = session2settings()
    locations = settings['locations']
    if len(locations) != 1:
        return
    location = locations[0]
    ages = {
        'Age0_4': 'wiek 0-4',
        'Age5_9': 'wiek 5-9',
        'Age10_14': 'wiek 10-14',
        'Age15_17': 'wiek 15-17',
        'Age18_24': 'wiek 18-24',
        'Age25_49': 'wiek 25-49',
        'Age50_59': 'wiek 50-59',
        'Age60_69': 'wiek 60-69',
        'Age70_79': 'wiek 70-79',
        'Age80+': 'wiek 80+',
        'ALL': 'wszystkie grupy wiekowe',
    }
    brands = {
        'MOD': ['rgb(255, 179, 179)', 'rgb(230, 0, 0)', 'rgb(64, 19, 10)', 'Moderna'],  # czerwień
        'COM': ['rgb(153, 255, 204)', 'rgb(0, 179, 89)', 'rgb(5, 61, 23)', 'Comirnaty'],  # zieleń
        'AZ': ['rgb(153, 194, 255)', 'rgb(0, 92, 230)', 'rgb(20, 12, 110)', 'AstraZeneca'],  # niebieski
        'JANSS': ['rgb(223, 191, 159)', 'rgb(153, 102, 51)', 'rgb(54, 33, 5)', 'Janssen'],  # brązowy
        'NVXD': ['rgb(223, 91, 159)', 'rgb(153, 02, 51)', 'rgb(54, 3, 5)', 'Novavax']
    }
    df = pd.read_csv(constants.data_files['ecdc_vacc']['data_fn'])
    df = df[['date', 'location', 'dawka_1', 'dawka_2', 'dawka_3', 'grupa', 'denom', 'marka']]
    df = df[(df['location'] == location) & (df['grupa'] == _age)].copy()
    df.sort_values(by=['date', 'grupa'], inplace=True)

    df['dawka_123'] = df['dawka_1'] + df['dawka_2'] + df['dawka_3']
    df['total_da1'] = df.groupby(['date', 'grupa'])['dawka_1'].transform('sum')
    df['total_da2'] = df.groupby(['date', 'grupa'])['dawka_2'].transform('sum')
    df['total_da3'] = df.groupby(['date', 'grupa'])['dawka_3'].transform('sum')
    df['total_da123'] = df.groupby(['date', 'grupa'])['dawka_123'].transform('sum')
    df['total_d1'] = df.groupby(['date'])['dawka_1'].transform('sum')
    df['total_d2'] = df.groupby(['date'])['dawka_2'].transform('sum')
    df['total_d3'] = df.groupby(['date'])['dawka_3'].transform('sum')
    df['total_d123'] = df.groupby(['date'])['dawka_123'].transform('sum')
    traces = []
    base = 0
    for brand in brands.keys():
        dfx = df[df['marka'] == brand].copy()
        x = list(dfx['date'])
        y = list(dfx['dawka_1'] / dfx['total_d123'] * 100)
        fig_data = go.Bar(
            x=x,
            y=y,
            text=[round(x, 2) for x in y],
            marker=dict(color=brands[brand][0]),
            textposition='auto',
            insidetextfont=dict(color='black'),
            outsidetextfont=dict(color='white'),
            name=brands[brand][3] + ' dawka 1'
        )
        traces.append(fig_data)
        if brand != 'JANSS':
            y = list(dfx['dawka_2'] / dfx['total_d123'] * 100)
            fig_data = go.Bar(
                x=x,
                y=y,
                text=[round(x, 2) for x in y],
                marker=dict(color=brands[brand][1]),
                textposition='auto',
                insidetextfont=dict(color='black'),
                outsidetextfont=dict(color='white'),
                name=brands[brand][3] + ' dawka 2'
            )
            traces.append(fig_data)
        if brand != 'JANSS':
            y = list(dfx['dawka_3'] / dfx['total_d123'] * 100)
            fig_data = go.Bar(
                x=x,
                y=y,
                text=[round(x, 2) for x in y],
                marker=dict(color=brands[brand][2]),
                textposition='auto',
                insidetextfont=dict(color='white'),
                outsidetextfont=dict(color='white'),
                name=brands[brand][3] + ' dawka 3'
            )
            traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            barmode='stack',
            title=dict(text='Udział rodzajów szczepionek w liczbie tygodniowych szczepień, ' + ages[_age] +
                            '<br>(dane ECDC z dnia ' + str(dt.today())[:10] + ')'),
            height=float(settings['plot_height']) * 700,
            paper_bgcolor=get_session_color(1),
            plot_bgcolor=get_session_color(1),
            legend=dict(x=1, y=1., orientation='v')
        )
    )
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


###########################################################
#  Szczepienia w Polsce - tygodniowo
###########################################################

def layout_more_ecdc_vacc11(_0, _age, _2, _3):
    # session['template'] = 'white'
    import datetime
    template = get_template()
    settings = session2settings()
    age = _age
    if _age == 'Age80PLUS':
        age = 'Age80+'
    df = pd.read_csv(constants.data_files['ecdc_vacc']['data_fn'])
    df = df[['date', 'location', 'dawka_1', 'dawka_2', 'dawka_3', 'dawka_4', 'grupa', 'denom', 'marka']]
    df = df[df['grupa'] == age].copy()
    df = df[df['location'] == 'Polska'].copy()
    last_d = list(df['date'].unique())[-1]
    weekday =datetime.datetime.strptime(last_d, '%Y-%m-%d').weekday()
    if weekday != 0:
        df = df[df['date'] != last_d].copy()
    def f(r):
        if r.marka == 'JANSS':
            r['dawka_first'] = 0
            r['dawka_12'] = r['dawka_1']
            r['janss'] = r['dawka_1']
            r['full'] = 0
        else:
            r['dawka_first'] = r['dawka_1']
            r['dawka_12'] = r['dawka_1'] + r['dawka_2']
            r['full'] = r['dawka_2']
            r['janss'] = 0
        return r
    df = df.apply(f, axis=1)
    df['total_first'] = df.groupby(['date'])['dawka_first'].transform('sum')
    df['total_12'] = df.groupby(['date'])['dawka_12'].transform('sum')
    df['total_3'] = df.groupby(['date'])['dawka_3'].transform('sum')
    df['total_4'] = df.groupby(['date'])['dawka_4'].transform('sum')
    df['total_full'] = df.groupby(['date'])['full'].transform('sum')
    df['total_janss'] = df.groupby(['date'])['janss'].transform('sum')
    df = df.drop_duplicates(subset=['date']).copy()

    df.sort_values(by=['date'], inplace=True)

    all_traces = {
        'total_first': 'pierwsza dawka szczepionki dwudawkowej',
        'total_full': 'druga dawka szczepionki dwudawkowej',
        'total_3': 'pierwsza dawka przypominająca',
        'total_4': 'druga dawka przypominająca',
        # 'total_12': 'suma wszystkich dawek',
        'total_janss': 'szczepionka jednodawkowa'
    }
    traces = []
    for trace in all_traces.keys():
        x = list(df['date'])
        y = list(df[trace])
        fig_data = go.Scatter(
            x=x,
            y=y,
            # stackgroup='one',
            name=all_traces[trace]
        )
        traces.append(fig_data)
    wdstr = ''
    if weekday != 0:
        wdstr = '<br>* nie zawiera danych z bieżącego, niepełnego tygodnia'
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            # barmode='stack',
            title=dict(text='Barometr szczepień<br><sub>' +
                            'dane ECDC (tygodniowe) z dnia ' + str(dt.today())[:10] + wdstr),
            height=float(settings['plot_height']) * 660,
            margin=dict(l=80, r=50, b=50, t=140, pad=2),
            legend=dict(x=0, y=1.02, orientation='h')
        )
    )
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


###########################################################
#  Spadek efektywnego zaszczepienia populacji Polski
###########################################################

def layout_more_ecdc_vacc1(_0, _1, _2, _3):
    template = get_template()
    settings = session2settings()
    locations = ['Polska']
    location_txt = locations[0]
    date6 = '2022-02-04'
    ages = {
        'Age0_4': 'wiek 0-4',
        'Age5_9': 'wiek 5-9',
        'Age10_14': 'wiek 10-14',
        'Age15_17': 'wiek 15-17',
        'Age18_24': 'wiek 18-24',
        'Age25_49': 'wiek 25-49',
        'Age50_59': 'wiek 50-59',
        'Age60_69': 'wiek 60-69',
        'Age70_79': 'wiek 70-75',
        'Age80+': 'wiek 80+'}
    ages_sort = {
        'Age0_4': 0,
        'Age5_9': 1,
        'Age10_14': 2,
        'Age15_17': 3,
        'Age18_24': 4,
        'Age25_49': 5,
        'Age50_59': 6,
        'Age60_69': 7,
        'Age70_79': 8,
        'Age80+': 9}
    df = pd.read_csv(constants.data_files['ecdc_vacc']['data_fn'])
    df = df[['date', 'location', 'dawka_1', 'dawka_2', 'dawka_3', 'dawka_4', 'grupa', 'denom', 'marka']]
    df = df[(df['location'].isin(locations)) & (df['grupa'].isin(ages.keys()))].copy()
    df['grupa_key'] = df['grupa'].map(ages_sort)
    df.sort_values(by=['grupa_key', 'marka'], inplace=True)

    df['dawka_12'] = df['dawka_2']
    df.loc[df.marka == 'JANSS', 'dawka_12'] = df['dawka_1']

    df6 = df[df['date'] > date6].copy()

    df['total_12'] = df.groupby(['grupa'])['dawka_12'].transform('sum')
    df['total_3'] = df.groupby(['grupa'])['dawka_3'].transform('sum')
    df['total_4'] = df.groupby(['grupa'])['dawka_4'].transform('sum')
    df['share_12'] = df['total_12'] / df['denom']
    df['share_3'] = df['total_3'] / df['denom']
    df['share_4'] = df['total_4'] / df['denom']
    df = df.drop_duplicates(subset=['grupa']).copy()

    # szczepienia wykonane nie później niż 5 miesięcy temu

    # df6 = df[df['date'] > date6].copy()
    df6['total_12'] = df6.groupby(['grupa'])['dawka_12'].transform('sum')
    df6['total_3'] = df6.groupby(['grupa'])['dawka_3'].transform('sum')
    df6['total_4'] = df6.groupby(['grupa'])['dawka_4'].transform('sum')
    df6['share_12'] = df6['total_12'] / df6['denom']
    df6['share_3'] = df6['total_3'] / df6['denom']
    df6['share_4'] = df6['total_4'] / df6['denom']
    df6 = df6.drop_duplicates(subset=['grupa']).copy()
    traces = []
    text_w = 'Wiek<br><br>0-4<br>5-9<br>10-14<br>15-17<br>18-24<br>25-49<br>50-59<br>60-69<br>70-79<br>80+'
    text1 = 'Populacja<br>'
    text1 += '<br>' + str(int(df[df['grupa'] == 'Age0_4'].iloc[0]['denom']))
    text1 += '<br>' + str(int(df[df['grupa'] == 'Age5_9'].iloc[0]['denom']))
    text1 += '<br>' + str(int(df[df['grupa'] == 'Age10_14'].iloc[0]['denom']))
    text1 += '<br>' + str(int(df[df['grupa'] == 'Age15_17'].iloc[0]['denom']))
    text1 += '<br>' + str(int(df[df['grupa'] == 'Age18_24'].iloc[0]['denom']))
    text1 += '<br>' + str(int(df[df['grupa'] == 'Age25_49'].iloc[0]['denom']))
    text1 += '<br>' + str(int(df[df['grupa'] == 'Age50_59'].iloc[0]['denom']))
    text1 += '<br>' + str(int(df[df['grupa'] == 'Age60_69'].iloc[0]['denom']))
    text1 += '<br>' + str(int(df[df['grupa'] == 'Age70_79'].iloc[0]['denom']))
    text1 += '<br>' + str(int(df[df['grupa'] == 'Age80+'].iloc[0]['denom']))

    text3a = 'w pełni<br>'
    text3a += '<br>' + str(int(df[df['grupa'] == 'Age0_4']['total_12'].sum()))
    text3a += '<br>' + str(int(df[df['grupa'] == 'Age5_9']['total_12'].sum()))
    text3a += '<br>' + str(int(df[df['grupa'] == 'Age10_14']['total_12'].sum()))
    text3a += '<br>' + str(int(df[df['grupa'] == 'Age15_17']['total_12'].sum()))
    text3a += '<br>' + str(int(df[df['grupa'] == 'Age18_24']['total_12'].sum()))
    text3a += '<br>' + str(int(df[df['grupa'] == 'Age25_49']['total_12'].sum()))
    text3a += '<br>' + str(int(df[df['grupa'] == 'Age50_59']['total_12'].sum()))
    text3a += '<br>' + str(int(df[df['grupa'] == 'Age60_69']['total_12'].sum()))
    text3a += '<br>' + str(int(df[df['grupa'] == 'Age70_79']['total_12'].sum()))
    text3a += '<br>' + str(int(df[df['grupa'] == 'Age80+']['total_12'].sum()))

    text3b = '%<br>'
    text3b += '<br>' + str(round(df[df['grupa'] == 'Age0_4']['share_12'].sum()*100, 2))+' %'
    text3b += '<br>' + str(round(df[df['grupa'] == 'Age5_9']['share_12'].sum()*100, 2))+' %'
    text3b += '<br>' + str(round(df[df['grupa'] == 'Age10_14']['share_12'].sum()*100, 2))+' %'
    text3b += '<br>' + str(round(df[df['grupa'] == 'Age15_17']['share_12'].sum()*100, 2))+' %'
    text3b += '<br>' + str(round(df[df['grupa'] == 'Age18_24']['share_12'].sum()*100, 2))+' %'
    text3b += '<br>' + str(round(df[df['grupa'] == 'Age25_49']['share_12'].sum()*100, 2))+' %'
    text3b += '<br>' + str(round(df[df['grupa'] == 'Age50_59']['share_12'].sum()*100, 2))+' %'
    text3b += '<br>' + str(round(df[df['grupa'] == 'Age60_69']['share_12'].sum()*100, 2))+' %'
    text3b += '<br>' + str(round(df[df['grupa'] == 'Age70_79']['share_12'].sum()*100, 2))+' %'
    text3b += '<br>' + str(round(df[df['grupa'] == 'Age80+']['share_12'].sum()*100, 2))+' %'

    text4a = 'III dawka<br>'
    text4a += '<br>' + str(int(df[df['grupa'] == 'Age0_4']['total_3'].sum()))
    text4a += '<br>' + str(int(df[df['grupa'] == 'Age5_9']['total_3'].sum()))
    text4a += '<br>' + str(int(df[df['grupa'] == 'Age10_14']['total_3'].sum()))
    text4a += '<br>' + str(int(df[df['grupa'] == 'Age15_17']['total_3'].sum()))
    text4a += '<br>' + str(int(df[df['grupa'] == 'Age18_24']['total_3'].sum()))
    text4a += '<br>' + str(int(df[df['grupa'] == 'Age25_49']['total_3'].sum()))
    text4a += '<br>' + str(int(df[df['grupa'] == 'Age50_59']['total_3'].sum()))
    text4a += '<br>' + str(int(df[df['grupa'] == 'Age60_69']['total_3'].sum()))
    text4a += '<br>' + str(int(df[df['grupa'] == 'Age70_79']['total_3'].sum()))
    text4a += '<br>' + str(int(df[df['grupa'] == 'Age80+']['total_3'].sum()))

    text4b = '%<br>'
    text4b += '<br>' + str(round(df[df['grupa'] == 'Age0_4']['share_3'].sum()*100, 2))+' %'
    text4b += '<br>' + str(round(df[df['grupa'] == 'Age5_9']['share_3'].sum()*100, 2))+' %'
    text4b += '<br>' + str(round(df[df['grupa'] == 'Age10_14']['share_3'].sum()*100, 2))+' %'
    text4b += '<br>' + str(round(df[df['grupa'] == 'Age15_17']['share_3'].sum()*100, 2))+' %'
    text4b += '<br>' + str(round(df[df['grupa'] == 'Age18_24']['share_3'].sum()*100, 2))+' %'
    text4b += '<br>' + str(round(df[df['grupa'] == 'Age25_49']['share_3'].sum()*100, 2))+' %'
    text4b += '<br>' + str(round(df[df['grupa'] == 'Age50_59']['share_3'].sum()*100, 2))+' %'
    text4b += '<br>' + str(round(df[df['grupa'] == 'Age60_69']['share_3'].sum()*100, 2))+' %'
    text4b += '<br>' + str(round(df[df['grupa'] == 'Age70_79']['share_3'].sum()*100, 2))+' %'
    text4b += '<br>' + str(round(df[df['grupa'] == 'Age80+']['share_3'].sum()*100, 2))+' %'

    text5a = 'IV dawka<br>'
    text5a += '<br>' + str(int(df[df['grupa'] == 'Age0_4']['total_4'].sum()))
    text5a += '<br>' + str(int(df[df['grupa'] == 'Age5_9']['total_4'].sum()))
    text5a += '<br>' + str(int(df[df['grupa'] == 'Age10_14']['total_4'].sum()))
    text5a += '<br>' + str(int(df[df['grupa'] == 'Age15_17']['total_4'].sum()))
    text5a += '<br>' + str(int(df[df['grupa'] == 'Age18_24']['total_4'].sum()))
    text5a += '<br>' + str(int(df[df['grupa'] == 'Age25_49']['total_4'].sum()))
    text5a += '<br>' + str(int(df[df['grupa'] == 'Age50_59']['total_4'].sum()))
    text5a += '<br>' + str(int(df[df['grupa'] == 'Age60_69']['total_4'].sum()))
    text5a += '<br>' + str(int(df[df['grupa'] == 'Age70_79']['total_4'].sum()))
    text5a += '<br>' + str(int(df[df['grupa'] == 'Age80+']['total_4'].sum()))

    text5b = '%<br>'
    text5b += '<br>' + str(round(df[df['grupa'] == 'Age0_4']['share_4'].sum()*100, 2))+' %'
    text5b += '<br>' + str(round(df[df['grupa'] == 'Age5_9']['share_4'].sum()*100, 2))+' %'
    text5b += '<br>' + str(round(df[df['grupa'] == 'Age10_14']['share_4'].sum()*100, 2))+' %'
    text5b += '<br>' + str(round(df[df['grupa'] == 'Age15_17']['share_4'].sum()*100, 2))+' %'
    text5b += '<br>' + str(round(df[df['grupa'] == 'Age18_24']['share_4'].sum()*100, 2))+' %'
    text5b += '<br>' + str(round(df[df['grupa'] == 'Age25_49']['share_4'].sum()*100, 2))+' %'
    text5b += '<br>' + str(round(df[df['grupa'] == 'Age50_59']['share_4'].sum()*100, 2))+' %'
    text5b += '<br>' + str(round(df[df['grupa'] == 'Age60_69']['share_4'].sum()*100, 2))+' %'
    text5b += '<br>' + str(round(df[df['grupa'] == 'Age70_79']['share_4'].sum()*100, 2))+' %'
    text5b += '<br>' + str(round(df[df['grupa'] == 'Age80+']['share_4'].sum()*100, 2))+' %'
    ###################

    text6a = 'w pełni<br>'
    text6a += '<br>' + str(int(df6[df6['grupa'] == 'Age0_4']['total_12'].sum()))
    text6a += '<br>' + str(int(df6[df6['grupa'] == 'Age5_9']['total_12'].sum()))
    text6a += '<br>' + str(int(df6[df6['grupa'] == 'Age10_14']['total_12'].sum()))
    text6a += '<br>' + str(int(df6[df6['grupa'] == 'Age15_17']['total_12'].sum()))
    text6a += '<br>' + str(int(df6[df6['grupa'] == 'Age18_24']['total_12'].sum()))
    text6a += '<br>' + str(int(df6[df6['grupa'] == 'Age25_49']['total_12'].sum()))
    text6a += '<br>' + str(int(df6[df6['grupa'] == 'Age50_59']['total_12'].sum()))
    text6a += '<br>' + str(int(df6[df6['grupa'] == 'Age60_69']['total_12'].sum()))
    text6a += '<br>' + str(int(df6[df6['grupa'] == 'Age70_79']['total_12'].sum()))
    text6a += '<br>' + str(int(df6[df6['grupa'] == 'Age80+']['total_12'].sum()))

    text6b = '%<br>'
    text6b += '<br>' + str(round(df6[df6['grupa'] == 'Age0_4']['share_12'].sum()*100, 2))+' %'
    text6b += '<br>' + str(round(df6[df6['grupa'] == 'Age5_9']['share_12'].sum()*100, 2))+' %'
    text6b += '<br>' + str(round(df6[df6['grupa'] == 'Age10_14']['share_12'].sum()*100, 2))+' %'
    text6b += '<br>' + str(round(df6[df6['grupa'] == 'Age15_17']['share_12'].sum()*100, 2))+' %'
    text6b += '<br>' + str(round(df6[df6['grupa'] == 'Age18_24']['share_12'].sum()*100, 2))+' %'
    text6b += '<br>' + str(round(df6[df6['grupa'] == 'Age25_49']['share_12'].sum()*100, 2))+' %'
    text6b += '<br>' + str(round(df6[df6['grupa'] == 'Age50_59']['share_12'].sum()*100, 2))+' %'
    text6b += '<br>' + str(round(df6[df6['grupa'] == 'Age60_69']['share_12'].sum()*100, 2))+' %'
    text6b += '<br>' + str(round(df6[df6['grupa'] == 'Age70_79']['share_12'].sum()*100, 2))+' %'
    text6b += '<br>' + str(round(df6[df6['grupa'] == 'Age80+']['share_12'].sum()*100, 2))+' %'

    text7a = 'III dawka<br>'
    text7a += '<br>' + str(int(df6[df6['grupa'] == 'Age0_4']['total_3'].sum()))
    text7a += '<br>' + str(int(df6[df6['grupa'] == 'Age5_9']['total_3'].sum()))
    text7a += '<br>' + str(int(df6[df6['grupa'] == 'Age10_14']['total_3'].sum()))
    text7a += '<br>' + str(int(df6[df6['grupa'] == 'Age15_17']['total_3'].sum()))
    text7a += '<br>' + str(int(df6[df6['grupa'] == 'Age18_24']['total_3'].sum()))
    text7a += '<br>' + str(int(df6[df6['grupa'] == 'Age25_49']['total_3'].sum()))
    text7a += '<br>' + str(int(df6[df6['grupa'] == 'Age50_59']['total_3'].sum()))
    text7a += '<br>' + str(int(df6[df6['grupa'] == 'Age60_69']['total_3'].sum()))
    text7a += '<br>' + str(int(df6[df6['grupa'] == 'Age70_79']['total_3'].sum()))
    text7a += '<br>' + str(int(df6[df6['grupa'] == 'Age80+']['total_3'].sum()))

    text7b = '%<br>'
    text7b += '<br>' + str(round(df6[df6['grupa'] == 'Age0_4']['share_3'].sum()*100, 2))+' %'
    text7b += '<br>' + str(round(df6[df6['grupa'] == 'Age5_9']['share_3'].sum()*100, 2))+' %'
    text7b += '<br>' + str(round(df6[df6['grupa'] == 'Age10_14']['share_3'].sum()*100, 2))+' %'
    text7b += '<br>' + str(round(df6[df6['grupa'] == 'Age15_17']['share_3'].sum()*100, 2))+' %'
    text7b += '<br>' + str(round(df6[df6['grupa'] == 'Age18_24']['share_3'].sum()*100, 2))+' %'
    text7b += '<br>' + str(round(df6[df6['grupa'] == 'Age25_49']['share_3'].sum()*100, 2))+' %'
    text7b += '<br>' + str(round(df6[df6['grupa'] == 'Age50_59']['share_3'].sum()*100, 2))+' %'
    text7b += '<br>' + str(round(df6[df6['grupa'] == 'Age60_69']['share_3'].sum()*100, 2))+' %'
    text7b += '<br>' + str(round(df6[df6['grupa'] == 'Age70_79']['share_3'].sum()*100, 2))+' %'
    text7b += '<br>' + str(round(df6[df6['grupa'] == 'Age80+']['share_3'].sum()*100, 2))+' %'

    text8a = 'IV dawka<br>'
    text8a += '<br>' + str(int(df6[df6['grupa'] == 'Age0_4']['total_4'].sum()))
    text8a += '<br>' + str(int(df6[df6['grupa'] == 'Age5_9']['total_4'].sum()))
    text8a += '<br>' + str(int(df6[df6['grupa'] == 'Age10_14']['total_4'].sum()))
    text8a += '<br>' + str(int(df6[df6['grupa'] == 'Age15_17']['total_4'].sum()))
    text8a += '<br>' + str(int(df6[df6['grupa'] == 'Age18_24']['total_4'].sum()))
    text8a += '<br>' + str(int(df6[df6['grupa'] == 'Age25_49']['total_4'].sum()))
    text8a += '<br>' + str(int(df6[df6['grupa'] == 'Age50_59']['total_4'].sum()))
    text8a += '<br>' + str(int(df6[df6['grupa'] == 'Age60_69']['total_4'].sum()))
    text8a += '<br>' + str(int(df6[df6['grupa'] == 'Age70_79']['total_4'].sum()))
    text8a += '<br>' + str(int(df6[df6['grupa'] == 'Age80+']['total_4'].sum()))

    text8b = '%<br>'
    text8b += '<br>' + str(round(df6[df6['grupa'] == 'Age0_4']['share_4'].sum()*100, 2))+' %'
    text8b += '<br>' + str(round(df6[df6['grupa'] == 'Age5_9']['share_4'].sum()*100, 2))+' %'
    text8b += '<br>' + str(round(df6[df6['grupa'] == 'Age10_14']['share_4'].sum()*100, 2))+' %'
    text8b += '<br>' + str(round(df6[df6['grupa'] == 'Age15_17']['share_4'].sum()*100, 2))+' %'
    text8b += '<br>' + str(round(df6[df6['grupa'] == 'Age18_24']['share_4'].sum()*100, 2))+' %'
    text8b += '<br>' + str(round(df6[df6['grupa'] == 'Age25_49']['share_4'].sum()*100, 2))+' %'
    text8b += '<br>' + str(round(df6[df6['grupa'] == 'Age50_59']['share_4'].sum()*100, 2))+' %'
    text8b += '<br>' + str(round(df6[df6['grupa'] == 'Age60_69']['share_4'].sum()*100, 2))+' %'
    text8b += '<br>' + str(round(df6[df6['grupa'] == 'Age70_79']['share_4'].sum()*100, 2))+' %'
    text8b += '<br>' + str(round(df6[df6['grupa'] == 'Age80+']['share_4'].sum()*100, 2))+' %'

    x = list(ages.keys())
    y_denom = list(df['denom'])
    fig_data = go.Bar(
        x=x,
        y=y_denom,
        base=0,
        offsetgroup=0,
        name='niezaszczepieni'
    )
    traces.append(fig_data)
    y_total_12 = list(df['total_12'])
    fig_data = go.Bar(
        x=x,
        y=y_total_12,
        base=0,
        offsetgroup=0,
        name='w pełni'
    )
    traces.append(fig_data)
    y_total_3 = list(df['total_3'])
    fig_data = go.Bar(
        x=x,
        y=y_total_3,
        base=0,
        offsetgroup=0,
        name='III dawka'
    )
    traces.append(fig_data)
    y_total_4 = list(df['total_4'])
    fig_data = go.Bar(
        x=x,
        y=y_total_4,
        base=0,
        offsetgroup=0,
        name='IV dawka'
    )
    traces.append(fig_data)
    y_total6_12 = list(df6['total_12'])
    fig_data = go.Bar(
        x=x,
        y=y_total6_12,
        base=0,
        offsetgroup=1,
        name='5M w pełni'
    )
    traces.append(fig_data)
    y_total6_3 = list(df6['total_3'])
    fig_data = go.Bar(
        x=x,
        y=y_total6_3,
        base=y_total6_12,
        offsetgroup=1,
        name='5M III dawka'
    )
    traces.append(fig_data)
    y_total6_4 = list(df6['total_4'])
    fig_data = go.Bar(
        x=x,
        y=y_total6_4,
        base=y_total6_3,
        offsetgroup=1,
        name='5M IV dawka'
    )
    traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            # barmode='stack',
            title=dict(text='<b>Spadek efektywnego zaszczepienia populacji Polski (dane ECDC z dnia ' +
                            str(dt.now())[:10] + ')</b><br>' + \
                            '<sub>Wszystkie szczepienia vs. szczepienia wykonane w ciągu ostatnich 5 m-cy<br>'),
            height=float(settings['plot_height']) * 760,
            paper_bgcolor=get_session_color(1),
            plot_bgcolor=get_session_color(1),
            legend=dict(x=0.4, y=1.03, orientation='h')
        )
    )
    figure.add_annotation(text='Wszystkie szczepienia',
                          xref="paper",
                          yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.25,
                          y=0.93,
                          showarrow=False)
    figure.add_annotation(text='Szczepienia w ciągu ostatnich 5 miesięcy',
                          xref="paper",
                          yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.8,
                          y=0.93,
                          showarrow=False)
    figure.add_annotation(text=text_w,
                          xref="paper",
                          yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.01,
                          y=0.85,
                          showarrow=False)
    # populacja
    figure.add_annotation(text=text1,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.07,
                          y=0.85,
                          showarrow=False)
    figure.add_annotation(text=text3a,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.14,
                          y=0.85,
                          showarrow=False)
    figure.add_annotation(text=text3b,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.20,
                          y=0.85,
                          showarrow=False)
    figure.add_annotation(text=text4a,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.26,
                          y=0.85,
                          showarrow=False)
    figure.add_annotation(text=text4b,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.32,
                          y=0.85,
                          showarrow=False)
    figure.add_annotation(text=text5a,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.39,
                          y=0.85,
                          showarrow=False)
    figure.add_annotation(text=text5b,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.45,
                          y=0.85,
                          showarrow=False)

    # 5M

    figure.add_annotation(text=text6a,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.63,
                          y=0.85,
                          showarrow=False)
    figure.add_annotation(text=text6b,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.71,
                          y=0.85,
                          showarrow=False)
    figure.add_annotation(text=text7a,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.77,
                          y=0.85,
                          showarrow=False)
    figure.add_annotation(text=text7b,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.83,
                          y=0.85,
                          showarrow=False)
    figure.add_annotation(text=text8a,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.88,
                          y=0.85,
                          showarrow=False)
    figure.add_annotation(text=text8b,
                          xref="paper",
                          yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.93,
                          y=0.85,
                          showarrow=False)
    figure.add_annotation(text=text_w,
                          xref="paper",
                          yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.98,
                          y=0.85,
                          showarrow=False)
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


###########################################################
#  Bilans efektywnego zaszczepienia populacji Polski ECDC
###########################################################

def layout_more_ecdc_vacc2(_0, _1, _2, _3):
    template = get_template()
    settings = session2settings()
    locations = ['Polska']
    location_txt = locations[0]
    date6 = '2021-11-14'
    ages = {
        'Age0_4': 'wiek 0-4',
        'Age5_9': 'wiek 5-9',
        'Age10_14': 'wiek 10-14',
        'Age15_17': 'wiek 15-17',
        'Age18_24': 'wiek 18-24',
        'Age25_49': 'wiek 25-49',
        'Age50_59': 'wiek 50-59',
        'Age60_69': 'wiek 60-69',
        'Age70_79': 'wiek 70-75',
        'Age80+': 'wiek 80+'}
    ages_sort = {
        'Age0_4': 0,
        'Age5_9': 1,
        'Age10_14': 2,
        'Age15_17': 3,
        'Age18_24': 4,
        'Age25_49': 5,
        'Age50_59': 6,
        'Age60_69': 7,
        'Age70_79': 8,
        'Age80+': 9}
    df = pd.read_csv(constants.data_files['ecdc_vacc']['data_fn'])
    df = df[['date', 'location', 'dawka_1', 'dawka_2', 'dawka_3', 'grupa', 'denom', 'marka']]
    df = df[(df['location'].isin(locations)) & (df['grupa'].isin(ages.keys()))].copy()
    df['grupa_key'] = df['grupa'].map(ages_sort)
    df.sort_values(by=['grupa_key', 'marka'], inplace=True)

    df['dawka_12'] = df['dawka_2']
    df.loc[df.marka == 'JANSS', 'dawka_12'] = df['dawka_1']

    # szczepienia wykonane nie później niż 6 miesięcy temu

    df6 = df[df['date'] > date6].copy()

    df6['total_12'] = df6.groupby(['grupa'])['dawka_12'].transform('sum')
    df6['total_3'] = df6.groupby(['grupa'])['dawka_3'].transform('sum')
    df6['share_12'] = df6['total_12'] / df6['denom']
    df6['share_3'] = df6['total_3'] / df6['denom']
    df6['share_denom'] = 1 - df6['share_12'] - df6['share_3']
    df6['1'] = 1
    df6 = df6.drop_duplicates(subset=['grupa']).copy()
    traces = []

    x = list(ages.keys())
    y_denom = list(df6['1'])
    fig_data = go.Bar(
        x=x,
        y=y_denom,
        text=list(round(df6['share_denom'], 2)),
        textposition='inside',
        base=0,
        offsetgroup=0,
        name='niezaszczepieni'
    )
    traces.append(fig_data)
    y_total_3 = list(df6['share_3'])
    fig_data = go.Bar(
        x=x,
        y=y_total_3,
        text=list(round(df6['share_3'], 2)),
        textposition='outside',
        base=0,
        offsetgroup=0,
        name='III dawka'
    )
    traces.append(fig_data)
    y_total_12 = list(df6['share_12'])
    fig_data = go.Bar(
        x=x,
        y=y_total_12,
        text=list(round(df6['share_12'], 2)),
        base=0,
        textposition='outside',
        offsetgroup=0,
        name='w pełni'
    )
    traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text='<b>Efektywne zaszczepienie w grupach wiekowych (dane ECDC z dnia ' +
                            str(dt.now())[:10] + ')</b><br>' + \
                            '<sub>Szczepienia wykonane w ciągu ostatnich 6 m-cy<br>'),
        height=float(settings['plot_height']) * 760,
            paper_bgcolor=get_session_color(1),
            plot_bgcolor=get_session_color(1),
            legend=dict(x=0.6, y=1.03, orientation='h')
        )
    )
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


############################################
#  Zmienność wieku w przypadkach zachorowań
############################################
def layout_more_mz_cases1(_0, _1, _2, _3):
    template = get_template()
    settings = session2settings()
    from_date = settings['from_date'][:10]
    to_date = settings['to_date'][:10]
    bins = constants.age_bins['bin']
    bins_rev = {bins[x]: x for x in bins}

    df0 = pd.read_csv(constants.data_files['basiw_i']['data_fn'])
    df = df0[df0['date'].between(from_date, to_date)].copy()
    df = df[['date', 'sex', 'age', 'bin', 'i_all']].copy()
    df['sum'] = df.groupby(['date', 'bin'])['i_all'].transform('sum')
    df['sum_day'] = df.groupby(['date'])['i_all'].transform('sum')
    df = df.drop_duplicates(subset=['date', 'bin']).copy()
    traces = []
    df.sort_values(by=['date'], inplace=True)

    i = 0
    if settings['line_dash'] == 'solid':
        dash = 'solid'
    else:
        dash = constants.dashes[random.randint(0, len(constants.dashes) - 1)]
    for bin in bins_rev:
        df_f = df[df['bin'] == bins_rev[bin]].copy()
        df_f['y'] = df_f['sum'] / df_f['sum_day'] * 100
        x = list(df_f['date'])
        y = list(df_f['y'].rolling(7, min_periods=7).mean())
        fig_data = go.Scatter(
            x=x,
            y=y,
            line=dict(width=settings['linewidth_basic'], dash=dash),
            name=bin
        )
        i += 1
        traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text='Udział % grup wiekowych w dziennej liczbie infekcji (średnia 7d)'
                            '<br><sub>Dane: Ministerstwo Zdrowia (BASiW)'),
            height=float(settings['plot_height']) * 760,
            legend=dict(x=1, y=1., orientation='v'),
            yaxis={'type': 'log' if settings['radio_scale'] == 'log' else 'linear'},
        )
    )
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


############################################
#  Statystyka zakażeń w grupach wiekowych
############################################
def layout_more_mz_cases2(_0, _1, _2, _3):
    template = get_template()
    settings = session2settings()
    od_daty = settings['from_date'][:10]
    do_daty = settings['to_date'][:10]
    df = pd.read_csv(constants.data_files['basiw_i']['data_fn'])
    df = df[['date', 'sex', 'wojew', 'wpelni', 'bin', 'bin5', 'bin10', 'bin_ecdc', 'i_all']].copy()


    def wykres(df, df_pop, od_daty, do_daty):
        traces = []
        bin = 'bin5'
        x0 = list(constants.age_bins[bin].values())
        df.sort_values(by=[bin], inplace=True)

        def f(query):
            dfx = df.query(query)
            sum1 = dfx['i_all'].sum()
            dfx['sum_bin_' + bin] = dfx.groupby([bin])['i_all'].transform('sum').drop_duplicates()
            x = [constants.age_bins[bin][i] for i in list(dfx[bin])]
            y = list(dfx['sum_bin_' + bin])
            return x, y, sum1

        xz, yz, sum_z = f("wpelni == 'T'")
        xn, yn, sum_n = f("wpelni == 'N'")

        sum_all = sum_z + sum_n

        anno_text = '<b>Razem: ' + str(int(sum_z+sum_n)) + ' (100%)</b><br><br>' + \
                    'Zaszczepionych: ' + str(int(sum_z)) + ' (' + str(round((sum_z)/(sum_all)*100, 2)) + '%)<br>' + \
                    'Niezaszczepionych: ' + str(int(sum_n)) + ' (' + str(round(sum_n/(sum_all)*100, 2)) + '%)'

        yz1 = []
        yn1 = []
        for i in x0:
            if i in xz:
                yz1.append(yz[xz.index(i)])
            else:
                yz1.append(0)
            if i in xn:
                yn1.append(yn[xn.index(i)])
            else:
                yn1.append(0)

        df_pop['population_5'] = df_pop['population_5m'] + df_pop['population_5k']
        population = df_pop['population_5m'].sum() + df_pop['population_5k'].sum()

        # zaszczepieni

        fig_data = go.Bar(
            x=x0,
            y=yz1,
            name='Zaszczepieni',
        )
        traces.append(fig_data)

        # niezaszczepieni

        fig_data = go.Bar(
            x=x0,
            y=yn1,
            name='Niezaszczepieni',
        )
        traces.append(fig_data)

        # populacja

        fig_data = go.Scatter(
            x=x0,
            y=[np.nan] + list(df_pop['population_5'] / population),
            yaxis='y2',
            name='Udział grupy wiekowe w populacji',
        )
        traces.append(fig_data)

        tytul = 'Statystyka zakażeń w grupach wiekowych ' \
                '(od ' + str(od_daty)[:10] + ' do ' + str(do_daty)[:10] + ')' + \
                '<br><sub>Dane: Ministerstwo Zdrowia (BASiW), GUS'
        figure = go.Figure(
            data=traces,
            layout=dict(
                template=template,
                # barmode='group',
                title=dict(text=tytul),
                height=float(settings['plot_height']) * 760,
                legend=dict(x=0.02, y=1., orientation='h'),
                xaxis=dict(tickmode='array', tickvals=x0, ticktext=x0),
                margin=dict(l=settings['marginl'] + 50, r=settings['marginr'] + 70, b=settings['marginb'] + 50,
                            t=settings['margint'] + 70,
                            pad=4),
                yaxis2=dict(
                    title='% populacji danej płci',
                    anchor="x",
                    overlaying="y",
                    side="right",
                    # range=[0., 1.]
                ),
            )
        )
        figure.add_annotation(text=anno_text,
                              xref="paper", yref="paper",
                              align='left',
                              xanchor='left',
                              font=dict(color=settings['color_4'], size=14),
                              x=0.75, y=0.9,
                              showarrow=False)
        figure = add_copyright(figure, settings)
        return figure

    df = df[df['date'].between(od_daty, do_daty)].copy()

    # dane populacyjne

    df_pop = pd.read_csv('data/dict/slownik_ludnosc_31.12.2020.csv')
    def f5(x):
        for i in range(1, len(constants.age_bins_5)):
            if x >= constants.age_bins_5[i-1] and x < constants.age_bins_5[i]:
                return i - 1
        return 0
    df_pop['bin5'] = df_pop['wiek'].apply(f5)
    df_pop['population_5'] = df_pop.groupby(['bin5'])['population'].transform('sum')
    df_pop['population_5m'] = df_pop.groupby(['bin5'])['population_m'].transform('sum')
    df_pop['population_5k'] = df_pop.groupby(['bin5'])['population_k'].transform('sum')
    df_pop.drop_duplicates(subset=['bin5'], inplace=True)

    figure = wykres(df, df_pop, od_daty, do_daty)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2'),
    ]
    return ret_val


##################################################
#  Liczba przypadków według płci
##################################################

def layout_more_mz_cases3(_0, _age, _2, _3):
    template = get_template()
    settings = session2settings()
    bins = {'M': 'mężczyźni', 'K': 'kobiety'}
    ages = {'wszyscy': 0, 'b.d.': 1, '<12': 2, '12-19': 3, '20-39': 4, '40-59': 5,
            '60-69': 6, '70+': 7}
    df = pd.read_csv(constants.data_files['basiw_i']['data_fn'])
    age = ages[_age]
    if age != 0:
        df = df[df['bin'] == age].copy()
    df = df[df['sex'].isin(['M', 'K'])].copy()
    df = df[df['date'] >= settings['from_date']].copy()
    df['sum'] = df.groupby(['date', 'sex'])['i_all'].transform('sum')
    df = df.drop_duplicates(subset=['date', 'sex']).copy()
    traces = []
    df.sort_values(by=['date'], inplace=True)

    for bin in bins.keys():
        df_f = df[df['sex'] == bin].copy()
        df_f['y'] = df_f['sum']
        x = list(df_f['date'])
        y = list(df_f['y'].rolling(7, min_periods=7).mean())
        fig_data = go.Scatter(
            x=x,
            y=y,
            line=dict(shape='linear', dash='solid', width=settings['linewidth_basic']),
            name=bins[bin]
        )
        traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text='Dzienne liczby potwierdzonych infekcji według płci (średnia 7d)'
                            '<br>' + _age +
                            '<br><sub>Dane: Ministerstwo Zdrowia'),
            height=float(settings['plot_height']) * 660,
            legend=dict(x=1, y=1., orientation='v'),
            yaxis={'type': 'log' if settings['radio_scale'] == 'log' else 'linear'},
        )
    )
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


######################################
#  Porównanie zakażeń BASiW i MZ
######################################

def layout_more_mz_cases4(DF, _1, _2, _3):
    template = get_template()
    settings = session2settings()

    df_basiw_d = pd.read_csv(constants.data_files['basiw_d']['data_fn'])
    df_basiw_d = df_basiw_d[['date', 'd_all', 'wojew']].copy()
    df_basiw_d = df_basiw_d[df_basiw_d['date'] >= '2021-01-02'].copy()
    df_basiw_d.sort_values(by=['date'], inplace=True)
    df_basiw_d['new_deaths_b'] = df_basiw_d.groupby(['date'])['d_all'].transform('sum')
    df_basiw_d.drop_duplicates(subset=['date'], inplace=True)
    df_basiw_d['total_deaths_b'] = df_basiw_d['new_deaths_b'].cumsum()

    df_basiw_i = pd.read_csv(constants.data_files['basiw_i']['data_fn'])
    df_basiw_i = df_basiw_i[['date', 'i_all', 'wojew']].copy()
    df_basiw_i = df_basiw_i[df_basiw_i['date'] >= '2021-01-02'].copy()
    df_basiw_i.sort_values(by=['date'], inplace=True)
    df_basiw_i['new_cases_b'] = df_basiw_i.groupby(['date'])['i_all'].transform('sum')
    df_basiw_i.drop_duplicates(subset=['date'], inplace=True)
    df_basiw_i['total_cases_b'] = df_basiw_i['new_cases_b'].cumsum()

    date_max = max(df_basiw_i['date'])

    df_mz = DF['world']
    df_mz = df_mz[['date', 'new_cases', 'new_deaths', 'location']].copy()
    df_mz = df_mz[df_mz['location'] == 'Polska'].copy()
    df_mz = df_mz[df_mz['date'] >= '2021-01-02'].copy()
    df_mz = df_mz[df_mz['date'] <= date_max].copy()
    df_mz['total_cases'] = df_mz['new_cases'].cumsum()
    df_mz['total_deaths'] = df_mz['new_deaths'].cumsum()

    df = pd.merge(df_mz, df_basiw_d[['date', 'new_deaths_b', 'total_deaths_b']], how='left',
                  left_on=['date'], right_on=['date'])
    df = pd.merge(df, df_basiw_i[['date', 'new_cases_b', 'total_cases_b']], how='left',
                  left_on=['date'], right_on=['date']).copy()

    traces1 = []
    anno_text = '<b>Podsumowanie</b>'
    x = list(df['date'])
    y1 = list(df['new_cases'].rolling(7, min_periods=7).mean())
    y2 = list(df['new_cases_b'].rolling(7, min_periods=7).mean())
    y3 = list(df['new_deaths'].rolling(7, min_periods=7).mean())
    y4 = list(df['new_deaths_b'].rolling(7, min_periods=7).mean())
    y5 = list(df['total_deaths'])
    y6 = list(df['total_deaths_b'])
    y7 = list(df['total_cases'])
    y8 = list(df['total_cases_b'])

    def wykres(x, y1, y2, name1, name2):
        traces = []
        fig_data = go.Scatter(
            x=x,
            y=y1,
            name=name1,
        )
        traces.append(fig_data)
        fig_data = go.Scatter(
            x=x,
            y=y2,
            name=name2,
        )
        traces.append(fig_data)
        tytul = name1 + ' vs ' + name2 + '<br><sub>Dane: Ministerstwo Zdrowia, BASiW'
        figure = go.Figure(
            data=traces,
            layout=dict(
                template=template,
                title=dict(text=tytul),
                height=float(settings['plot_height']) * 380,
                legend=dict(x=0.02, y=1., orientation='h'),
                # xaxis=dict(tickmode='array', tickvals=x0, ticktext=x0)
            )
        )
        figure.add_annotation(text=anno_text,
                              xref="paper", yref="paper",
                              align='left',
                              xanchor='left',
                              font=dict(color=settings['color_4'], size=14),
                              x=0.6, y=0.9,
                              showarrow=False)
        figure = add_copyright(figure, settings)
        return figure

    figure_1 = wykres(x, y1, y2, 'Infekcje dzienne MZ', 'Infekcje dzienne BASiW')
    figure_2 = wykres(x, y3, y4, 'Zgony dzienne MZ', 'Zgony dzienne BASiW')
    figure_3 = wykres(x, y5, y6, 'Infekcje razem MZ', 'Infekcje razem BASiW')
    figure_4 = wykres(x, y7, y8, 'Zgony razem MZ', 'Zgony razem BASiW')
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig1 = dcc.Graph(id='xxxx', figure=figure_1, config=config)
    fig2 = dcc.Graph(id='xxxx', figure=figure_2, config=config)
    fig3 = dcc.Graph(id='xxxx', figure=figure_3, config=config)
    fig4 = dcc.Graph(id='xxxx', figure=figure_4, config=config)
    ret_val = [
        dbc.Row([
            dbc.Col(fig1, width=6, className='mb-2 mt-2'),
            dbc.Col(fig2, width=6, className='mb-2 mt-2')
        ]),
        dbc.Row([
            dbc.Col(fig3, width=6, className='mb-2 mt-2'),
            dbc.Col(fig4, width=6, className='mb-2 mt-2')
        ]),
    ]
    return ret_val


###################################################################
#  Piramida wieku populacji, zachorowań i przypadków śmiertelnych
###################################################################

def layout_more_mz_cases5(_0, _rodzaj, _status, _3):
    template = get_template()
    settings = session2settings()
    rodzaj = _rodzaj if _rodzaj != 'wszystko' else 'populacja, infekcje, zgony'
    colors = ['green', 'gray', 'red']

    if settings['from_date'] is None:
        return
    if settings['to_date'] is None:
        return
    oddaty = str(settings['from_date'])[:10]
    dodaty = str(settings['to_date'])[:10]
    print(oddaty, dodaty)

    # infekcje

    df = pd.read_csv(constants.data_files['basiw_i']['data_fn'])
    df.sort_values(by=['sex', 'age'], inplace=True)
    if _status == 'w pełni zaszczepieni':
        df = df[df['wpelni'] == 'T'].copy()
    elif _status == 'niezaszczepieni':
        df = df[df['wpelni'] == 'N'].copy()
    df = df[df['sex'].isin(['M', 'K'])].copy()
    df = df[df['bin'] > 0].copy()
    df = df[df['age'] < 90].copy()
    df = df[df['date'] >= oddaty].copy()
    df = df[df['date'] <= dodaty].copy()
    df['sum'] = df.groupby(['sex'])['i_all'].transform('sum')
    df['sum_wiek'] = df.groupby(['sex', 'age'])['i_all'].transform('sum')
    df.drop_duplicates(subset=['sex', 'age'], inplace=True)
    df.sort_values(by=['sex', 'age'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['wsk_i'] = df['sum_wiek'] / df['sum'] * 100
    df = df[['sex', 'age', 'wsk_i']].copy()
    df.rename(columns={'sex': 'plec'}, inplace=True)
    df.rename(columns={'age': 'wiek'}, inplace=True)

    # dane basiw o zgonach

    df_d = pd.read_csv(constants.data_files['basiw_d']['data_raw_fn'])
    if _status == 'w pełni zaszczepieni':
        df_d = df_d[df_d['wpelni'] == 'T'].copy()
    elif _status == 'niezaszczepieni':
        df_d = df_d[df_d['wpelni'] == 'N'].copy()
    df_d = df_d[df_d['date'] >= oddaty].copy()
    df_d = df_d[df_d['date'] <= dodaty].copy()
    df_d = df_d[['date', 'd_all', 'age', 'sex']].copy()
    df_d.columns = ['date', 'deaths', 'wiek', 'plec']
    df_d = df_d[df_d['plec'] != 'b.d.'].copy()
    df_d['deaths_all'] = df_d.groupby(['plec'])['deaths'].transform('sum')
    df_d['deaths_sex'] = df_d.groupby(['plec', 'wiek'])['deaths'].transform('sum')
    df_d['wsk_d'] = df_d['deaths_sex'] / df_d['deaths_all'] * 100
    df_d.drop_duplicates(subset=['plec', 'wiek'], inplace=True)
    df_d.sort_values(by=['plec', 'wiek'], inplace=True)
    df_d.reset_index(drop=True, inplace=True)
    df_d = df_d[['plec', 'wiek', 'wsk_d']].copy()
    df = pd.merge(df, df_d, how='left', left_on=['plec', 'wiek'], right_on=['plec', 'wiek']).copy()

    # dane populacyjne

    df_pop_m = pd.read_csv('data/dict/slownik_ludnosc_31.12.2020.csv')
    df_pop_k = df_pop_m.copy()
    df_pop_m['population_all'] = df_pop_m['population_m'].sum()
    df_pop_m['wsk_p'] = df_pop_m['population_m'] / df_pop_m['population_all'] * 100
    df_pop_m = df_pop_m[['wiek', 'wsk_p']].copy()
    df_pop_m['plec'] = 'M'
    df_pop_k['population_all'] = df_pop_k['population_k'].sum()
    df_pop_k['wsk_p'] = df_pop_k['population_k'] / df_pop_k['population_all'] * 100
    df_pop_k = df_pop_k[['wiek', 'wsk_p']].copy()
    df_pop_k['plec'] = 'K'
    df_pop = df_pop_m.append(df_pop_k, ignore_index=True).copy()
    df = pd.merge(df, df_pop, how='left', left_on=['wiek', 'plec'], right_on=['wiek', 'plec']).copy()
    df['wsk_p'].fillna(0, inplace=True)
    df['wsk_d'].fillna(0, inplace=True)
    df['wsk_i'].fillna(0, inplace=True)

    df = df[df['wiek'] >= 0].copy()

    bins_i_k = list(df[df['plec'] == 'K']['wsk_i'])
    bins_i_m = list(df[df['plec'] == 'M']['wsk_i'])
    bins_d_k = list(df[df['plec'] == 'K']['wsk_d'])
    bins_d_m = list(df[df['plec'] == 'M']['wsk_d'])
    bins_p_k = list(df[df['plec'] == 'K']['wsk_p'])
    bins_p_m = list(df[df['plec'] == 'M']['wsk_p'])

    y = list(df['wiek'].unique())

    figure = go.Figure()

    # wykres I

    if 'infekcje' in _rodzaj:
        figure.add_trace(go.Bar(
            y=y,
            x=bins_i_m,
            orientation='h',
            name='Mężczyźni infekcje',
            marker=dict(color=colors[0], opacity=0.3, line=dict(color=colors[0], width=0.3)),
        ))
        figure.add_trace(go.Scatter(
            y=y,
            x=bins_i_m,
            orientation='h',
            name='',
            line=dict(color=colors[0], width=4),
        ))
        figure.add_trace(go.Bar(
            y=y,
            x=[-1 * x for x in bins_i_k],
            orientation='h',
            name='Kobiety infekcje',
            marker=dict(color=colors[0], opacity=0.3, line=dict(color=colors[0], width=0.5)),
        ))
        figure.add_trace(go.Scatter(
            y=y,
            x=[-1 * x for x in bins_i_k],
            orientation='h',
            name='',
            line=dict(color=colors[0], width=4),
        ))

    # wykres P

    if 'populacja' in _rodzaj:
        figure.add_trace(go.Bar(
            y=y,
            x=bins_p_m,
            orientation='h',
            name='Mężczyźni populacja',
            opacity=1,
            marker=dict(color=colors[1], opacity=0.3, line=dict(color=colors[1], width=0.3)),
        ))
        figure.add_trace(go.Scatter(
            y=y,
            x=bins_p_m,
            orientation='h',
            name='',
            opacity=1,
            line=dict(color=colors[1], width=4),
        ))
        figure.add_trace(go.Bar(
            y=y,
            x=[-1 * x for x in bins_p_k],
            orientation='h',
            name='Kobiety populacja',
            opacity=1,
            marker=dict(color=colors[1], opacity=0.3, line=dict(color=colors[1], width=0.5)),
        ))
        figure.add_trace(go.Scatter(
            y=y,
            x=[-1 * x for x in bins_p_k],
            orientation='h',
            name='',
            opacity=1,
            line=dict(color=colors[1], width=4),
        ))

    # wykres D

    if 'zgony' in _rodzaj:
        figure.add_trace(go.Bar(
            y=y,
            x=bins_d_m,
            orientation='h',
            name='Mężczyźni zgony',
            opacity=1,
            marker=dict(color=colors[2], opacity=0.3, line=dict(color=colors[2], width=0.3)),
        ))
        figure.add_trace(go.Scatter(
            y=y,
            x=bins_d_m,
            orientation='h',
            name='',
            opacity=1,
            line=dict(color=colors[2], width=4),
        ))
        figure.add_trace(go.Bar(
            y=y,
            x=[-1 * x for x in bins_d_k],
            orientation='h',
            name='Kobiety zgony',
            opacity=1,
            marker=dict(color=colors[2], opacity=0.3, line=dict(color=colors[2], width=0.5)),
        ))
        figure.add_trace(go.Scatter(
            y=y,
            x=[-1 * x for x in bins_d_k],
            orientation='h',
            name='',
            opacity=1,
            line=dict(color=colors[2], width=4),
        ))
    figure.add_annotation(text='Kobiety',
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'], size=28),
                          x=0.01, y=0.1,
                          showarrow=False)
    figure.add_annotation(text='Mężczyźni',
                          xref="paper", yref="paper",
                          align='right',
                          font=dict(color=settings['color_4'], size=28),
                          x=0.99, y=0.1,
                          showarrow=False)
    figure.update_layout(
        title='Piramida wieku: ' + rodzaj + ', status szczepienia: ' + _status + \
              '<br><sub>od ' + settings['from_date'][:10] + ' do ' + settings['to_date'][:10],
        template=template,
        yaxis=dict(title=dict(text='Wiek',
                              font=dict(
                                  size=int(settings['font_size_xy']),
                                  color=get_session_color(7)
                              )),
                   ),
        xaxis=dict(title=dict(text='Udział %',
                              font=dict(
                                  size=int(settings['font_size_xy']),
                                  color=get_session_color(7)
                              )),
                   ),
        barmode='overlay',
        height=float(settings['plot_height']) * 660,
        bargap=0
    )
    figure = add_copyright(figure, settings)
    config = {
        'displaylogo': False,
        'responsive': True,
        'toImageButtonOptions': {
            'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
            'filename': 'docent_image', 'height': 700, 'width': 1200}
    }
    ret_val = [
        dbc.Col(dcc.Graph(id='pyramid', figure=figure, config=config), width=12, className='mb-2 mt-2')
    ]
    return ret_val


#######################################################################################
#  Mediana i średni wiek infekcji
########################################################################################

def layout_more_mz_cases6(_0, _1, _2, _3):
    template = get_template()
    settings = session2settings()
    df = pd.read_csv(constants.data_files['basiw_i']['data_fn'])
    df = df[['date', 'sex', 'age', 'i_all']].copy()
    df = df[df['sex'].isin(['M', 'K'])].copy()
    df = df[df['age'] != -1].copy()
    df.sort_values(by=['date', 'sex'], inplace=True)
    df['suma_lat'] = df['age'] * df['i_all']
    df['sum_day'] = df.groupby(['date', 'sex'])['suma_lat'].transform('sum')
    df['count_day'] = df.groupby(['date', 'sex'])['i_all'].transform('sum')
    df = df.drop_duplicates(subset=['date', 'sex']).copy()
    df['sredni wiek'] = round(df['sum_day'] / df['count_day'], 2)

    dfx = df[df['sex'] == 'K'].copy()
    x = list(dfx['date'])
    y = list(dfx['sredni wiek'].rolling(7, min_periods=1).mean())
    traces = []
    fig_data = go.Scatter(
        x=x,
        y=y,
        line=dict(width=4),
        name='kobiety'
    )
    traces.append(fig_data)

    dfx = df[df['sex'] == 'M'].copy()
    # x = list(dfx['date'])
    y = list(dfx['sredni wiek'].rolling(7, min_periods=1).mean())
    fig_data = go.Scatter(
        x=x,
        y=y,
        line=dict(width=4),
        name='mężczyźni'
    )
    traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text='Średnia dzienna wieku osób zakażonych SARS-CoV-2 w Polsce<br><sub>(na podstawie danych MZ)'),
            height=float(settings['plot_height']) * 660,
            legend=dict(x=1, y=1., orientation='v')
        )
    )
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


##############################################
#  Zmienność wieku w przypadkach zachorowań 2
##############################################

def layout_more_mz_cases7(_0, _plec, _2, _3):
    template = get_template()
    settings = session2settings()

    bins = constants.age_bins['bin']
    # bins = {1: 'b.d.', 2: '<12', 3: '12-19', 4: '20-39', 5: '40-59', 6: '60-69', 7: '70+'}
    df = pd.read_csv(constants.data_files['basiw_i']['data_fn'])

    plci = {'mężczyźni': 'M', 'kobiety': 'K', 'wszyscy': 'A'}
    plec = plci[_plec]
    if plec != 'A':
        df = df[df['sex'] == plec].copy()
    df.sort_values(by=['bin', 'date'], inplace=True)
    df['sum_bin'] = df.groupby(['bin', 'date'])['i_all'].transform('sum')
    df = df.drop_duplicates(subset=['bin', 'date']).copy()
    df = df[['date', 'bin', 'sum_bin']].copy()

    # resample week

    df.index = pd.to_datetime(df['date'])
    df = df.groupby('bin')['sum_bin'].resample('W').sum()
    df = df.reset_index()
    # df['date'] = df.index
    # df.reset_index(drop=True, inplace=True)

    df['sum_day'] = df.groupby(['date'])['sum_bin'].transform('sum')
    df['wynik'] = (df['sum_bin'] / df['sum_day']) * 100
    traces = []

    i =0
    for bin in bins.keys():
        dfx = df[df['bin'] == bin].copy()
        dfx['wynik'] = dfx['wynik']
        x = list(dfx['date'])
        y = list(dfx['wynik'])
        fig_data = go.Bar(
            x=x,
            y=y,
            name=bins[bin]
        )
        traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            barmode='stack',
            bargap=0.3,
            title=dict(text='Udział procentowy grup wiekowych w zachorowaniach - tygodniowo' +
                            '<br>(dane MZ z dnia ' + str(dt.today())[:10] + ')'),
            height=float(settings['plot_height']) * 660,
            legend=dict(x=0, y=1.02, orientation='h')
        )
    )
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


##################################################
#  Heatmap Liczba przypadków w grupach wiekowych
##################################################

def layout_more_mz_cases8(_0, _plec, _typ, _3):
    template = get_template()
    settings = session2settings()
    from_date = settings['from_date']
    to_date = settings['to_date']
    ages = {x: constants.ages_5_pop[x][0] for x in constants.ages_5_pop}
    df = pd.read_csv(constants.data_files['basiw_i']['data_fn'])
    df = df[df['sex'].isin(['M', 'K'])].copy()
    df = df[df['bin5'] > 1].copy()
    df['sum'] = df.groupby(['date', 'bin5'])['i_all'].transform('sum')
    df['sum_day'] = df.groupby(['date'])['i_all'].transform('sum')
    df = df.drop_duplicates(subset=['date', 'bin5']).copy()
    if _typ == 'brutto':
        df['liczba'] = df['sum'] / df['sum_day']
        title_1 = "Udział zakażeń w grupach wiekowych w dziennej liczbie potwierdzonych zakażeń"
    else:
        title_1 = "Liczba potwierdzonych zakażeń w grupach wiekowych na 100k populacji grupy"
        def f(r):
            r['liczba'] = r['sum'] / constants.ages_5_pop[r['bin5']][1] * 100000
            return r
        df = df.apply(f, axis=1)
    df.sort_values(by=['bin5', 'date'], inplace=True)
    df = df[df['date'].between(from_date, to_date)].copy()
    df.reset_index(drop=True, inplace=True)

    df['liczba'] = df['liczba'].rolling(7, min_periods=1).mean()
    title = title_1 + "<br><sub>Dane MZ"
    fig = heatmap(df, 'bin5', 'date', 'liczba', settings, template, ages, title, gapx_0=0, gapy_0=0)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


##################################################
#  Dzienne infekcje Covid-19 w grupach wiekowych
##################################################

def layout_more_mz_cases10(_0, _typ, _age, _3):
    template = get_template()
    settings = session2settings()
    # dane GUS 31.12.2020
    bins0 = {
        3: ['12-19', 0.4365, 3011688, 'red'],
        4: ['20-39', 0.5229, 10418299, 'green'],
        5: ['40-59', 0.6523, 10373837, 'blue'],
        6: ['60-69', 0.735, 5185843, 'orange'],
        7: ['70+', 0.8104, 4614390, 'purple']
    }
    bins2 = {'12-19': 3, '20-39': 4, '40-59': 5, '60-69': 6, '70+': 7}
    df = pd.read_csv(constants.data_files['basiw_i']['data_fn'])
    if _age == 'wszyscy':
        bins = bins0
    else:
        bins = {bins2[_age]: bins0[bins2[_age]]}
    if _typ == 'brutto':
        tytul = 'Dzienne zakażenia Covid-19 w grupach wiekowych<br><sub>w podziale na zaszczepionych / niezaszczepionych'
    else: # na 100k
        tytul = 'Dzienne zakażenia Covid-19 w grupach wiekowych na 100k populacji grupy wiekowej' \
                '<br><sub>w podziale na zaszczepionych / niezaszczepionych'
    traces = []
    for bin in bins.keys():

        # niezaszczepieni

        dfx = df[(df['bin_ecdc'] == bin) & (df['wpelni'] != 'T')].copy()
        dfx['total'] = dfx.groupby(['date', 'bin'])['i_all'].transform('sum')
        dfx = dfx.drop_duplicates(subset=['date', 'bin']).copy()
        dfx.sort_values(by=['date'], inplace=True)
        if _typ == 'brutto':
            dfx['data'] = dfx['total']
        elif _typ == 'na 100k':
            dfx['data'] = dfx['total'] / bins[bin][2] * 100000
        else:
            dfx['data'] = dfx['total'] / bins[bin][2] * 100000
        x = list(dfx['date'])
        y = list(dfx['data'].rolling(settings['average_days'], min_periods=1).mean())
        fig_data = go.Scatter(
            x=x,
            y=y,
            name=bins[bin][0] + ' niezaszczepieni',
            line=dict(color=bins[bin][3], width=settings['linewidth_basic'], dash='solid')
        )
        traces.append(fig_data)

        # zaszczepieni

        dfx = df[(df['bin'] == bin) & (df['wpelni'] == 'T')].copy()
        dfx['total'] = dfx.groupby(['date', 'bin'])['i_all'].transform('sum')
        dfx = dfx.drop_duplicates(subset=['date', 'bin']).copy()
        dfx.sort_values(by=['date'], inplace=True)
        if _typ == 'brutto':
            dfx['data'] = dfx['total']
        elif _typ == 'na 100k':
            dfx['data'] = dfx['total'] / bins[bin][2] * bins[bin][1] * 100000
        else:
            dfx['data'] = dfx['total'] / bins[bin][2] * bins[bin][1] * 100000
        x = list(dfx['date'])
        y = list(dfx['data'].rolling(settings['average_days'], min_periods=1).mean())
        fig_data = go.Scatter(
            x=x,
            y=y,
            name=bins[bin][0] + ' zaszczepieni',
            line=dict(color=bins[bin][3], width=settings['linewidth_basic'], dash='dash')
        )
        traces.append(fig_data)

    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text=tytul),
            height=float(settings['plot_height']) * 760,
            legend=dict(x=1.02, y=1., orientation='v')
        )
    )
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


##########################################################################
#  Dzienne infekcje Covid-19 w grupach zaszczepionych i niezaszczepionych
##########################################################################

def layout_more_mz_cases11(_0, _age, _2, _3):
    template = get_template()
    settings = session2settings()
    # dane GUS 31.12.2020
    bins = {
        '0-4':   [1, 1902236, 'Age0_4'],
        '5-9':   [2, 1910470, 'Age5_9'],
        '10-14': [3, 2065628, 'Age10_14'],
        '15-17': [4, 1075305, 'Age15_17'],
        '18-24': [5, 2688690, 'Age18_24'],
        '25-49': [6, 14216985, 'Age25_49'],
        '50-59': [7, 4605466, 'Age50_59'],
        '60-69': [8, 5185843, 'Age60_69'],
        '70-79': [9, 2930420, 'Age70_79'],
        '80+': [10, 1683970, 'Age80+']
    }

    # szczepienia ECDC

    df_v = pd.read_csv(constants.data_files['ecdc_vacc']['data_fn'])
    df_v = df_v[['date', 'location', 'dawka_1', 'dawka_2', 'grupa', 'denom', 'marka']]
    df_v = df_v[df_v['grupa'] == bins[_age][2]].copy()
    df_v = df_v[df_v['location'] == 'Polska'].copy()

    df_v['full_brand'] = df_v['dawka_2']
    df_v.loc[df_v.marka == 'JANSS', 'full_brand'] = df_v['dawka_1']

    # def f(r):
    #     if r.marka == 'JANSS':
    #         r['full_brand'] = r['dawka_1']
    #     else:
    #         r['full_brand'] = r['dawka_2']
    #     return r
    # df_v = df_v.apply(f, axis=1)
    df_v['full'] = df_v.groupby(['date'])['full_brand'].transform('sum')
    df_v = df_v.drop_duplicates(subset=['date']).copy()

    # resample day

    df_v.index = pd.to_datetime(df_v['date'], format='%Y-%m-%d')
    df_v = df_v.resample('D').mean().fillna(method='ffill')
    df_v['full'] = df_v['full'] / 7
    df_v['date'] = df_v.index.astype(str)
    df_v = df_v[['date', 'full']].copy()
    df_v.reset_index(drop=True, inplace=True)

    df_v['full_cum'] = df_v['full'].cumsum()

    df = pd.read_csv(constants.data_files['basiw_i']['data_fn'])
    df = df[df['bin_ecdc'] == bins[_age][0]].copy()

    df['total'] = df.groupby(['date', 'bin_ecdc', 'wpelni'])['i_all'].transform('sum')
    df = df.drop_duplicates(subset=['date', 'bin_ecdc', 'wpelni']).copy()
    df.sort_values(by=['date'], inplace=True)
    df = pd.merge(df, df_v, how='left', left_on=['date'], right_on=['date']).copy()

    traces = []

    # niezaszczepieni

    dfx = df[df['wpelni'] == 'N'].copy()

    # ułamek niezaszczepionych w populacji wieku

    dfx['population'] = bins[_age][1]
    dfx['wsp_n'] = (dfx['population'] - dfx['full_cum']) / dfx['population']
    dfx['total'] = dfx['total'].rolling(settings['average_days'], min_periods=1).mean()
    dfx['data'] = dfx['total'] / (dfx['population'] * dfx['wsp_n']) * 100000
    xn = list(dfx['date'])
    yn = list(dfx['data'])
    fig_data = go.Scatter(
        x=xn,
        y=yn,
        name=_age + ' niezaszczepieni',
        line=dict(width=settings['linewidth_basic'], dash='solid')
    )
    traces.append(fig_data)

    # zaszczepieni

    dfx = df[df['wpelni'] == 'T'].copy()

    # ułamek zaszczepionych w populacji wieku

    dfx['population'] = bins[_age][1]
    dfx['wsp_t'] = dfx['full_cum'] / dfx['population']
    dfx['total'] = dfx['total'].rolling(settings['average_days'], min_periods=1).mean()
    dfx['data'] = dfx['total'] / (dfx['population'] * dfx['wsp_t']) * 100000
    # dfx['data'].fillna(0, inplace=True)

    xz = list(dfx['date'])
    yz = list(dfx['data'])
    fig_data = go.Scatter(
        x=xz,
        y=yz,
        name=_age + ' zaszczepieni',
        line=dict(width=settings['linewidth_basic'], dash='dash')
    )
    traces.append(fig_data)

    # procent zaszczepienia

    wz = list(dfx['wsp_t'])
    fig_data = go.Scatter(
        x=xz,
        y=wz,
        name=_age + ' % zaszczepienia',
        line=dict(width=1., dash='dot'),
        yaxis='y2'
    )
    traces.append(fig_data)


    dn = {xn[i]: yn[i] for i in range(len(xn))}
    dz = {xz[i]: yz[i] for i in range(len(xz))}

    sum_n = 0
    sum_z = 0
    sum_all = 0
    for i in dz:
        if i in dn:
            part_n = dn[i] if not np.isnan(dn[i]) else 0
            part_z = dz[i] if not np.isnan(dz[i]) else 0
            sum_n += part_n
            sum_z += part_z
            sum_all += part_n + part_z
    anno_text = 'Grupa wiekowa: ' + _age + \
        '<br><br>Liczba zakażeń razem / 100k: ' + str(int(sum_all)) + \
        '<br>- zakażenia spośród osób zaszczepionych / 100k: : ' + str(int(sum_z)) + ' (' + str(round(sum_z/sum_all*100, 2)) + '%)' + \
        '<br>- zakażenia spośród osób niezaszczepionych / 100k: : ' + str(int(sum_n)) + ' (' + str(round(sum_n/sum_all*100, 2)) + '%)'


    tytul = 'Dzienne zakażenia Covid-19 / 100k w grupach zaszczepionych / niezaszczepionych' \
            '<br>Grupa wiekowa: '+ _age + \
            '<br><sub>(z uwzględnieniem bieżącego wskaźnika zaszczepienia)<br>Dane: ECDC, MZ (BASIW)'
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text=tytul),
            height=float(settings['plot_height']) * 760,
            legend=dict(x=1.02, y=1., orientation='v'),
            yaxis2=dict(
                title='',
                anchor="x",
                overlaying="y",
                side="right",
                range=[0., 1.]
            ),
        )
    )
    figure = add_copyright(figure, settings)
    figure.add_annotation(text=anno_text,
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=.75, y=0.8,
                          # x=0.01, y=0.8,
                          showarrow=False)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


####################################
#  Statystyka wg statusu odporności
####################################
def layout_more_mz_cases12(_0, _oddaty, _mnoznik, _3):
    template = get_template()
    settings = session2settings()
    oddaty = str(_oddaty)[:10]
    oddaty = '2021-01-01'
    mnoznik = int(_mnoznik)

    ages_labels = constants.age_bins['bin_ecdc']
    ages_ecdc = constants.ages_ecdc
    x0 = list(ages_labels.keys())
    x0l = list(ages_labels.values())

    # dane o infekcjach

    df_i = pd.read_csv(constants.data_files['basiw_i']['data_fn'])
    df_i = df_i[['date', 'i_all', 'bin_ecdc', 'dawka', 'wpelni']].copy()
    df_i = df_i[df_i['date'] >= oddaty].copy()
    df_i['dawka'].fillna('brak', inplace=True)
    df_i['total_vacc0'] = df_i.groupby(['bin_ecdc', 'dawka', 'wpelni'])['i_all'].transform('sum')
    df_i.drop_duplicates(subset=['bin_ecdc', 'dawka', 'wpelni'], inplace=True)
    df_i['status_szczepienia'] = -1
    df_i.loc[df_i['dawka'] == 'przypominajaca', 'status_szczepienia'] = 3
    df_i.loc[df_i['dawka'] == 'uzupelniajaca', 'status_szczepienia'] = 2
    df_i.loc[df_i['dawka'] == 'pelna_dawka', 'status_szczepienia'] = 2
    df_i.loc[df_i['dawka'] == 'jedna_dawka', 'status_szczepienia'] = 1
    df_i.loc[df_i['dawka'] == 'brak', 'status_szczepienia'] = 0
    df_i['total_vacc'] = df_i.groupby(['bin_ecdc', 'status_szczepienia'])['total_vacc0'].transform('sum')
    df_i['total_vacc'] = df_i['total_vacc'] * mnoznik
    df_i.sort_values(by=['bin_ecdc'], inplace=True)
    df_i.drop_duplicates(subset=['bin_ecdc', 'status_szczepienia'], inplace=True)

    del df_i['i_all']
    del df_i['dawka']
    del df_i['wpelni']
    del df_i['date']
    del df_i['total_vacc0']

    def f_status(r):
        if r['status_szczepienia'] == 0:
            r['n_status_00'] = r['total_vacc']
        if r['status_szczepienia'] == 1:
            r['n_status_10'] = r['total_vacc']
        if r['status_szczepienia'] == 2:
            r['n_status_20'] = r['total_vacc']
        if r['status_szczepienia'] == 3:
            r['n_status_30'] = r['total_vacc']
        return r
    df_i = df_i.apply(f_status, axis=1)
    df_i.fillna(0, inplace=True)
    df_i.sort_values(by=['bin_ecdc'], inplace=True)
    df_i['n_status_0'] = df_i.groupby(['bin_ecdc'])['n_status_00'].transform('sum')
    df_i['n_status_1'] = df_i.groupby(['bin_ecdc'])['n_status_10'].transform('sum')
    df_i['n_status_2'] = df_i.groupby(['bin_ecdc'])['n_status_20'].transform('sum')
    df_i['n_status_3'] = df_i.groupby(['bin_ecdc'])['n_status_30'].transform('sum')
    df_i.drop_duplicates(subset=['bin_ecdc'], inplace=True)
    del df_i['n_status_00']
    del df_i['n_status_10']
    del df_i['n_status_20']
    del df_i['n_status_30']
    del df_i['status_szczepienia']
    del df_i['total_vacc']

    # dane populacyjne

    df_pop = pd.read_csv('data/dict/slownik_ludnosc_31.12.2020.csv')
    def f_ecdc(x):
        for i in range(1, len(constants.age_bins_ecdc)):
            if x >= constants.age_bins_ecdc[i-1] and x < constants.age_bins_ecdc[i]:
                return i - 1
        return 0
    df_pop['bin_ecdc'] = df_pop['wiek'].apply(f_ecdc)
    df_pop['population_ecdc'] = df_pop.groupby(['bin_ecdc'])['population'].transform('sum')
    df_pop.drop_duplicates(subset=['bin_ecdc'], inplace=True)
    df_pop = df_pop[['bin_ecdc', 'population_ecdc']]

    # dane o szczepieniach

    df = pd.read_csv(constants.data_files['ecdc_vacc']['data_fn'])
    df = df[df['location'] == 'Polska'].copy()
    df = df[df['grupa'] != 'ALL'].copy()
    df = df[['date', 'dawka_1', 'dawka_2', 'dawka_3', 'grupa', 'marka']]
    df = df[df['date'] >= oddaty].copy()
    df.sort_values(by=['date'], inplace=True)
    def f_cykl(r):
        r['cykl_n'] = r['dawka_1']
        r['cykl_d'] = r['dawka_3']
        if r['marka'] == 'AZ':
            r['cykl_p'] = r['dawka_1']
        else:
            r['cykl_p'] = r['dawka_2']
        return r
    df = df.apply(f_cykl, axis=1)
    df['n_n0'] = df.groupby(['grupa'])['cykl_n'].transform('sum')
    df['n_p0'] = df.groupby(['grupa'])['cykl_p'].transform('sum')
    df['n_d0'] = df.groupby(['grupa'])['cykl_d'].transform('sum')
    df.drop_duplicates(subset=['grupa'], inplace=True)
    df['n_dn'] = df['n_n0'] - df['n_p0']
    df['n_dp'] = df['n_p0'] - df['n_d0']
    df['n_dd'] = df['n_d0']
    df['bin_ecdc'] = df['grupa'].map(ages_ecdc)
    df = df[df['grupa'].isin(ages_ecdc.keys())].copy()
    del df['dawka_1']
    del df['dawka_2']
    del df['dawka_3']
    del df['date']
    del df['grupa']
    del df['cykl_n']
    del df['cykl_p']
    del df['cykl_d']
    del df['n_n0']
    del df['n_p0']
    del df['n_d0']
    del df['marka']

    # połączenie

    df = pd.merge(df, df_pop, how='left', left_on=['bin_ecdc'], right_on=['bin_ecdc']).copy()
    df = pd.merge(df, df_i, how='left', left_on=['bin_ecdc'], right_on=['bin_ecdc']).copy()
    df = df[df['bin_ecdc'] > 0].copy()
    df.sort_values(by=['bin_ecdc'], inplace=True)

    # obliczenia

    # odporność po szczepieniu

    df['n_v_all'] = df['n_dn'] + df['n_dp'] + df['n_dd']
    df['p_sn'] = df['n_dn'] / df['population_ecdc']
    df['p_sp'] = df['n_dp'] / df['population_ecdc']
    df['p_sd'] = df['n_dd'] / df['population_ecdc']

    # odporność po infekcji po 0, 1, 2, 3 dawce

    df['p_i0'] = df['n_status_0'] / df['population_ecdc']
    df['p_i1'] = df['n_status_1'] / df['population_ecdc']
    df['p_i2'] = df['n_status_2'] / df['population_ecdc']
    df['p_i3'] = df['n_status_3'] / df['population_ecdc']

    df['total_vacc'] = df['n_status_0'] + df['n_status_1'] + df['n_status_2'] + df['n_status_3']
    df['p_i_all'] = df['p_i0'] + df['p_i1'] + df['p_i2'] + df['p_i3']
    df['p_s_all'] = df['p_sn'] + df['p_sp'] + df['p_sd']
    df['n_naive'] = df['population_ecdc'] - df['n_v_all'] - df['total_vacc']
    df['p_naive'] = 1 - df['p_s_all'] - df['p_i_all']

    df['p_test'] = df['p_sn'] + df['p_sp'] + df['p_sd'] + \
                   df['p_i0'] + df['p_i1'] + df['p_i2'] + df['p_i3'] + \
                   df['p_naive']
    df.sort_values(by=['bin_ecdc'], inplace=True)

    traces = []

    # infekcja po 3 dawkach

    x = list(df['bin_ecdc'])
    y = list(df['p_i3'])
    fig_data = go.Bar(
        x=x,
        y=y,
        text=['inf.3d<br>' + str(round(x, 2)) for x in y],
        textposition='auto',
        name='infekcja po 3 dawce (inf.3d)'
    )
    traces.append(fig_data)

    # 3 dawki

    y = list(df['p_sd'])
    fig_data = go.Bar(
        x=x,
        y=y,
        text=['3d<br>' + str(round(x, 2)) for x in y],
        textposition='auto',
        name='dawka przypominająca (3d)'
    )
    traces.append(fig_data)

    # infekcja po 2 dawkach

    x = list(df['bin_ecdc'])
    y = list(df['p_i2'])
    fig_data = go.Bar(
        x=x,
        y=y,
        text=['inf.2d<br>' + str(round(x, 2)) for x in y],
        textposition='auto',
        name='infekcja po 2 dawce (inf.2d)'
    )
    traces.append(fig_data)

    # 2 dawki

    x = list(df['bin_ecdc'])
    y = list(df['p_sp'])
    fig_data = go.Bar(
        x=x,
        y=y,
        text=['2d<br>' + str(round(x, 2)) for x in y],
        textposition='auto',
        name='dawka pełna (2d)'
    )
    traces.append(fig_data)

    # infekcja po 1 dawce

    x = list(df['bin_ecdc'])
    y = list(df['p_i1'])
    fig_data = go.Bar(
        x=x,
        y=y,
        text=['inf.1d<br>' + str(round(x, 2)) for x in y],
        textposition='auto',
        name='infekcja po 1 dawce (inf.1d)'
    )
    traces.append(fig_data)

    # 1 dawka

    x = list(df['bin_ecdc'])
    y = list(df['p_sn'])
    fig_data = go.Bar(
        x=x,
        y=y,
        text=['1d<br>' + str(round(x, 2)) for x in y],
        textposition='auto',
        name='dawka niepełna (1d)'
    )
    traces.append(fig_data)

    # infekcja bez szczepienia

    x = list(df['bin_ecdc'])
    y = list(df['p_i0'])
    fig_data = go.Bar(
        x=x,
        y=y,
        text=['inf.0d<br>' + str(round(x, 2)) for x in y],
        textposition='auto',
        name='infekcja bez szczepienia (inf.0d)'
    )
    traces.append(fig_data)

    # naive

    x = list(df['bin_ecdc'])
    y = list(df['p_naive'])
    fig_data = go.Bar(
        x=x,
        y=y,
        text=['b.o.<br>' + str(round(x, 2)) for x in y],
        textposition='auto',
        name='brak odporności (b.o.)'
    )
    traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            barmode='stack',
            title=dict(text='Źródła immunizacji'),
            xaxis=dict(tickmode='array', tickvals=x0, ticktext=x0l),
            height=float(settings['plot_height']) * 760,
            legend=dict(x=0.02, y=1.03, orientation='h')
        )
    )

    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2'),
    ]
    return ret_val


#  ###############################################
#  Reinfekcje vs. infekcje wg rodzaju szczepionki
#  ###############################################
def layout_more_mz_cases13(_0, _1, _2, _3):
    template = get_template()
    settings = session2settings()

    marki = ['niezaszczepiony', 'Pfizer', 'Johnson&Johnson', 'Astra Zeneca', 'Moderna']

    # dane o infekcjach

    df = pd.read_csv(constants.data_files['basiw_i']['data_fn'])
    df = df[['date', 'i_all', 'marka', 'dawka', 'wpelni', 'nr_inf']].copy()
    df = df[df.date > '2021-10-01'].copy()
    # df = df[(df.date > '2022-01-01') & (df.wpelni == 'T')].copy()
    df.sort_values(by=['date'], inplace=True)
    df['marka'].fillna('niezaszczepiony', inplace=True)
    df['dawka'].fillna('brak', inplace=True)
    df['total_day'] = df.groupby(['date'])['i_all'].transform('sum')
    df['total_brand_day'] = df.groupby(['date', 'marka'])['i_all'].transform('sum')
    df['i_reinf'] = df['nr_inf'].map({1: 0, 2: 1, 3: 1, 4: 1}) * df['i_all']

    df['total_reinf_day'] = df.groupby(['date'])['i_reinf'].transform('sum')
    df['total_reinf_brand_day'] = df.groupby(['date', 'marka'])['i_reinf'].transform('sum')
    df.drop_duplicates(subset=['date', 'marka'], inplace=True)

    traces = []

    for marka in marki:
        dfx = df[df['marka'] == marka].copy()
        dfx['ile'] = df['total_reinf_brand_day'] / df['total_brand_day']
        x = list(dfx['date'])
        y = list(dfx['ile'])
        fig_data = go.Scatter(
            x=x,
            y=y,
            # text=y,
            name=marka
        )
        traces.append(fig_data)

    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            barmode='stack',
            title=dict(text='Reinfekcje vs. infekcje wg rodzaju szczepionki'),
            height=float(settings['plot_height']) * 760,
            legend=dict(x=0.02, y=1.03, orientation='h')
        )
    )

    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2'),
    ]
    return ret_val


#  ###############################################
#  Reinfekcje vs. infekcje wg statusu szczepienia
#  ###############################################
def layout_more_mz_cases14(_0, _1, _2, _3):
    template = get_template()
    settings = session2settings()

    dawki = ['niezaszczepiony', 'pelna_dawka', 'przypominajaca', 'jedna_dawka']
    # dawki = ['niezaszczepiony', 'pelna_dawka', 'przypominajaca', 'jedna_dawka', 'uzupełniająca']

    # dane o infekcjach

    df = pd.read_csv(constants.data_files['basiw_i']['data_fn'])
    df = df[['date', 'i_all', 'marka', 'dawka', 'wpelni', 'nr_inf']].copy()
    df = df[df.date > '2021-12-01'].copy()
    # df = df[(df.date > '2022-01-01') & (df.wpelni == 'T')].copy()
    df.sort_values(by=['date'], inplace=True)
    df['dawka'].fillna('niezaszczepiony', inplace=True)
    df['total_day'] = df.groupby(['date'])['i_all'].transform('sum')
    df['total_dawka_day'] = df.groupby(['date', 'dawka'])['i_all'].transform('sum')
    df['i_reinf'] = df['nr_inf'].map({1: 0, 2: 1, 3: 1, 4: 1}) * df['i_all']

    df['total_reinf_day'] = df.groupby(['date'])['i_reinf'].transform('sum')
    df['total_reinf_dawka_day'] = df.groupby(['date', 'dawka'])['i_reinf'].transform('sum')
    df.drop_duplicates(subset=['date', 'dawka'], inplace=True)

    traces = []

    for dawka in dawki:
        dfx = df[df['dawka'] == dawka].copy()
        dfx['ile'] = df['total_reinf_dawka_day'] / df['total_dawka_day']
        x = list(dfx['date'])
        y = list(dfx['ile'])
        fig_data = go.Scatter(
            x=x,
            y=y,
            # text=y,
            name=dawka
        )
        traces.append(fig_data)

    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            barmode='stack',
            title=dict(text='Reinfekcje vs. infekcje wg statusu szczepienia'),
            height=float(settings['plot_height']) * 760,
            legend=dict(x=0.02, y=1.03, orientation='h')
        )
    )

    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2'),
    ]
    return ret_val


#  ###############################################
#  Reinfekcje i infekcje ilościowo
#  ###############################################
def layout_more_mz_cases15(_0, _1, _2, _3):
    template = get_template()
    settings = session2settings()

    dawki = ['niezaszczepiony', 'pelna_dawka', 'przypominajaca']

    # dane o infekcjach

    df = pd.read_csv(constants.data_files['basiw_i']['data_fn'])
    df = df[['date', 'i_all', 'marka', 'dawka', 'wpelni', 'nr_inf']].copy()
    df = df[df.nr_inf > 1].copy()
    df.sort_values(by=['date'], inplace=True)
    df['dawka'].fillna('niezaszczepiony', inplace=True)
    df['dawka'] = df['dawka'].str.replace('jedna_dawka', 'niezaszczepiony')
    df['total_day'] = df.groupby(['date'])['i_all'].transform('sum')
    df['total_status_day'] = df.groupby(['date', 'dawka'])['i_all'].transform('sum')
    df.drop_duplicates(subset=['date', 'dawka'], inplace=True)

    traces = []

    for dawka in dawki:
        dfx = df[df['dawka'] == dawka].copy()
        dfx['ile'] = (dfx['total_status_day'] / dfx['total_day'])
        dfx['ile'] = dfx['ile'].rolling(7, min_periods=1).mean()
        x = list(dfx['date'])
        y = list(dfx['ile'])
        fig_data = go.Scatter(
            x=x,
            y=y,
            line=dict(width=settings['linewidth_basic']),
            name=dawka
        )
        traces.append(fig_data)

    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            barmode='stack',
            title=dict(text='Reinfekcje - Udział przypadków wg statusu zaszczepienia'),
            height=float(settings['plot_height']) * 760,
            legend=dict(x=0.02, y=1.03, orientation='h')
        )
    )

    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2'),
    ]
    return ret_val


#  ###############################################
#  LC Long Covid
#  ###############################################
def layout_more_lc1(_0, _1, _2, _3):
    template = get_template()
    settings = session2settings()

    # dane o LC

    df_lc = pd.read_csv('data/csv/lc.csv', header=1)
    df_lc.columns = ['NFZ', 'year', 'month', 'age', 'n_episode', 'n_patient', 'code']
    df_lc = df_lc[df_lc['year'] == 2021].copy()
    df_lc = df_lc[df_lc['month'] > 3].copy()
    df_lc = df_lc[['age', 'n_patient', 'n_episode']].copy()
    df_lc['n_age_lc_patient'] = df_lc.groupby(['age'])['n_patient'].transform('sum')
    df_lc['n_age_lc_episode'] = df_lc.groupby(['age'])['n_episode'].transform('sum')
    df_lc = df_lc.drop_duplicates(subset=['age']).copy()

    df_i = pd.read_csv(constants.data_files['basiw_i']['data_raw_fn'])
    df_i = df_i[df_i['age'].between(0, 19)].copy()
    df_i = df_i[df_i['date'] < '2021-10-01'].copy()
    df_i['n_age_i'] = df_i.groupby(['age'])['i_all'].transform('sum')
    df_i = df_i.drop_duplicates(subset=['age']).copy()

    df = pd.merge(df_lc, df_i, how='left', left_on=['age'], right_on=['age'])
    df = df[['age', 'n_age_lc_patient', 'n_age_lc_episode', 'n_age_i']].copy()
    df['ile_patient'] = df['n_age_lc_patient'] / df['n_age_i']
    df['ile_episode'] = df['n_age_lc_episode'] / df['n_age_i']

    df.sort_values(by=['age'], inplace=True)

    traces = []

    x = list(df['age'])
    y = list(df['n_age_i'])
    fig_data = go.Bar(
        x=x,
        y=y,
        name='Zakażenia'
    )
    traces.append(fig_data)
    y = list(df['n_age_lc_patient'])
    fig_data = go.Bar(
        x=x,
        y=y,
        name='Hospitalizacje LC'
    )
    traces.append(fig_data)
    # y = list(df['n_age_lc_episode'])
    # fig_data = go.Bar(
    #     x=x,
    #     y=y,
    #     name='Ambulatoryjnie LC'
    # )
    # traces.append(fig_data)

    y = list(df['ile_patient'])
    fig_data = go.Scatter(
        x=x,
        y=y,
        yaxis='y2',
        name='Hospitalizacje LC/Zakażenia'
    )
    traces.append(fig_data)
    # y = list(df['ile_episode'])
    # fig_data = go.Scatter(
    #     x=x,
    #     y=y,
    #     yaxis='y2',
    #     name='Ambulatoryjnie LC/Zakażenia'
    # )
    # traces.append(fig_data)

    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            # barmode='stack',
            title=dict(text='Long Covid. Zakażenia w grupach wiekowych i liczba hospitalizacji związanych z Covid-19' + \
                       '<br><sub>Infekcje od 1.01.2021 do 1.10.2021, hospitalizacje od 1.04.2021 do 31.12.2021'),
            height=float(settings['plot_height']) * 760,
            legend=dict(x=0.02, y=1.03, orientation='h'),
            xaxis=dict(
                dtick=1,
                title=dict(text='Grupa wiekowa')),
            yaxis=dict(
                title='Liczba przypadków',
            ),
            yaxis2=dict(
                title='Hospitalizacje LC/Zakażenia',
                anchor="x",
                range=[0, max(df['ile_patient'])],
                overlaying="y",
                side="right",
            ),
        )
    )

    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2'),
    ]
    return ret_val


#  ###############################################
#  LC PIMS
#  ###############################################
def layout_more_lc2(_0, _1, _2, _3):
    template = get_template()
    settings = session2settings()

    # dane o PIMS

    df_lc = pd.read_csv('data/csv/pims.csv', header=3)
    df_lc.columns = ['NFZ', 'year', 'month', 'age', 'n_episode', 'n_patient']
    df_lc = df_lc[df_lc['year'] == 2021].copy()
    df_lc = df_lc[df_lc['month'] > 3].copy()
    df_lc = df_lc[['age', 'n_patient', 'n_episode']].copy()
    df_lc['n_age_lc_patient'] = df_lc.groupby(['age'])['n_patient'].transform('sum')
    df_lc['n_age_lc_episode'] = df_lc.groupby(['age'])['n_episode'].transform('sum')
    df_lc = df_lc.drop_duplicates(subset=['age']).copy()

    df_i = pd.read_csv(constants.data_files['basiw_i']['data_raw_fn'])
    df_i = df_i[df_i['age'].between(0, 19)].copy()
    df_i = df_i[df_i['date'] < '2021-10-01'].copy()
    df_i['n_age_i'] = df_i.groupby(['age'])['i_all'].transform('sum')
    df_i = df_i.drop_duplicates(subset=['age']).copy()

    df = pd.merge(df_lc, df_i, how='left', left_on=['age'], right_on=['age'])
    df = df[['age', 'n_age_lc_patient', 'n_age_lc_episode', 'n_age_i']].copy()
    df['ile_patient'] = df['n_age_lc_patient'] / df['n_age_i']
    df['ile_episode'] = df['n_age_lc_episode'] / df['n_age_i']

    df.sort_values(by=['age'], inplace=True)

    traces = []

    x = list(df['age'])
    y = list(df['n_age_i'])
    fig_data = go.Bar(
        x=x,
        y=y,
        name='Zakażenia'
    )
    traces.append(fig_data)
    y = list(df['n_age_lc_patient'])
    fig_data = go.Bar(
        x=x,
        y=y,
        name='Hospitalizacje LC'
    )
    traces.append(fig_data)
    # y = list(df['n_age_lc_episode'])
    # fig_data = go.Bar(
    #     x=x,
    #     y=y,
    #     name='Ambulatoryjnie LC'
    # )
    # traces.append(fig_data)

    y = list(df['ile_patient'])
    fig_data = go.Scatter(
        x=x,
        y=y,
        yaxis='y2',
        name='Hospitalizacje LC/Zakażenia'
    )
    traces.append(fig_data)
    # y = list(df['ile_episode'])
    # fig_data = go.Scatter(
    #     x=x,
    #     y=y,
    #     yaxis='y2',
    #     name='Ambulatoryjnie LC/Zakażenia'
    # )
    # traces.append(fig_data)

    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            # barmode='stack',
            title=dict(text='PIMS. Zakażenia w grupach wiekowych i liczba przypadków PIMS' + \
                       '<br><sub>Infekcje od 1.01.2021 do 1.10.2021, PIMS od 1.04.2021 do 31.12.2021'),
            height=float(settings['plot_height']) * 760,
            legend=dict(x=0.02, y=1.03, orientation='h'),
            xaxis=dict(
                dtick=1,
                title=dict(text='Grupa wiekowa')),
            yaxis=dict(
                title='Liczba przypadków',
            ),
            yaxis2=dict(
                title='Hospitalizacje LC/Zakażenia',
                range=[0, max(df['ile_patient'])],
                anchor="x",
                overlaying="y",
                side="right",
            ),
        )
    )

    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2'),
    ]
    return ret_val


##################################################
#  Mapa discrete colors województwa
##################################################

def layout_more_map_discrete_woj(DF, deltat, _scale, n):

    if deltat is None:
        return
    template = get_template()
    settings = session2settings()
    # map_palette: ciągły, dyskretny
    # map_cut: równe, kwantyle, własne
    map_cut = settings['map_cut']
    colors = constants.color_scales[settings['color_order']]

    df = DF['poland'].copy()
    to_date = settings['to_date']
    d = list(df.date.unique())
    d = [dd for dd in d if dd <= to_date]
    dates = []
    for i in range(n):
        dates.append(d[-1 - deltat * i])

    chart_types = settings['chart_type']
    if len(chart_types) != 1:
        return "Wybierz tylko jedną wielkość"

    chart_type = chart_types[0]
    if n == 1:
        rows = 1; cols = 1; zoom = 5.5; width = 12
        height = int(settings['map_height'])
    elif n == 2:
        rows = 1; cols = 2; zoom = 5.5; width = 6
        height = int(settings['map_height'])
    elif n <= 4:
        rows = 2; cols = 2; zoom = 4.5; width = 6
        height = int(settings['map_height']) / 2
    elif n <= 6:
        rows = 2; cols = 3; zoom = 4.5; width = 4
        height = int(settings['map_height']) / 2
    else:
        rows = 3; cols = 3; zoom = 4.0; width = 4
        height = int(settings['map_height']) / 3

    mapbox_access_token = open("data/mapbox.token").read()
    mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=19.18, lat=52.11), zoom=zoom)
    mapbox['style'] = constants.mapbox_styles[settings['map_mapbox']]

    json_file = r'data/geojson/woj.min.geo.json'
    featuredikey = 'properties.nazwa'
    with open(json_file, encoding="utf-8") as f:
        geojson = json.load(f)
    figures = []

    dfxs = pd.DataFrame()
    for date in dates:
        locations = list(constants.wojew_cap.values())[1:]
        dfx = prepare_data(settings, DF,
                          scope='poland',
                          locations=locations,
                          date=date,
                          chart_types=chart_types,
                          all_columns=True)
        dfx['date'] = date
        dfxs = dfxs.append(dfx, ignore_index=True).copy()

    s, bins, bin_labels = get_bins(dfxs[chart_type], len(colors), map_cut)

    if len(bins) == 0:
        return bin_labels
    dfxs['bin'] = s
    dfxs['bin'] = round(dfxs['bin'], 3)
    color_scales = [((0., colors[i]), (1., colors[i])) for i in range(len(bins)-1)] * 2
    if map_cut == 'kwantyle':
        cbtext = 'Kwantyle (' + str(len(colors)) + ': ' +settings['color_order'] + ')'
    elif map_cut == 'równe':
        cbtext = 'Równe przedziały (' + str(len(colors)) + ': ' + settings['color_order'] + ')'
    else:
        cbtext = 'Predefiniowane (' + str(len(colors)) + ': ' + settings['color_order'] + ')'
    dfxs.sort_values(by=['bin'], inplace=True)
    for date in dates[::-1]:
        dfx = dfxs[dfxs['date'] == date].copy()
        dfx.dropna(subset=['Long', 'Lat'], inplace=True)
        dfx['Long'] = dfx['location'].apply(lambda x: constants.wojew_mid[x][1])
        dfx['Lat'] = dfx['location'].apply(lambda x: constants.wojew_mid[x][0])
        dfx = dfx[dfx['location'] != 'Polska'].copy()
        dfx.sort_values(by=['bin'], inplace=True)
        figure = go.Figure()
        first = True
        for i, data in enumerate(dfxs['bin'].unique()):
            dfp = dfx[dfx['bin'] == data]
            locations = [i.lower() for i in list(dfp['location'].unique())]
            figure.add_trace(go.Choroplethmapbox(
                geojson=geojson, featureidkey=featuredikey, locations=locations,
                z=[float(data)] * len(dfp),
                marker=dict(line=dict(color='black', width=0.5 if 'drawborder' in settings['map_options'] else 0.)),
                # hoverinfo='text',
                colorscale=color_scales[i],
                marker_opacity=settings['map_opacity'],
                colorbar=dict(len=0.1,
                              tickmode="array",
                              tickvals=[float(data)],
                              ticktext=[str(data)+'+'],
                              ticks="outside",
                              y=0.1 + 0.1 * i
                              ),
            ))
            if 'annotations' in settings['map_options']:
                if '0c' in settings['map_options']: acc = 0
                elif '1c' in settings['map_options']: acc = 1
                elif '2c' in settings['map_options']: acc = 2
                else: acc = 3
                if acc == 0:
                    anno_text = dfp[chart_type].astype(int).astype(str)
                else:
                    anno_text = round(dfp[chart_type], acc).astype(str)
                figure.add_trace(
                    go.Scattermapbox(
                        lat=dfp.Lat, lon=dfp.Long,
                        mode='text',
                        text=anno_text,
                        textfont=dict(size=settings['font_size_anno'], color=get_session_color(4)),
                    ))
        figure.update_layout(
            images=constants.image_logo_map,
            template=template,
            autosize=True,
            showlegend=False,
            title=dict(
                text=date + ' ' + get_title(settings, chart_type, date) + '<br><sup>' + cbtext,
                x=0.05, y=settings['titleypos'],
                xanchor='left',
                font=dict(size=settings['font_size_title'], color=get_session_color(7))
            ),
            height=height,
            margin=dict(l=settings['marginl'], r=settings['marginr'] + 10, b=settings['marginb'], t=settings['margint'],
                        pad=0),
            paper_bgcolor=get_session_color(1),
            plot_bgcolor=get_session_color(1),
            mapbox=mapbox,
        )
        figure = add_copyright(figure, settings)
        config = {
            'displaylogo': False,
            'responsive': True,
            'toImageButtonOptions': {
                'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
                'filename': 'docent_image', 'height': 700, 'width': 1200}
        }
        f = dbc.Col(dcc.Graph(id='mapa1' + chart_type, figure=figure, config=config), width=width, className=constants.nomargins)
        figures.append(f)
    ret_val = []
    for i in range(rows):
        row = []
        for j in range(cols):
            nelem = i*cols+j
            if nelem < len(figures):
                row.append(figures[nelem])
        ret_val.append(dbc.Row(row))

    return dbc.Col(ret_val)


##################################################
#  Mapa discrete colors powiaty
##################################################

def layout_more_map_discrete_pow(DF, deltat, _scale, n):

    if deltat is None:
        return
    template = get_template()
    settings = session2settings()
    # map_palette: ciągły, dykretny
    # map_cut: równe, kwantyle, własne
    map_cut = settings['map_cut']
    colors = constants.color_scales[settings['color_order']]

    df = DF['poland'].copy()
    to_date = settings['to_date']
    d = list(df.date.unique())
    d = [dd for dd in d if dd <= to_date]
    dates = []
    for i in range(n):
        dates.append(d[-1 - deltat * i])

    chart_types = settings['chart_type']
    if len(chart_types) != 1:
        return "Wybierz tylko jedną wielkość"

    chart_type = chart_types[0]
    if n == 1:
        rows = 1; cols = 1; zoom = 5.5; width = 12
        height = int(settings['map_height'])
    elif n == 2:
        rows = 1; cols = 2; zoom = 5.5; width = 6
        height = int(settings['map_height'])
    elif n <= 4:
        rows = 2; cols = 2; zoom = 4.5; width = 6
        height = int(settings['map_height']) / 2
    elif n <= 6:
        rows = 2; cols = 3; zoom = 4.5; width = 4
        height = int(settings['map_height']) / 2
    else:
        rows = 3; cols = 3; zoom = 4.0; width = 4
        height = int(settings['map_height']) / 3

    mapbox_access_token = open("data/mapbox.token").read()
    mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=19.18, lat=52.11), zoom=zoom)
    mapbox['style'] = constants.mapbox_styles[settings['map_mapbox']]

    json_file = r'data/geojson/powiaty-min-ws.geojson'
    featuredikey = 'properties.nazwa'
    with open(json_file, encoding="utf-8") as f:
        geojson = json.load(f)
    figures = []

    dfxs = pd.DataFrame()
    for date in dates:
        locations = list(DF['cities'].location.unique())
        dfx = prepare_data(settings, DF,
                          scope='cities',
                          locations=locations,
                          date=date,
                          chart_types=chart_types,
                          all_columns=True)
        dfxs = dfxs.append(dfx, ignore_index=True).copy()

    s, bins, bin_labels = get_bins(dfxs[chart_type], len(colors), map_cut)
    if len(bins) == 0:
        return bin_labels
    dfxs['bin'] = s
    dfxs['bin'] = round(dfxs['bin'], 3)
    color_scales = [((0., colors[i]), (1., colors[i])) for i in range(len(bins)-1)] * 2
    if map_cut == 'kwantyle':
        cbtext = 'Kwantyle (' + str(len(colors)) + ': ' +settings['color_order'] + ')'
    elif map_cut == 'równe':
        cbtext = 'Równe przedziały (' + str(len(colors)) + ': ' + settings['color_order'] + ')'
    else:
        cbtext = 'Predefiniowane (' + str(len(colors)) + ': ' + settings['color_order'] + ')'
    dfxs.sort_values(by=['bin'], inplace=True)
    for date in dates[::-1]:
        dfx = dfxs[dfxs['date'] == date].copy()
        dfx.dropna(subset=['Long', 'Lat'], inplace=True)
        dfx = dfx[dfx['location'] != 'Polska'].copy()
        dfx.sort_values(by=['bin'], inplace=True)
        figure = go.Figure()
        first = True
        for i, data in enumerate(dfxs['bin'].unique()):
            dfp = dfx[dfx['bin'] == data]
            locations = list(dfp.location.unique())
            figure.add_trace(go.Choroplethmapbox(
                geojson=geojson, featureidkey=featuredikey, locations=locations,
                z=[float(data)] * len(dfp),
                marker=dict(line=dict(color='black', width=0.5 if 'drawborder' in settings['map_options'] else 0.)),
                colorscale=color_scales[i],
                marker_opacity=settings['map_opacity'],
                colorbar=dict(len=0.1,
                              tickmode="array",
                              tickvals=[float(data)],
                              ticktext=[str(data)+'+'],
                              ticks="outside",
                              y=0.1 + 0.1 * i
                              ),
            ))
            if 'annotations' in settings['map_options']:
                if '0c' in settings['map_options']: acc = 0
                elif '1c' in settings['map_options']: acc = 1
                elif '2c' in settings['map_options']: acc = 2
                else: acc = 3
                if acc == 0:
                    anno_text = dfp[chart_type].astype(int).astype(str)
                else:
                    anno_text = round(dfp[chart_type], acc).astype(str)
                figure.add_trace(
                    go.Scattermapbox(
                        lat=dfp.Lat, lon=dfp.Long,
                        mode='text',
                        text=anno_text,
                        textfont=dict(size=settings['font_size_anno'], color=get_session_color(4)),
                    ))
        figure.update_layout(
            images=constants.image_logo_map,
            template=template,
            autosize=True,
            showlegend=False,
            title=dict(
                text=date + ' ' + get_title(settings, chart_type, date) + '<br><sup>' + cbtext,
                x=0.05, y=settings['titleypos'],
                xanchor='left',
                font=dict(size=settings['font_size_title'], color=get_session_color(7))
            ),
            height=height,
            margin=dict(l=settings['marginl'], r=settings['marginr'] + 10, b=settings['marginb'], t=settings['margint'],
                        pad=0),
            paper_bgcolor=get_session_color(1),
            plot_bgcolor=get_session_color(1),
            mapbox=mapbox,
        )
        figure = add_copyright(figure, settings)
        config = {
            'displaylogo': False,
            'responsive': True,
            'toImageButtonOptions': {
                'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
                'filename': 'docent_image', 'height': 700, 'width': 1200}
        }
        f = dbc.Col(dcc.Graph(id='mapa1' + chart_type, figure=figure, config=config), width=width, className=constants.nomargins)
        figures.append(f)
    ret_val = []
    for i in range(rows):
        row = []
        for j in range(cols):
            nelem = i*cols+j
            if nelem < len(figures):
                row.append(figures[nelem])
        ret_val.append(dbc.Row(row))

    return dbc.Col(ret_val)


##################################################
#  Wykres 2 osie
##################################################

def layout_more_2axes(DF, field_1, _2, _3):
    template = get_template()
    settings = session2settings()
    chart_types = settings['chart_type_data'] + settings['chart_type_calculated']
    if len(settings['locations']) == 0:
        return 'Wybierz lokalizację...'
    if field_1 == '<brak>':
        return 'Nie wybrano pola do lewej osi'
    if len(chart_types) == 0:
        return 'Nie wybrano pól do prawej osi'
    if field_1 in chart_types:
        return "Powtarzające się pole dla lewej i prawej osi " + field_1
    prepare_types = chart_types + [field_1]

    filtered_df = prepare_data(settings, DF,
                               locations=settings['locations'],
                               chart_types=prepare_types,
                               scope=settings['scope'],
                               )
    fields = [field_1]
    types1 = ['scatter'] * len(fields)
    types2 = ['scatter'] * len(chart_types)
    ret_val = layout_timeline_two_axes(filtered_df, 'date', fields, chart_types,
                                       t1=types1, t2=types2,
                                       color1=get_session_color('13'),
                                       color2=get_session_color('14'),
                                       height=600)

    return ret_val


##################################################
#  Wykres histogram
##################################################

def layout_more_histogram(DF, hist_agg, hist_date, _3):
    template = get_template()
    settings = session2settings()
    scope = settings['scope']
    data_modifier = settings['data_modifier']
    table_mean = settings['table_mean']
    from_date = settings['from_date']
    dzielna = settings['dzielna']
    locations = settings['locations']
    chart_types = settings['chart_type']
    if len(locations) == 0:
        return
    if len(chart_types) != 1:
        return
    if hist_agg == 'oneday':
        from_date = ''
        date = hist_date
        okres = hist_date
    else:
        okres = str(from_date) + ' - ' + str(hist_date)
        date = ''
    chart_type = chart_types[0]
    if constants.trace_props[chart_type]['category'] == 'data':
        trace_name = get_trace_name(chart_type) + {1: '',
                                                   2: ' na 100 000 osób',
                                                   3: ' na 1000 km2'}[settings['data_modifier']]
    else:
        trace_name = get_trace_name(chart_type)
    df0 = prepare_data(settings, DF,
                      scope=scope,
                      locations=locations,
                      date=date,
                      chart_types=chart_types)
    if len(df0) == 0:
        return 'Brak rekordów spełniających podane warunki'
    if table_mean != 'oneday':
        df = df0[df0['date'] <= hist_date].copy()
    else:
        df = df0.copy()
    print(df.date.min(), df.date.max())
    if hist_agg == 'sum':
        hist = df.groupby('location')[chart_type].sum()
    elif hist_agg == 'median':
        hist = df.groupby('location')[chart_type].median()
    elif hist_agg == 'max':
        hist = df.groupby('location')[chart_type].max()
    elif hist_agg == 'mean':
        hist = df.groupby('location')[chart_type].mean()
    else:
        hist = df.groupby('location')[chart_type].mean()
    # figure = go.Figure(data=[go.Histogram(x=hist)])
    hmax = hist.max()
    if hmax < 1:
        step = 0.1
    elif hmax < 2:
        step = 0.2
    elif hmax < 5:
        step = 0.5
    elif hmax < 10:
        step = 0.1
    elif hmax < 20:
        step = 2
    elif hmax < 50:
        step = 5
    elif hmax < 100:
        step = 10
    elif hmax < 200:
        step = 20
    elif hmax < 500:
        step = 5
    elif hmax < 1000:
        step = 100
    elif hmax < 2000:
        step = 200
    else:
        step = 500
    mybins = [x * step for x in range(11)]
    counts, bins = np.histogram(hist, bins=mybins)
    # fig.data[0].text = list(zip(counts, bins.round(1)))
    def gen_text(lhist, l0, l1):
        h = hist[hist.between(l0, l1)]
        if len(h) > 0:
            t = [hh[0]+'<br>'+str(round(hh[1], 2))+'<br>' for hh in h.iteritems()]
        else:
            t = ''
        print(t)
        if len(t) > 15:
            t = t[:15] + ['..... więcej']
        return ''.join(t)

    texts = [gen_text(hist, mybins[i], mybins[i+1]) for i in range(len(mybins)-1)]
    annotations = [
                      dict(
                          x=xpos,
                          y=ypos,
                          xref='x',
                          yref='y',
                          text=text,
                          font=dict(size=10, color=settings['color_4']),
                          bgcolor=settings['color_5'],
                          # height=300,
                          width=100,
                          showarrow=False,
                          arrowhead=1
                      ) for xpos, ypos, text in zip(bins, counts, texts)
                  ],
    fig = go.Figure(go.Bar(
        x=bins,
        y=counts,
        marker=dict(opacity=0.3, color='rgba(0,0,0,0)', line=dict(color='white', width=4)),
    ))
    # fig.data[0].text = texts
    # fig.data[0].text = counts
    fig.update_traces(textposition='outside', textfont_size=18)
    fig.update_layout(bargap=0.02)
    figure = fig

    figure.update_layout(height=700,
                         template=template,
                         annotations=annotations[0],
                         title=dict(text='Histogram - ' + trace_name + '<br><sub>rodzaj agregacji: '
                                         + hist_agg + '<br>' + okres),
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
    ret_val = [dbc.Col(dcc.Graph(id='ternary', figure=fig))]

    return ret_val


##################################################
#  Heatmap - struktura wiekowa zachorowań
##################################################

def layout_more_mzage2(_0, _data, rodzaj, _3):
    template = get_template()
    settings = session2settings()
    ages5 = constants.age_bins['bin5']
    oddaty = str(_data)[:10]
    df = pd.read_csv(constants.data_files['basiw_d']['data_fn'])

    df['sum'] = df.groupby(['date', 'bin5'])['d_all'].transform('sum')
    df['sum_day'] = df.groupby(['date'])['d_all'].transform('sum')
    if rodzaj == 'brutto':
        title = "Dzienne liczby zgonów Covid-19 w kategoriach wiekowych"
        df['liczba'] = df['sum']
    else:
        title = "Udział grup wiekowych w dziennych zgonach Covid-19"
        df['liczba'] = df['sum'] / df['sum_day']
    df = df.drop_duplicates(subset=['date', 'bin5']).copy()
    df.sort_values(by=['bin5', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[df['date'] > oddaty].copy()
    fig = heatmap(df, 'bin5', 'date', 'liczba', settings, template, ages5, title)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


##########################################################################
#  Dzienne zgony Covid-19 w grupie wiekowej zaszczepieni/niezaszczepieni
##########################################################################

def layout_more_mzage3(_0, _oddaty, _dodaty, _age):
    template = get_template()
    settings = session2settings()
    # dane GUS 31.12.2020
    population_all = 38265013
    oddaty = str(_oddaty)[:10]
    dodaty = str(_dodaty)[:10]
    bins = {
        '10-14': [2, 2065628, 'Age10_14'],
        '15-17': [3, 1075305, 'Age15_17'],
        '18-24': [4, 2688690, 'Age18_24'],
        '25-49': [5, 14216985, 'Age25_49'],
        '50-59': [6, 4605466, 'Age50_59'],
        '60-69': [7, 5185843, 'Age60_69'],
        '70-79': [8, 2930420, 'Age70_79'],
        '80+': [9, 1683970, 'Age80+']
    }

    # szczepienia ECDC

    df_v = pd.read_csv(constants.data_files['ecdc_vacc']['data_fn'])
    df_v = df_v[['date', 'location', 'dawka_1', 'dawka_2', 'grupa', 'denom', 'marka']]
    if _age == 'ALL':
        df_v = df_v[df_v['grupa'] == 'ALL'].copy()
    else:
        df_v = df_v[df_v['grupa'] == bins[_age][2]].copy()
    df_v = df_v[df_v['location'] == 'Polska'].copy()

    df_v['full_brand'] = df_v['dawka_2']
    df_v.loc[df_v.marka == 'JANSS', 'full_brand'] = df_v['dawka_1']

    df_v['full'] = df_v.groupby(['date'])['full_brand'].transform('sum')
    df_v = df_v.drop_duplicates(subset=['date']).copy()

    # resample day

    df_v.index = pd.to_datetime(df_v['date'], format='%Y-%m-%d')
    df_v = df_v.resample('D').mean().fillna(method='ffill')
    df_v['full'] = df_v['full'] / 7
    df_v['date'] = df_v.index.astype(str)
    df_v = df_v[['date', 'full']].copy()
    df_v.reset_index(drop=True, inplace=True)

    df_v['full_cum'] = df_v['full'].cumsum()

    df = pd.read_csv(constants.data_files['basiw_d']['data_fn'])
    df = df[df['date'].between(oddaty, dodaty)]
    if _age == 'ALL':
        df['total'] = df.groupby(['date', 'wpelni'])['d_all'].transform('sum')
        df = df.drop_duplicates(subset=['date', 'wpelni']).copy()
    else:
        bin = bins[_age][0]
        df = df[df['bin_ecdc'] == bin].copy()
        df['total'] = df.groupby(['date', 'bin_ecdc', 'wpelni'])['d_all'].transform('sum')
        df = df.drop_duplicates(subset=['date', 'bin_ecdc', 'wpelni']).copy()
    df.sort_values(by=['date'], inplace=True)
    df = pd.merge(df, df_v, how='left', left_on=['date'], right_on=['date']).copy()

    df = df[df['full_cum'].notna()]
    dodaty = min(dodaty, max(df.date))

    traces = []

    # niezaszczepieni

    dfx = df[df['wpelni'] == 'N'].copy()

    # ułamek niezaszczepionych w populacji wieku

    if _age == 'ALL':
        dfx['population'] = population_all
    else:
        dfx['population'] = bins[_age][1]
    dfx['wsp_n'] = (dfx['population'] - dfx['full_cum']) / dfx['population']
    dfx['total'] = dfx['total'].rolling(session['average_days'], min_periods=1).mean()
    dfx['data'] = dfx['total'] / (dfx['population'] * dfx['wsp_n']) * 100000
    xn = list(dfx['date'])
    yn = list(dfx['data'])
    fig_data = go.Scatter(
        x=xn,
        y=yn,
        name=_age + ' niezaszczepieni',
        yaxis='y1',
        line=dict(width=settings['linewidth_basic'], dash='solid')
    )
    traces.append(fig_data)

    # zaszczepieni

    dfx = df[df['wpelni'] == 'T'].copy()

    # ułamek zaszczepionych w populacji wieku

    if _age == 'ALL':
        dfx['population'] = population_all
    else:
        dfx['population'] = bins[_age][1]
    dfx['wsp_t'] = dfx['full_cum'] / dfx['population']
    dfx['total'] = dfx['total'].rolling(session['average_days'], min_periods=1).mean()
    dfx['data'] = dfx['total'] / (dfx['population'] * dfx['wsp_t']) * 100000
    xz = list(dfx['date'])
    yz = list(dfx['data'])
    fig_data = go.Scatter(
        x=xz,
        y=yz,
        name=_age + ' zaszczepieni',
        yaxis='y1',
        line=dict(width=settings['linewidth_basic'], dash='dash')
    )
    traces.append(fig_data)

    # procent zaszczepienia

    wz = list(round(dfx['wsp_t'] * 100, 2))
    fig_data = go.Scatter(
        x=xz,
        y=wz,
        name=_age + ' % zaszczepienia',
        line=dict(width=1., dash='dot'),
        yaxis='y2'
    )
    traces.append(fig_data)

    # proporcja

    dn = {xn[i]: yn[i] for i in range(len(xn))}
    dz = {xz[i]: yz[i] for i in range(len(xz))}

    sum_n = round(sum(yn), 2)
    sum_z = round(sum(yz), 2)
    sum_all = sum_n + sum_z
    sum_total = df.total.sum()
    anno_text = '<b>Grupa wiekowa:</b> ' + _age + \
        '<br>Okres: (od ' + oddaty + ' do ' + dodaty + ')' + \
        '<br><br>Liczebność grupy: ' + str(int(dfx.iloc[0]['population'])) + \
        '<br>Liczba zgonów razem: ' + str(int(sum_total)) + \
        '<br>- zgony spośród osób zaszczepionych / 100k: : ' + str(sum_z) + ' (' + str(round(sum_z/sum_all*100, 2)) + '%)' + \
        '<br>- zgony spośród osób niezaszczepionych / 100k: : ' + str(sum_n) + ' (' + str(round(sum_n/sum_all*100, 2)) + '%)'


    tytul = 'Dzienne zgony Covid-19 / 100k w grupach zaszczepionych / niezaszczepionych (od ' + oddaty + ' do ' + dodaty + ')' + \
            '<br>Grupa wiekowa: '+ _age + \
            '<br><sub>(z uwzględnieniem bieżącego wskaźnika zaszczepienia)<br>Dane: ECDC, GUS, MZ (BASIW)'
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text=tytul),
            height=float(settings['plot_height']) * 760,
            legend=dict(x=1.02, y=1., orientation='v'),
            yaxis1=dict(
                title='% Liczba zgonów w grupie wiekowej zaszczepionych / niezaszczepionych',
            ),
            yaxis2=dict(
                title='% zaszczepienia',
                anchor="x",
                ticksuffix='%',
                overlaying="y",
                side="right",
                range=[0., 100.]
            ),
        )
    )
    figure = add_copyright(figure, settings)
    figure.add_annotation(text=anno_text,
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=settings['annoxpos'], y=settings['annoypos'],
                          # x=0.01, y=0.8,
                          showarrow=False)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val

##########################################
#  Statystyka zgonów w grupach wiekowych
##########################################

def layout_more_mzage1(_0, _1, _2, _3):
    template = get_template()
    settings = session2settings()
    _oddaty = str(settings['from_date'])[:10]
    _dodaty = str(settings['to_date'])[:10]
    df = pd.read_csv(constants.data_files['basiw_d']['data_fn'])
    df = df[['date', 'sex', 'wojew', 'wpelni', 'age_group', 'bin_mz', 'd_all', 'dawka']].copy()

    def wykres(df, df_pop, od_daty, do_daty):
        traces = []
        bin = 'bin_mz'
        x0 = list(constants.age_bins[bin].values())
        df.sort_values(by=[bin], inplace=True)

        def f(query):
            dfx = df.query(query)
            sum1 = dfx['d_all'].sum()
            dfx['sum_bin_' + bin] = dfx.groupby([bin])['d_all'].transform('sum')
            dfx.drop_duplicates(subset=[bin], inplace=True)
            x = [constants.age_bins[bin][i] for i in list(dfx[bin])]
            y = list(dfx['sum_bin_' + bin])
            return x, y, sum1

        xz2, yz2, sum_z2 = f("dawka == 'pelna_dawka'")
        xz3, yz3, sum_z3 = f("dawka == 'przypominajaca'")
        # xz, yz, sum_z = f("wpelni == 'T'")
        xn, yn, sum_n = f("wpelni == 'N'")

        sum_all = sum_z2 + sum_z3 + sum_n

        anno_text = '<b>Razem: ' + str(int(sum_z2+sum_z3+sum_n)) + ' (100%)</b><br><br>' + \
                    'Zaszczepionych pełną dawką: ' + str(int(sum_z2)) + ' (' + str(round((sum_z2)/(sum_all)*100, 2)) + '%)<br>' + \
                    'Zaszczepionych dawką przypominającą: ' + str(int(sum_z3)) + ' (' + str(round((sum_z3)/(sum_all)*100, 2)) + '%)<br>' + \
                    'Niezaszczepionych: ' + str(int(sum_n)) + ' (' + str(round(sum_n/(sum_all)*100, 2)) + '%)<br>'

        yz12 = []
        yz13 = []
        yn1 = []
        for i in x0:
            if i in xz2:
                yz12.append(yz2[xz2.index(i)])
            else:
                yz12.append(0)
            if i in xz3:
                yz13.append(yz3[xz3.index(i)])
            else:
                yz13.append(0)
            if i in xn:
                yn1.append(yn[xn.index(i)])
            else:
                yn1.append(0)

        df_pop['population_mz'] = df_pop['population_mzm'] + df_pop['population_mzk']
        population = df_pop['population_mzm'].sum() + df_pop['population_mzk'].sum()

        # zaszczepieni pełną dawką

        fig_data = go.Bar(
            x=x0,
            y=yz12,
            yaxis='y1',
            name='Zaszczepieni pełną dawką',
        )
        traces.append(fig_data)

        # niezaszczepieni

        fig_data = go.Bar(
            x=x0,
            y=yn1,
            yaxis='y1',
        name='Niezaszczepieni',
        )
        traces.append(fig_data)

        # zaszczepieni dawką przypominającą

        fig_data = go.Bar(
            x=x0,
            y=yz13,
            yaxis='y1',
            name='Zaszczepieni dawką przypominającą',
        )
        traces.append(fig_data)

        # populacja

        fig_data = go.Scatter(
            x=x0,
            y=[np.nan] + list(df_pop['population_mz'] / population * 100),
            yaxis='y2',
            name='Udział grupy wiekowej w populacji',
        )
        traces.append(fig_data)

        tytul = 'Statystyka zgonów w grupach wiekowych ' \
                '(od ' + str(od_daty)[:10] + ' do ' + str(do_daty)[:10] + ')' + \
                '<br><sub>Dane: Ministerstwo Zdrowia (BASiW), GUS'
        figure = go.Figure(
            data=traces,
            layout=dict(
                template=template,
                title=dict(text=tytul),
                height=float(settings['plot_height']) * 760,
                legend=dict(x=0.02, y=1., orientation='h'),
                xaxis=dict(tickmode='array',
                           tickvals=x0,
                           ticktext=x0),
                margin=dict(l=settings['marginl'] + 50, r=settings['marginr'] + 70, b=settings['marginb'] + 50,
                            t=settings['margint'] + 70,
                            pad=4),
                yaxis1=dict(
                    title=dict(text='Liczba zgonów',
                               font=dict(color=settings['color_3'],
                                         size=settings['font_size_xy'],
                                         )),
                ),
                yaxis2=dict(
                    title=dict(text='% udziału grupy wiekowej w populacji',
                               font=dict(color=settings['color_3'],
                                         size=settings['font_size_xy'],
                                         )),
                    anchor="x",
                    ticksuffix='%',
                    overlaying="y",
                    side="right",
                ),
            )
        )
        figure.add_annotation(text=anno_text,
                              xref="paper", yref="paper",
                              align='left',
                              xanchor='left',
                              font=dict(color=settings['color_4'], size=14),
                              x=0.05, y=0.9,
                              showarrow=False)
        figure = add_copyright(figure, settings)
        return figure

    od_daty = str(_oddaty)[:10]
    do_daty = str(_dodaty)[:10]
    df = df[df['date'].between(od_daty, do_daty)].copy()

    # dane populacyjne

    df_pop = pd.read_csv('data/dict/slownik_ludnosc_31.12.2020.csv')
    def f(x):
        for i in range(1, len(constants.age_bins_mz)):
            if x >= constants.age_bins_mz[i-1] and x < constants.age_bins_mz[i]:
                return i
        return 0
    df_pop['bin_mz'] = df_pop['wiek'].apply(f)
    df_pop['population_mz'] = df_pop.groupby(['bin_mz'])['population'].transform('sum')
    df_pop['population_mzm'] = df_pop.groupby(['bin_mz'])['population_m'].transform('sum')
    df_pop['population_mzk'] = df_pop.groupby(['bin_mz'])['population_k'].transform('sum')
    df_pop.drop_duplicates(subset=['bin_mz'], inplace=True)

    figure = wykres(df, df_pop, od_daty, do_daty)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2'),
    ]
    return ret_val


##########################################################################
#  Dzienne zgony Covid-19 w grupach wiekowych
##########################################################################

def layout_more_mzage4(_0, _typ, _2, _3):
    template = get_template()
    settings = session2settings()

    # _typ
    # brutto, na 100k, proporcja
    # dane GUS 31.12.2020
    # bin, opis, % zaszcz., populacja
    bins = {
        # 2: ['< 12', 0.000001, 4660956],
        1: ['12-19', 0.4365, 3011688, 'red'],
        2: ['20-39', 0.5229, 10418299, 'green'],
        3: ['40-59', 0.6523, 10373837, 'blue'],
        4: ['60-69', 0.735, 5185843, 'orange'],
        5: ['70+', 0.8104, 4614390, 'purple']
    }
    df = pd.read_csv(constants.data_files['basiw_d']['data_fn'])
    if _typ == 'brutto':
        tytul = 'Dzienne zgony Covid-19 w grupach wiekowych'
    else:
        tytul = 'Dzienne zgony Covid-19 w grupach wiekowych na 100k populacji grupy wiekowej'
    traces = []

    anno_text = 'Podsumowanie zgonów w grupach wiekowych:<br>'
    for bin in bins.keys():

        # wszyscy

        dfx = df[df['bin'] == bin].copy()
        dfx['total'] = dfx.groupby(['date', 'bin'])['d_all'].transform('sum')
        dfx = dfx.drop_duplicates(subset=['date', 'bin']).copy()
        dfx.sort_values(by=['date'], inplace=True)
        if _typ == 'brutto':
            dfx['data'] = dfx['total']
        elif _typ == 'na 100k':
            dfx['data'] = dfx['total'] / bins[bin][2] * 100000
        else:
            dfx['data'] = dfx['total'] / bins[bin][2] * 100000

        anno_text += '<br>' + bins[bin][0] + ': ' + str(round(dfx['total'].sum() / bins[bin][2] * 100, 3)) + '%'
        x = list(dfx['date'])
        y = list(dfx['data'].rolling(session['average_days'], min_periods=1).mean())
        fig_data = go.Scatter(
            x=x,
            y=y,
            name=bins[bin][0] + ' niezaszczepieni',
            line=dict(color=bins[bin][3], width=settings['linewidth_basic'], dash='solid')
        )
        traces.append(fig_data)

        # zaszczepieni

        dfx = df[(df['bin'] == bin) & (df['wpelni'] == 'T')].copy()
        dfx['total'] = dfx.groupby(['date', 'bin'])['d_all'].transform('sum')
        dfx = dfx.drop_duplicates(subset=['date', 'bin']).copy()
        dfx.sort_values(by=['date'], inplace=True)
        if _typ == 'brutto':
            dfx['data'] = dfx['total']
        elif _typ == 'na 100k':
            dfx['data'] = dfx['total'] / bins[bin][2] * bins[bin][1] * 100000
        else:
            dfx['data'] = dfx['total'] / bins[bin][2] * bins[bin][1] * 100000
        x = list(dfx['date'])
        y = list(dfx['data'].rolling(session['average_days'], min_periods=1).mean())
        fig_data = go.Scatter(
            x=x,
            y=y,
            name=bins[bin][0] + ' zaszczepieni',
            line=dict(color=bins[bin][3], width=settings['linewidth_basic'], dash='dash')
        )
        traces.append(fig_data)

    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text=tytul),
            height=float(settings['plot_height']) * 760,
            legend=dict(x=1.02, y=1., orientation='v'),
            yaxis={
                'exponentformat': 'none',
                'type': 'log' if settings['radio_scale'] == 'log' else 'linear',
            },
        )
    )
    figure = add_copyright(figure, settings)
    figure.add_annotation(text=anno_text,
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.8, y=0.9,
                          # x=0.01, y=0.8,
                          showarrow=False)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


##################################################
#  CFR w falach pandemii
##################################################

def layout_more_cfr1(_0, _data, rodzaj, _3):
    template = get_template()
    settings = session2settings()
    nn = 22
    fale = {
        'cały rok 2021': ['2021-01-01', '2021-12-31'],
        'jesień 2020': ['2021-01-01', '2021-02-03'],
        'wiosna 2021': ['2021-02-04', '2021-05-31'],
        'jesień 2021': ['2021-10-10', '2022-01-04'],
        'zima 2022': ['2022-01-05', '2022-03-07']
    }
    ages5_0 = constants.age_bins['bin5']
    ages5 = {k: ages5_0[k] for k in range(0, nn)}
    # ages5 = {k: ages5_0[k] for k in [0] + list(range(11, 22))}
    oddaty = str(_data)[:10]
    df_d = pd.read_csv(constants.data_files['basiw_d']['data_fn'])
    df_d = df_d[['date', 'bin5', 'd_all', 'wpelni']].copy()
    df_d = df_d[df_d['bin5'] < nn].copy()
    if rodzaj == 'zaszczepieni':
        df_d = df_d[df_d.wpelni == 'T'].copy()
    if rodzaj == 'niezaszczepieni':
        df_d = df_d[df_d.wpelni == 'N'].copy()
    df_d['d_all'] = df_d['d_all']
    # df_d['d_all'] = df_d['d_all'].shift(-14)
    df_d.sort_values(by=['bin5'], inplace=True)
    df_i = pd.read_csv(constants.data_files['basiw_i']['data_fn'])
    df_i = df_i[['date', 'bin5', 'i_all', 'wpelni']].copy()
    df_i = df_i[df_i['bin5'] < nn].copy()
    if rodzaj == 'zaszczepieni':
        df_i = df_i[df_i.wpelni == 'T'].copy()
    if rodzaj == 'niezaszczepieni':
        df_i = df_i[df_i.wpelni == 'N'].copy()
    df_i.sort_values(by=['bin5'], inplace=True)

    traces = []
    # x = [n for n in sorted(list(df_i['bin5'].unique()))]
    x = [ages5[n] for n in sorted(list(df_i['bin5'].unique()))]
    for fala in fale:
        df_dx = df_d[df_d['date'].between(fale[fala][0], fale[fala][1])].copy()
        df_dx['total_d_bin'] = df_dx.groupby(['bin5'])['d_all'].transform('sum')
        df_dx = df_dx.drop_duplicates(subset=['bin5']).copy()

        df_ix = df_i[df_i['date'].between(fale[fala][0], fale[fala][1])].copy()
        df_ix['total_i_bin'] = df_ix.groupby(['bin5'])['i_all'].transform('sum')
        df_ix['total_i_bin'] = df_ix['total_i_bin']
        df_ix = df_ix.drop_duplicates(subset=['bin5']).copy()
        df = pd.merge(df_dx, df_ix, how='left',
                      left_on=['bin5'], right_on=['bin5']).copy()
        df['liczba'] = df['total_d_bin'] / df['total_i_bin']

        df.reset_index(drop=True, inplace=True)
        df.sort_values(by=['bin5'], inplace=True)
        # df = df[df['date'] > oddaty].copy()
        idx = df.index
        y = []
        b0 = list(df.bin5)
        l0 = list(df.liczba)
        for i in list(df_i['bin5'].unique()):
            if i in b0:
                y.append(l0[b0.index(i)])
            else:
                y.append(0)

        fig_data = go.Bar(
            x=x,
            y=y,
            name=fala + ' (' +fale[fala][0] + ' ' + fale[fala][1] + ')'
        )
        traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text='CFR<br>status zaszczepienia: ' + rodzaj),
            height=float(settings['plot_height']) * 760,
            margin=dict(l=80, r=50, b=50, t=140, pad=2),
            legend=dict(x=0, y=1.02, orientation='v'),
            yaxis=dict(type='log')
        )
    )
    fig = dcc.Graph(id='xxxx', figure=figure)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


##################################################
#  CFR a status szczepienia
##################################################

def layout_more_cfr2(_0, _1, _2, _3):

    from datetime import datetime, timedelta

    template = get_template()
    settings = session2settings()
    duration_d = settings['duration_d']
    ages5 = constants.age_bins['bin5'].copy()
    del ages5[0]
    del ages5[21]
    ages5[20] = '95+'
    oddaty = str(settings['from_date'])[:10]
    dodaty = str(settings['to_date'])[:10]

    data_0d = datetime.strptime(oddaty, '%Y-%m-%d')
    data_14d = str(data_0d - timedelta(days=duration_d))[:10]

    df_d = pd.read_csv(constants.data_files['basiw_d']['data_fn'])
    df_d = df_d[['date', 'bin5', 'd_all', 'wpelni']].copy()
    df_d = df_d[df_d['bin5'] > 0].copy()
    df_d = df_d[df_d['date'].between(oddaty, dodaty)].copy()
    df_d['bin5'] = df_d['bin5'].replace(21, 20)
    df_d.sort_values(by=['bin5'], inplace=True)

    df_i = pd.read_csv(constants.data_files['basiw_i']['data_fn'])
    df_i = df_i[['date', 'bin5', 'i_all', 'wpelni']].copy()
    df_i = df_i[df_i['bin5'] > 0].copy()

    data_max = df_i.date.max()
    data_maxd = datetime.strptime(dodaty, '%Y-%m-%d')
    data_max14d = str(data_maxd - timedelta(days=duration_d))[:10]

    # df_i['date'] = (pd.to_datetime(df_i.date) - pd.Timedelta(days=duration_d)).astype(str).str.slice(0, 10)
    df_i = df_i[df_i['date'] > oddaty].copy()
    df_i['bin5'] = df_i['bin5'].replace(21, 20)
    df_i = df_i[df_i['date'].between(data_14d, data_max14d)].copy()
    df_i.sort_values(by=['bin5'], inplace=True)

    traces = []
    x = [ages5[n] for n in sorted(list(df_i['bin5'].unique()))]
    anno = '<b>Średni współczynnik CFR ('+str(duration_d)+' dni):</b><br><br>'
    for status in [1, 2]:
        if status == 1:
            df_dx = df_d[df_d.wpelni == 'T'].copy()
            df_ix = df_i[df_i.wpelni == 'T'].copy()
            name = 'zaszczepieni'
        else:
            df_dx = df_d[df_d.wpelni == 'N'].copy()
            df_ix = df_i[df_i.wpelni == 'N'].copy()
            name = 'niezaszczepieni'
        df_dx['total_d_bin'] = df_dx.groupby(['bin5'])['d_all'].transform('sum')
        df_dx = df_dx.drop_duplicates(subset=['bin5']).copy()

        df_ix['total_i_bin'] = df_ix.groupby(['bin5'])['i_all'].transform('sum')
        df_ix['total_i_bin'] = df_ix['total_i_bin']
        df_ix = df_ix.drop_duplicates(subset=['bin5']).copy()
        df = pd.merge(df_dx, df_ix, how='left',
                      left_on=['bin5'], right_on=['bin5']).copy()
        df['liczba'] = round(df['total_d_bin'] / df['total_i_bin'] * 100, 1)
        cfr = round(df['total_d_bin'].sum() / df['total_i_bin'].sum() * 100, 2)
        anno += name + ' ' + str(cfr) + '%<br>'
        df.reset_index(drop=True, inplace=True)
        df.sort_values(by=['bin5'], inplace=True)
        y = []
        b0 = list(df.bin5)
        l0 = list(df.liczba)
        for i in list(df_i['bin5'].unique()):
            if i in b0:
                y.append(l0[b0.index(i)])
            else:
                y.append(0)

        fig_data = go.Bar(
            x=x,
            y=y,
            text=y,
            textposition='outside',
            name=name
        )
        traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text='Współczynnik śmiertelności CFR vs. status zaszczepienia' + \
                            '<br>w okresie od ' + oddaty + '  do ' + dodaty),
            height=float(settings['plot_height']) * 760,
            margin=dict(l=80, r=50, b=50, t=140, pad=2),
            legend=dict(x=0, y=1.02, orientation='v'),
            xaxis=dict(title=dict(text='Grupa wiekowa',
                                  font=dict(
                                      size=int(settings['font_size_xy']),
                                      color=get_session_color(7)
                                  )),
                       ),
            yaxis=dict(title=dict(text='CFR (%)',
                                  font=dict(
                                      size=int(settings['font_size_xy']),
                                      color=get_session_color(7)
                                  )),
                       ),
        )
    )
    figure.add_annotation(text=anno,
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.02, y=0.8,
                          showarrow=False)
    fig = dcc.Graph(id='xxxx', figure=figure)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


##################################################
#  CFR vs. rodzaj szczepionki
##################################################

def layout_more_cfr3(_0, _data, _2, _3):
    template = get_template()
    settings = session2settings()
    bin_type = 'bin_ecdc'
    ages = constants.age_bins['bin_ecdc'].copy()
    options = ['dawka przypominająca', 'szczepionki mRNA (Pfizer, Moderna)', 'Astra Zeneca', 'Johnson&Johnson',
               'wszystkie szczepionki', 'niezaszczepieni', 'wszyscy']
    df_d = pd.read_csv(constants.data_files['basiw_d']['data_fn'])
    df_d = df_d[['date', bin_type, 'd_all', 'wpelni', 'marka', 'dawka']].copy()
    df_d = df_d[df_d[bin_type] >= 7].copy()
    df_d['d_all'] = df_d['d_all'].shift(-16)
    df_d.sort_values(by=[bin_type], inplace=True)
    df_i = pd.read_csv(constants.data_files['basiw_i']['data_fn'])
    df_i = df_i[['date', bin_type, 'i_all', 'wpelni', 'marka', 'dawka']].copy()
    df_i = df_i[df_i[bin_type] >= 7].copy()
    df_i.sort_values(by=[bin_type], inplace=True)

    traces = []
    x = [ages[n] for n in sorted(list(df_i[bin_type].unique()))]
    anno = '<b>Średni współczynnik CFR za okres od 1.01.2021 (delta_t = 16 dni):</b><br><br>'
    brands = ['Johnson&Johnson', 'Astra Zeneca']
    mrna = ['Pfizer', 'Moderna']
    for option in options:
        if option in brands:
            df_dx = df_d[(df_d.wpelni == 'T') & (df_d.marka == option)].copy()
            df_ix = df_i[(df_i.wpelni == 'T') & (df_i.marka == option)].copy()
            name = 'pełne zaszczepienie - ' + option
        elif option == 'wszystkie szczepionki':
            df_dx = df_d[(df_d.wpelni == 'T') & (df_d.marka.isin(brands))].copy()
            df_ix = df_i[(df_i.wpelni == 'T') & (df_i.marka.isin(brands))].copy()
            name = 'pełne zaszczepienie - dowolna szczepionka'
        elif option == 'dawka przypominająca':
            df_dx = df_d[df_d.dawka == 'przypominajaca'].copy()
            df_ix = df_i[df_i.dawka == 'przypominajaca'].copy()
            name = 'dawka przypominająca'
        elif option == 'niezaszczepieni':
            df_dx = df_d[df_d.wpelni == 'N'].copy()
            df_ix = df_i[df_i.wpelni == 'N'].copy()
            name = 'niezaszczepieni lub nie w pełni zaszczepieni'
        elif option == 'szczepionki mRNA (Pfizer, Moderna)':
            df_dx = df_d[(df_d.wpelni == 'T') & (df_d.marka.isin(mrna))].copy()
            df_ix = df_i[(df_i.wpelni == 'T') & (df_i.marka.isin(mrna))].copy()
            name = 'szczepionki mRNA (Pfizer, Moderna)'
        else:
            df_dx = df_d.copy()
            df_ix = df_i.copy()
            name = 'wszyscy'
        df_dx['total_d_bin'] = df_dx.groupby([bin_type])['d_all'].transform('sum')
        df_dx = df_dx.drop_duplicates(subset=[bin_type]).copy()

        df_ix['total_i_bin'] = df_ix.groupby([bin_type])['i_all'].transform('sum')
        df_ix = df_ix.drop_duplicates(subset=[bin_type]).copy()
        df = pd.merge(df_dx, df_ix, how='left',
                      left_on=[bin_type], right_on=[bin_type]).copy()
        df.reset_index(drop=True, inplace=True)
        df.sort_values(by=[bin_type], inplace=True)
        df['liczba'] = round(df['total_d_bin'] / df['total_i_bin'] * 100, 1)
        cfr = round(df['total_d_bin'].sum() / df['total_i_bin'].sum() * 100, 1)
        anno += name + ' ' + str(cfr) + '%<br>'
        y = []
        b0 = list(df[bin_type])
        l0 = list(df.liczba)
        for i in list(df_i[bin_type].unique()):
            if i in b0:
                y.append(l0[b0.index(i)])
            else:
                y.append(0)

        fig_data = go.Bar(
            x=x,
            y=y,
            text=y,
            textposition='outside',
            name=name
        )
        traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text='Współczynnik śmiertelności CFR vs. dokładny status zaszczepienia' + \
                       '<br>Dane: MZ, od 1.01.2021, grupy wiekowe 50+'),
            height=float(settings['plot_height']) * 760,
            margin=dict(l=80, r=50, b=50, t=140, pad=2),
            legend=dict(x=0, y=1.02, orientation='h'),
            xaxis=dict(title=dict(text='Grupa wiekowa',
                                  font=dict(
                                      size=int(settings['font_size_xy']),
                                      color=get_session_color(7)
                                  )),
                       ),
            yaxis=dict(title=dict(text='CFR (%)',
                                  font=dict(
                                      size=int(settings['font_size_xy']),
                                      color=get_session_color(7)
                                  )),
                       ),
        )
    )
    figure.update_layout(uniformtext_minsize=12)
    figure.add_annotation(text=anno,
                          xref="paper", yref="paper",
                          align='left',
                          font=dict(color=settings['color_4'],
                                    size=settings['font_size_anno']),
                          x=0.02, y=0.8,
                          showarrow=False)
    fig = dcc.Graph(id='xxxx', figure=figure)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


##########################################
# % nadmiarowych zgonów w województwach
##########################################
def layout_more_nzwall(DF, _1, _2, _3):
    # session['template'] = 'white'
    # session['template_change'] = True
    template = get_template()
    settings = session2settings()

    # dane Eurostat do obliczenia średniej

    df_euro = pd.read_csv(constants.data_files['mortality_eurostat']['data_fn'])
    df_euro = df_euro[
        (df_euro['age'] == 'TOTAL') &
        (df_euro['short'].str.startswith('PL'))] \
        .copy()

    locations = df_euro[df_euro.location.str.islower()]['location'].unique()
    locations.sort()

    means = {}
    df_means = df_euro.loc[(df_euro['year'].between(2016, 2019))].copy()
    for i in locations:
        print(i)
        dx = df_means[df_means['location'] == i]['total']
        if len(dx) > 0:
            means[i] = dx.mean()
    df_euro = df_euro.loc[df_euro['year'].isin([2020, 2021, 2022])].copy()

    # dane Covid

    df_w = DF['poland'].copy()
    df_w['location'] = df_w['location'].str.lower()
    df_we = df_w[df_w['location'].isin(locations)].copy()

    # sumowanie tygodniowe

    df_we.index = df_we['date'].astype('datetime64[ns]')
    df_we = df_we.groupby('location').resample('W-mon').sum().reset_index()
    df_we['date'] = df_we['date'].dt.strftime('%Y-%m-%d')

    # połączenie z danymi Covid

    df_euro['total'] = df_euro['total'].replace(0, np.nan)
    df_euro = pd.merge(df_euro, df_we[['date', 'location', 'new_deaths', 'population']], how='right',
                       left_on=['date', 'location'],
                       right_on=['date', 'location']).copy()
    df_euro.dropna(axis='rows', subset=['year'], inplace=True)
    cols = 6
    row_height = 220
    rows = int((len(locations) - 1) / cols) + 1
    figure = make_subplots(rows=rows,
                           cols=cols,
                           subplot_titles=locations,
                           print_grid=False,
                           )
    row = 1
    col = 0

    # przygotowanie danych w jednej tabeli

    for location in locations:
        filtr = df_euro['location'] == location
        df_euro.loc[filtr, 'percent'] = round(
            (df_euro[filtr]['new_deaths'] / (df_euro[filtr]['total'] - means[location])) * 100, 2)
        df_euro.loc[filtr, 'mean'] = round(means[location], 0)
    df_euro[df_euro['date'] >= '2020-08-31'].to_csv('data/excess_deaths_woj.csv', index=False)

    value_min = 0
    value_max = max(df_euro['total'])
    for location in locations:
        filtered = df_euro[(df_euro['location'] == location) & (df_euro['date'] > '2020-10-01')]
        filtered.sort_values(by=['date'], inplace=True)
        filtered.reset_index(drop=True, inplace=True)

        # wspólne skalowanie osi

        # value_min = 0
        # value_max = max(filtered['total'])
        xvals = list(filtered['date'])
        col += 1
        if col > cols:
            row += 1
            col = 1

        # Liczba zgonów euro

        yvals = list(filtered['total'])
        fig_data = go.Scatter(x=xvals,
                              y=yvals,
                              line=dict(color='blue', width=1),
                              name=location)
        figure.add_trace(fig_data, row=row, col=col)

        # Liczba zgonów Covid

        yvals = list(filtered['new_deaths'])
        yvals = [0 if math.isnan(i) else i for i in yvals]
        fig_data = go.Bar(x=xvals,
                          y=yvals,
                          marker={'color': 'red'},
                          name=location)
        figure.add_trace(fig_data, row=row, col=col)

        # średnia

        xvals = list(filtered['date'])
        yvals = [means[location] for i in range(len(xvals))]
        fig_data = go.Scatter(x=xvals,
                              y=yvals,
                              line=dict(color='green', width=1),
                              name=location)
        figure.add_trace(fig_data, row=row, col=col)

        figure.update_yaxes(range=[value_min, value_max], row=row, col=col)

    height = 100 + row_height * row
    figure.update_layout(height=height,
                         template=template,
                         title=dict(text='Nadmiarowe zgony (<i>excess deaths</i>) oraz zgony Covid w województwach<br>'
                                         'w porównaniu ze średnią 2018-2019 (sumy tygodniowe, od 01.10.2020)'),
                         showlegend=False),
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'docent_image',
            'width': 1200, 'height': height, 'scale': 1}
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)

    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


##################################################
#  Heatmap - zgony nadmiarowe
##################################################

def layout_more_nz_heatmap(DF, _location, _2, _3):
    template = get_template()
    settings = session2settings()

    df = pd.read_csv(constants.data_files['mortality_eurostat']['data_fn'])
    df = df[(df['age'] == 'TOTAL') & (df['location'] == _location)].copy()
    df.sort_values(by=['date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    # years = list(df.year.unique())
    # ymap = {x: str(x) for x in years}
    title = 'Zgony w latach 2000-2021 (tygodniowo)<br>' + _location
    fig = heatmap_bis(df, 'year', 'week', 'total', settings, template, title=title)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val



###########################################################
# Poównanie zgonów nadwmiarowych w województwach za okres
###########################################################
def layout_more_nzwcmp(DF, _1, _2, _3):
    settings = session2settings()

    df_i = DF['poland'].copy()
    df_i = df_i[['wojew', 'Lat', 'Long']].copy()
    df_i.drop_duplicates(subset=['wojew'], inplace=True)

    df_m = pd.read_csv(constants.data_files['mortality_eurostat']['data_fn'])
    df_m = df_m[
        (df_m['age'] == 'TOTAL') &
        (df_m['short'] != 'PL') &
        (df_m['short'].str.startswith('PL'))] \
        .copy()

    df_m = df_m[df_m['total'] != 0].copy()
    df_m.sort_values(by=['location'], inplace=True)
    sum1519 = df_m[df_m['year'].between(2015, 2019)].groupby('location')['total'].sum()
    sum2020 = df_m[df_m['year'] == 2020].groupby('location')['total'].sum()
    sum2021 = df_m[df_m['year'] == 2021].groupby('location')['total'].sum()
    sum2022 = df_m[df_m['year'] == 2022].groupby('location')['total'].sum()
    weeks1519 = df_m[df_m['year'].between(2015, 2019)].groupby('location')['total'].count()
    weeks2020 = df_m[df_m['year'] == 2020].groupby('location')['total'].count()
    weeks2021 = df_m[df_m['year'] == 2021].groupby('location')['total'].count()
    weeks2022 = df_m[df_m['year'] == 2022].groupby('location')['total'].count()
    df_m.drop_duplicates(subset=['location'], inplace=True)
    df_m['mean1519'] = list(sum1519 / weeks1519)
    df_m['mean2020'] = list(sum2020 / weeks2020)
    df_m['mean2021'] = list(sum2021 / weeks2021)
    df_m['mean2022'] = list(sum2022 / weeks2022)
    df_m['ile2020'] = round((df_m['mean2020'] / df_m['mean1519'] - 1) * 100, 2)
    df_m['ile2021'] = round((df_m['mean2021'] / df_m['mean1519'] - 1) * 100, 2)
    df_m['ile2022'] = round((df_m['mean2022'] / df_m['mean1519'] - 1) * 100, 2)

    df_m = pd.merge(df_m, df_i, how='left', left_on=['location'], right_on=['wojew'])

    df_m['Long'] = df_m['wojew'].apply(lambda x: constants.wojew_mid_lower[x][1])
    df_m['Lat'] = df_m['wojew'].apply(lambda x: constants.wojew_mid_lower[x][0])

    tytul1 = 'Nadwyżka zgonów nadmiarowych w roku 2020<br><sup>w porównaniu do średniej z lat 2015-1019'
    tytul2 = 'Nadwyżka zgonów nadmiarowych w roku 2021<br><sup>w porównaniu do średniej z lat 2015-1019'
    tytul3 = 'Nadwyżka zgonów nadmiarowych w roku 2022<br><sup>w porównaniu do średniej z lat 2015-1019'

    mapbox_access_token = open("data/mapbox.token").read()
    mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=19.1, lat=52.3), zoom=5.1)
    # mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=19.1, lat=52.3), zoom=4.82)
    mapbox['style'] = constants.mapbox_styles[settings['map_mapbox']]

    json_file = r'data/geojson/woj.min.geo.json'
    locations = df_m['location']
    featuredikey = 'properties.nazwa'
    with open(json_file, encoding="utf-8") as f:
        geojson = json.load(f)

    figure1 = go.Figure()
    figure1.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        featureidkey=featuredikey,
        locations=locations,
        z=df_m['ile2020'],
        showscale=False,
        text='',
        marker=dict(line=dict(color='black', width=0.5 if 'drawborder' in settings['map_options'] else 0.)),
        hoverinfo='text',
        colorscale=settings['map_color_scale'],
        reversescale=True if 'reversescale' in settings['map_options'] else False,
        marker_opacity=settings['map_opacity'],
        colorbar=dict(
            len=0.5,
            tickcolor='red',
            tickfont=dict(color=get_session_color(10)),
            orientation='h',
            # bgcolor='white',
            title=dict(text='2020'),
            ticksuffix='%',
        ),
    ))
    anno_text1 = df_m['ile2020'].round(2).astype(str) + '%'
    figure1.add_trace(
        go.Scattermapbox(
            lat=df_m.Lat, lon=df_m.Long,
            mode='text',
            hoverinfo='none',
            below="''",
            marker=dict(allowoverlap=True),
            text=anno_text1,
            textfont=dict(family='Bebas Neue', size=settings['font_size_anno'], color=get_session_color(4)),
        ))
    figure1.update_layout(
        images=constants.image_logo_map,
        autosize=True,
        hovermode='closest',
        title=dict(
            text=tytul1,
            x=settings['titlexpos'], y=settings['titleypos'] - 0.04,
            font=dict(size=settings['font_size_title'], color=get_session_color(7)
                      )
        ),
        height=int(settings['map_height']),
        margin=dict(l=settings['marginl'], r=settings['marginr'], b=settings['marginb'], t=settings['margint'],
                    pad=0),
        paper_bgcolor=get_session_color(1),
        plot_bgcolor=get_session_color(1),
        mapbox=mapbox,
    )
    figure2 = go.Figure()
    figure2.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        featureidkey=featuredikey,
        locations=locations,
        z=df_m['ile2021'],
        showscale=False,
        text='',
        marker=dict(line=dict(color='black', width=0.5 if 'drawborder' in settings['map_options'] else 0.)),
        hoverinfo='text',
        colorscale=settings['map_color_scale'],
        reversescale=True if 'reversescale' in settings['map_options'] else False,
        marker_opacity=settings['map_opacity'],
        colorbar=dict(
            len=0.5,
            # bgcolor='white',
            tickcolor='red',
            orientation='h',
            tickfont=dict(color=get_session_color(10)),
            title=dict(text='2021'),
            ticksuffix='%',
        ),
    ))
    anno_text2 = df_m['ile2021'].round(2).astype(str) + '%'
    figure2.add_trace(
        go.Scattermapbox(
            lat=df_m.Lat, lon=df_m.Long,
            mode='text',
            hoverinfo='none',
            below="''",
            marker=dict(allowoverlap=True),
            text=anno_text2,
            textfont=dict(family='Bebas Neue', size=settings['font_size_anno'], color=get_session_color(4)),
        ))
    figure2.update_layout(
        images=constants.image_logo_map,
        autosize=True,
        hovermode='closest',
        title=dict(
            text=tytul2,
            x=settings['titlexpos'], y=settings['titleypos'] - 0.04,
            font=dict(size=settings['font_size_title'], color=get_session_color(7)
                      )
        ),
        height=int(settings['map_height']),
        margin=dict(l=settings['marginl'], r=settings['marginr'], b=settings['marginb'], t=settings['margint'],
                    pad=0),
        paper_bgcolor=get_session_color(1),
        plot_bgcolor=get_session_color(1),
        mapbox=mapbox,
    )
    figure3 = go.Figure()
    figure3.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        featureidkey=featuredikey,
        locations=locations,
        z=df_m['ile2022'],
        showscale=False,
        text='',
        marker=dict(line=dict(color='black', width=0.5 if 'drawborder' in settings['map_options'] else 0.)),
        hoverinfo='text',
        colorscale=settings['map_color_scale'],
        reversescale=True if 'reversescale' in settings['map_options'] else False,
        marker_opacity=settings['map_opacity'],
        colorbar=dict(
            len=0.5,
            tickcolor='red',
            orientation='h',
            tickfont=dict(color=get_session_color(10)),
            title=dict(text='2022'),
            ticksuffix='%',
        ),
    ))
    anno_text3 = df_m['ile2022'].round(2).astype(str) + '%'
    figure3.add_trace(
        go.Scattermapbox(
            lat=df_m.Lat, lon=df_m.Long,
            mode='text',
            hoverinfo='none',
            below="''",
            marker=dict(allowoverlap=True),
            text=anno_text3,
            textfont=dict(family='Bebas Neue', size=settings['font_size_anno'], color=get_session_color(4)),
        ))
    figure3.update_layout(
        images=constants.image_logo_map,
        autosize=True,
        hovermode='closest',
        title=dict(
            text=tytul3,
            x=settings['titlexpos'], y=settings['titleypos'] - 0.04,
            font=dict(size=settings['font_size_title'], color=get_session_color(7)
                      )
        ),
        height=int(settings['map_height']),
        margin=dict(l=settings['marginl'], r=settings['marginr'], b=settings['marginb'], t=settings['margint'],
                    pad=0),
        paper_bgcolor=get_session_color(1),
        plot_bgcolor=get_session_color(1),
        mapbox=mapbox,
    )
    figure1 = add_copyright(figure1, settings)
    config = {
        'displaylogo': False,
        'responsive': True,
        'toImageButtonOptions': {
            'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
            'filename': 'docent_image', 'height': 700, 'width': 1200}
    }
    ret_val = [
        dbc.Col(dcc.Graph(id='mapa1', figure=figure1, config=config), className='pr-0 pl-0 mr-0 ml-0'),
        dbc.Col(dcc.Graph(id='mapa2', figure=figure2, config=config), className='pr-0 pl-0 mr-0 ml-0'),
        dbc.Col(dcc.Graph(id='mapa3', figure=figure3, config=config), className='pr-0 pl-0 mr-0 ml-0')
    ]

    return ret_val


############################################################
# Mapa. Zgony nadmiarowe 2020-2022 - poróœnanie ze średnią
############################################################
def layout_more_nzwcmp_1(DF, _age, _lata, _option):
    settings = session2settings()

    df_i = DF['poland'].copy()
    df_i = df_i[['wojew', 'population', 'Lat', 'Long']].copy()
    df_i.drop_duplicates(subset=['wojew'], inplace=True)
    # ['Y_LT20', 'UNK', 'Y20-39', 'Y40-59', 'TOTAL', 'Y_GE80', 'Y60-79']
    df_m = pd.read_csv(constants.data_files['mortality_eurostat']['data_fn'])
    df_m = df_m[
        (df_m['age'] == _age) &
        (df_m['short'] != 'PL') &
        (df_m['short'].str.startswith('PL'))] \
        .copy()

    df_m = df_m[df_m['total'] != 0].copy()
    df_m.sort_values(by=['location'], inplace=True)
    sum1619 = df_m[df_m['year'].between(2016, 2019)].groupby('location')['total'].sum()
    weeks1619 = df_m[df_m['year'].between(2016, 2019)].groupby('location')['total'].count()
    if _lata == '2020-2022':
        sum22 = df_m[df_m['year'] == 2020].groupby('location')['total'].sum() + \
                df_m[df_m['year'] == 2021].groupby('location')['total'].sum() + \
                df_m[df_m['year'] == 2022].groupby('location')['total'].sum()
        weeks22 = df_m[df_m['year'] == 2020].groupby('location')['total'].count() + \
                  df_m[df_m['year'] == 2021].groupby('location')['total'].count() + \
                  df_m[df_m['year'] == 2022].groupby('location')['total'].count()
    else:
        sum22 = df_m[df_m['year'] == 2021].groupby('location')['total'].sum() + \
                df_m[df_m['year'] == 2022].groupby('location')['total'].sum()
        weeks22 = df_m[df_m['year'] == 2021].groupby('location')['total'].count() + \
                  df_m[df_m['year'] == 2022].groupby('location')['total'].count()
    df_m.drop_duplicates(subset=['location'], inplace=True)
    df_m['mean1619'] = list(sum1619 / weeks1619)
    df_m['mean22'] = list(sum22 / weeks22)
    df_m['ile22'] = round((df_m['mean22'] / df_m['mean1619'] - 1) * 100, 2)
    df_m = pd.merge(df_m, df_i, how='left', left_on=['location'], right_on=['wojew'])
    subtyt = ''
    rounding = 2
    if _option == 'na 100k':
        df_m['mean1619'] = list(df_m['mean1619'] / df_m['population'] * 100000)
        df_m['mean22'] = list(df_m['mean22'] / df_m['population'] * 100000)
        subtyt = ' na 100k '
        rounding = 2
    tytul = 'Nadwyżka zgonów nadmiarowych' + subtyt + 'w latach ' + _lata + \
            '<br><sub>w porównaniu do średniej z lat 2016-2019 (sumy tygodniowe)' + \
            'grupa wiekowa: ' + _age


    df_m['Long'] = df_m['location'].apply(lambda x: constants.wojew_mid_lower[x][1])
    df_m['Lat'] = df_m['location'].apply(lambda x: constants.wojew_mid_lower[x][0])

    mapbox_access_token = open("data/mapbox.token").read()
    mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=19.1, lat=52.3), zoom=5.5)
    mapbox['style'] = constants.mapbox_styles[settings['map_mapbox']]

    json_file = r'data/geojson/woj.min.geo.json'
    locations = df_m['location']
    featuredikey = 'properties.nazwa'
    with open(json_file, encoding="utf-8") as f:
        geojson = json.load(f)

    figure1 = go.Figure()
    figure1.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        featureidkey=featuredikey,
        locations=locations,
        z=df_m['ile22'],
        text='',
        marker=dict(line=dict(color='black', width=0.5 if 'drawborder' in settings['map_options'] else 0.)),
        hoverinfo='text',
        colorscale=settings['map_color_scale'],
        reversescale=True if 'reversescale' in settings['map_options'] else False,
        marker_opacity=settings['map_opacity'],
        colorbar=dict(
            len=0.5,
            tickcolor='red',
            tickfont=dict(color=get_session_color(10)),
            title=dict(text=''),
            ticksuffix='%',
        ),
    ))
    anno_text1 = df_m['ile22'].round(2).astype(str) + '%' + \
                 '<br>' + df_m['mean22'].round(rounding).astype(str) + \
                 '<br>' + df_m['mean1619'].round(rounding).astype(str)
    figure1.add_trace(
        go.Scattermapbox(
            lat=df_m.Lat, lon=df_m.Long,
            mode='text',
            hoverinfo='none',
            below="''",
            marker=dict(allowoverlap=True),
            text=anno_text1,
            textfont=dict(family='Bebas Neue', size=settings['font_size_anno'], color=get_session_color(4)),
        ))
    figure1.update_layout(
        images=constants.image_logo_map,
        autosize=True,
        hovermode='closest',
        title=dict(
            text=tytul,
            x=settings['titlexpos'], y=settings['titleypos'] - 0.04,
            font=dict(size=settings['font_size_title'], color=get_session_color(7)
                      )
        ),
        height=int(settings['map_height']),
        margin=dict(l=settings['marginl'], r=settings['marginr'], b=settings['marginb'], t=settings['margint'],
                    pad=0),
        paper_bgcolor=get_session_color(1),
        plot_bgcolor=get_session_color(1),
        mapbox=mapbox,
    )
    figure = add_copyright(figure1, settings)
    config = {
        'displaylogo': False,
        'responsive': True,
        'toImageButtonOptions': {
            'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
            'filename': 'docent_image', 'height': 700, 'width': 1200}
    }
    ret_val = [
        dbc.Col(dcc.Graph(id='mapa1', figure=figure1, config=config), className='pr-0 pl-0 mr-0 ml-0'),
    ]

    return ret_val


############################################################
# Mapa. Porównanie umieralności
############################################################
def layout_more_nzwcmp_2(DF, _age, _years, _option):
    settings = session2settings()
    template = get_template()
    df_i = DF['poland'].copy()
    df_i = df_i[['wojew', 'population', 'Lat', 'Long']].copy()
    df_i.drop_duplicates(subset=['wojew'], inplace=True)
    # ['Y_LT20', 'UNK', 'Y20-39', 'Y40-59', 'TOTAL', 'Y_GE80', 'Y60-79']
    df_m = pd.read_csv(constants.data_files['mortality_eurostat']['data_fn'])
    df_m = df_m[
        (df_m['age'] == _age) &
        (df_m['short'] != 'PL') &
        (df_m['short'].str.startswith('PL'))] \
        .copy()

    df_m = df_m[df_m['total'] != 0].copy()
    df_m.sort_values(by=['location'], inplace=True)
    sum1619 = df_m[df_m['year'].between(2016, 2019)].groupby('location')['total'].sum()
    weeks1619 = df_m[df_m['year'].between(2016, 2019)].groupby('location')['total'].count()
    if _years == '2020-2022':
        sum22 = df_m[df_m['year'] == 2020].groupby('location')['total'].sum() + \
                df_m[df_m['year'] == 2021].groupby('location')['total'].sum() + \
                df_m[df_m['year'] == 2022].groupby('location')['total'].sum()
        weeks22 = df_m[df_m['year'] == 2020].groupby('location')['total'].count() + \
                  df_m[df_m['year'] == 2021].groupby('location')['total'].count() + \
                  df_m[df_m['year'] == 2022].groupby('location')['total'].count()
    elif _years == '2020-2021':
        sum22 = df_m[df_m['year'] == 2020].groupby('location')['total'].sum() + \
                df_m[df_m['year'] == 2021].groupby('location')['total'].sum()
        weeks22 = df_m[df_m['year'] == 2020].groupby('location')['total'].count() + \
                  df_m[df_m['year'] == 2021].groupby('location')['total'].count()
    elif _years == '2020-2020':
        sum22 = df_m[df_m['year'] == 2020].groupby('location')['total'].sum()
        weeks22 = df_m[df_m['year'] == 2020].groupby('location')['total'].count()
    elif _years == '2021-2021':
        sum22 = df_m[df_m['year'] == 2021].groupby('location')['total'].sum()
        weeks22 = df_m[df_m['year'] == 2021].groupby('location')['total'].count()
    else: #
        sum22 = df_m[df_m['year'] == 2022].groupby('location')['total'].sum()
        weeks22 = df_m[df_m['year'] == 2022].groupby('location')['total'].count()
    df_m.drop_duplicates(subset=['location'], inplace=True)
    df_m['mean1619'] = list(sum1619 / weeks1619)
    df_m['mean22'] = list(sum22 / weeks22)
    df_m = pd.merge(df_m, df_i, how='left', left_on=['location'], right_on=['wojew'])
    df_m['mean1619'] = list(df_m['mean1619'] / df_m['population'] * 100000)
    df_m['mean22'] = list(df_m['mean22'] / df_m['population'] * 100000)
    df_m['excess'] = list(df_m['mean22'] - df_m['mean1619'])
    rounding = 2

    df_m['Long'] = df_m['location'].apply(lambda x: constants.wojew_mid_lower[x][1])
    df_m['Lat'] = df_m['location'].apply(lambda x: constants.wojew_mid_lower[x][0])

    mapbox_access_token = open("data/mapbox.token").read()
    mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=19.1, lat=52.3), zoom=5.5)
    mapbox['style'] = constants.mapbox_styles[settings['map_mapbox']]

    json_file = r'data/geojson/woj.min.geo.json'
    locations = df_m['location']
    featuredikey = 'properties.nazwa'
    with open(json_file, encoding="utf-8") as f:
        geojson = json.load(f)

    if _option == 'nadmiarowe':
        tytul1 = '<b>Zgony nadmiarowe na 100k w latach ' + _years
        z = df_m['excess']
    else:
        tytul1 = '<b>Tygodniowa liczba zgonów na 100k w latach 2016-2019'
        z = df_m['mean1619']
    figure1 = go.Figure()
    figure1.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        featureidkey=featuredikey,
        locations=locations,
        z=z,
        text='',
        marker=dict(line=dict(color='black', width=0.5 if 'drawborder' in settings['map_options'] else 0.)),
        hoverinfo='text',
        colorscale=settings['map_color_scale'],
        reversescale=True if 'reversescale' in settings['map_options'] else False,
        marker_opacity=settings['map_opacity'],
        colorbar=dict(
            len=0.5,
            tickcolor='red',
            tickfont=dict(color=get_session_color(10)),
            title=dict(text=''),
            ticksuffix='%',
        ),
    ))
    anno_text1 = z.round(rounding).astype(str)
    figure1.add_trace(
        go.Scattermapbox(
            lat=df_m.Lat, lon=df_m.Long,
            mode='text',
            hoverinfo='none',
            below="''",
            marker=dict(allowoverlap=True),
            text=anno_text1,
            textfont=dict(family='Bebas Neue', size=settings['font_size_anno'], color=get_session_color(4)),
        ))
    figure1.update_layout(
        template=template,
        autosize=True,
        hovermode='closest',
        title=dict(
            text=tytul1,
            x=settings['titlexpos'], y=settings['titleypos'] - 0.04,
            font=dict(size=settings['font_size_title'], color=get_session_color(7)
                      )
        ),
        height=int(settings['map_height']),
        margin=dict(l=settings['marginl'], r=settings['marginr'], b=settings['marginb'], t=settings['margint'],
                    pad=0),
        paper_bgcolor=get_session_color(1),
        plot_bgcolor=get_session_color(1),
        mapbox=mapbox,
    )
    figure1 = add_copyright(figure1, settings)

    tytul2 = '<b>Tygodniowa liczba zgonów na 100k w latach ' + _years
    figure2 = go.Figure()
    figure2.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        featureidkey=featuredikey,
        locations=locations,
        z=df_m['mean22'],
        text='',
        marker=dict(line=dict(color='black', width=0.5 if 'drawborder' in settings['map_options'] else 0.)),
        hoverinfo='text',
        colorscale=settings['map_color_scale'],
        reversescale=True if 'reversescale' in settings['map_options'] else False,
        marker_opacity=settings['map_opacity'],
        colorbar=dict(
            len=0.5,
            tickcolor='red',
            tickfont=dict(color=get_session_color(10)),
            title=dict(text=''),
            ticksuffix='%',
        ),
    ))
    anno_text1 = df_m['mean22'].round(rounding).astype(str)
    figure2.add_trace(
        go.Scattermapbox(
            lat=df_m.Lat, lon=df_m.Long,
            mode='text',
            hoverinfo='none',
            below="''",
            marker=dict(allowoverlap=True),
            text=anno_text1,
            textfont=dict(family='Bebas Neue', size=settings['font_size_anno'], color=get_session_color(4)),
        ))
    figure2.update_layout(
        template=template,
        autosize=True,
        hovermode='closest',
        title=dict(
            text=tytul2,
            x=settings['titlexpos'], y=settings['titleypos'] - 0.04,
            font=dict(size=settings['font_size_title'], color=get_session_color(7)
                      )
        ),
        height=int(settings['map_height']),
        margin=dict(l=settings['marginl'], r=settings['marginr'], b=settings['marginb'], t=settings['margint'],
                    pad=0),
        paper_bgcolor=get_session_color(1),
        plot_bgcolor=get_session_color(1),
        mapbox=mapbox,
    )
    figure2= add_copyright(figure2, settings)
    config = {
        'displaylogo': False,
        'responsive': True,
        'toImageButtonOptions': {
            'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
            'filename': 'docent_image', 'height': 700, 'width': 1200}
    }
    ret_val = [
        dbc.Col(dcc.Graph(id='mapa1', figure=figure1, config=config), className='pr-0 pl-0 mr-0 ml-0'),
        dbc.Col(dcc.Graph(id='mapa2', figure=figure2, config=config), className='pr-0 pl-0 mr-0 ml-0'),
    ]

    return ret_val


##############################################
# porównanie II, III i IV fali epidemii w Polsce
##############################################

def layout_more_cmpf(DF, location, _2, _3):
    session['template'] = 'white'
    session['template_change'] = True
    session.modified = True
    template = get_template()
    settings = session2settings()
    df = DF['poland']
    df_pl = df[df['location'] == location].copy()
    df_j = df_pl[(df_pl['date'] >= '2020-09-16') & (df_pl['date'] < '2021-02-16')].copy()
    df_w = df_pl[(df_pl['date'] >= '2021-02-16') & (df_pl['date'] < '2021-07-17')].copy()
    df_l = df_pl[df_pl['date'] >= '2021-07-18'].copy()
    df_j.reset_index(drop=True, inplace=True)
    df_w.reset_index(drop=True, inplace=True)
    df_l.reset_index(drop=True, inplace=True)
    df_j['day'] = df_j.index
    df_w['day'] = df_w.index
    df_l['day'] = df_l.index
    cols = 3
    rows = 3
    row_height = 230
    titles = [
        "R(t)",
        "Nowe infekcje / 100k",
        "Nowe przypadki śmiertelne / 100K",
        "Wykrywalność",
        "Nowe testy",
        "Zajęte łózka - suma",
        "Zajęte respiratory - suma",
        "Zajęte łózka - przyrost",
        "Zajęte respiratory - przyrost"
    ]
    figure = make_subplots(rows=rows,
                           cols=cols,
                           subplot_titles=titles,
                           print_grid=False,
                           )
    row = 1
    col = 0
    xvals_j = list(df_j['day'])
    xvals_w = list(df_w['day'])
    xvals_l = list(df_l['day'])

    # Rt
    ########################

    yvals_j = list(df_j['reproduction_rate'])
    fig_data1 = go.Scatter(x=xvals_j,
                           y=yvals_j,
                           line=dict(color='blue', width=1),
                           name='jesień 2020')
    yvals_w = list(df_w['reproduction_rate'])
    fig_data2 = go.Scatter(x=xvals_w,
                           y=yvals_w,
                           line=dict(color='green', width=1),
                           name='wiosna 2021')
    yvals_l = list(df_l['reproduction_rate'])
    fig_data3 = go.Scatter(x=xvals_l,
                           y=yvals_l,
                           line=dict(color='red', width=3),
                           name='lato 2021')
    col += 1
    if col > cols:
        row += 1
        col = 1
    figure.add_trace(fig_data1, row=row, col=col)
    figure.add_trace(fig_data2, row=row, col=col)
    figure.add_trace(fig_data3, row=row, col=col)

    # nowe infekcje / 100000
    ##########################

    yvals_j = list((df_j['new_cases'] / df_j['population'] * 100000).rolling(7, min_periods=1).mean())
    fig_data1 = go.Scatter(x=xvals_j,
                           y=yvals_j,
                           line=dict(color='blue', width=1),
                           name='jesień 2020')
    yvals_w = list((df_w['new_cases'] / df_w['population'] * 100000).rolling(7, min_periods=1).mean())
    fig_data2 = go.Scatter(x=xvals_w,
                           y=yvals_w,
                           line=dict(color='green', width=1),
                           name='wiosna 2021')
    yvals_l = list((df_l['new_cases'] / df_l['population'] * 100000).rolling(7, min_periods=1).mean())
    fig_data3 = go.Scatter(x=xvals_l,
                           y=yvals_l,
                           line=dict(color='red', width=3),
                           name='lato 2021')

    col += 1
    if col > cols:
        row += 1
        col = 1
    figure.add_trace(fig_data1, row=row, col=col)
    figure.add_trace(fig_data2, row=row, col=col)
    figure.add_trace(fig_data3, row=row, col=col)

    # nowe śmiertelne / 100000
    ##########################

    yvals_j = list((df_j['new_deaths'] / df_j['population'] * 100000).rolling(7, min_periods=1).mean())
    fig_data1 = go.Scatter(x=xvals_j,
                           y=yvals_j,
                           line=dict(color='blue', width=1),
                           name='jesień 2020')
    yvals_w = list((df_w['new_deaths'] / df_w['population'] * 100000).rolling(7, min_periods=1).mean())
    fig_data2 = go.Scatter(x=xvals_w,
                           y=yvals_w,
                           line=dict(color='green', width=1),
                           name='wiosna 2021')
    yvals_l = list((df_l['new_deaths'] / df_l['population'] * 100000).rolling(7, min_periods=1).mean())
    fig_data3 = go.Scatter(x=xvals_l,
                           y=yvals_l,
                           line=dict(color='red', width=3),
                           name='lato 2021')

    col += 1
    if col > cols:
        row += 1
        col = 1
    figure.add_trace(fig_data1, row=row, col=col)
    figure.add_trace(fig_data2, row=row, col=col)
    figure.add_trace(fig_data3, row=row, col=col)

    # Wykrywalnosc
    ######################

    yvals_j = list(df_j['positive_rate'].rolling(7, min_periods=1).mean())
    fig_data1 = go.Scatter(x=xvals_j,
                           y=yvals_j,
                           line=dict(color='blue', width=1),
                           name='jesień 2020')
    yvals_w = list(df_w['positive_rate'].rolling(7, min_periods=1).mean())
    fig_data2 = go.Scatter(x=xvals_w,
                           y=yvals_w,
                           line=dict(color='green', width=1),
                           name='wiosna 2021')
    yvals_l = list(df_l['positive_rate'].rolling(7, min_periods=1).mean())
    fig_data3 = go.Scatter(x=xvals_l,
                           y=yvals_l,
                           line=dict(color='red', width=3),
                           name='lato 2021')
    col += 1
    if col > cols:
        row += 1
        col = 1
    figure.add_trace(fig_data1, row=row, col=col)
    figure.add_trace(fig_data2, row=row, col=col)
    figure.add_trace(fig_data3, row=row, col=col)

    # Testy nowe
    ######################

    yvals_j = list(df_j['new_tests'].rolling(7, min_periods=1).mean())
    fig_data1 = go.Scatter(x=xvals_j,
                           y=yvals_j,
                           line=dict(color='blue', width=1),
                           name='jesień 2020')
    yvals_w = list(df_w['new_tests'].rolling(7, min_periods=1).mean())
    fig_data2 = go.Scatter(x=xvals_w,
                           y=yvals_w,
                           line=dict(color='green', width=1),
                           name='wiosna 2021')
    yvals_l = list(df_l['new_tests'].rolling(7, min_periods=1).mean())
    fig_data3 = go.Scatter(x=xvals_l,
                           y=yvals_l,
                           line=dict(color='red', width=3),
                           name='lato 2021')
    col += 1
    if col > cols:
        row += 1
        col = 1
    figure.add_trace(fig_data1, row=row, col=col)
    figure.add_trace(fig_data2, row=row, col=col)
    figure.add_trace(fig_data3, row=row, col=col)

    # Zajęte łózka - suma
    ######################

    yvals_j = list(df_j['hosp_patients'].rolling(7, min_periods=1).mean())
    fig_data1 = go.Scatter(x=xvals_j,
                           y=yvals_j,
                           line=dict(color='blue', width=1),
                           name='jesień 2020')
    yvals_w = list(df_w['hosp_patients'].rolling(7, min_periods=1).mean())
    fig_data2 = go.Scatter(x=xvals_w,
                           y=yvals_w,
                           line=dict(color='green', width=1),
                           name='wiosna 2021')
    yvals_l = list(df_l['hosp_patients'].rolling(7, min_periods=1).mean())
    fig_data3 = go.Scatter(x=xvals_l,
                           y=yvals_l,
                           line=dict(color='red', width=3),
                           name='lato 2021')
    col += 1
    if col > cols:
        row += 1
        col = 1
    figure.add_trace(fig_data1, row=row, col=col)
    figure.add_trace(fig_data2, row=row, col=col)
    figure.add_trace(fig_data3, row=row, col=col)

    # Zajęte respiratory - suma
    #############################

    yvals_j = list(df_j['icu_patients'].rolling(7, min_periods=1).mean())
    fig_data1 = go.Scatter(x=xvals_j,
                           y=yvals_j,
                           line=dict(color='blue', width=1),
                           name='jesień 2020')
    yvals_w = list(df_w['icu_patients'].rolling(7, min_periods=1).mean())
    fig_data2 = go.Scatter(x=xvals_w,
                           y=yvals_w,
                           line=dict(color='green', width=1),
                           name='wiosna 2021')
    yvals_l = list(df_l['icu_patients'].rolling(7, min_periods=1).mean())
    fig_data3 = go.Scatter(x=xvals_l,
                           y=yvals_l,
                           line=dict(color='red', width=3),
                           name='lato 2021')
    col += 1
    if col > cols:
        row += 1
        col = 1
    figure.add_trace(fig_data1, row=row, col=col)
    figure.add_trace(fig_data2, row=row, col=col)
    figure.add_trace(fig_data3, row=row, col=col)

    # Zajęte łózka - przyrost

    df_j['hp'] = df_j['hosp_patients'].diff()
    df_w['hp'] = df_w['hosp_patients'].diff()
    df_l['hp'] = df_l['hosp_patients'].diff()
    yvals_j = list(df_j['hp'].rolling(7, min_periods=1).mean())
    fig_data1 = go.Scatter(x=xvals_j,
                           y=yvals_j,
                           line=dict(color='blue', width=1),
                           name='jesień 2020')
    yvals_w = list(df_w['hp'].rolling(7, min_periods=1).mean())
    fig_data2 = go.Scatter(x=xvals_w,
                           y=yvals_w,
                           line=dict(color='green', width=1),
                           name='wiosna 2021')
    yvals_l = list(df_l['hp'].rolling(7, min_periods=1).mean())
    fig_data3 = go.Scatter(x=xvals_l,
                           y=yvals_l,
                           line=dict(color='red', width=3),
                           name='lato 2021')
    col += 1
    if col > cols:
        row += 1
        col = 1
    figure.add_trace(fig_data1, row=row, col=col)
    figure.add_trace(fig_data2, row=row, col=col)
    figure.add_trace(fig_data3, row=row, col=col)

    # Zajęte respiratory - przyrost

    df_j['ip'] = df_j['icu_patients'].diff()
    df_w['ip'] = df_w['icu_patients'].diff()
    df_l['ip'] = df_l['icu_patients'].diff()
    yvals_j = list(df_j['ip'].rolling(7, min_periods=1).mean())
    fig_data1 = go.Scatter(x=xvals_j,
                           y=yvals_j,
                           line=dict(color='blue', width=1),
                           name='jesień 2020')
    yvals_w = list(df_w['ip'].rolling(7, min_periods=1).mean())
    fig_data2 = go.Scatter(x=xvals_w,
                           y=yvals_w,
                           line=dict(color='green', width=1),
                           name='wiosna 2021')
    yvals_l = list(df_l['ip'].rolling(7, min_periods=1).mean())
    fig_data3 = go.Scatter(x=xvals_l,
                           y=yvals_l,
                           line=dict(color='red', width=3),
                           name='lato 2021')
    col += 1
    if col > cols:
        row += 1
        col = 1
    figure.add_trace(fig_data1, row=row, col=col)
    figure.add_trace(fig_data2, row=row, col=col)
    figure.add_trace(fig_data3, row=row, col=col)

    height = 220 + row_height * row
    figure.update_layout(height=height,
                         template=template,
                         title=dict(text='Porównanie fali jesiennej 2020, wiosennej 2021 i letniej 2021 (stan na ' +
                                         str(dt.now())[:10] + ') - ' + location + '<br>'                                        
                                         '<sub>Oś X: kolejny dzień epidemii, '
                                         'niebieski - fala jesienna, zielony - fala wiosenna, czrerwony - fala letnia'),
                         showlegend=False),
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'docent_image',
            'width': 1200, 'height': height, 'scale': 1}
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)

    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val

##############################################
# porównanie epidemii w Polsce 2020 vs. 2021
##############################################

def layout_more_cmpf_2(DF, location, _2, _3):
    # session['template'] = 'white'
    # session['template_change'] = True
    # session.modified = True
    template = get_template()
    settings = session2settings()
    df = DF['poland']
    df_pl = df[df['location'] == location].copy()
    date_start = '2020-07-01'
    dates = list(df_pl['date'])
    date_end = max(df_pl['date'])
    date_end_2020 = dates[dates.index('2021' + date_end[4:])+14]
    df_j = df_pl[(df_pl['date'] >= date_start) & (df_pl['date'] < date_end_2020)].copy()
    # df_j = df_pl[(df_pl['date'] >= date_start) & (df_pl['date'] < ('2020' + date_end[4:]))].copy()
    df_l = df_pl[df_pl['date'] >= ('2021' + date_start[4:])].copy()
    df_j.reset_index(drop=True, inplace=True)
    df_l.reset_index(drop=True, inplace=True)
    df_j['day'] = df_j.index
    df_l['day'] = df_l.index
    cols = 3
    rows = 3
    row_height = 230
    titles = [
        "R(t)",
        "Nowe infekcje / 100k",
        "Nowe przypadki śmiertelne / 100K",
        "Wykrywalność (%)",
        "Nowe testy",
        "Zajęte łózka - suma",
        "Zajęte respiratory - suma",
        "Zajęte łózka - przyrost",
        "Zajęte respiratory - przyrost"
    ]
    figure = make_subplots(rows=rows,
                           cols=cols,
                           subplot_titles=titles,
                           print_grid=False,
                           )
    row = 1
    col = 0
    xvals_j = list(df_j['day'])
    xvals_l = list(df_l['day'])

    # Rt
    ########################

    yvals_j = list(df_j['reproduction_rate'])
    fig_data1 = go.Scatter(x=xvals_j,
                           y=yvals_j,
                           line=dict(color='blue', width=1),
                           name='jesień 2020')
    yvals_l = list(df_l['reproduction_rate'])
    fig_data3 = go.Scatter(x=xvals_l,
                           y=yvals_l,
                           line=dict(color='red', width=3),
                           name='lato 2021')
    col += 1
    if col > cols:
        row += 1
        col = 1
    figure.add_trace(fig_data1, row=row, col=col)
    figure.add_trace(fig_data3, row=row, col=col)

    # nowe infekcje / 100000
    ##########################

    yvals_j = list((df_j['new_cases'] / df_j['population'] * 100000).rolling(7, min_periods=1).mean())
    fig_data1 = go.Scatter(x=xvals_j,
                           y=yvals_j,
                           line=dict(color='blue', width=1),
                           name='jesień 2020')
    yvals_l = list((df_l['new_cases'] / df_l['population'] * 100000).rolling(7, min_periods=1).mean())
    fig_data3 = go.Scatter(x=xvals_l,
                           y=yvals_l,
                           line=dict(color='red', width=3),
                           name='lato 2021')

    col += 1
    if col > cols:
        row += 1
        col = 1
    figure.add_trace(fig_data1, row=row, col=col)
    figure.add_trace(fig_data3, row=row, col=col)

    # nowe śmiertelne / 100000
    ##########################

    yvals_j = list((df_j['new_deaths'] / df_j['population'] * 100000).rolling(7, min_periods=1).mean())
    fig_data1 = go.Scatter(x=xvals_j,
                           y=yvals_j,
                           line=dict(color='blue', width=1),
                           name='jesień 2020')
    yvals_l = list((df_l['new_deaths'] / df_l['population'] * 100000).rolling(7, min_periods=1).mean())
    fig_data3 = go.Scatter(x=xvals_l,
                           y=yvals_l,
                           line=dict(color='red', width=3),
                           name='lato 2021')

    col += 1
    if col > cols:
        row += 1
        col = 1
    figure.add_trace(fig_data1, row=row, col=col)
    figure.add_trace(fig_data3, row=row, col=col)

    # Wykrywalnosc
    ######################

    yvals_j = list(df_j['positive_rate'].rolling(7, min_periods=1).mean())
    fig_data1 = go.Scatter(x=xvals_j,
                           y=yvals_j,
                           line=dict(color='blue', width=1),
                           name='jesień 2020')
    yvals_l = list(df_l['positive_rate'].rolling(7, min_periods=1).mean())
    fig_data3 = go.Scatter(x=xvals_l,
                           y=yvals_l,
                           line=dict(color='red', width=3),
                           name='lato 2021')
    col += 1
    if col > cols:
        row += 1
        col = 1
    figure.add_trace(fig_data1, row=row, col=col)
    figure.add_trace(fig_data3, row=row, col=col)

    # Testy nowe
    ######################

    yvals_j = list(df_j['new_tests'].rolling(7, min_periods=1).mean())
    fig_data1 = go.Scatter(x=xvals_j,
                           y=yvals_j,
                           line=dict(color='blue', width=1),
                           name='jesień 2020')
    yvals_l = list(df_l['new_tests'].rolling(7, min_periods=1).mean())
    fig_data3 = go.Scatter(x=xvals_l,
                           y=yvals_l,
                           line=dict(color='red', width=3),
                           name='lato 2021')
    col += 1
    if col > cols:
        row += 1
        col = 1
    figure.add_trace(fig_data1, row=row, col=col)
    figure.add_trace(fig_data3, row=row, col=col)

    # Zajęte łózka - suma
    ######################

    yvals_j = list(df_j['hosp_patients'].rolling(7, min_periods=1).mean())
    fig_data1 = go.Scatter(x=xvals_j,
                           y=yvals_j,
                           line=dict(color='blue', width=1),
                           name='jesień 2020')
    yvals_l = list(df_l['hosp_patients'].rolling(7, min_periods=1).mean())
    fig_data3 = go.Scatter(x=xvals_l,
                           y=yvals_l,
                           line=dict(color='red', width=3),
                           name='lato 2021')
    col += 1
    if col > cols:
        row += 1
        col = 1
    figure.add_trace(fig_data1, row=row, col=col)
    figure.add_trace(fig_data3, row=row, col=col)

    # Zajęte respiratory - suma
    #############################

    yvals_j = list(df_j['icu_patients'].rolling(7, min_periods=1).mean())
    fig_data1 = go.Scatter(x=xvals_j,
                           y=yvals_j,
                           line=dict(color='blue', width=1),
                           name='jesień 2020')
    yvals_l = list(df_l['icu_patients'].rolling(7, min_periods=1).mean())
    fig_data3 = go.Scatter(x=xvals_l,
                           y=yvals_l,
                           line=dict(color='red', width=3),
                           name='lato 2021')
    col += 1
    if col > cols:
        row += 1
        col = 1
    figure.add_trace(fig_data1, row=row, col=col)
    figure.add_trace(fig_data3, row=row, col=col)

    # Zajęte łózka - przyrost

    df_j['hp'] = df_j['hosp_patients'].diff()
    df_l['hp'] = df_l['hosp_patients'].diff()
    yvals_j = list(df_j['hp'].rolling(7, min_periods=1).mean())
    fig_data1 = go.Scatter(x=xvals_j,
                           y=yvals_j,
                           line=dict(color='blue', width=1),
                           name='jesień 2020')
    yvals_l = list(df_l['hp'].rolling(7, min_periods=1).mean())
    fig_data3 = go.Scatter(x=xvals_l,
                           y=yvals_l,
                           line=dict(color='red', width=3),
                           name='lato 2021')
    col += 1
    if col > cols:
        row += 1
        col = 1
    figure.add_trace(fig_data1, row=row, col=col)
    figure.add_trace(fig_data3, row=row, col=col)

    # Zajęte respiratory - przyrost

    df_j['ip'] = df_j['icu_patients'].diff()
    df_l['ip'] = df_l['icu_patients'].diff()
    yvals_j = list(df_j['ip'].rolling(7, min_periods=1).mean())
    fig_data1 = go.Scatter(x=xvals_j,
                           y=yvals_j,
                           line=dict(color='blue', width=1),
                           name='jesień 2020')
    yvals_l = list(df_l['ip'].rolling(7, min_periods=1).mean())
    fig_data3 = go.Scatter(x=xvals_l,
                           y=yvals_l,
                           line=dict(color='red', width=3),
                           name='lato 2021')
    col += 1
    if col > cols:
        row += 1
        col = 1
    figure.add_trace(fig_data1, row=row, col=col)
    figure.add_trace(fig_data3, row=row, col=col)

    height = 220 + row_height * row
    figure.update_layout(height=height,
                         template=template,
                         xaxis=dict(title=dict(text='Kolejny dzień od ' + date_start[5:],
                                               font=dict(color=get_session_color(4))
                                               ),
                                    ),
                         title=dict(text='Synchroniczne porównanie fal epidemicznych 2020 i 2021 od ' + date_start[8:] + date_start[4:7] + ' do ' +
                                         date_end[8:] + date_end[4:7] + ' (' + location + ')<br>'                                        
                                         '<sup>Oś X: kolejny dzień, '
                                         'niebieski - 2020, czerwony - 2021'),
                         showlegend=False),
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'docent_image',
            'width': 1200, 'height': height, 'scale': 1}
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)

    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


def restrykcje():
    df = pd.read_csv('data/sources/response_graphs_data_0.csv')
    return df


############################################
# GREG R i nadmiarowe zgony w województwach
############################################
def layout_more_greg(DF, _1, _2, _3):
    from sklearn.linear_model import LinearRegression
    template = get_template()
    settings = session2settings()

    # dane Eurostat do obliczenia średniej

    df_euro = pd.read_csv(constants.data_files['mortality_eurostat']['data_fn'])
    df_euro = df_euro[df_euro['age'] == 'TOTAL'].copy()

    df_gus = pd.read_csv(constants.data_files['mortality_gus_woj']['data_fn'])
    df_gus['location'] = df_gus['location'].str.lower()
    df_pl = df_gus.groupby('date').sum()
    df_pl['location'] = 'Polska'
    df_pl['date'] = df_pl.index
    df_pl.reset_index(drop=True, inplace=True)
    df_gus = df_gus.append(df_pl, ignore_index=True)

    locations = df_euro[
        df_euro.location.str.islower() |
        (df_euro.location == 'Polska')
        ]['location'].unique()
    locations.sort()

    means = {}
    df_means = df_euro.loc[(df_euro['year'].between(2018, 2019))].copy()
    for i in locations:
        print(i)
        dx = df_means[df_means['location'] == i]['total']
        if len(dx) > 0:
            means[i] = dx.mean()

    # dane GUS

    df_gus['date'] = df_gus.date.astype(str)

    # dane Covid

    df_w = DF['poland'].copy()
    df_w['location'] = df_w['location'].apply(lambda x: x if x == 'Polska' else x.lower())
    # df_w['location'] = df_w['location'].str.lower()

    df_we = df_w[df_w['location'].isin(locations)].copy()

    # sumowanie tygodniowe

    df_we.index = df_we['date'].astype('datetime64[ns]')
    df_we = df_we.groupby('location').resample('W-mon').sum().reset_index()
    df_we['date'] = df_we['date'].dt.strftime('%Y-%m-%d')

    # połączenie z danymi Covid

    df_gus = pd.merge(df_gus, df_we[['date', 'location', 'new_deaths', 'reproduction_rate', 'population']], how='right',
                      left_on=['date', 'location'],
                      right_on=['date', 'location']).copy()
    df_gus['total'].replace(0, np.nan)
    df_gus.dropna(axis='rows', subset=['total'], inplace=True)
    cols = 4
    row_height = 250
    rows = int((len(locations) - 1) / cols) + 1
    figure = make_subplots(rows=rows,
                           cols=cols,
                           subplot_titles=locations,
                           print_grid=False,
                           )
    row = 1
    col = 0

    # przygotowanie danych w jednej tabeli

    for location in locations:
        filtr = df_gus['location'] == location
        df_gus.loc[filtr, 'percent'] = round(
            (df_gus[filtr]['new_deaths'] / (df_gus[filtr]['total'] - means[location])) * 100, 2)
        df_gus.loc[filtr, 'mean'] = round(means[location], 0)
    df_gus[df_gus['date'] >= '2020-10-01'].to_csv('data/excess_deaths_woj.csv', index=False)

    df = df_gus[df_gus['date'] >= '2020-10-01'].copy()
    df['nadmiar'] = (df['new_deaths'] / df['population']) * 1000000 * 7
    df.dropna(inplace=True)
    df['nadmiar'] = df['nadmiar'].astype(np.float64)
    df['nadmiar_cum'] = df.groupby(['location'])['nadmiar'].cumsum()
    df['reproduction_rate'] = df['reproduction_rate'] / 7
    for location in locations:
        print(location)
        filtered = df[(df['location'] == location)].copy()
        filtered.sort_values(by=['nadmiar'], inplace=True)
        filtered.reset_index(drop=True, inplace=True)

        # wspólne skalowanie osi

        xvals = list(filtered['nadmiar_cum'])

        # Wskaźnik R

        yvals = list((filtered['reproduction_rate']))
        fig_data1 = go.Scatter(x=xvals,
                               y=yvals,
                               mode='markers',
                               marker={'symbol': 'circle', 'size': 6, 'color': 'red'},
                               )

        # linia regresji liniowej

        xvals = filtered['nadmiar_cum'].values.reshape(-1, 1)
        yvals = filtered['reproduction_rate'].values.reshape(-1, 1)
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(xvals, yvals)  # perform linear regression
        score = linear_regressor.score(xvals, yvals)
        yvals = linear_regressor.predict(xvals)
        xvals = list(xvals.flatten())
        yvals = list(yvals.flatten())
        yorg = list((filtered['reproduction_rate']))
        err = [yvals[i] - yorg[i] for i in range(len(yvals))]
        fig_data2 = go.Scatter(x=xvals,
                               y=yvals,
                               mode='lines',
                               error_y=dict(type='data', array=err, visible=True),
                               )
        col += 1
        if col > cols:
            row += 1
            col = 1
        figure.add_trace(fig_data1, row=row, col=col)
        figure.add_trace(fig_data2, row=row, col=col)

        figure.update_xaxes(title={'text': 'Zgony nadmiarowe na 1 mln<br>R0=' +
                                           str(round(linear_regressor.predict(np.array([0]).reshape(-1, 1))[0][0],
                                                     2)) + ', score=' + str(round(score, 3))},
                            row=row, col=col)
        figure.update_yaxes(title={'text': 'R(t)'}, row=row, col=col)

    height = 100 + row_height * row
    figure.update_layout(height=height,
                         template=template,
                         title=dict(
                             text='Zależność R(t) i ZN na 1 mln narastająco dla województw, regresja liniowa',
                             ),
                         showlegend=False),
    figure = add_copyright(figure, settings)
    config = {
        'responsive': True,
        'displaylogo': False,
        'locale': 'pl-PL',
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'docent_image',
            'width': 1200, 'height': height, 'scale': 1}
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)

    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


############################################
# Reinfekcje
############################################
def layout_more_mz_reinf(DF, _typ, _2, _3):
    template = get_template()
    settings = session2settings()

    df = pd.read_csv('data/sources/mz_reinf.csv', sep=';')
    df.columns = ['Rok', 'Miesiąc', 'Rok.1', 'Miesiąc.1', 'producent', 'czy_druga_dawka', \
       'kat_wiek', 'kat_zakazenia', 'plec', 'ile_osob']

    # kategoria_zakażenia, wiek

    df = df[['producent', 'kat_wiek', 'kat_zakazenia', 'plec', 'ile_osob']].copy()
    df.sort_values(by=['kat_zakazenia', 'kat_wiek'], inplace=True)
    df['ile_osob'] = df['ile_osob'].str.replace(' ', '')
    df['ile_osob'] = df['ile_osob'].astype(np.float64)

    # osoby zaszczepione razem

    df['total'] = df['ile_osob'].sum()

    # suma infekcji

    df['total_infekcje'] = df[df['kat_zakazenia'] != 'Brak pozytywnego wyniku COVID']['ile_osob'].sum()

    # osoby kat_zakażenia

    df['total_kategoria'] = df.groupby(['kat_zakazenia'])['ile_osob'].transform('sum')

    # osoby wg kategorii i wieku

    df['total_kategoria_wiek'] = df.groupby(['kat_zakazenia', 'kat_wiek'])['ile_osob'].transform('sum')
    df['procent_kategoria_wiek'] = round((df['total_kategoria_wiek'] / df['total_kategoria'] * 100), 2)
    df = df.drop_duplicates(subset=['kat_zakazenia', 'kat_wiek']).copy()
    del df['plec']
    del df['ile_osob']
    del df['producent']

    bins = ['Brak pozytywnego wyniku COVID', 'do 14 dni po drugiej dawce',
       'do 14 dni po pierwszej dawce', 'powyżej 14 dni po drugiej dawce',
       'powyżej 14 dni po pierwszej dawce, przed drugą dawką']
    traces = []
    for bin in bins:
        if bin == 'Brak pozytywnego wyniku COVID':
            width = 4
        else:
            width = 1
        df_f = df[df['kat_zakazenia'] == bin].copy()
        df_f['y'] = df_f['procent_kategoria_wiek']
        x = list(df_f['kat_wiek'])
        y = list(df_f['y'])
        pass
        fig_data = go.Scatter(
            x=x,
            y=y,
            line=dict(width=width),
            name=bin
        )
        traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text='Udział % grup wiekowych w liczbie infekcji po szczepieniu'
                       '<br><sub>(gruba czerwona linia jest linią odniesienia)'),
            height=660,
            legend=dict(x=1, y=1., orientation='v'),
        )
    )
    fig = dcc.Graph(id='xxxx', figure=figure)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


############################################
# Reinfekcje 2
############################################
def layout_more_mz_reinf2(DF, _1, _2, _3):
    template = get_template()
    settings = session2settings()

    # ['Pfizer', 'Johnson&Johnson', 'Moderna', 'Niezaszczepiony', 'Astra Zeneca']
    # ['NEGATYWNY', 'POZYTYWNY', 'NIEDIAGNOSTYCZNY', 'NIEROZSTRZYGAJACY']
    # ['wynik', 'producent', 'teryt', 'ile', 'date_wynik', 'date_1_dawka',
    #  'date_2_dawka', 'tyg_1_dawka', 'tyg_2_dawka']

    df = pd.read_csv(constants.data_files['reinf']['data_fn'])
    df = df[['wynik', 'producent', 'ile', 'tyg_2_dawka']].copy()
    df = df[df['producent'] != 'Niezaszczepiony'].copy()
    df = df[df['wynik'].isin(['NEGATYWNY', 'POZYTYWNY'])].copy()
    df = df[df['tyg_2_dawka'] >= 0].copy()
    df.sort_values(by=['tyg_2_dawka'], inplace=True)
    df['total_producent'] = df.groupby(['tyg_2_dawka', 'producent'])['ile'].transform('sum')
    df['total_producent_wynik'] = df.groupby(['tyg_2_dawka', 'producent', 'wynik'])['ile'].transform('sum')
    df = df.drop_duplicates(subset=['tyg_2_dawka', 'producent', 'wynik']).copy()
    df.reset_index(drop=True, inplace=True)

    bins = ['Pfizer', 'Moderna', 'Astra Zeneca']
    traces = []
    for bin in bins:
        df_f = df[df['producent'] == bin].copy()
        df_f = df_f[df_f['wynik'] == 'POZYTYWNY'].copy()
        df_f['y'] = (df_f['total_producent_wynik'] / df_f['total_producent']) * 100
        x = list(df_f['tyg_2_dawka'])
        y = list(df_f['y'])
        fig_data = go.Scatter(
            x=x,
            y=y,
            line=dict(width=3),
            name=bin
        )
        traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text='Udział % pozytywnych testów w kolejnych tygodniach po zaszczepieniu 2. dawką'),
            height=460,
            legend=dict(x=1, y=1., orientation='v'),
        )
    )
    fig = dcc.Graph(id='xxxx', figure=figure)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


############################################
# Reinfekcje 3
############################################
def layout_more_mz_reinf3(DF, _typ, _2, _3):
    template = get_template()
    settings = session2settings()

    # ['Pfizer', 'Johnson&Johnson', 'Moderna', 'Niezaszczepiony', 'Astra Zeneca']
    # ['NEGATYWNY', 'POZYTYWNY', 'NIEDIAGNOSTYCZNY', 'NIEROZSTRZYGAJACY']
    # ['wynik', 'producent', 'teryt', 'ile', 'date_wynik', 'date_1_dawka',
    #  'date_2_dawka', 'tyg_1_dawka', 'tyg_2_dawka']

    df = pd.read_csv(constants.data_files['reinf']['data_fn'])
    df['producent'] = df['producent'].str.replace('Pfizer', 'Zaszczepiony')
    df['producent'] = df['producent'].str.replace('Astra Zeneca', 'Zaszczepiony')
    df['producent'] = df['producent'].str.replace('Moderna', 'Zaszczepiony')
    df['producent'] = df['producent'].str.replace('Johnson&Johnson', 'Zaszczepiony')
    df['status'] = df['producent']
    # df = df[['wynik', 'status', 'ile', 'date_wynik']].copy()
    df = df[df['wynik'].isin(['NEGATYWNY', 'POZYTYWNY'])].copy()

    df.loc[(df['status'] == 'Zaszczepiony') & (df['tyg_2_dawka'] < 3), 'status'] = 'Niezaszczepiony'

    df.sort_values(by=['date_wynik'], inplace=True)
    df['total_status'] = df.groupby(['date_wynik', 'status'])['ile'].transform('sum')
    df['total_status_wynik'] = df.groupby(['date_wynik', 'status', 'wynik'])['ile'].transform('sum')
    df = df.drop_duplicates(subset=['date_wynik', 'status', 'wynik']).copy()
    df = df[df['wynik'] == 'POZYTYWNY'].copy()
    df.reset_index(drop=True, inplace=True)

    traces = []
    df_f = df[df['status'] == 'Niezaszczepiony'].copy()

    df_f['date'] = pd.to_datetime(df_f['date_wynik'], format='%Y-%m-%d')

    df_f['y'] = (df_f['total_status_wynik'] / df_f['total_status']) * 100
    x = list(df_f['date'])
    y = list(df_f['y'])
    fig_data = go.Scatter(
        x=x,
        y=y,
        line=dict(width=3),
        name='Niezaszczepieni'
    )
    traces.append(fig_data)
    df_f = df[df['status'] != 'Niezaszczepiony'].copy()
    df_f['date'] = pd.to_datetime(df_f['date_wynik'], format='%Y-%m-%d')
    df_f['y'] = (df_f['total_status_wynik'] / df_f['total_status']) * 100
    x = list(df_f['date'])
    y = list(df_f['y'])
    fig_data = go.Scatter(
        x=x,
        y=y,
        line=dict(width=3),
        name='Zaszczepieni'
    )
    traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text='Udział % pozytywnych testów w grupie zaszczepionych i niezaszczepionych'),
            height=660,
            legend=dict(x=1, y=1., orientation='v'),
        )
    )
    fig = dcc.Graph(id='xxxx', figure=figure)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


############################################
# Reinfekcje 4
############################################
def layout_more_mz_reinf4(DF, _typ, _2, _3):
    template = get_template()
    settings = session2settings()

    # ['Pfizer', 'Johnson&Johnson', 'Moderna', 'Niezaszczepiony', 'Astra Zeneca']
    # ['NEGATYWNY', 'POZYTYWNY', 'NIEDIAGNOSTYCZNY', 'NIEROZSTRZYGAJACY']
    # ['wynik', 'producent', 'teryt', 'ile', 'date_wynik', 'date_1_dawka',
    #  'date_2_dawka', 'tyg_1_dawka', 'tyg_2_dawka']

    df = pd.read_csv(constants.data_files['reinf']['data_fn'])
    df['producent'] = df['producent'].str.replace('Pfizer', 'Zaszczepiony')
    df['producent'] = df['producent'].str.replace('Astra Zeneca', 'Zaszczepiony')
    df['producent'] = df['producent'].str.replace('Moderna', 'Zaszczepiony')
    df['producent'] = df['producent'].str.replace('Johnson&Johnson', 'Zaszczepiony')
    df['status'] = df['producent']
    df = df[df['wynik'].isin(['NEGATYWNY', 'POZYTYWNY'])].copy()

    df.loc[(df['status'] == 'Zaszczepiony') & (df['tyg_2_dawka'] < 0), 'status'] = 'Niezaszczepiony'

    df.sort_values(by=['date_wynik'], inplace=True)
    df['total_status'] = df.groupby(['date_wynik', 'status'])['ile'].transform('sum')
    df['total_status_wynik'] = df.groupby(['date_wynik', 'status', 'wynik'])['ile'].transform('sum')
    df = df.drop_duplicates(subset=['date_wynik', 'status', 'wynik']).copy()
    df = df[df['wynik'] == 'POZYTYWNY'].copy()
    df.reset_index(drop=True, inplace=True)

    traces = []
    df_f1 = df[df['status'] == 'Niezaszczepiony'].copy()
    df_f1['date'] = pd.to_datetime(df_f1['date_wynik'], format='%Y-%m-%d')
    df_f1['y'] = (df_f1['total_status_wynik'] / df_f1['total_status']) * 100
    df_f2 = df[df['status'] != 'Niezaszczepiony'].copy()
    df_f2['date'] = pd.to_datetime(df_f2['date_wynik'], format='%Y-%m-%d')
    df_f2['y'] = (df_f2['total_status_wynik'] / df_f2['total_status']) * 100
    df_f = pd.merge(df_f1, df_f2, how='left',
                           left_on=['date'], right_on=['date']).copy()

    df_f['coeff'] = df_f['y_y'] / df_f['y_x']
    x = list(df_f['date'])
    y = list(df_f['coeff'])
    fig_data = go.Scatter(
        x=x,
        y=y,
        line=dict(width=3),
        name='coeff'
    )
    traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text='Proporcja udziału % pozytywnych testów w grupie zaszczepionych i niezaszczepionych'),
            height=660,
            legend=dict(x=1, y=1., orientation='v'),
        )
    )
    fig = dcc.Graph(id='xxxx', figure=figure)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


############################################
# Reinfekcje 5
############################################
def layout_more_mz_reinf5(DF, _typ, _2, _3):
    template = get_template()
    settings = session2settings()

    # ['Pfizer', 'Johnson&Johnson', 'Moderna', 'Niezaszczepiony', 'Astra Zeneca']
    # ['NEGATYWNY', 'POZYTYWNY', 'NIEDIAGNOSTYCZNY', 'NIEROZSTRZYGAJACY']
    # ['wynik', 'producent', 'teryt', 'ile', 'date_wynik', 'date_1_dawka',
    #  'date_2_dawka', 'tyg_1_dawka', 'tyg_2_dawka']

    df = pd.read_csv(constants.data_files['reinf']['data_fn'])

    # tylko szczepionki dwudawkowe

    df = df[df['producent'].isin(['Pfizer', 'Moderna', 'Astra Zeneca'])].copy()
    df = df[df['aktywny'] == 'wpelni'].copy()
    df = df[df['wynik'].isin(['NEGATYWNY', 'POZYTYWNY'])].copy()

    df['tyg_ile'] = df.groupby(['tyg_2_dawka', 'producent'])['ile'].transform('sum')
    df['tyg_wynik_ile'] = df.groupby(['tyg_2_dawka', 'producent', 'wynik'])['ile'].transform('sum')
    df = df.drop_duplicates(subset=['tyg_2_dawka', 'producent', 'wynik']).copy()
    df = df[df['wynik'] == 'POZYTYWNY'].copy()
    df['coeff'] = df['tyg_wynik_ile'] / df['tyg_ile']

    df.sort_values(by=['tyg_2_dawka', 'wynik'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    traces = []
    for producent in ['Pfizer', 'Moderna', 'Astra Zeneca']:
        df1 = df[df['producent'] == producent].copy()
        x = list(df1['tyg_2_dawka'])
        y = list(df1['coeff'])
        fig_data = go.Scatter(
            x=x,
            y=y,
            line=dict(width=3),
            name=producent
        )
        traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text='Wykrywalność po zaszczepieniu w funkcji liczby tygodni od 2 dawki'),
            height=660,
            legend=dict(x=1, y=1., orientation='v'),
        )
    )
    fig = dcc.Graph(id='xxxx', figure=figure)
    ret_val = [
        dbc.Col(fig, width=12, className='mb-2 mt-2')
    ]
    return ret_val


################################################
#  Multi mapa województwa
################################################

def layout_more_multimap_woj(DF, last_date, _2, _3):
    template = get_template()
    settings = session2settings()

    last_date = last_date[:10]

    from_date = settings['from_date']
    to_date = settings['to_date']
    locations = list(constants.wojew_cap.values())
    chart_types = settings['chart_type']
    if len(chart_types) == 0:
        return 'Wybierz co najmniej jeden typ danych'
    df = pd.DataFrame()
    for location in locations:
        df0 = prepare_data(settings, DF,
                          scope='poland',
                          locations=[location],
                          date=last_date,
                          chart_types=chart_types,
                          all_columns=True)
        df = df.append(df0, ignore_index=True).copy()

    if len(df) == 0:
        return 'Brak rekordów spełniających podane warunki'
    df.dropna(subset=['Long', 'Lat'], inplace=True)

    df['Long'] = df['location'].apply(lambda x: constants.wojew_mid[x][1])
    df['Lat'] = df['location'].apply(lambda x: constants.wojew_mid[x][0])

    df = df[df['location'] != 'Polska'].copy()
    if len(chart_types) == 1:
        rows = 1; cols = 1; zoom = 5.5; width = 12
        height = int(settings['map_height'])
    elif len(chart_types) == 2:
        rows = 1; cols = 2; zoom = 5.5; width = 6; height = int(settings['map_height'])
    elif len(chart_types) <= 4:
        rows = 2; cols = 2; zoom = 4.5; width = 6
        height = int(settings['map_height']) / 2
    elif len(chart_types) <= 6:
        rows = 2; cols = 3; zoom = 4.6; width = 4
        height = int(settings['map_height']) / 2
    else:
        rows = 3; cols = 3; zoom = 4.0; width = 4
        height = int(settings['map_height']) / 3
    mapbox_access_token = open("data/mapbox.token").read()
    center = dict(lon=19.18, lat=52.11)
    # zoom, center = zoom_center(
    #     lons=[min(df.Long), max(df.Long)],
    #     lats=[min(df.Lat), max(df.Lat)]
    # )
    # zoom /= 2
    mapbox = dict(accesstoken=mapbox_access_token, center=center, zoom=zoom)
    mapbox['style'] = constants.mapbox_styles[settings['map_mapbox']]

    json_file = r'data/geojson/woj.min.geo.json'
    locations = df['wojew']
    featuredikey = 'properties.nazwa'
    with open(json_file, encoding="utf-8") as f:
        geojson = json.load(f)

    figures = []
    for chart_type in chart_types:
        tickvals, ticktext = get_ticks(df[chart_type])
        figure = go.Figure()
        if settings['map_opt'] == 'log':
            z = np.log10(df[chart_type] + 0.001)
        else:
            z = df[chart_type]
        figure.add_trace(go.Choroplethmapbox(
            geojson=geojson,
            featureidkey=featuredikey,
            locations=locations,
            z=z,
            marker=dict(line=dict(color='black', width=0.5 if 'drawborder' in settings['map_options'] else 0.)),
            hoverinfo='text',
            # showscale=False,
            colorscale=settings['map_color_scale'],
            reversescale=True if 'reversescale' in settings['map_options'] else False,
            marker_opacity=settings['map_opacity'],
            colorbar=dict(
                len=0.5,
                bgcolor=get_session_color(1),
                title='',
                # title=dict(text=chart_type),
                tickfont=dict(color=get_session_color(4)),
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
            anno_text = round(df[chart_type], acc).astype(str)
            figure.add_trace(
                go.Scattermapbox(
                    lat=df.Lat, lon=df.Long,
                    mode='text',
                    hoverinfo='none',
                    below="''",
                    marker=dict(allowoverlap=True),
                    text=anno_text,
                    textfont=dict(family='Bebas Neue', size=settings['font_size_anno'], color=get_session_color(4)),
                ))
        figure.update_layout(
            images=constants.image_logo_map,
            template=template,
            autosize=True,
            title=dict(
                text=get_title(settings, chart_type),
                x=0.05, y=settings['titleypos'],
                xanchor='left',
                font=dict(size=settings['font_size_title'], color=get_session_color(7))
            ),
            height=height,
            margin=dict(l=settings['marginl'], r=settings['marginr'] + 10, b=settings['marginb'], t=settings['margint'],
                        pad=0),
            paper_bgcolor=get_session_color(1),
            plot_bgcolor=get_session_color(1),
            mapbox=mapbox,
        )
        figure = add_copyright(figure, settings)
        config = {
            'displaylogo': False,
            'responsive': True,
            'toImageButtonOptions': {
                'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
                'filename': 'docent_image', 'height': 700, 'width': 1200}
        }
        f = dbc.Col(dcc.Graph(id='mapa1' + chart_type, figure=figure, config=config), width=width, class_name=constants.nomargins)
        figures.append(f)
    ret_val = []
    for i in range(rows):
        row = []
        for j in range(cols):
            nelem = i * cols + j
            if nelem < len(figures):
                row.append(figures[nelem])
        ret_val.append(dbc.Row(row))

    return dbc.Col(ret_val)


################################################
#  Mapa chrono województwa
################################################

def layout_more_chronomap_woj(DF, deltat, _scale, n):
    if deltat is None:
        return
    template = get_template()
    settings = session2settings()
    d = list(DF['poland'].date.unique())
    dates = []
    for i in range(n):
        dates.append(d[-1 - deltat * i])

    chart_types = settings['chart_type']
    if len(chart_types) != 1:
        return "Wybierz tylko jedną wielkość"

    chart_type = chart_types[0]
    if n == 1:
        rows = 1; cols = 1; zoom = 5.5; width = 12
        height = int(settings['map_height'])
    elif n == 2:
        rows = 1; cols = 2; zoom = 5.5; width = 6
        height = int(settings['map_height'])
    elif n <= 4:
        rows = 2; cols = 2; zoom = 4.5; width = 6
        height = int(settings['map_height']) / 2
    elif n <= 6:
        rows = 2; cols = 3; zoom = 4.5; width = 4
        height = int(settings['map_height']) / 2
    else:
        rows = 3; cols = 3; zoom = 4.0; width = 4
        height = int(settings['map_height']) / 3

    mapbox_access_token = open("data/mapbox.token").read()
    mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=19.18, lat=52.11), zoom=zoom)
    mapbox['style'] = constants.mapbox_styles[settings['map_mapbox']]

    json_file = r'data/geojson/woj.min.geo.json'
    featuredikey = 'properties.nazwa'
    with open(json_file, encoding="utf-8") as f:
        geojson = json.load(f)
    figures = []

    dfxs = pd.DataFrame()
    for date in dates:
        locations = list(constants.wojew_cap.values())[1:]
        dfx = prepare_data(settings, DF,
                          scope='poland',
                          locations=locations,
                          date=date,
                          chart_types=chart_types,
                          all_columns=True)
        dfxs = dfxs.append(dfx, ignore_index=True).copy()
    vmax = max(dfxs[chart_type])
    vmin = min(dfxs[chart_type])
    dx = (vmax - vmin) / 5
    tickvals, ticktext = get_nticks(dfxs[chart_type])

    for date in dates[::-1]:
        dfx = dfxs[dfxs['date'] == date].copy()
        dfx.dropna(subset=['Long', 'Lat'], inplace=True)

        dfx['Long'] = dfx['location'].apply(lambda x: constants.wojew_mid[x][1])
        dfx['Lat'] = dfx['location'].apply(lambda x: constants.wojew_mid[x][0])

        dfx = dfx[dfx['location'] != 'Polska'].copy()
        if _scale == 'wspólna skala':
            locations = ['xxx'] + [i.lower() for i in list(dfx['location'].unique())] + ['yyy']
            z = [vmin] + list(dfx[chart_type]) + [vmax]
        else:
            locations = [i.lower() for i in list(dfx['location'].unique())]
            z = list(dfx[chart_type])
        figure = go.Figure()
        figure.add_trace(go.Choroplethmapbox(
            geojson=geojson,
            featureidkey=featuredikey,
            locations=locations,
            z=z,
            marker=dict(line=dict(color='black', width=0.5 if 'drawborder' in settings['map_options'] else 0.)),
            hoverinfo='text',
            colorscale=constants.color_scales[settings['map_color_scale']],
            # colorscale=settings['map_color_scale'],
            reversescale=True if 'reversescale' in settings['map_options'] else False,
            marker_opacity=settings['map_opacity'],
            colorbar=dict(
                len=0.9,
                bgcolor=get_session_color(1),
                title='',
                # title=dict(text=chart_type),
                tickfont=dict(color=get_session_color(4)),
                # tickmode="array",
                tickmode="auto",
                tick0=0,
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
            anno_text = round(dfx[chart_type], acc).astype(str)
            figure.add_trace(
                go.Scattermapbox(
                    lat=dfx.Lat, lon=dfx.Long,
                    mode='text',
                    hoverinfo='none',
                    below="''",
                    marker=dict(allowoverlap=True),
                    text=anno_text,
                    textfont=dict(family='Bebas Neue', size=settings['font_size_anno'], color=get_session_color(4)),
                ))
        figure.update_layout(
            images=constants.image_logo_map,
            template=template,
            autosize=True,
            title=dict(
                text=get_title(settings, chart_type),
                x=0.05, y=settings['titleypos'],
                xanchor='left',
                font=dict(size=settings['font_size_title'], color=get_session_color(7))
            ),
            height=height,
            margin=dict(l=settings['marginl'], r=settings['marginr'] + 10, b=settings['marginb'], t=settings['margint'],
                        pad=0),
            paper_bgcolor=get_session_color(1),
            plot_bgcolor=get_session_color(1),
            mapbox=mapbox,
        )
        figure = add_copyright(figure, settings)
        config = {
            'displaylogo': False,
            'responsive': True,
            'toImageButtonOptions': {
                'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
                'filename': 'docent_image', 'height': 700, 'width': 1200}
        }
        f = dbc.Col(dcc.Graph(id='mapa1' + chart_type, figure=figure, config=config), width=width, className=constants.nomargins)
        figures.append(f)
    ret_val = []
    for i in range(rows):
        row = []
        for j in range(cols):
            nelem = i*cols+j
            if nelem < len(figures):
                row.append(figures[nelem])
        ret_val.append(dbc.Row(row))
    # ret_val = figures

    return dbc.Col(ret_val)

################################################
#  Multi mapa powiaty
################################################

def layout_more_multimap_pow(DF, last_date, _2, _3):
    template = get_template()
    settings = session2settings()

    chart_types = settings['chart_type']
    if len(chart_types) == 0:
        return 'Wybierz co najmniej jeden typ danych'

    last_date = last_date[:10]
    locations = list(settings['locations'])
    df = pd.DataFrame()
    for location in locations:
        df0 = prepare_data(settings, DF,
                          scope='cities',
                          locations=[location],
                          date=last_date,
                          chart_types=chart_types,
                          all_columns=True)
        df = df.append(df0, ignore_index=True).copy()

    if len(df) == 0:
        return 'Brak rekordów spełniających podane warunki'
    df.dropna(subset=['Long', 'Lat'], inplace=True)

    df = df[df['location'] != 'Polska'].copy()
    if len(chart_types) == 1:
        rows = 1
        cols = 1
        zoom = 5.5
        width = 12
        height = int(settings['map_height'])
    elif len(chart_types) == 2:
        rows = 1
        cols = 2
        zoom = 5.5
        width = 6
        height = int(settings['map_height'])
    elif len(chart_types) <= 4:
        rows = 2
        cols = 2
        zoom = 4.5
        width = 6
        height = int(settings['map_height']) / 2
    elif len(chart_types) <= 6:
        rows = 2
        cols = 3
        zoom = 4.5
        width = 4
        height = int(settings['map_height']) / 2
    else:
        rows = 3
        cols = 3
        zoom = 4.0
        width = 4
        height = int(settings['map_height']) / 3
    mapbox_access_token = open("data/mapbox.token").read()
    center = dict(lon=19.18, lat=52.11)
    # zoom, center = zoom_center(
    #     lons=tuple[min(df.Long), max(df.Long)],
    #     lats=[min(df.Lat), max(df.Lat)]
    # )
    # zoom = zoom * 0.95
    mapbox = dict(accesstoken=mapbox_access_token, center=center, zoom=zoom)
    mapbox['style'] = constants.mapbox_styles[settings['map_mapbox']]

    json_file = r'data/geojson/powiaty-min-ws.geojson'
    locations = df['location']
    featuredikey = 'properties.nazwa'
    with open(json_file, encoding="utf-8") as f:
        geojson = json.load(f)

    figures = []
    for chart_type in chart_types:
        tickvals, ticktext = get_nticks(df[chart_type])
        figure = go.Figure()
        if settings['map_opt'] == 'log':
            z = np.log10(df[chart_type] + 0.001)
        else:
            z = df[chart_type]
        figure.add_trace(go.Choroplethmapbox(
            geojson=geojson,
            featureidkey=featuredikey,
            locations=locations,
            z=z,
            marker=dict(line=dict(color='black', width=0.5 if 'drawborder' in settings['map_options'] else 0.)),
            hoverinfo='text',
            # showscale=False,
            colorscale=settings['map_color_scale'],
            reversescale=True if 'reversescale' in settings['map_options'] else False,
            marker_opacity=settings['map_opacity'],
            colorbar=dict(
                len=0.5,
                bgcolor=get_session_color(1),
                title='',
                # title=dict(text=chart_type),
                tickfont=dict(color=get_session_color(4)),
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
                anno_text = dfx['color2'] + constants.trace_props[chart_type]['postfix']
            else:
                anno_text = dfx['location2'] + '<br>' + dfx['color2'] + constants.trace_props[chart_type]['postfix']
            # anno_text = round(df[chart_type], acc).astype(str)
            figure.add_trace(
                go.Scattermapbox(
                    lat=dfx.Lat, lon=dfx.Long,
                    mode='text',
                    hoverinfo='none',
                    below="''",
                    marker=dict(allowoverlap=True),
                    text=anno_text,
                    textfont=dict(family='Bebas Neue', size=settings['font_size_anno'], color=get_session_color(4)),
                ))
        figure.update_layout(
            images=constants.image_logo_map,
            template=template,
            autosize=True,
            title=dict(
                text=get_title(settings, chart_type),
                x=0.05, y=settings['titleypos'],
                xanchor='left',
                font=dict(size=settings['font_size_title'], color=get_session_color(7))
            ),
            height=height,
            margin=dict(l=settings['marginl'], r=settings['marginr'] + 10, b=settings['marginb'], t=settings['margint'],
                        pad=0),
            paper_bgcolor=get_session_color(1),
            plot_bgcolor=get_session_color(1),
            mapbox=mapbox,
        )
        figure = add_copyright(figure, settings)
        config = {
            'displaylogo': False,
            'responsive': True,
            'toImageButtonOptions': {
                'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
                'filename': 'docent_image', 'height': 700, 'width': 1200}
        }
        f = dbc.Col(dcc.Graph(id='mapa1' + chart_type, figure=figure, config=config), width=width, className=constants.nomargins)
        figures.append(f)
    ret_val = []
    for i in range(rows):
        row = []
        for j in range(cols):
            nelem = i*cols+j
            if nelem < len(figures):
                row.append(figures[nelem])
        ret_val.append(dbc.Row(row))

    return dbc.Col(ret_val)


################################################
#  Mapa chronologiczna powiaty
################################################

def layout_more_chronomap_pow(DF, deltat, _scale, n):
    if deltat is None:
        return
    template = get_template()
    settings = session2settings()
    from_date = str(settings['from_date'])[:10]
    to_date = str(settings['to_date'])[:10]

    d = list(DF['poland'].date.unique())
    if to_date not in d:
        to_date = max(d)
    d = d[:d.index(to_date[:10]) + 1]
    dates = []
    for i in range(n):
        dates.append(d[-1 - deltat * i])

    data_modifier = settings['data_modifier']
    dzielna = settings['dzielna']
    chart_types = settings['chart_type']
    if len(chart_types) != 1:
        return "Wybierz tylko jedną wielkość"

    chart_type = chart_types[0]
    if n == 1:
        rows = 1
        cols = 1
        zoom = 5.5
        width = 12
        height = int(settings['map_height'])
    elif n == 2:
        rows = 1
        cols = 2
        zoom = 5.5
        width = 6
        height = int(settings['map_height'])
    elif n <= 4:
        rows = 2
        cols = 2
        zoom = 4.5
        width = 6
        height = int(settings['map_height']) / 2
    elif n <= 6:
        rows = 2
        cols = 3
        zoom = 4.5
        width = 4
        height = int(settings['map_height']) / 2
    else:
        rows = 3
        cols = 3
        zoom = 4.0
        width = 4
        height = int(settings['map_height']) / 3

    mapbox_access_token = open("data/mapbox.token").read()
    mapbox = dict(accesstoken=mapbox_access_token, center=dict(lon=19.18, lat=52.11), zoom=zoom)
    mapbox['style'] = constants.mapbox_styles[settings['map_mapbox']]

    json_file = r'data/geojson/powiaty-min-ws.geojson'
    featuredikey = 'properties.nazwa'
    with open(json_file, encoding="utf-8") as f:
        geojson = json.load(f)

    figures = []

    dfxs = pd.DataFrame()
    for date in dates:
        if settings['scope'] == 'cities' and len(settings['locations']) > 0:
            locations = list(settings['locations'])
        else:
            locations = list(DF['cities'].location.unique())
        dfx = prepare_data(settings, DF,
                          scope='cities',
                          locations=locations,
                          date=date,
                          chart_types=chart_types,
                          all_columns=True)
        dfxs = dfxs.append(dfx, ignore_index=True).copy()
    vmax = max(dfxs[chart_type])
    vmin = min(dfxs[chart_type])
    dx = (vmax - vmin) / 5
    tick0 = vmin
    dtick = dx

    for date in dates[::-1]:
        dfx = dfxs[dfxs['date'] == date].copy()
        dfx.dropna(subset=['Long', 'Lat'], inplace=True)

        dfx = dfx[dfx['location'] != 'Polska'].copy()
        locations = list(dfx['location'].unique())
        if _scale == 'wspólna skala':
            locations = ['xxx'] + locations + ['yyy']
            z = [vmin] + list(dfx[chart_type]) + [vmax]
        else:
            z = list(dfx[chart_type])
        figure = go.Figure()
        figure.add_trace(go.Choroplethmapbox(
            geojson=geojson,
            featureidkey=featuredikey,
            locations=locations,
            z=z,
            marker=dict(line=dict(color='black', width=0.5 if 'drawborder' in settings['map_options'] else 0.)),
            hoverinfo='text',
            # colorscale=settings['map_color_scale'],
            colorscale=settings['map_color_scale'],
            # colorscale=constants.color_scales[settings['map_color_scale']],
            reversescale=True if 'reversescale' in settings['map_options'] else False,
            marker_opacity=settings['map_opacity'],
            colorbar=dict(
                len=0.5,
                bgcolor=get_session_color(1),
                title='',
                tickfont=dict(color=get_session_color(4)),
                tickmode="auto",
                tick0=tick0,
                dtick=dtick,
                ticks="outside",
            ),
        ))
        #########
        if 'annotations' in settings['map_options']:
            if '0c' in settings['map_options']:
                acc = 0
            elif '1c' in settings['map_options']:
                acc = 1
            elif '2c' in settings['map_options']:
                acc = 2
            else:
                acc = 3
            dfx2 = dfx.copy()
            dfx2[chart_type] = dfx2[chart_type].round(acc).astype(str)
            if constants.trace_props[chart_type]['category'] == 'data':
                dfx2['color2'] = dfx2.groupby('Long')[chart_type].transform(lambda x: ', '.join(x))
                dfx2['location2'] = dfx2.groupby('Long')['location'].transform(lambda x: ', '.join(x))
                dfx2[['location2']].drop_duplicates()
            else:
                dfx2['color2'] = dfx2[chart_type]
                dfx2['location2'] = dfx2['location']
            if 'number' in settings['map_options']:
                anno_text = dfx2['color2'] + constants.trace_props[chart_type]['postfix']
            else:
                anno_text = dfx2['location2'] + '<br>' + dfx2['color2'] + constants.trace_props[chart_type]['postfix']
            figure.add_trace(
                go.Scattermapbox(
                    lat=dfx2.Lat, lon=dfx2.Long,
                    mode='text',
                    hoverinfo='none',
                    below="''",
                    marker=dict(allowoverlap=True),
                    text=anno_text,
                    textfont=dict(family='Bebas Neue', size=settings['font_size_anno'], color=get_session_color(4)),
                ))
        #########
        figure.update_layout(
            images=constants.image_logo_map,
            template=template,
            autosize=True,
            title=dict(
                text=get_title(settings, chart_type),
                x=0.05, y=settings['titleypos'],
                xanchor='left',
                font=dict(size=settings['font_size_title'], color=get_session_color(7))
            ),
            height=height,
            margin=dict(l=settings['marginl'], r=settings['marginr'] + 10, b=settings['marginb'], t=settings['margint'],
                        pad=0),
            paper_bgcolor=get_session_color(1),
            plot_bgcolor=get_session_color(1),
            mapbox=mapbox,
        )
        figure = add_copyright(figure, settings)
        config = {
            'displaylogo': False,
            'responsive': True,
            'toImageButtonOptions': {
                'format': session.get('options_graph_format') if session.get('options_graph_format') else 'svg',
                'filename': 'docent_image', 'height': 700, 'width': 1200}
        }
        f = dbc.Col(dcc.Graph(id='mapa1' + chart_type, figure=figure, config=config), width=width, className=constants.nomargins)
        figures.append(f)
    ret_val = []
    for i in range(rows):
        row = []
        for j in range(cols):
            nelem = i*cols+j
            if nelem < len(figures):
                row.append(figures[nelem])
        ret_val.append(dbc.Row(row))

    return dbc.Col(ret_val)

####################################################
#  Szpital tymczasowy Poznań
####################################################

def layout_more_analiza_poznan(DF, _shift, _date, _3):
    template = get_template()
    settings = session2settings()

    df_p = pd.read_csv('data/sources/inne/Poznań szpital tymczasowy.csv')
    df_p.columns = ['_1', 'date', 'hospi', 'respi', 'zgony', 'in', 'out']
    date_min = min(df_p.date)
    date_max = max(df_p.date)
    df_c = DF['cities']
    df_m = df_c[df_c['location'] == 'Poznań'].copy()
    df_m = df_m[(df_m.date >= date_min) & (df_m.date <= date_max)].copy()
    df_m = df_m[['date', 'new_cases', 'new_deaths']].copy()
    df = pd.merge(df_m, df_p, how='left', left_on=['date'], right_on=['date'])
    df['in'] = df['in'].rolling(7, min_periods=1).mean()
    df['out'] = df['out'].rolling(7, min_periods=1).mean()
    df['respi'] = df['respi'].rolling(7, min_periods=1).mean()
    df['zgony'] = df['zgony'].rolling(7, min_periods=1).mean()
    df['hospi'] = df['hospi'].rolling(7, min_periods=1).mean()

    # przyjęcia/wypisy

    traces = []
    xvals = list(df['date'])
    yvals = list(df['out'])
    fig_data = go.Scatter(
        x=xvals,
        y=yvals,
        line=dict(width=settings['linewidth_basic']),
        name='wypisy'
    )
    traces.append(fig_data)
    yvals = list(df['in'])
    fig_data = go.Scatter(
        x=xvals,
        y=yvals,
        line=dict(width=settings['linewidth_basic']),
        name='przyjęcia'
    )
    traces.append(fig_data)
    figure1 = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text='Szpital Tymczasowy Poznań<br><sub>przyjęcia i wypisy'),
            height=float(settings['plot_height']) * 360,
            legend=dict(x=0., y=1., orientation='h'),
    )
    )

    # zgony

    traces = []
    xvals = list(df['date'])
    yvals = list(df['zgony'])
    fig_data = go.Scatter(
        x=xvals,
        y=yvals,
        line=dict(width=settings['linewidth_basic']),
        name='zgony'
    )
    traces.append(fig_data)
    figure2 = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text='Szpital Tymczasowy Poznań<br><sub>zgony'),
            height=float(settings['plot_height']) * 360,
            legend=dict(x=0., y=1., orientation='h'),
    )
    )

    # zajęte łóżka

    traces = []
    xvals = list(df['date'])
    yvals = list(df['hospi'])
    fig_data = go.Scatter(
        x=xvals,
        y=yvals,
        line=dict(width=settings['linewidth_basic']),
        name='zajęte łóżka'
    )
    traces.append(fig_data)
    figure3 = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text='Szpital Tymczasowy Poznań<br><sub>zajęte łóżka'),
            height=float(settings['plot_height']) * 360,
            legend=dict(x=0., y=1., orientation='h'),
    )
    )

    # zajęte respiratory

    traces = []
    xvals = list(df['date'])
    yvals = list(df['respi'])
    fig_data = go.Scatter(
        x=xvals,
        y=yvals,
        line=dict(width=settings['linewidth_basic']),
        name='zajęte respiratory'
    )
    traces.append(fig_data)
    figure4 = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text='Szpital Tymczasowy Poznań<br><sub>zajęte respiratory'),
            height=float(settings['plot_height']) * 360,
            legend=dict(x=0., y=1., orientation='h'),
    )
    )
    config = {
        'responsive': True,
        'displaylogo': False,

        'locale': 'pl-PL',
    }
    fig1 = dcc.Graph(id='xxxx', figure=figure1, config=config)
    fig2 = dcc.Graph(id='xxxx', figure=figure2, config=config)
    fig3 = dcc.Graph(id='xxxx', figure=figure3, config=config)
    fig4 = dcc.Graph(id='xxxx', figure=figure4, config=config)
    ret_val = [
        dbc.Col(fig1, width=6, className=constants.nomargins),
        dbc.Col(fig2, width=6, className=constants.nomargins),
        dbc.Col(fig3, width=6, className=constants.nomargins),
        dbc.Col(fig4, width=6, className=constants.nomargins),
    ]

    return ret_val


####################################################
#  Analiza dni tygodnia
####################################################

def layout_more_analiza_dni_tygodnia(DF, _shift, _date, _3):
    template = get_template()
    settings = session2settings()

    dows = {
        'Monday': 'Poniedziałek',
        'Tuesday': 'Wtorek',
        'Wednesday': 'Środa',
        'Thursday': 'Czwartek',
        'Friday': 'Piątek',
        'Saturday': 'Sobota',
        'Sunday': 'Niedziela'
    }
    df = DF['poland'].copy()

    df = df[['date', 'location', 'new_cases', 'new_tests']].copy()
    df = df[df['location'] == 'Polska'].copy()

    df['dow'] = pd.to_datetime(df['date']).dt.day_name()
    # przyjęcia/wypisy

    traces = []
    for dow in dows.keys():
        dfx = df[df['dow'] == dow].copy()
        dfx['nc'] = dfx['new_cases'].shift()
        dfx['R'] = dfx['new_cases'] / dfx['nc']
        xvals = list(dfx['date'])
        yvals = list(dfx['R'])
        fig_data = go.Scatter(
            x=xvals,
            y=yvals,
            line=dict(width=settings['linewidth_basic']),
            name=dows[dow]
        )
        traces.append(fig_data)
    figure = go.Figure(
        data=traces,
        layout=dict(
            template=template,
            title=dict(text='Dynamika zachorowań t/t w poszczególnych dniach tygodnia'),
            height=float(settings['plot_height']) * 750,
            legend=dict(x=0., y=1., orientation='h'),
    )
    )
    config = {
        'responsive': True,
        'displaylogo': False,

        'locale': 'pl-PL',
    }
    fig = dcc.Graph(id='xxxx', figure=figure, config=config)
    ret_val = [
        dbc.Col(fig, width=12, className=constants.nomargins),
    ]

    return ret_val


################################################
#  Porównanie synchroniczne wielu przebiegów
################################################

def layout_more_analiza_2(DF, _shift, _date, _3):
    template = get_template()
    settings = session2settings()

    locations = settings['locations']
    if len(locations) > 1:
        return 'Wybierz nie więcej niż jedną lokalizację'
    if len(locations) == 0:
        location = 'Polska'
    else:
        location = locations[0]
    data_od = str(_date)[:10]
    chart_types = settings['chart_type']
    if len(chart_types) == 0:
        return 'Wybierz co najmniej jedną wielkość'
    figures = []
    if len(chart_types) == 1:
        height = 720
        width = 12
    elif len(chart_types) == 2:
        height = 360
        width = 12
    else:
        height = 360
        width = 6
    for trace in chart_types:
        tail = constants.table_mean[settings['table_mean']]
        if constants.trace_props[trace]['category'] == 'data':
            tail += {1: '',
                     2: ' - na 100 000 osób',
                     3: ' - na 1000 km2'}[settings['data_modifier']]
        df = filter_data(settings, DF, loc=location, trace=trace, scope='poland')
        df[trace] = df['data']
        df = df[['date', trace]].copy()
        df2020 = df[df['date'].str.startswith('2020')].copy()
        df2021 = df[df['date'].str.startswith('2021')].copy()
        df2020['short'] = df2020['date'].str.slice(5, 10)
        df2021['short'] = df2021['date'].str.slice(5, 10)
        short = data_od[5:]
        df = pd.merge(df2020, df2021, how='left', left_on=['short'], right_on=['short']).copy()
        df[trace] = df[trace + '_x']
        df[trace + '_x'] = df[trace].shift(int(_shift))
        df = df[df['short'] >= short].copy()
        i = list(df[df[trace + '_y'].isnull()].index)[0]
        df = df.loc[:i+14].copy()
        df.reset_index(drop=True, inplace=True)
        df['day'] = df.index
        if len(df[trace + '_x']) == 0:
            return 'zerowa długość: ' + trace + '_x'
        if len(df[trace + '_y']) == 0:
            return 'zerowa długość: ' + trace + '_y'
        traces = []
        xvals = list(df['day'])
        yvals = list(df[trace + '_x'])
        fig_data = go.Scatter(
            x=xvals,
            y=yvals,
            line=dict(width=settings['linewidth_basic']),
            name='2020 shifted'
        )
        traces.append(fig_data)
        yvals = list(df[trace])
        fig_data = go.Scatter(
            x=xvals,
            y=yvals,
            line=dict(width=settings['linewidth_basic'], dash='dot'),
            name='2020'
        )
        traces.append(fig_data)
        yvals = list(df[trace + '_y'])
        fig_data = go.Scatter(
            x=xvals,
            y=yvals,
            line=dict(width=settings['linewidth_basic']),
            name='2021'
        )
        traces.append(fig_data)
        trace_name = get_trace_name(trace)
        if settings['dzielna'] != '<brak>' and constants.trace_props[trace]['category'] == 'data':
            trace_name = get_trace_name(trace) + ' / ' + get_trace_name(settings['dzielna']).lower()
        figure = go.Figure(
            data=traces,
            layout=dict(
                template=template,
                title=dict(text=trace_name + '<br><sub>' + tail + ', ' + location + \
                                ', przesunięcie = ' + str(_shift) + ', od daty ' + data_od),
                height=float(settings['plot_height']) * height,
                legend=dict(x=0., y=1., orientation='h'),
                yaxis={'type': 'log' if settings['radio_scale'] == 'log' else 'linear'},
        )
        )
        figure = add_copyright(figure, settings)
        config = {
            'responsive': True,
            'displaylogo': False,
            'locale': 'pl-PL',
        }
        fig = dcc.Graph(id='xxxx', figure=figure, config=config)
        figures.append(dbc.Col(fig, width=width, className=constants.nomargins))
    ret_val = figures

    return ret_val


    # if settings['from_date'] is None:
    #     return
    # if settings['to_date'] is None:
    #     return
