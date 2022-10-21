import dash
import flask

from covid_2 import DF
from layouts import layout_timeline_location, layout_timeline_multi_axes, layout_app_header, layout_footer, \
    layout_header, layout_timeline_type
from dash import html
from urllib.parse import urlsplit, parse_qs, unquote

# def get(l, index, default=''):
#     return l[index] if index < len(l) else default
from layouts_more import layout_more_mz_vacc2, \
    layout_more_mz_vacc1, layout_more_mz_vacc3, layout_more_cmpf, layout_more_ecdc_vacc, \
    layout_more_ecdc_vacc0, layout_more_ecdc_vacc7, layout_more_mz_vacc4, layout_more_mz_vacc5, \
    layout_more_mz_psz2
import constants
from util import get_trace_name, filter_data, defaults2session
from flask import session

def get_options(src):
    # składnia opcja1:wartość1;opcja2:wartość2..........

    ret_val = {}
    if src:
        if '+' in src:
            opts = src.split(sep='+')
        else:
            opts = src.split(sep=' ')
        for option in opts:
            opt_item = option.split(sep=':')
            if len(opt_item) == 2:
                ret_val[opt_item[0].strip()] = opt_item[1].strip()
    return ret_val


def render_api(url):
    args = parse_qs(urlsplit(unquote(url)).query)
    args = {x: args[x][0] for x in args}
    ret_val = 'Nieznany błąd'

    p_type = args.get('type', 'unknown')
    if p_type == 'unknown':
        return ['Nieznany szablon parametrów']

    opt = get_options(args.get('opt'))


    # Wykres czasowy dla jednej lokalizacji
    # type=timeloc&scope={cities}{poland}{world}&location={location}&chart={chart_types}&from_date={from_date}

    if p_type == 'timeloc':
        p_scope = args.get('scope')
        if p_scope in ['cities', 'poland', 'world']:
            p_location = args.get('location').lower()
            p_chart = args.get('chart')
            chart_types = [x.strip() for x in p_chart.split(sep=',')]
            if len(chart_types) == 1:
                tail = '- ' +get_trace_name(chart_types[0])
            else:
                tail = ''
            from_date = args.get('from_date', '')
            ret_val = [
                layout_timeline_location(DF,
                                         location=p_location,
                                         data_types=chart_types,
                                         scope=p_scope,
                                         from_date=from_date,
                                         opt=opt,
                                         height='auto'),
                # layout_header(p_location.upper() + ' ' + tail),
                layout_footer()
            ]
        else:
            ret_val = ['nieprawidłowy parametr {scope}']


    # Wykres czasowy dla jednego typu danych
    # type=timeloc&scope={cities}{poland}{world}&location={location}&chart={chart_types}&from_date={from_date}

    if p_type == 'timetype':
        p_scope = args.get('scope')
        if p_scope in ['cities', 'poland', 'world']:
            p_locations = args.get('location')
            p_chart_type = args.get('chart')
            locations = [x.strip() for x in p_locations.split(sep=',')]
            if len(locations) == 1:
                tail = '- ' +locations[0]
            else:
                tail = ''
            from_date = args.get('from_date', '')
            ret_val = [
                # layout_header(get_trace_name(p_chart_type).upper() + ' ' + tail),
                layout_timeline_type(DF,
                                     locations=locations,
                                     data_type=p_chart_type,
                                     scope=p_scope,
                                     from_date=from_date,
                                     opt=opt,
                                     height='auto'),
                layout_footer()
            ]
        else:
            ret_val = ['nieprawidłowy parametr {scope}']

    # Pobranie danych dla jednej lokalizacji i jednego typu danych w formacie csv
    # type=data&scope={cities}{poland}{world}&location={location}&chart={chart_type}&from_date={from_date}

    if p_type == 'data':
        p_scope = args.get('scope')
        if p_scope in ['cities', 'poland', 'world']:
            p_location = args.get('location').lower()
            p_chart = args.get('chart')
            filtered_df = filter_data(DF,
                                      loc=p_location,
                                      trace=p_chart,
                                      total_min=0,
                                      scope=p_scope,
                                      data_modifier=1,
                                      win_type='równe wagi'
                                      )
            filtered_df = filtered_df[['date', 'location', p_chart]]
            ret_val = [
                html.Pre(filtered_df.to_csv(index=False))
                # html.P(children=x, style={'height': 1.5}) for x in filtered_df.to_csv(index=False).split(sep='\n')
            ]
        else:
            ret_val = ['nieprawidłowy parametr {scope}']


    # Zajętość łóżek
    # type=beds&location={województwo|{}}

    if p_type == 'beds':
        p_location = args.get('location', 'Polska')
        if p_location == 'Polska':
            ret_val = [
                layout_header('Polska - zajętość łóżek'),
                layout_timeline_location(DF, 'polska', ['hosp_patients', 'total_beds'],
                                         'resources', height='auto'),
                layout_footer()
            ]
        else:
            ret_val = [
                layout_header('Województwo ' + p_location + ' - zajętość łóżek'),
                layout_timeline_location(DF, p_location.lower(), ['hosp_patients', 'total_beds'],
                                         'resources', height='auto'),
                layout_footer()
            ]
        return ret_val

    # Zajętość respiratorów
    # type=resp&location={województwo|{}}

    if p_type == 'resp':
        p_location = args.get('location', 'Polska')
        if p_location == 'Polska':
            ret_val = [
                layout_header('Polska - zajętość respiratorów'),
                layout_timeline_location(DF, 'polska',
                                         ['total_in_resp', 'total_resp'],
                                         'resources', height='auto'),
                layout_footer()
            ]
        else:
            ret_val = [
                layout_header('Województwo ' + p_location + ' - zajętość respiratorów'),
                layout_timeline_location(DF, p_location.lower(),
                                         ['total_in_resp', 'total_resp'],
                                         'resources', height='auto'),
                layout_footer()
            ]

    # Wykorzystanie zasobów
    # type=resources&location={województwo|{}}

    if p_type == 'resources':
        p_location = args.get('location', 'Polska')
        if p_location == 'Polska':
            ret_val = [
                layout_header('Polska - Użycie zasobów (%)'),
                layout_timeline_location(DF, 'polska',
                                         ['used_resp', 'used_beds'],
                                         'resources', height='auto'),
                layout_footer()
            ]
        else:
            ret_val = [
                layout_header('Województwo ' + p_location + ' - Użycie zasobów (%)'),
                layout_timeline_location(DF, p_location.lower(),
                                         ['used_resp', 'used_beds'],
                                         'resources', height='auto'),
                layout_footer()
            ]

    # Wykresy czasowe (dane sumaryczne)
    # type=total_data&location={województwo|{}}

    if p_type == 'total_data':
        p_location = args.get('location', 'Polska')
        if p_location == 'Polska':
            ret_val = [
                layout_header('Polska - Dane sumaryczne'),
                layout_timeline_multi_axes(DF, 'polska',
                                           ['total_cases', 'total_deaths', 'total_recoveries', 'total_active'],
                                           'world', height='auto'),
                layout_footer()
            ]
        else:
            ret_val = [
                layout_header('Województwo ' + p_location + ' - Dane sumaryczne'),
                layout_timeline_multi_axes(DF, p_location.lower(),
                                         ['new_cases', 'new_deaths', 'new_recoveries', 'new_tests'],
                                         'poland', height='auto'),
                layout_footer()
            ]
        return ret_val


    # Wykresy czasowe (przyrosty dzienne)
    # type=new_data&location={województwo|{}}

    if p_type == 'new_data':
        p_location = args.get('location', 'Polska')
        if p_location == 'Polska':
            ret_val = [
                layout_header('Polska - Przyrosty dzienne'),
                layout_timeline_multi_axes(DF, 'polska',
                                           ['new_cases', 'new_deaths', 'new_recoveries', 'new_tests'],
                                           'world', height='auto'),
                layout_footer()
            ]
        else:
            ret_val = [
                layout_header('Województwo ' + p_location + ' - Przyrosty dzienne'),
                layout_timeline_multi_axes(DF, p_location.lower(),
                                         ['new_cases', 'new_deaths', 'new_recoveries', 'new_tests'],
                                         'poland', height='auto'),
                layout_footer()
            ]


    # Wykresy czasowe podstawowych wskaźników epidemicznych
    # type=indicators&location={województwo|{}}

    if p_type == 'indicators':
        p_location = args.get('location', 'Polska')
        if p_location == 'Polska':
            ret_val = [
                layout_header('Polska - Podstawowe wskaźniki epidemiczne'),
                layout_timeline_multi_axes(DF, 'polska',
                                           ['zapadalnosc', 'umieralnosc', 'smiertelnosc', 'positive_rate'],
                                           'world', height='auto'),
                layout_footer()
            ]
        else:
            ret_val = [
                layout_header('Województwo ' + p_location + ' - Podstawowe wskaźniki epidemiczne'),
                layout_timeline_multi_axes(DF, p_location.lower(),
                                         ['zapadalnosc', 'umieralnosc', 'smiertelnosc', 'positive_rate'],
                                         'poland', height='auto'),
                layout_footer()
            ]

    # Raporty
    # type=reports&unit={typ}}

    if p_type == 'reports':
        defaults2session()
        p_unit = args.get('unit', '-none-')
        p_plot_height = args.get('plot_height', '1.0')
        p_template = args.get('template', 'default')
        session['template'] = p_template
        for key in constants.user_templates[p_template].keys():
            session.pop(key, None)
            session[key] = constants.user_templates[p_template].get(key)
        session['plot_height'] = float(p_plot_height)
        session['template_change'] = True
        session.modified = True

        # http://127.0.0.1:8050/api?type=reports&unit=mz_vacc1
        # OK

        if p_unit == 'mz_vacc1':
            # ['Wiek narastająco', 'Płeć narastająco', 'Wiek przyrosty dzienne',
            # 'Płeć przyrosty dzienne', 'Bilans magazynu', 'Bilans punktów']
            p_variant = args.get('variant', 'Wiek przyrosty dzienne')
            ret_val = [
                layout_header('MZ Szczepienia - podstawowy'),
                layout_more_mz_vacc1(DF, p_variant, '', '')[1],
                layout_footer()
            ]

        # http://127.0.0.1:8050/api?type=reports&unit=mz_vacc2
        # OK

        if p_unit == 'mz_vacc2':
            p_dawka = args.get('dawka', 'dawka_1')
            ret_val = [
                layout_header('Udział grup wiekowych w tygodniowych szczepieniach'),
                layout_more_mz_vacc2(DF, p_dawka, '', '')[1],
                layout_footer()
            ]

        # http://127.0.0.1:8050/api?type=reports&unit=mz_vacc3
        # OK

        elif p_unit == 'mz_vacc3':
            p_location = args.get('location', '--miasta wojewódzkie')
            p_order = args.get('order', 'malejąco')  # malejąco, rosnąco, alfabetycznie
            ret_val = [
                layout_header(' Procent zaszczepionych w powiatach (MZ)'),
                layout_more_mz_vacc3(DF, p_location, p_order, '')[1],
                layout_footer()
            ]

        # http://127.0.0.1:8050/api?type=reports&unit=mz_vacc4&type=[type]
        # OK

        elif p_unit == 'mz_vacc4':
            p_subtype = args.get('subtype', 'wszystkie gminy')
            ret_val = [
                layout_header(' Ranking wojewódzki szczepień wg rodzaju gmin (MZ)'),
                layout_more_mz_vacc4(DF, p_subtype, '', '')[1],
                layout_footer()
            ]

        # http://127.0.0.1:8050/api?type=reports&unit=mz_vacc5&wojew=[wojew]
        # OK

        elif p_unit == 'mz_vacc5':
            p_wojew = args.get('wojew', 'mazowieckie')
            ret_val = [
                layout_header(' Ranking powiatowy szczepień w województwie (MZ)'),
                layout_more_mz_vacc5(DF, p_wojew, '', '')[1],
                layout_footer()
            ]

        # http://127.0.0.1:8050/api?type=reports&unit=mz_psz25&subtype=[subtype]&field=[field]
        # OK

        elif p_unit == 'mz_psz2':
            p_subtype = args.get('subtype', 'szczepienia na 100k')
            p_field = args.get('field', 'total_1d')
            session['map_height'] = 900
            session['map_opacity'] = 0.9
            ret_val = [
                layout_header(' Mapa powiatowych wskaźników szczepień (MZ)'),
                layout_more_mz_psz2(DF, p_subtype, p_field, ''),
                layout_footer()
            ]

        # http://127.0.0.1:8050/api?type=reports&unit=ecdc_vacc&age=Age_70_79&brand=COM&shot=dawka_12
        # OK

        elif p_unit == 'ecdc_vacc':
            p_location = args.get('location', 'Polska')
            p_age = args.get('age', 'ALL')
            p_brand = args.get('brand', 'ALL')
            p_shot = args.get('shot', 'dawka_1')
            session['locations'] = [p_location]
            ret_val = [
                layout_header(' Szczepienia - sumy tygodniowe (ECDC)'),
                layout_more_ecdc_vacc(DF, p_age, p_brand, p_shot),
                layout_footer()
            ]

        # http://127.0.0.1:8050/api?type=reports&unit=ecdc_vacc0
        # OK

        elif p_unit == 'ecdc_vacc0':
            p_location = args.get('location', 'Polska')
            p_age = args.get('age', 'ALL')
            p_brand = args.get('brand', 'ALL')
            session['locations'] = [p_location]
            ret_val = [
                layout_header(' Szczepienia - bilans razem (ECDC)'),
                layout_more_ecdc_vacc0(DF, p_age, p_brand, ''),
                layout_footer()
            ]

        # http://127.0.0.1:8050/api?type=reports&unit=ecdc_vacc7

        elif p_unit == 'ecdc_vacc7':
            p_location = args.get('location', 'Polska')
            p_age = args.get('age', 'ALL')
            session['locations'] = p_location
            ret_val = [
                layout_header(' Procent zaszczepienia w województwach (ECDC)'),
                layout_more_ecdc_vacc7(DF, p_age, '', '')[1],
                layout_footer()
            ]

        # http://127.0.0.1:8050/api?type=reports&unit=cmpf
        # http://127.0.0.1:8050/api?type=reports&unit=cmpf&location=Podlaskie
        # OK

        elif p_unit == 'cmpf':
            p_location = args.get('location', 'Polska')
            session.modified = True
            ret_val = [
                layout_header(' porównanie fali jesiennej i wiosennej'),
                layout_more_cmpf(DF, p_location, '', '')[0],
                layout_footer()
            ]

        # http://127.0.0.1:8050/api?type=reports&unit=rmaw2
        # OK

        elif p_unit == 'rmaw2':
            p_location = args.get('location', 'Polska')
            ret_val = [
                layout_header(' Raport mobilności wg Apple'),
                layout_more_rmaw2(DF, '', '', '')[0],
                layout_footer()
            ]

    return ret_val
