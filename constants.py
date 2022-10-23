from random import random

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json
from collections import OrderedDict
from datetime import datetime as dt, timedelta
import inspect
import plotly
from flask import session
import plotly.graph_objects as go

top_opt_class = 'border-left border-danger p-2'
top_row_class = 'shadow p-1 mt-0 bg-green mr-1 text-wrap ml-1'
top_row_class_inner = 'shadow p-1 mt-0 mb-0 bg-green mr-1 text-wrap'

news_text = '''
**17 XI 2020** dodanie wykresu liczby zgonów Covid-19 na tle ogólnej liczby zgonów.

**14 XI 2020** dodanie wykresu wskaźnika reprodukcji R(t) dla Polski.

**13 XI 2020** w tabeli porównawczej województw dodano wskaźnik średniej liczby nowych zakażeń na 100 000 osób z ostatnich 7 dni (kryterium wprowadzenia kwarantanny narodowej).

**9 XI 2020** dodano wykresy wykorzystania łóżek i respiratorów dla Polski i poszczególnych województw
'''
image_url = '/assets/covidepl.png'
image_woj49 = '/assets/stare_woj.png'
image_logo_timeline = [{
    "source": image_url,
    "xref": "paper",
    "yref": "paper",
    "x": 1,
    "y": 0.03,
    "sizex": 0.05,
    "sizey": 0.05,
    "xanchor": "right",
    "yanchor": "top",
    'opacity': 0.1
}]
image_logo_map = [{
    "source": image_url,
    "xref": "paper",
    "yref": "paper",
    "x": 0.1,
    "y": 0.05,
    "sizex": 0.08,
    "sizey": 0.08,
    "xanchor": "right",
    "yanchor": "top",
    'opacity': 0.4
}]
image_stare_woj = [{
    "source": image_woj49,
    "xref": "paper",
    "yref": "paper",
    "x": 0.,
    "y": 0.,
    "sizex": 1,
    "sizey": 1,
    "xanchor": "left",
    "yanchor": "bottom",
    'opacity': 1.
}]
image_logo_dynamics = [{
    "source": image_url,
    "xref": "paper",
    "yref": "paper",
    "x": 0,
    "y": 1.1,
    "sizex": 0.05,
    "sizey": 0.05,
    "xanchor": "left",
    "yanchor": "top",
    'opacity': 0.1
}]

# sortowanie list "po polsku"

chars_p = " '" + '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
chars_l = '01234567890aąbcćdeęfghijklłmnńoópqrsśtuvwxyzźż'
chars_u = '01234567890AĄBCĆDEĘFGHIJKLŁMNŃOÓPQRSŚTUVWXYZŹŻ'
tp = {chars_p[i]: chr(i) for i in range(len(chars_p))}
tl = {chars_l[i]: chr(len(chars_p) + i) for i in range(len(chars_l))}
tu = {chars_u[i]: chr(len(chars_p + chars_l) + i) for i in range(len(chars_u))}
tui = {chars_u[i]: chr(len(chars_p) + i) for i in range(len(chars_u))}
XTABLE = {**tp, **tl, **tu}
XTABLE_IC = {**tp, **tl, **tui}

slownik = pd.read_csv("data/dict/sl_nazw_panstw_ang_pol.csv", sep='\t')
country_dict = {list(slownik.eng.str.strip())[i]: list(slownik.pl.str.strip())[i] for i in range(len(slownik))}


def pl_case(x):
    ret_val = ''.join([XTABLE[c] for c in x])
    return ret_val


def pl_icase(x):
    ret_val = ''.join([XTABLE_IC[c] for c in x])
    return ret_val


# użycie
# s_case = sorted(arr, key=pl_case)
# s_icase = sorted(arr, key=pl_icase)

######################
# LISTY GEOGRAFICZNE
######################


continents = ["Europa", "Afryka", "Ameryka Północna", "Ameryka Południowa", "Oceania", "Azja"]

continents_eng = {"Europa": 'Europe',
                  "Afryka": 'Africa',
                  "Ameryka Północna": 'North America',
                  "Ameryka Południowa": 'South America',
                  "Oceania": 'Oceania',
                  "Azja": 'Asia'}
latin_america_list = ['Antigua i Barbuda', 'Argentyna', 'Bahamy', 'Barbados', 'Belize',
                      'Boliwia', 'Brazylia', 'Chile', 'Kolumbia', 'Kostaryka',
                      'Kuba', 'Dominika', 'Dominikana', 'Ekwador', 'Salwador', 'Grenada',
                      'Gwatemala', 'Gujana', 'Haiti', 'Honduras', 'Jamajka', 'Meksyk',
                      'Nikaragua', 'Panama', 'Paragwaj', 'Peru', 'Saint Kitts and Nevis',
                      'Saint Lucia', 'Saint Vincent and the Grenadines',
                      'Sint Maarten (Dutch part)', 'Surinam', 'Trynidad i Tobago',
                      'Turks and Caicos Islands', 'Urugwaj',
                      'Wenezuela']
arab_countries_list = ['Algieria', 'Arabia Saudyjska', 'Bahrajn', 'Egipt', 'Irak', 'Jemen', 'Jordania',
                       'Katar', 'Kuwejt', 'Liban', 'Libia', 'Maroko', 'Mauretania', 'Sahara Zachodnia',
                       'Oman', 'Palestyna', 'Sudan', 'Sudan Południowy', 'Syria', 'Tunezja',
                       'Zjednoczone Emiraty Arabskie']
eu_countries_list = ['Austria', 'Belgia', 'Bułgaria', 'Chorwacja', 'Cypr', 'Czechy', 'Dania', 'Estonia', 'Finlandia',
                     'Francja', 'Niemcy', 'Grecja', 'Węgry', 'Irlandia', 'Włochy', 'Łotwa', 'Litwa', 'Luksemburg',
                     'Malta', 'Holandia', 'Polska', 'Portugalia', 'Rumunia', 'Słowacja', 'Słowenia',
                     'Hiszpania', 'Szwecja']
wojewodzkie_list = ['Białystok', 'Gdańsk', 'Katowice', 'Kielce', 'Kraków', 'Lublin', 'Olsztyn', 'Opole',
                    'Poznań', 'Rzeszów', 'Szczecin', 'Warszawa', 'Wrocław', 'Zielona Góra', 'Łódź', 'Bydgoszcz']
wojewodztwa_list = ['podlaskie', 'pomorskie', 'śląskie', 'świętokrzyskie', 'małopolskie', 'lubelskie',
                    'warmińsko-mazurskie', 'opolskie', 'wielkopolskie', 'podkarpackie', 'zachodniopomorskie',
                    'mazowieckie', 'dolnośląskie', 'lubuskie', 'łódzkie', 'kujawsko-pomorskie']
wojewodztwa_pn_zach = ['Pomorskie', 'Warmińsko-Mazurskie', 'Wielkopolskie', 'Zachodniopomorskie',
                       'Lubuskie', 'Kujawsko-Pomorskie']
wojewodztwa_pd_wsch = ['Podlaskie', 'Śląskie', 'Świętokrzyskie', 'Małopolskie', 'Lubelskie',
                       'Opolskie', 'Podkarpackie', 'Mazowieckie', 'Dolnośląskie', 'Łódzkie']
skandynawskie_list = ['Dania', 'Szwecja', 'Norwegia', 'Islandia', 'Finlandia', 'Estonia']
slowianskie_list = ['Polska', 'Czechy', 'Słowacja', 'Białoruś', 'Rosja', 'Ukraina',
                    'Bośnia i Hercegowina', 'Bułgaria', 'Chorwacja', 'Czarnogóra', 'Macedonia', 'Serbia', 'Słowenia']
bw_rozwiniete_list = ['Norwegia', 'Australia', 'Szwajcaria', 'Niemcy', 'Dania', 'Singapur',
                      'Holandia', 'Irlandia', 'Islandia', 'Kanada', 'Stany Zjednoczone', 'Hongkong', 'Nowa Zelandia',
                      'Szwecja', 'Liechtenstein', 'Wielka Brytania', 'Japonia', 'Korea Południowa', 'Izrael',
                      'Luksemburg', 'Francja', 'Belgia', 'Finlandia', 'Austria', 'Słowenia', 'Włochy', 'Czechy',
                      'Grecja', 'Brunei', 'Estonia', 'Andora', 'Cypr', 'Malta', 'Katar', 'Polska', 'Litwa', 'Chile',
                      'Arabia Saudyjska', 'Słowacja', 'Portugalia', 'Zjednoczone Emiraty Arabskie', 'Węgry', 'Łotwa',
                      'Argentyna', 'Chorwacja', 'Bahrajn', 'Czarnogóra', 'Rosja', 'Rumunia', 'Kuwejt']
demoludy_list = ['Bułgaria', 'Chorwacja', 'Czechy', 'Estonia', 'Serbia', 'Albania', 'Bośnia i Hercegowina',
                 'Węgry', 'Łotwa', 'Litwa', 'Polska', 'Rumunia', 'Słowacja', 'Słowenia', 'Mołdawia', 'Rosja',
                 'Białoruś', 'Ukraina', 'Czarnogóra', 'Macedonia']
powiaty_list = ['Białystok', 'Bydgoszcz', 'Gdańsk', 'Gorzów Wielkopolski', 'Katowice', 'Kielce', 'Kraków', 'Lublin',
                'Olsztyn', 'Opole', 'Poznań', 'Rzeszów', 'St. Warszawa', 'Szczecin', 'Wałbrzych', 'Wrocław',
                'Zielona Góra', 'aleksandrowski', 'augustowski', 'bartoszycki', 'bełchatowski', 'białobrzeski',
                'białogardzki', 'bielski', 'bieruńsko-lędziński', 'bieszczadzki', 'biłgorajski', 'bocheński',
                'bolesławiecki', 'braniewski', 'brodnicki', 'brzeski', 'brzeziński', 'brzozowski', 'buski', 'bytowski',
                'będziński', 'chełmiński', 'chodzieski', 'chojnicki', 'choszczeński', 'chrzanowski', 'ciechanowski',
                'cieszyński', 'czarnkowsko-trzcianecki', 'człuchowski', 'drawski', 'działdowski',
                'dzierżoniowski', 'dąbrowski', 'dębicki', 'ełcki', 'garwoliński', 'gdański', 'giżycki', 'gnieźnieński',
                'goleniowski', 'golubsko-dobrzyński', 'gorlicki', 'gostyniński', 'gostyński', 'gołdapski', 'grajewski',
                'grodziski', 'gryficki', 'gryfiński', 'grójecki', 'górowski', 'głogowski',
                'głubczycki', 'hajnowski', 'hrubieszowski', 'inowrocławski', 'iławski', 'janowski', 'jarociński',
                'jarosławski', 'jasielski', 'jaworski', 'jędrzejowski', 'kamiennogórski', 'kamieński',
                'kartuski', 'kazimierski', 'kluczborski', 'kolbuszowski', 'kolneński', 'kolski', 'konecki',
                'kozienicki', 'kołobrzeski', 'kościański', 'kościerski', 'krapkowicki', 'krasnostawski', 'kraśnicki',
                'krotoszyński', 'krośnieński', 'kutnowski', 'kwidzyński', 'kędzierzyńsko-kozielski', 'kępiński',
                'kętrzyński', 'kłobucki', 'kłodzki', 'legionowski', 'leski', 'leżajski', 'lidzbarski', 'limanowski',
                'lipnowski', 'lipski', 'lubaczowski', 'lubartowski', 'lubański', 'lubiński', 'lubliniecki', 'lwówecki',
                'lęborski', 'makowski', 'malborski', 'miechowski', 'mielecki', 'mikołowski', 'milicki', 'międzychodzki',
                'międzyrzecki', 'miński', 'mogileński', 'moniecki', 'mrągowski', 'myszkowski',
                'myślenicki', 'myśliborski', 'mławski', 'nakielski', 'namysłowski', 'nidzicki', 'niżański',
                'nowodworski', 'nowomiejski', 'nowosolski', 'nowotarski', 'nowotomyski', 'nyski', 'obornicki', 'olecki',
                'oleski', 'oleśnicki', 'olkuski', 'opatowski', 'opoczyński', 'opolski', 'ostrowiecki', 'ostrowski',
                'ostrzeszowski', 'ostródzki', 'otwocki', 'oławski', 'oświęcimski', 'pabianicki', 'pajęczański',
                'parczewski', 'piaseczyński', 'pilski', 'piski', 'pińczowski', 'pleszewski', 'poddębicki', 'policki',
                'polkowicki', 'proszowicki', 'prudnicki', 'pruszkowski', 'przasnyski', 'przeworski', 'przysuski',
                'pszczyński', 'pucki', 'puławski', 'pułtuski', 'pyrzycki', 'płoński', 'raciborski', 'radomszczański',
                'radziejowski', 'radzyński', 'rawicki', 'rawski', 'ropczycko-sędziszowski', 'rycki',
                'rypiński', 'sandomierski', 'sanocki', 'sejneński', 'siemiatycki', 'sieradzki', 'sierpecki',
                'skarżyski', 'sochaczewski', 'sokołowski', 'sokólski', 'stalowowolski', 'starachowicki', 'stargardzki',
                'starogardzki', 'staszowski', 'strzelecki', 'strzelecko-drezdenecki', 'strzeliński',
                'strzyżowski', 'sulęciński', 'suski', 'szamotulski', 'szczecinecki', 'szczycieński', 'sztumski',
                'szydłowiecki', 'sępoleński', 'sławieński', 'słubicki', 'słupecki', 'tarnogórski', 'tatrzański',
                'tczewski', 'tomaszowski', 'trzebnicki', 'tucholski', 'turecki', 'wadowicki',
                'warszawski zachodni', 'wałecki', 'wejherowski', 'wielicki', 'wieluński', 'wieruszowski',
                'wodzisławski', 'wolsztyński', 'wołomiński', 'wołowski',
                'wrzesiński', 'wschowski', 'wysokomazowiecki', 'wyszkowski', 'wąbrzeski', 'wągrowiecki', 'węgorzewski',
                'węgrowski', 'włodawski', 'włoszczowski', 'zambrowski', 'zawierciański', 'zduńskowolski', 'zgierski',
                'zgorzelecki', 'zwoleński', 'ząbkowicki', 'złotoryjski', 'złotowski', 'Łódź',
                'łaski', 'łańcucki', 'łobeski', 'łosicki', 'łowicki', 'łukowski', 'łęczycki', 'łęczyński', 'średzki',
                'śremski', 'świdnicki', 'świdwiński', 'świebodziński', 'świecki', 'żagański', 'żarski', 'żniński',
                'żuromiński', 'żyrardowski', 'żywiecki'
                ]
wojew_cap = {
    'Polska': 'Polska',
    'podlaskie': 'Podlaskie',
    'pomorskie': 'Pomorskie',
    'śląskie': 'Śląskie',
    'świętokrzyskie': 'Świętokrzyskie',
    'małopolskie': 'Małopolskie',
    'lubelskie': 'Lubelskie',
    'warmińsko-mazurskie': 'Warmińsko-Mazurskie',
    'opolskie': 'Opolskie',
    'wielkopolskie': 'Wielkopolskie',
    'podkarpackie': 'Podkarpackie',
    'zachodniopomorskie': 'Zachodniopomorskie',
    'mazowieckie': 'Mazowieckie',
    'dolnośląskie': 'Dolnośląskie',
    'lubuskie': 'Lubuskie',
    'łódzkie': 'Łódzkie',
    'kujawsko-pomorskie': 'Kujawsko-Pomorskie'
}
wojew_pop = {
    'podlaskie': 1176576,
    'pomorskie': 2346717,
    'śląskie': 4508078,
    'świętokrzyskie': 1230044,
    'małopolskie': 3413931,
    'lubelskie': 2103342,
    'warmińsko-mazurskie': 1420514,
    'opolskie': 980771,
    'wielkopolskie': 3500361,
    'podkarpackie': 2125901,
    'zachodniopomorskie': 1693219,
    'mazowieckie': 5428031,
    'dolnośląskie': 2898525,
    'lubuskie': 1010177,
    'łódzkie': 2448713,
    'kujawsko-pomorskie': 2069273
}

wojew_mid = {  # Lat, Long
    'Polska': [52.378793, 19.736883],
    'Podlaskie': [53.253191, 23.028253],
    'Pomorskie': [54.125766, 17.912459],
    'Śląskie': [50.648878, 19.053345],
    'Świętokrzyskie': [50.768348, 20.687523],
    'Małopolskie': [49.852807, 20.223355],
    'Lubelskie': [51.154431, 22.842407],
    'Warmińsko-Mazurskie': [53.821080, 20.702638],
    'Opolskie': [50.644156, 17.784384],
    'Wielkopolskie': [52.241848, 17.223725],
    'Podkarpackie': [49.935197, 22.061772],
    'Zachodniopomorskie': [53.528877, 15.435255],
    'Mazowieckie': [52.447561, 21.070414],
    'Dolnośląskie': [51.155511, 16.219449],
    'Lubuskie': [52.183999, 15.196372],
    'Łódzkie': [51.694337, 19.374116],
    'Kujawsko-Pomorskie': [53.141944, 18.421349]
}
wojew_mid_lower = {  # Lat, Long
    'polska': [52.378793, 19.736883],
    'podlaskie': [53.253191, 23.028253],
    'pomorskie': [54.125766, 17.912459],
    'śląskie': [50.648878, 19.053345],
    'świętokrzyskie': [50.768348, 20.687523],
    'małopolskie': [49.852807, 20.223355],
    'lubelskie': [51.154431, 22.842407],
    'warmińsko-mazurskie': [53.821080, 20.702638],
    'opolskie': [50.644156, 17.784384],
    'wielkopolskie': [52.241848, 17.223725],
    'podkarpackie': [49.935197, 22.061772],
    'zachodniopomorskie': [53.528877, 15.435255],
    'mazowieckie': [52.447561, 21.070414],
    'dolnośląskie': [51.155511, 16.219449],
    'lubuskie': [52.183999, 15.196372],
    'łódzkie': [51.694337, 19.374116],
    'kujawsko-pomorskie': [53.141944, 18.421349]
}

terc_woj = {
    2: 'dolnośląskie', 4: 'kujawsko-pomorskie', 6: 'lubelskie', 8: 'lubuskie',
    10: 'łódzkie', 12: 'małopolskie', 14: 'mazowieckie', 16: 'opolskie',
    18: 'podkarpackie', 20: 'podlaskie', 22: 'pomorskie', 24: 'śląskie',
    26: 'świętokrzyskie', 28: 'warmińsko-mazurskie', 30: 'wielkopolskie',
    32: 'zachodniopomorskie',
}

# PL92X PL9
eurostat_nuts = {}
eurostat_nuts['PL'] = 'Polska'
eurostat_nuts['PL84'] = 'podlaskie'
eurostat_nuts['PL63'] = 'pomorskie'
eurostat_nuts['PL22'] = 'śląskie'
eurostat_nuts['PL72'] = 'świętokrzyskie'
eurostat_nuts['PL21'] = 'małopolskie'
eurostat_nuts['PL81'] = 'lubelskie'
eurostat_nuts['PL62'] = 'warmińsko-mazurskie'
eurostat_nuts['PL52'] = 'opolskie'
eurostat_nuts['PL41'] = 'wielkopolskie'
eurostat_nuts['PL82'] = 'podkarpackie'
eurostat_nuts['PL42'] = 'zachodniopomorskie'
eurostat_nuts['PL9'] = 'mazowieckie'
eurostat_nuts['PL51'] = 'dolnośląskie'
eurostat_nuts['PL43'] = 'lubuskie'
eurostat_nuts['PL71'] = 'łódzkie'
eurostat_nuts['PL61'] = 'kujawsko-pomorskie'
eurostat_nuts_reverse = {}
eurostat_nuts_reverse['Polska'] = 'PL'
eurostat_nuts_reverse['podlaskie'] = 'PL84'
eurostat_nuts_reverse['pomorskie'] = 'PL63'
eurostat_nuts_reverse['śląskie'] = 'PL22'
eurostat_nuts_reverse['świętokrzyskie'] = 'PL72'
eurostat_nuts_reverse['małopolskie'] = 'PL21'
eurostat_nuts_reverse['lubelskie'] = 'PL81'
eurostat_nuts_reverse['warmińsko-mazurskie'] = 'PL62'
eurostat_nuts_reverse['opolskie'] = 'PL52'
eurostat_nuts_reverse['wielkopolskie'] = 'PL41'
eurostat_nuts_reverse['podkarpackie'] = 'PL82'
eurostat_nuts_reverse['zachodniopomorskie'] = 'PL42'
eurostat_nuts_reverse['mazowieckie'] = 'PL9'
eurostat_nuts_reverse['dolnośląskie'] = 'PL51'
eurostat_nuts_reverse['lubuskie'] = 'PL43'
eurostat_nuts_reverse['łódzkie'] = 'PL71'
eurostat_nuts_reverse['kujawsko-pomorskie'] = 'PL61'

nuts_countries = {'AT': 'Austria', 'BE': 'Belgia', 'BG': 'Bułgaria', 'CY': 'Cypr', 'CZ': 'Czechy', 'DE': ' Niemcy',
                  'DK': 'Dania', 'EE': 'Estonia', 'EL': 'Grecja', 'ES': 'Hiszpania', 'FI': 'Finlandia', 'FR': 'Francja',
                  'HR': ' Chorwacja', 'HU': 'Węgry', 'IE': 'Irlandia', 'IS': 'Islandia', 'IT': ' Włochy',
                  'LI': 'Liechtenstein', 'LT': 'Litwa', 'LU': 'Luksemburg', 'LV': 'Łotwa', 'MT': 'Czarnogóra',
                  'NL': 'Holandia', 'NO': 'Norwegia', 'PL': 'Polska', 'PT': 'Portugalia', 'RO': 'Rumunia',
                  'SE': 'Serbia', 'SI': 'Słowenia', 'SK': 'Słowacja'}

powiat_translation_table = [
    dict(location='bielski', wojew='śląskie', new_location='bielski (śląskie)', kod=2402),
    dict(location='bielski', wojew='podlaskie', new_location='bielski (podlaskie)', kod=2003),
    dict(location='brzeski', wojew='małopolskie', new_location='brzeski (małopolskie)', kod=1202),
    dict(location='brzeski', wojew='opolskie', new_location='brzeski (opolskie)', kod=1601),
    dict(location='grodziski', wojew='mazowieckie', new_location='grodziski (mazowieckie)', kod=1405),
    dict(location='grodziski', wojew='wielkopolskie', new_location='grodziski (wielkopolskie)', kod=3005),
    dict(location='krośnieński', wojew='lubuskie', new_location='krośnieński (lubuskie)', kod=802),
    dict(location='krośnieński', wojew='podkarpackie', new_location='krośnieński (podkarpackie)', kod=1807),
    dict(location='nowodworski', wojew='mazowieckie', new_location='nowodworski (mazowieckie)', kod=1414),
    dict(location='nowodworski', wojew='pomorskie', new_location='nowodworski (pomorskie)', kod=2210),
    dict(location='ostrowski', wojew='mazowieckie', new_location='ostrowski (mazowieckie)', kod=1416),
    dict(location='ostrowski', wojew='wielkopolskie', new_location='ostrowski (wielkopolskie)', kod=3017),
    dict(location='tomaszowski', wojew='lubelskie', new_location='tomaszowski (lubelskie)', kod=618),
    dict(location='tomaszowski', wojew='łódzkie', new_location='tomaszowski (łódzkie)', kod=1016),
    dict(location='średzki', wojew='dolnośląskie', new_location='średzki (dolnośląskie)', kod=218),
    dict(location='średzki', wojew='wielkopolskie', new_location='średzki (wielkopolskie)', kod=3025),
    dict(location='świdnicki', wojew='dolnośląskie', new_location='świdnicki (dolnośląskie)', kod=219),
    dict(location='świdnicki', wojew='lubelskie', new_location='świdnicki (lubelskie)', kod=617),
    dict(location='opolski', wojew='opolskie', new_location='opolski (opolskie)', kod=1609),
    dict(location='opolski', wojew='lubelskie', new_location='opolski (lubelskie)', kod=612),
    # dict(location='dąbrowski', wojew='śląskie', new_location='dąbrowski (śląskie)', kod=2465),
    # dict(location='dąbrowski', wojew='małopolskie', new_location='dąbrowski (małopolskie)', kod=1204)
]


def get_subgroups():
    file = r'data/geojson/world.geo.json'
    with open(file) as json_file:
        data = json.load(json_file)
    columns = data['features'][0]['properties']
    df_data = {col: [] for col in columns}
    for x in data['features']:
        for col in columns:
            df_data[col].append(x['properties'][col])

    country_iso_translate = {df_data['adm0_a3'][i]: df_data['iso_a2'][i] for i in range(len(df_data['adm0_a3']))}
    df = pd.DataFrame(df_data)
    subs = {
        'Europa Południowa': list(df.loc[df.subregion == 'Southern Europe', 'adm0_a3']),
        'Europa Pólnocna': list(df.loc[df.subregion == 'Northern Europe', 'adm0_a3']),
        'Europa Zachodnia': list(df.loc[df.subregion == 'Western Europe', 'adm0_a3']),
        'Europa Wschodnia': list(df.loc[df.subregion == 'Eastern Europe', 'adm0_a3']),
        'Karaiby': list(df.loc[df.subregion == 'Carribean', 'adm0_a3']),
        'Ameryka Środkowa': list(df.loc[df.subregion == 'Central America', 'adm0_a3']),
        'Azja Południowa': list(df.loc[df.subregion == 'Southern Asia', 'adm0_a3']),
        'Azja Srodkowa': list(df.loc[df.subregion == 'Central Asia', 'adm0_a3']),
        'Afryka Centralna': list(df.loc[df.subregion == 'Central Africa', 'adm0_a3']),
        'Afryka Wschodnia': list(df.loc[df.subregion == 'Eastern Africa', 'adm0_a3']),
        'Afryka Zachodnia': list(df.loc[df.subregion == 'Western Africa', 'adm0_a3']),
        'Afryka Południowa': list(df.loc[df.subregion == 'Southern Africa', 'adm0_a3']),
        'Afryka Północna': list(df.loc[df.subregion == 'Northern Africa', 'adm0_a3']),
        'Wyspy Pacyfiku': list(df.loc[df.subregion.isin(['Polynesia', 'Melanesia', 'Micronesia']), 'adm0_a3']),
        'Kraje rozwijające się': list(df.loc[df.economy == '6. Developing region', 'adm0_a3']),
        'Kraje rozwinięte (poza G7)': list(df.loc[df.economy == '2. Developed region: nonG7', 'adm0_a3']),
        'Kraje rozwinięte (G7)': list(df.loc[df.economy == '1. Developed region: G7', 'adm0_a3']),
        'Gospodarki wschodzące': list(df.loc[df.economy.isin(['3. Emerging region: BRIC',
                                                              '5. Emerging region: G20',
                                                              '4. Emerging region: MIKT']), 'adm0_a3']),
        'Kraje najmniej rozwinięte': list(df.loc[df.economy == '7. Least developed region', 'adm0_a3'])
    }
    return subs, country_iso_translate


subgroups, country_iso_translate = get_subgroups()

####################
# DANE SYSTEMOWE
####################

fields = {
    'date': {'dtype': np.object},
    'iso_code': {'dtype': np.object},
    'location': {'dtype': np.object},
    'Continent_Name': {'dtype': np.object},
    'wojew': {'dtype': np.object},
    'total_cases': {'dtype': np.float64},
    'population': {'dtype': np.float64},
    'area': {'dtype': np.float64},
    'new_cases': {'dtype': np.float64},
    'new_cases_total': {'dtype': np.float64},
    'total_deaths': {'dtype': np.float64},
    'new_deaths': {'dtype': np.float64},
    'mort': {'dtype': np.float64},
    'new_excess': {'dtype': np.float64},
    'total_excess': {'dtype': np.float64},
    'new_recoveries': {'dtype': np.float64},
    'total_recoveries': {'dtype': np.float64},
    'total_active': {'dtype': np.float64},
    'total_tests': {'dtype': np.float64},
    'total_vaccinations': {'dtype': np.float64},
    'new_vaccinations': {'dtype': np.float64},
    'people_vaccinated': {'dtype': np.float64},
    'people_fully_vaccinated': {'dtype': np.float64},
    'new_tests': {'dtype': np.float64},
    'double_days': {'dtype': np.float64},
    'zapadalnosc': {'dtype': np.float64},
    'umieralnosc': {'dtype': np.float64},
    'smiertelnosc': {'dtype': np.float64},
    'wyzdrawialnosc': {'dtype': np.float64},
    'dynamikaI': {'dtype': np.float64},
    'dynamikaD': {'dtype': np.float64},
    'kwarantanna': {'dtype': np.float64},
    'new_reinf': {'dtype': np.float64},
    'reproduction_rate': {'dtype': np.float64},
    'icu_patients': {'dtype': np.float64},
    'positive_rate': {'dtype': np.float64},
    'Lat': {'dtype': np.float64},
    'Long': {'dtype': np.float64},
    'hosp_patients': {'dtype': np.float64},
    'total_quarantine': {'dtype': np.float64},
    'total_nadzor': {'dtype': np.float64},
    'new_deaths_c': {'dtype': np.float64},
    'new_deaths_nc': {'dtype': np.float64},
    'new_tests_poz': {'dtype': np.float64},
}

scale_names = [name
               for name, body in inspect.getmembers(getattr(px.colors, 'sequential'))
               if isinstance(body, list)]
color_scales = {x: eval('plotly.colors.sequential.' + x) for x in scale_names}
color_scales['Retro Metro'] = ["#ea5545", "#f46a9b", "#ef9b20", "#edbf33", "#ede15b", "#bdcf32", "#87bc45", "#27aeef",
                               "#b33dc6"]
color_scales['Dutch Field'] = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff",
                               "#00bfa0"]
color_scales['River Nights'] = ["#b30000", "#7c1158", "#4421af", "#1a53ff", "#0d88e6", "#00b7c7", "#5ad45a", "#8be04e",
                                "#ebdc78"]
color_scales['Spring Pastels'] = ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db",
                                  "#fdcce5", "#8bd3c7"]
color_scales['Default'] = ["dodgerblue", "#E31A1C", "green", "#6A3D9A", "#FF7F00", "black", "gold", "skyblue",
                           "#FB9A99",
                           "palegreen", "#CAB2D6", "#FDBF6F", "gray", "khaki", "maroon", "orchid", "deeppink",
                           "blue", "steelblue", "darkturquoise", "green", "yellow", "yellow", "darkorange", "brown"]
color_scales['ECDC greens5'] = ['#e8e8ba', '#b3d06e', '#65b52a', '#1a8232', '#044623']
color_scales['Discrete reds6'] = ['#ffc300', '#ff5733', '#c70039', '#900c3e', '#571845']
color_scales['Discrete browns4'] = ['#ede0d4', '#e6ccb2', '#b08968', '#9c6644']
color_scales['Discrete brick5'] = ["#ea8c55", "#c75146", "#ad2e24", "#81171b", "#540804"]
color_scales['Discrete greys10'] = ["#f8f9fa", "#e9ecef", "#dee2e6", "#ced4da", "#adb5bd", "#6c757d", "#495057",
                                    "#343a40", "#212529"]
color_scales['Discrete browns6'] = ['#ede0d4', '#e6ccb2', '#ddb892', '#b08968', '#7f5539', '#9c6644']
color_scales['Discrete red/green2'] = ['#80b918', '#f08080']
color_scales['Discrete green/red2'] = ['#f08080', '#80b918']
color_scales['Discrete lightbrown4'] = ['#c9cba3', '#ffe1a8', '#e26d5c', '#723d46']

color_scales['ECDC mix5'] = ['#aec086', '#b9c5c7', '#d7d2cc', '#a08c7d', '#b3504b']

color_scales = OrderedDict(color_scales.items())

trace_props = OrderedDict([
    ('new_cases', {'title': 'Infekcje nowe',
                   'category': 'data',
                   'subcategory': 'new',
                   'round': 0,
                   'disable_pl': False,
                   'disable_world': False,
                   'disable_cities': False,
                   'log_scale': True,
                   'postfix': '',
                   'color': 'lightblue'}),
    ('new_deaths', {'title': 'Śmiertelne nowe',
                    'category': 'data',
                    'subcategory': 'new',
                    'round': 0,
                    'disable_pl': False,
                    'disable_world': False,
                    'disable_cities': False,
                    'log_scale': False,
                    'postfix': '',
                    'color': 'pink'}),
    ('new_recoveries', {'title': 'Wyzdrowienia nowe',
                        'category': 'data',
                        'subcategory': 'new',
                        'round': 0,
                        'disable_pl': False,
                        'disable_world': False,
                        'disable_cities': False,
                        'log_scale': False,
                        'postfix': '',
                        'color': 'lightgreen'}),
    ('total_cases', {'title': 'Infekcje razem',
                     'category': 'data',
                     'subcategory': 'total',
                     'round': 0,
                     'disable_pl': False,
                     'disable_world': False,
                     'disable_cities': False,
                     'log_scale': True,
                     'postfix': '',
                     'color': 'orange'}),
    ('new_cases_total', {'title': 'Infekcje + reinfekcje nowe',
                         'category': 'data',
                         'subcategory': 'total',
                         'round': 0,
                         'disable_pl': False,
                         'disable_world': True,
                         'disable_cities': False,
                         'log_scale': True,
                         'postfix': '',
                         'color': 'orange'}),
    ('total_deaths', {'title': 'Śmiertelne razem',
                      'category': 'data',
                      'subcategory': 'total',
                      'round': 0,
                      'disable_pl': False,
                      'disable_world': False,
                      'disable_cities': False,
                      'log_scale': True,
                      'postfix': '',
                      'color': 'red'}),
    ('total_recoveries', {'title': 'Wyzdrowienia razem',
                          'category': 'data',
                          'subcategory': 'total',
                          'round': 0,
                          'disable_pl': False,
                          'disable_world': False,
                          'disable_cities': False,
                          'log_scale': False,
                          'postfix': '',
                          'color': 'green'}),
    ('total_active', {'title': 'Aktywne razem',
                      'category': 'data',
                      'subcategory': 'total',
                      'round': 0,
                      'disable_pl': False,
                      'disable_world': False,
                      'disable_cities': False,
                      'log_scale': False,
                      'postfix': '',
                      'color': 'lightgray'}),
    ('hosp_patients', {'title': 'Hospitalizowani razem',
                       'category': 'data',
                       'subcategory': 'total',
                       'round': 2,
                       'disable_pl': False,
                       'disable_world': False,
                       'disable_cities': True,
                       'log_scale': False,
                       'postfix': '',
                       'color': 'white'}),
    ('icu_patients', {'title': 'Respiratory w użyciu',
                      'category': 'data',
                      'subcategory': 'total',
                      'round': 2,
                      'disable_pl': False,
                      'disable_world': False,
                      'disable_cities': True,
                      'log_scale': False,
                      'postfix': '',
                      'color': 'white'}),
    ('new_tests', {'title': 'Testy nowe',
                   'category': 'data',
                   'subcategory': 'new',
                   'round': 0,
                   'disable_pl': False,
                   'disable_world': False,
                   'disable_cities': False,
                   'log_scale': False,
                   'postfix': '',
                   'color': 'yellow'}),
    ('total_tests', {'title': 'Testy razem',
                     'category': 'data',
                     'subcategory': 'total',
                     'round': 0,
                     'disable_pl': False,
                     'disable_world': False,
                     'disable_cities': False,
                     'log_scale': False,
                     'postfix': '',
                     'color': 'orange'}),
    ('total_quarantine', {'title': 'Kwarantanna razem',
                          'category': 'data',
                          'subcategory': 'total',
                          'round': 2,
                          'disable_pl': False,
                          'disable_world': False,
                          'disable_cities': False,
                          'log_scale': False,
                          'postfix': '',
                          'color': 'red'}),
    ('total_nadzor', {'title': 'Nadzór epidemiczny razem',
                      'category': 'data',
                      'subcategory': 'total',
                      'round': 2,
                      'disable_pl': False,
                      'disable_world': True,
                      'disable_cities': True,
                      'log_scale': False,
                      'postfix': ' %',
                      'color': 'red'}),
    ('new_tests_poz', {'title': 'Nowe zlecenia POZ',
                       'category': 'data',
                       'subcategory': 'new',
                       'round': 0,
                       'disable_pl': False,
                       'disable_world': True,
                       'disable_cities': False,
                       'log_scale': False,
                       'postfix': '',
                       'color': 'red'}),
    ('total_vaccinations', {'title': 'Szczepienia razem',
                            'category': 'data',
                            'subcategory': 'total',
                            'round': 0,
                            'disable_pl': False,
                            'disable_world': False,
                            'disable_cities': True,
                            'log_scale': True,
                            'postfix': '',
                            'color': 'orange'}),
    ('people_vaccinated', {'title': 'Osób zaszczepionych razem',
                           'category': 'data',
                           'subcategory': 'total',
                           'round': 0,
                           'disable_pl': False,
                           'disable_world': False,
                           'disable_cities': True,
                           'log_scale': True,
                           'postfix': '',
                           'color': 'orange'}),
    ('people_fully_vaccinated', {'title': 'Osób w pełni zaszczepionych razem',
                                 'category': 'data',
                                 'subcategory': 'total',
                                 'round': 0,
                                 'disable_pl': False,
                                 'disable_world': False,
                                 'disable_cities': True,
                                 'log_scale': True,
                                 'postfix': '',
                                 'color': 'orange'}),
    ('new_vaccinations', {'title': 'Szczepienia nowe',
                          'category': 'data',
                          'subcategory': 'total',
                          'round': 0,
                          'disable_pl': False,
                          'disable_world': False,
                          'disable_cities': True,
                          'log_scale': True,
                          'postfix': '',
                          'color': 'orange'}),
    ('mort', {'title': 'Nowe zgony (Eurostat)',
              'category': 'data',
              'subcategory': 'total',
              'round': 0,
              'disable_pl': False,
              'disable_world': False,
              'disable_cities': True,
              'log_scale': True,
              'postfix': '',
              'color': 'orange'}),
    ('new_excess', {'title': 'Nowe zgony nadmiarowe',
                    'category': 'data',
                    'subcategory': 'new',
                    'round': 0,
                    'disable_pl': False,
                    'disable_world': False,
                    'disable_cities': True,
                    'log_scale': True,
                    'postfix': '',
                    'color': 'orange'}),
    ('total_excess', {'title': 'zgony nadmiarowe razem',
                      'category': 'data',
                      'subcategory': 'total',
                      'round': 0,
                      'disable_pl': False,
                      'disable_world': False,
                      'disable_cities': True,
                      'log_scale': True,
                      'postfix': '',
                      'color': 'orange'}),
    ('double_days', {'title': 'Okres podwojenia',
                     'category': 'calculated',
                     'subcategory': 'calculated',
                     'round': 0,
                     'disable_pl': False,
                     'disable_world': False,
                     'disable_cities': False,
                     'log_scale': False,
                     'postfix': '',
                     'color': 'white'}),

    ('reproduction_rate', {'title': 'R(t)',
                           'category': 'calculated',
                           'subcategory': 'calculated',
                           'round': 2,
                           'disable_pl': False,
                           'disable_world': False,
                           'disable_cities': True,
                           'log_scale': False,
                           'postfix': '',
                           'color': 'white'}),
    ('positive_rate', {'title': 'Wykrywalność (%)',
                       'category': 'calculated',
                       'subcategory': 'calculated',
                       'round': 2,
                       'disable_pl': False,
                       'disable_world': False,
                       'disable_cities': False,
                       'log_scale': True,
                       'postfix': '',
                       'color': 'white'}),
    ('zapadalnosc', {'title': 'Zapadalność',
                     'category': 'calculated',
                     'subcategory': 'calculated',
                     'round': 1,
                     'disable_pl': False,
                     'disable_world': False,
                     'disable_cities': False,
                     'log_scale': False,
                     'postfix': '',
                     'color': 'lightbrown'}),
    ('umieralnosc', {'title': 'Umieralność',
                     'category': 'calculated',
                     'subcategory': 'calculated',
                     'round': 1,
                     'disable_pl': False,
                     'disable_world': False,
                     'disable_cities': False,
                     'log_scale': False,
                     'postfix': '',
                     'color': 'purple'}),
    ('smiertelnosc', {'title': 'CFR D/I [%]',
                      'category': 'calculated',
                      'subcategory': 'calculated',
                      'round': 3,
                      'disable_pl': False,
                      'disable_world': False,
                      'disable_cities': False,
                      'log_scale': False,
                      'postfix': ' %',
                      'color': 'red'}),
    ('dynamikaD', {'title': 'Dynamika D',
                   'category': 'calculated',
                   'subcategory': 'calculated',
                   'round': 3,
                   'disable_pl': False,
                   'disable_world': False,
                   'disable_cities': False,
                   'log_scale': False,
                   'postfix': ' %',
                   'color': 'red'}),
    ('wyzdrawialnosc', {'title': 'Wyzdrawialność (%)',
                        'category': 'calculated',
                        'subcategory': 'calculated',
                        'round': 2,
                        'disable_pl': False,
                        'disable_world': False,
                        'disable_cities': True,
                        'log_scale': False,
                        'postfix': ' %',
                        'color': 'red'}),
    ('dynamikaI', {'title': 'Dynamika I',
                   'category': 'calculated',
                   'subcategory': 'calculated',
                   'round': 2,
                   'disable_pl': False,
                   'disable_world': False,
                   'disable_cities': False,
                   'log_scale': True,
                   'postfix': '',
                   'color': 'red'}),
    ('population', {'title': 'Populacja',
                    'category': 'calculated',
                    'subcategory': 'calculated',
                    'round': 2,
                    'disable_pl': False,
                    'disable_world': False,
                    'disable_cities': False,
                    'log_scale': False,
                    'postfix': '',
                    'color': 'green'}),
    ('new_reinf', {'title': 'Reinfekcje nowe',
                   'category': 'data',
                   'subcategory': 'new',
                   'round': 2,
                   'disable_pl': False,
                   'disable_world': True,
                   'disable_cities': False,
                   'log_scale': False,
                   'postfix': '',
                   'color': 'red'}),
])

trace_names_list = {'data': [], 'calculated': []}
trace_disable_pl = {'data': [], 'calculated': []}
trace_disable_world = {'data': [], 'calculated': []}
trace_disable_cities = {'data': [], 'calculated': []}
for key in trace_props.keys():
    cat = trace_props[key]['category']
    trace_names_list[cat].append([key, trace_props[key]['title']])
    trace_disable_pl[cat].append(trace_props[key]['disable_pl'])
    trace_disable_world[cat].append(trace_props[key]['disable_world'])
    trace_disable_cities[cat].append(trace_props[key]['disable_cities'])

mapbox_styles = {
    'Przezroczysty - bez etykiet': 'mapbox://styles/docent-ws/ckrb2gn0v058d17pnhcw9jm0a',
    'Szary - bez etykiet': 'mapbox://styles/docent-ws/ckdebv96b2jft1in2agw5mwq2',
    'Szary - z etykietami': 'mapbox://styles/docent-ws/ckdebs1jt4xaw1inxgcp18zdt',
    'Grafitowy - bez etykiet': 'mapbox://styles/docent-ws/ckdhpoqxw02o91ilcmgtybwpq',
    'Czarny - bez etykiet': 'mapbox://styles/docent-ws/cke4s0xyf1czs19miiyluuss9'
}

input_controls = {
    # main
    'scope_id': {'id': '{"index":"scope_id","type":"params"}.value', 'type': 'param'},
    'subscope_id': {'id': '{"index":"subscope_id","type":"params"}.value', 'type': 'scope_dep'},
    'locations_id': {'id': '{"index":"locations_id","type":"params"}.value', 'type': 'scope_dep'},
    'options_yesno_id': {'id': '{"index":"options_yesno_id","type":"params"}.value', 'type': 'param'},
    'options_graph_format_id': {'id': '{"index":"options_graph_format_id","type":"params"}.value', 'type': 'param'},
    'chart_type_data_id': {'id': '{"index":"chart_type_data_id","type":"params"}.value', 'type': 'scope_dep'},
    'chart_type_calculated_id': {'id': '{"index":"chart_type_calculated_id","type":"params"}.value',
                                 'type': 'scope_dep'},
    'data_modifier_id': {'id': '{"index":"data_modifier_id","type":"params"}.value', 'type': 'scope_dep'},
    'dzielna_id': {'id': '{"index":"dzielna_id","type":"params"}.value', 'type': 'param'},
    'font_size_xy_id': {'id': '{"index":"font_size_xy_id","type":"params"}.value', 'type': 'param'},
    'font_size_title_id': {'id': '{"index":"font_size_title_id","type":"params"}.value', 'type': 'param'},
    'font_size_anno_id': {'id': '{"index":"font_size_anno_id","type":"params"}.value', 'type': 'param'},
    'font_size_legend_id': {'id': '{"index":"font_size_legend_id","type":"params"}.value', 'type': 'param'},
    'legend_place_id': {'id': '{"index":"legend_place_id","type":"params"}.value', 'type': 'param'},
    'line_dash_id': {'id': '{"index":"line_dash_id","type":"params"}.value', 'type': 'param'},
    'color_order_id': {'id': '{"index":"color_order_id","type":"params"}.value', 'type': 'param'},
    'linedraw_id': {'id': '{"index":"linedraw_id","type":"params"}.value', 'type': 'param'},
    'titlexpos_id': {'id': '{"index":"titlexpos_id","type":"params"}.value', 'type': 'param'},
    'titleypos_id': {'id': '{"index":"titleypos_id","type":"params"}.value', 'type': 'param'},
    'copyrightxpos_id': {'id': '{"index":"copyrightxpos_id","type":"params"}.value', 'type': 'param'},
    'copyrightypos_id': {'id': '{"index":"copyrightypos_id","type":"params"}.value', 'type': 'param'},
    'annoxpos_id': {'id': '{"index":"annoxpos_id","type":"params"}.value', 'type': 'param'},
    'annoypos_id': {'id': '{"index":"annoypos_id","type":"params"}.value', 'type': 'param'},
    'margint_id': {'id': '{"index":"margint_id","type":"params"}.value', 'type': 'param'},
    'marginb_id': {'id': '{"index":"marginb_id","type":"params"}.value', 'type': 'param'},
    'marginl_id': {'id': '{"index":"marginl_id","type":"params"}.value', 'type': 'param'},
    'marginr_id': {'id': '{"index":"marginr_id","type":"params"}.value', 'type': 'param'},
    'duration_d_id': {'id': '{"index":"duration_d_id","type":"params"}.value', 'type': 'param'},
    'duration_r_id': {'id': '{"index":"duration_r_id","type":"params"}.value', 'type': 'param'},
    'win_type_id': {'id': '{"index":"win_type_id","type":"params"}.value', 'type': 'param'},
    'rounding_id': {'id': '{"index":"rounding_id","type":"params"}.value', 'type': 'param'},
    'template_id': {'id': '{"index":"template_id","type":"params"}.value', 'type': 'param'},
    # timeline
    'radio_class_id': {'id': '{"index":"radio_class_id","type":"params"}.value', 'type': 'param'},
    'radio_flow_id': {'id': '{"index":"radio_flow_id","type":"params"}.value', 'type': 'param'},
    'average_days_id': {'id': '{"index":"average_days_id","type":"params"}.value', 'type': 'param'},
    'smooth_id': {'id': '{"index":"smooth_id","type":"params"}.value', 'type': 'param'},
    'smooth_method_id': {'id': '{"index":"smooth_method_id","type":"params"}.value', 'type': 'param'},
    'radio_type_id': {'id': '{"index":"radio_type_id","type":"params"}.value', 'type': 'param'},
    'radio_scale_id': {'id': '{"index":"radio_scale_id","type":"params"}.value', 'type': 'param'},
    'annotations_id': {'id': '{"index":"annotations_id","type":"params"}.value', 'type': 'param'},
    'anno_form_id': {'id': '{"index":"anno_form_id","type":"params"}.value', 'type': 'param'},
    'arrangement_id': {'id': '{"index":"arrangement_id","type":"params"}.value', 'type': 'param'},
    'plot_height_id': {'id': '{"index":"plot_height_id","type":"params"}.value', 'type': 'param'},
    'total_min_id': {'id': '{"index":"total_min_id","type":"params"}.value', 'type': 'param'},
    'from_date_id': {'id': '{"index":"from_date_id","type":"params"}.date', 'type': 'scope_dep'},
    'to_date_id': {'id': '{"index":"to_date_id","type":"params"}.date', 'type': 'scope_dep'},
    'timeline_opt_id': {'id': '{"index":"timeline_opt_id","type":"params"}.value', 'type': 'param'},
    'timeline_view_id': {'id': '{"index":"timeline_view_id","type":"params"}.value', 'type': 'param'},
    'timeline_highlight_id': {'id': '{"index":"timeline_highlight_id","type":"params"}.value', 'type': 'param'},
    'linewidth_basic_id': {'id': '{"index":"linewidth_basic_id","type":"params"}.value', 'type': 'param'},
    'linewidth_thin_id': {'id': '{"index":"linewidth_thin_id","type":"params"}.value', 'type': 'param'},
    'linewidth_thick_id': {'id': '{"index":"linewidth_thick_id","type":"params"}.value', 'type': 'param'},
    # core
    'core_view_id': {'id': '{"index":"core_view_id","type":"params"}.value', 'type': 'param'},
    'core_opt_id': {'id': '{"index":"core_opt_id","type":"params"}.value', 'type': 'param'},
    'core_agg_x_id': {'id': '{"index":"core_agg_x_id","type":"params"}.value', 'type': 'param'},
    'core_agg_y_id': {'id': '{"index":"core_agg_y_id","type":"params"}.value', 'type': 'param'},
    'core_highlight_id': {'id': '{"index":"core_highlight_id","type":"params"}.value', 'type': 'param'},
    'core_date_id': {'id': '{"index":"core_date_id","type":"params"}.date', 'type': 'param'},
    # rankings
    'rankings_highlight_id': {'id': '{"index":"rankings_highlight_id","type":"params"}.value', 'type': 'param'},
    'max_cut_id': {'id': '{"index":"max_cut_id","type":"params"}.value', 'type': 'param'},
    'bar_gap_id': {'id': '{"index":"bar_gap_id","type":"params"}.value', 'type': 'param'},
    'bar_fill_id': {'id': '{"index":"bar_fill_id","type":"params"}.value', 'type': 'param'},
    'bar_frame_id': {'id': '{"index":"bar_frame_id","type":"params"}.value', 'type': 'param'},
    'bar_mode_id': {'id': '{"index":"bar_mode_id","type":"params"}.value', 'type': 'param'},
    'options_rank_id': {'id': '{"index":"options_rank_id","type":"params"}.value', 'type': 'param'},
    # dynamics
    'dynamics_scaling_id': {'id': '{"index":"dynamics_scaling_id","type":"params"}.value', 'type': 'param'},
    'columns_id': {'id': '{"index":"columns_id","type":"params"}.value', 'type': 'param'},
    'dynamics_chart_height_id': {'id': '{"index":"dynamics_chart_height_id","type":"params"}.value', 'type': 'param'},
    # map
    'map_opt_id': {'id': '{"index":"map_opt_id","type":"params"}.value', 'type': 'param'},
    'map_date_id': {'id': '{"index":"map_date_id","type":"params"}.date', 'type': 'param'},
    'map_color_scale_id': {'id': '{"index":"map_color_scale_id","type":"params"}.value', 'type': 'param'},
    'map_mapbox_id': {'id': '{"index":"map_mapbox_id","type":"params"}.value', 'type': 'param'},
    'map_opacity_id': {'id': '{"index":"map_opacity_id","type":"params"}.value', 'type': 'param'},
    'map_options_id': {'id': '{"index":"map_options_id","type":"params"}.value', 'type': 'param'},
    'map_palette_id': {'id': '{"index":"map_palette_id","type":"params"}.value', 'type': 'param'},
    'map_cut_id': {'id': '{"index":"map_cut_id","type":"params"}.value', 'type': 'param'},
    'map_height_id': {'id': '{"index":"map_height_id","type":"params"}.value', 'type': 'param'},
    # table
    'table_rows_id': {'id': '{"index":"table_rows_id","type":"params"}.value', 'type': 'param'},
    'table_mean_id': {'id': '{"index":"table_mean_id","type":"params"}.value', 'type': 'param'},
    # 'table_to_date_id': {'id': '{"index":"table_to_date_id","type":"params"}.date', 'type': 'scope_dep'},
}

defaults = dict(
    scope='poland',
    subscope='world',
    locations=[],
    chart_type_data=[],
    chart_type=['new_cases'],
    chart_type_calculated=[],
    data_modifier=1,
    dzielna='<brak>',
    from_date=str(dt.today())[:10],
    to_date=str(dt.today())[:10],
    timeline_highlight='Brak',
    date_picker=str(dt.today() - timedelta(days=1))[:10],
    core_highlight='Brak',
    core_agg_x='mean',
    core_agg_y='mean',
    core_opt=[],
    core_view=['regresja', 'errors'],
    core_date=str(dt.today())[:10],
    rankings_highlight='Brak',
    max_cut='10',
    average_days=7,
    smooth=11,
    smooth_method='sawicki',
    total_min=100,
    options_yesno=[],
    duration_d=14,
    duration_r=21,
    win_type='równe wagi',
    rounding='2',
    font_size_xy=14,
    font_size_title=22,
    font_size_anno=14,
    font_size_legend=14,
    linewidth_basic=2,
    linewidth_thin=0.5,
    linewidth_thick=3.5,
    margint=10,
    marginb=25,
    marginl=25,
    marginr=0,
    titlexpos=0.5,
    titleypos=0.99,
    copyrightxpos=0.,
    copyrightypos=0.,
    annoxpos=0.05,
    annoypos=0.8,
    options_graph_format='jpeg',
    options_rank='',
    legend_place='lewo',
    line_dash='solid',
    color_order='Sunsetdark',
    linedraw='lines',
    annotations='max',
    anno_form='name',
    color_scale='Inferno',
    bar_gap=0.2,
    bar_fill='',
    bar_frame=0.,
    bar_mode='group',
    arrangement='smart',
    plot_height=1,
    radio_scale='linear',
    map_color_scale='Reds',
    map_mapbox='Przezroczysty - bez etykiet',
    map_opacity=0.8,
    map_height=750,
    map_options=['drawborder', 'annotations', '1c'],
    radio_type='scatter',
    timeline_opt=[],
    timeline_view=['legenda'],
    radio_class='types',
    radio_flow='real',
    dynamics_scaling='Porównanie kształtów',
    columns=5,
    dynamics_chart_height=300,
    map_date=str(dt.today() - timedelta(days=1))[:10],
    map_opt='linear',
    map_palette='dyskretny',
    map_cut='kwantyle',

    color_0='rgb(0,43,54)',  # kolor podkłądu
    color_1='rgb(0,43,54)',  # kolor tła wykresu
    color_2='red',  # kolor osi współrzędnych
    color_3='yellow',  # kolor etykiet osi
    color_4='orange',  # kolor adnotacji
    color_5='rgba(0,0,0,0)',  # kolor tła adnotacji
    color_6='rgba(0,0,0,0)',  # kolor tła legendy
    color_7='white',  # kolor tytułu
    color_8='rgb(102, 101, 101)',  # kolor słupków (dzienny)
    color_9='red',  # kolor wyróżnienia (dzienny)
    color_10='white',  # kolor legendy
    color_11='rgb(0,43,54)',  # interfejs (tło ogólne)
    color_12='rgb(0,43,54)',  # interfejs (tło listy lokalizacji)
    color_13='red',  # dodatkowy 1
    color_14='yellow',  # dodatkowy 2
    table_rows=16,
    table_mean='daily',
    template='default'
)
default_template = go.layout.Template(
    layout=dict(
        images=image_logo_timeline,
        title=dict(
            font=dict(
                size=defaults['font_size_title'],
                color=defaults['color_7'],
            ),
            x=defaults['titlexpos'],
            y=defaults['titleypos'],
            xanchor='center',
            yanchor='top',
            pad=dict(t=20, b=50, l=0, r=0),
        ),
        xaxis=dict(
            linecolor=defaults['color_2'],
            linewidth=1,
            showline=True,
            hoverformat="%d-%m-%Y",
            ticks='outside',
            showticklabels=True,
            tickfont=dict(
                size=defaults['font_size_xy'],
                color=defaults['color_3']
            ),
            title=dict(
                font=dict(
                    color=defaults['color_2'],
                    size=12,
                )
            )
        ),
        yaxis=dict(
            linecolor=defaults['color_2'],
            linewidth=1,
            showline=True,
            separatethousands=False,
            tickfont=dict(
                size=defaults['font_size_xy'],
                color=defaults['color_3']
            ),
            title=dict(
                font=dict(
                    color=defaults['color_2'],
                    size=12,
                )
            )
        ),
        colorway=color_scales.get(defaults['color_order']),
        legend=dict(
            bgcolor=defaults['color_6'],
            font=dict(
                size=defaults['font_size_legend'],
                color=defaults['color_10'],
            ),
        ),
        paper_bgcolor=defaults['color_1'],
        plot_bgcolor=defaults['color_1'],
    )
)

user_templates = {
    'default': dict(

        color_1='rgb(0,43,54)',  # kolor tła wykresu
        color_2='red',  # kolor osi współrzędnych
        color_3='yellow',  # kolor etykiet osi
        color_4='orange',  # kolor adnotacji
        color_5='rgba(0,0,0,0)',  # kolor tła adnotacji
        color_6='rgba(0,0,0,0)',  # kolor tła legendy
        color_7='white',  # kolor tytułu
        color_8='yellow',  # kolor słupków (dzienny)
        color_9='red',  # kolor wyróżnienia (dzienny)
        color_10='white',  # kolor legendy
        color_11='rgb(0,43,54)',  # interfejs (tło ogólne)

        # interfejs (tło listy lokalizacji)
    ),
    'beige': dict(
        color_0=' rgb(170,121,60)',
        color_1=' rgb(170,121,60)',
        color_2='red',
        color_3='rgb(16,16,16)',
        color_4='white',
        color_5='rgb(214, 249, 207)',
        color_6='rgba(0,0,0,0)',
        color_7='rgb(59, 59, 59)',
        color_10='rgb(0,0,0)',
        # color_order='RdBu',
        # font_size_legend=16
    ),
    'blue': dict(
        color_0='rgb(0, 51, 102)',
        color_1='rgb(0, 51, 102)',
        color_2='red',
        color_3='white',
        color_4='white',
        color_5='rgb(214, 249, 207)',
        color_6='rgba(0,0,0,0)',
        color_7='white',
        color_10='white',
        color_order='RdBu',
        # font_size_legend=16
    ),
    'darkblue': dict(
        color_0='rgb(0, 51, 102)',
        color_1='rgb(0, 51, 102)',
        color_2='red',
        color_3='white',
        color_4='white',
        color_5='rgb(214, 249, 207)',
        color_6='rgba(0,0,0,0)',
        color_7='white',
        color_10='white',
        # color_order='PuBu',
        # font_size_legend=16
    ),
    'green': dict(
        color_0='#666600',
        color_1='#666600',
        color_2='yellow',
        color_3='white',
        color_4='rgb(16,16,16)',
        color_5='rgb(214, 249, 207)',
        color_6='rgba(0,0,0,0)',
        color_7='white',
        color_10='white',
        # color_order='Greens_r',
        # font_size_legend=16
    ),
    'brick': dict(
        color_0='#666600',
        color_1='rgb(57, 45, 37) ',
        color_2='yellow',
        color_3='white',
        color_4='white',
        color_5='rgb(214, 249, 207)',
        color_6='rgba(0,0,0,0)',
        color_7='white',
        color_10='white',
        # color_order='Oranges_r',
        # font_size_legend=16
    ),
    'white': dict(
        color_0=' rgb(254,254,253)',
        color_1=' rgb(254,254,253)',
        color_2='red',
        color_3='rgb(120,14,40)',
        color_4='black',
        color_5='rgba(0,0,0,0)',
        color_6='rgba(0,0,0,0)',
        color_7='rgb(59, 59, 59)',
        color_8='green',
        color_9='red',
        color_10='black',
        color_order='Default',
        # font_size_legend=18,
        # font_size_anno=16
    ),
    'transparent': dict(
        color_0=' rgb(254,254,253)',
        color_1=' rgba(0,0,0,0)',
        color_2='red',
        color_3='rgb(120,14,40)',
        color_4='black',
        color_5='rgba(0,0,0,0)',
        color_6='rgba(0,0,0,0)',
        color_7='rgb(59, 59, 59)',
        color_8='green',
        color_9='red',
        color_10='black',
        # color_order='Rainbow_r',
        # font_size_legend=18,
        # font_size_anno=16
    ),
}


# defaults = dict(list(defaults.items()) + list(user_templates['white'].items()))


def local_session_or_die(key, default=True):
    try:
        session[key]
    except:
        return default
    else:
        ret_val = session.get(key)
        if ret_val is None:
            return default
        return ret_val


def get_defa():
    template = local_session_or_die('template', 'default')
    # print('template =', template)
    if template == 'default':
        ret_val = defaults
    else:
        ret_val = dict(list(defaults.items()) + list(user_templates[template].items()))
    return ret_val


settings_props = {
    'locations': dict(save=False, ),
    'scope': dict(save=False, ),
    'subscope': dict(save=False, ),
    'chart_type': dict(save=False, ),
    'chart_type_data': dict(save=False, ),
    'chart_type_calculated': dict(save=False, ),
    'data_modifier': dict(save=False, ),
    'dzielna': dict(save=False, ),
    'from_date': dict(save=False, ),
    'to_date': dict(save=False, ),
    'timeline_highlight': dict(save=True, ),
    'date_picker': dict(save=True, ),
    'rankings_highlight': dict(save=True, ),
    'core_highlight': dict(save=True, ),
    'core_opt': dict(save=True, ),
    'core_agg_x': dict(save=True, ),
    'core_agg_y': dict(save=True, ),
    'core_view': dict(save=True, ),
    'core_date': dict(save=True, ),
    'max_cut': dict(save=True, ),
    'options_yesno': dict(save=True, ),
    'average_days': dict(save=True, ),
    'smooth': dict(save=True, ),
    'smooth_method': dict(save=True, ),
    'options_rank': dict(save=True, ),
    'total_min': dict(save=True, ),
    'duration_d': dict(save=True, ),
    'duration_r': dict(save=True, ),
    'win_type': dict(save=True, ),
    'rounding': dict(save=True, ),
    'font_size_xy': dict(save=True, ),
    'font_size_title': dict(save=True, ),
    'font_size_anno': dict(save=True, ),
    'font_size_legend': dict(save=True, ),
    'linewidth_basic': dict(save=True, ),
    'linewidth_thin': dict(save=True, ),
    'linewidth_thick': dict(save=True, ),
    'margint': dict(save=True, ),
    'marginb': dict(save=True, ),
    'marginl': dict(save=True, ),
    'marginr': dict(save=True, ),
    'titlexpos': dict(save=True, ),
    'titleypos': dict(save=True, ),
    'copyrightxpos': dict(save=True, ),
    'copyrightypos': dict(save=True, ),
    'annoxpos': dict(save=True, ),
    'annoypos': dict(save=True, ),
    'options_graph_format': dict(save=True, ),
    'legend_place': dict(save=True, ),
    'line_dash': dict(save=True, ),
    'color_order': dict(save=True, ),
    'linedraw': dict(save=True, ),
    'annotations': dict(save=True, ),
    'anno_form': dict(save=True, ),
    'color_scale': dict(save=True, ),
    'bar_gap': dict(save=True, ),
    'bar_fill': dict(save=True, ),
    'bar_frame': dict(save=True, ),
    'bar_mode': dict(save=True, ),
    'arrangement': dict(save=True, ),
    'plot_height': dict(save=True, ),
    'radio_scale': dict(save=True, ),
    'map_color_scale': dict(save=True, ),
    'map_mapbox': dict(save=True, ),
    'map_opacity': dict(save=True, ),
    'map_height': dict(save=True, ),
    'map_options': dict(save=True, ),
    'map_palette': dict(save=True, ),
    'map_cut': dict(save=True, ),
    'radio_type': dict(save=True, ),
    'timeline_opt': dict(save=True, ),
    'timeline_view': dict(save=True, ),
    'radio_class': dict(save=True, ),
    'radio_flow': dict(save=True, ),
    'dynamics_scaling': dict(save=True, ),
    'columns': dict(save=True, ),
    'dynamics_chart_height': dict(save=True, ),
    'map_date': dict(save=True, ),
    'map_opt': dict(save=True, ),
    'table_rows': dict(save=True, ),
    'table_mean': dict(save=True, ),
    # 'table_from_date': dict(save=True,),
    # 'table_to_date': dict(save=True,),
    'color_1': dict(save=True, ),
    'color_2': dict(save=True, ),
    'color_3': dict(save=True, ),
    'color_4': dict(save=True, ),
    'color_5': dict(save=True, ),
    'color_6': dict(save=True, ),
    'color_7': dict(save=True, ),
    'color_8': dict(save=True, ),
    'color_9': dict(save=True, ),
    'color_10': dict(save=True, ),
    'color_11': dict(save=True, ),
    'color_12': dict(save=True, ),
    'template': dict(save=True, ),
}
settings_names = settings_props.keys()


# kolory ANSI do terminala
# użyj ENDC aby wrócić do koloru standardowego


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


data_files = {
    # Dane API z bazy MZ
    # Struktura wiekowa przypadków śmiertelnych
    # Bartek
    'mz_api_age_d':
        {
            'name': 'Dane API MZ deaths age',
            'type': 'api',
            'priority': 1,
            'url': "MZ API",
            'src_fn': 'data/sources/mz_api_age_d.csv',
            'data_fn': 'data/last/last_mz_api_age_d.csv'
        },
    'mz_api_age_v':
        {
            'name': 'Dane API MZ szczepienia',
            'type': 'api',
            'priority': 1,
            'url': "MZ API",
            'src_fn': 'data/sources/mz_api_age_vacc.csv',
            'data_fn': 'data/last/last_mz_api_age_vacc.csv'
        },
    'mz_api_age_v3':
        {
            'name': 'Dane API MZ szczepienia podsumowanie',
            'type': 'api',
            'priority': 1,
            'url': "MZ API",
            'src_fn': 'data/sources/mz_api_age_vacc3.csv',
            'data_fn': 'data/last/last_mz_api_age_vacc3.csv'
        },
    'mz_api_vacc_powiaty':
        {
            'name': 'Dane API MZ powiaty szczepienia',
            'type': 'api',
            'priority': 3,
            'url': "MZ_API",
            'src_fn': 'data/sources/mz_api_vacc_powiaty.csv',
            'data_fn': 'data/last/last_mz_api_vacc_powiaty.csv'
        },
    'mz_api_vacc_gminy':
        {
            'name': 'Dane API MZ gminy szczepienia',
            'type': 'json',
            'priority': 3,
            'url': "https://www.gov.pl/api/data/registers/search?pageId=20306374",
            'src_fn': 'data/sources/mz_api_vacc_gminy.csv',
            'data_fn': 'data/last/last_mz_api_vacc_gminy.csv'
        },
    'mz_psz':
        {
            'name': 'Punkty szczepień',
            'type': 'json',
            'priority': 1,
            'url': "https://www.gov.pl/api/data/covid-vaccination-point",
            'src_fn': 'data/sources/mz_psz.csv',
            'data_fn': 'data/last/last_mz_psz.csv'
        },
    'mz_api_vacc_wojew':
        {
            'name': 'Dane API MZ województwa szczepienia',
            'type': 'api',
            'priority': 1,
            'url': "MZ_API",
            'src_fn': 'data/sources/mz_api_vacc_wojew.csv',
            'data_fn': 'data/last/last_mz_api_vacc_wojew.csv'
        },
    # dane podstawowe 'cities'
    # dane do 23.11.2020
    # zawieszone pobieranie
    'cities_cases':
        {
            'name': 'Miasta i powiaty (przypadki)',
            'priority': 3,
            'type': 'csv',
            'url': 'https://docs.google.com/spreadsheets/d/1Tv6jKMUYdK6ws6SxxAsHVxZbglZfisC8x_HZ1jacmBM/gviz/tq?tqx=out:csv&sheet=Suma%20przypadk%C3%B3w',
            'src_fn': 'data/sources/src_cities_cases.csv',
            'data_fn': 'data/last/last_cities_cases.csv'
        },
    # dane podstawowe 'cities'
    # dane do 23.11.2020
    # zawieszone pobieranie
    'cities_deaths':
        {
            'name': 'Miasta i powiaty (zgony)',
            'priority': 3,
            'type': 'csv',
            'url': 'https://docs.google.com/spreadsheets/d/1Tv6jKMUYdK6ws6SxxAsHVxZbglZfisC8x_HZ1jacmBM/gviz/tq?tqx=out:csv&sheet=Suma%20zgon%C3%B3w',
            'src_fn': 'data/sources/src_cities_deaths.csv',
            'data_fn': ''
        },
    # dane pomocnicze 'poland'
    # dane do 23.11.2020
    # zawieszone pobieranie
    'poland_balance':
        {
            'name': 'Polska bilans',
            'priority': 3,
            'type': 'csv',
            'url': 'https://docs.google.com/spreadsheets/d/1ierEhD6gcq51HAm433knjnVwey4ZE5DCnu1bW7PRG3E/gviz/tq?tqx=out:csv&gid=1136959919',
            'src_fn': 'data/sources/src_balance_pl.csv',
            'data_fn': 'data/last/last_balance_pl.csv'
        },
    # dane podatawowe 'poland'
    # dane do 23.11.2020
    # zawieszone pobieranie
    'poland_all':
        {
            'name': 'Województwa (całość)',
            'type': 'csv',
            'priority': 3,
            'url': "https://docs.google.com/spreadsheets/d/1ierEhD6gcq51HAm433knjnVwey4ZE5DCnu1bW7PRG3E/gviz/tq?tqx=out:csv&gid=1841152698",
            'src_fn': 'data/sources/src_poland.csv',
            'data_fn': 'data/last/last_poland.csv'
        },
    # dane pomocnicze 'poland'
    # dane pomocnicze 'world' dla Polski
    # dane do 5.07.2021
    # zawieszone pobieranie
    'poland_resources':
        {
            'name': 'Polska (zasoby)',
            'type': 'csv',
            'priority': 3,
            'url': "https://docs.google.com/spreadsheets/d/1ierEhD6gcq51HAm433knjnVwey4ZE5DCnu1bW7PRG3E/gviz/tq?tqx=out:csv&gid=570911918",
            'src_fn': 'data/sources/src_resources.csv',
            'data_fn': 'data/last/last_resources.csv'
        },
    # dane pomocnicze 'poland'
    # dane do 23.11.2020
    # zawieszone pobieranie
    'wojew_tests':
        {
            'name': 'Województwa (testy)',
            'type': 'csv',
            'priority': 3,
            'url': "https://docs.google.com/spreadsheets/d/1ierEhD6gcq51HAm433knjnVwey4ZE5DCnu1bW7PRG3E/gviz/tq?tqx=out:csv&gid=1234400689",
            'src_fn': 'data/sources/src_tests_wojew.csv',
            'data_fn': ''
        },
    # dane pomocnicze 'poland' dla Polski
    # arkusze Google Adama Gapińskiego
    'poland_rt':
        {
            'name': 'Polska R(t)',
            'type': 'csv',
            'priority': 1,
            'url': "https://docs.google.com/spreadsheets/d/1bfQ4ZtDW8Q6KNt400-uJd4fEaGjgFEDmpD76EKwr8Uo/gviz/tq?tqx=out:csv&gid=682915191",
            'src_fn': 'data/sources/src_poland_rt.csv',
            'data_fn': 'data/last/last_poland_rt.csv'
        },
    # dane pomocnicze 'poland' dla województw
    # arkusze Google Adama Gapińskiego
    'woj_rt':
        {
            'name': 'Województwa R(t)',
            'type': 'csv',
            'priority': 1,
            'url': "https://docs.google.com/spreadsheets/d/1bfQ4ZtDW8Q6KNt400-uJd4fEaGjgFEDmpD76EKwr8Uo/gviz/tq?tqx=out:csv&gid=1590578101",
            'src_fn': 'data/sources/src_rt_woj.csv',
            'data_fn': 'data/last/last_rt_woj.csv'
        },
    # dane podstawowe 'world'
    'world':
        {
            'name': 'Świat OWID',
            'type': 'csv',
            'priority': 1,
            'url': "https://covid.ourworldindata.org/data/owid-covid-data.csv",
            'src_fn': 'data/sources/src_world.csv',
            'data_fn': 'data/last/last_world.csv'
        },
    # Dane bieżące z bazy BASIW
    # Struktura wiekowa zgonów ze statusem szczepienia
    # aktualizowane 1 x tyg. na stronie https://basiw.mz.gov.pl/index.html#/visualization?id=3653
    'basiw_d':
        {
            'name': 'Zgony BASIW + szczepienie',
            'type': 'manual',
            'priority': 1,
            'url': "",
            'src_fn': 'data/sources/basiw_d.csv',
            'data_fn': 'data/last/last_basiw_d.csv',
            'data_raw_fn': 'data/last/last_basiw_d_raw.csv'
        },
    # Dane bieżące z bazy BASIW
    # Struktura wiekowa infekcji ze statusem szczepienia
    # aktualizowane 1 x tyg. na stronie https://basiw.mz.gov.pl/index.html#/visualization?id=3653
    'basiw_i':
        {
            'name': 'Infekcje BASIW + szczepienie',
            'type': 'manual',
            'priority': 1,
            'url': "",
            'src_fn': 'data/sources/basiw_i.csv',
            'data_fn': 'data/last/last_basiw_i.csv',
            'data_raw_fn': 'data/last/last_basiw_i_raw.csv'
        },
    # Dane o infekcji wśród zaszczepionych
    # aktualizowane 1 x tyg. na stronie https://basiw.mz.gov.pl/index.html#/visualization?id=3653
    'reinf':
        {
            'name': 'Infekcje zaszczepionych',
            'type': 'manual',
            'priority': 1,
            'url': "manual",
            'src_fn': 'data/sources/src_reinf.csv',
            'data_fn': 'data/last/last_reinf.csv',
        },
    # Dane ECDC szczepienia w Polsce w grupach wiekowych
    # wg rodzaju szczepionki, dostawy
    'ecdc_vacc':
        {
            'name': 'Szczepienia PL ECDC',
            'type': 'csv',
            'priority': 2,
            'url': "https://opendata.ecdc.europa.eu/covid19/vaccine_tracker/csv/data.csv",
            'src_fn': 'data/sources/ecdc_vac_pl.csv',
            'data_fn': 'data/last/last_ecdc_vac_pl.csv'
        },
    # Śmiertelność z bazy Eurostat
    'mortality_eurostat':
        {
            'name': 'Śmiertelność wg. Eurostat',
            'type': 'csv',
            'priority': 2,
            'url': "Eurostat",
            'src_fn': 'data/sources/src_mortality_eurostat.csv',
            'data_fn': 'data/last/last_mortality_eurostat.csv'
        },
    # Zgony wg. tygodni wg GUS w regionach/województwach
    # użyte tylko do GREG
    # wymaga obróbki ręcznej
    'mortality_gus_woj':
        {
            'name': 'Śmiertelność wg. GUS w województwach',
            'type': 'file',
            'priority': 3,
            'url': "https://stat.gov.pl//download/gfx/portalinformacyjny/pl/defaultaktualnosci/5468/39/2/1/zgony_wedlug_tygodni.zip",
            'src_fn': 'data/sources/Zgony według tygodni w Polsce_2020.xlsx',
            'data_fn': 'data/last/last_zgony_wg_tygodni_woj.csv'
        },
    # Zgony nadmiarowe OWID
    'excess_mortality':
        {
            'name': 'Zgony nadmiarowe OWID',
            'type': "csv",
            'priority': 1,
            'url': 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/excess_mortality/excess_mortality.csv',
            'src_fn': 'data/sources/excess_mortality.csv',
            'data_fn': 'data/last/last_excess_mortality.csv'
        },
}

table_mean = {
    'daily': "Dane dzienne",
    'daily7': "Dane dzienne uśrednione",
    'weekly_mean': "Średnie tygodniowe",
    'biweekly_mean': "Średnie dwutygodniowe",
    'monthly_mean': "Średnie miesięczne",
    'mean': "Średnia okresu",
    'sum': "Suma okresu",
    'weekly_sum': "Sumy tygodniowe",
    'biweekly_sum': "Sumy dwutygodniowe",
    'monthly_sum': "Sumy miesięczne",
    'daily_diff': "Różnice dzienne",
    'daily7_diff': "Różnice dzienne uśrednione",
    'dynamics': "Dynamika t/t",
}

activities = {'Sklepy i rozrywka': 'retail_and_recreation_percent_change_from_baseline',
              'Drogerie i apteki': 'grocery_and_pharmacy_percent_change_from_baseline',
              # 'Parki': 'parks_percent_change_from_baseline',
              'Komunikacja miejska': 'transit_stations_percent_change_from_baseline',
              'Miejsce pracy': 'workplaces_percent_change_from_baseline',
              'Miejsce zamieszkania': 'residential_percent_change_from_baseline'
              }

woj_translate_dict = {
    'Greater Poland Voivodeship': "Wielkopolskie",
    'Kuyavian-Pomeranian Voivodeship': "Kujawsko-Pomorskie",
    'Lesser Poland Voivodeship': "Małopolskie",
    'Łódź Voivodeship': "Łódzkie",
    'Lower Silesian Voivodeship': "Dolnośląskie",
    'Lublin Voivodeship': "Lubelskie",
    'Lubusz Voivodeship': "Lubuskie",
    'Masovian Voivodeship': "Mazowieckie",
    'Opole Voivodeship': "Opolskie",
    'Podkarpackie Voivodeship': "Podkarpackie",
    'Podlaskie Voivodeship': "Podlaskie",
    'Pomeranian Voivodeship': "Pomorskie",
    'Silesian Voivodeship': "Śląskie",
    'Świętokrzyskie Voivodeship': "Świętokrzyskie",
    'Warmian-Masurian Voivodeship': "Warmińsko-Mazurskie",
    'West Pomeranian Voivodeship': "Zachodniopomorskie"
}
stare_woj = ['bialskopodlaskie', 'białostockie', 'bielskie', 'bydgoskie', 'chełmskie', 'ciechanowskie',
             'częstochowskie', 'elbląskie', 'gdańskie', 'gorzowskie', 'jeleniogórskie', 'kaliskie', 'katowickie',
             'kieleckie', 'konińskie', 'koszalińskie', '(miejskie) krakowskie', 'krośnieńskie', 'legnickie',
             'leszczyńskie', 'lubelskie', 'łomżyńskie', '(miejskie) łódzkie', 'nowosądeckie', 'olsztyńskie',
             'opolskie', 'ostrołęckie', 'pilskie', 'piotrkowskie', 'płockie', 'poznańskie', 'przemyskie',
             'radomskie', 'rzeszowskie', 'siedleckie', 'sieradzkie', 'skierniewickie', 'słupskie', 'suwalskie',
             'szczecińskie', 'tarnobrzeskie', 'tarnowskie', 'toruńskie', 'wałbrzyskie', '(stołeczne) warszawskie',
             'włocławskie', 'wrocławskie', 'zamojskie', 'zielonogórskie']

nomargins = 'pr-0 pl-0 mr-0 ml-0 pt-0 pb-0 mt-0 mb-0 g-0'

dropdown_style = {'margin-left': '20px'}

dashes = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot", "solid", "dot", "dash", "longdash", "dashdot",
          "longdashdot"]

ages_5_pop = {
    0: ['b.d.', 99999999999],
    1: ['0-4', 1902236],
    2: ['5-9', 1910470],
    3: ['10-14', 2065628],
    4: ['15-19', 1794310],
    5: ['20-24', 1969685],
    6: ['25-29', 2402721],
    7: ['30-34', 2820162],
    8: ['35-39', 3225731],
    9: ['40-44', 3075130],
    10: ['45-49', 2693241],
    11: ['50-54', 2282038],
    12: ['55-59', 2323428],
    13: ['60-64', 2680248],
    14: ['65-69', 2505595],
    15: ['70-74', 1916928],
    16: ['75-79', 1013492],
    17: ['80-84', 865717],
    18: ['85-89', 538356],
    19: ['90-95', 222498],
    20: ['96-99', 50517],
    21: ['100+', 6882],
}
Poland_population = 38265013
age_bins_0 = [-1, 0, 12, 20, 40, 60, 70, 1000]
age_bins_10 = [-1, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000]
age_bins_5 = [-1, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65,
              70, 75, 80, 85, 90, 95, 100, 1000]
age_bins_ecdc = [-1, 0, 5, 10, 15, 18, 25, 50, 60, 70, 80, 1000]
age_bins_mz = [-1, 0, 19, 25, 35, 45, 55, 65, 75, 85, 95, 1000]
ages_mz = ['BD', '0-18', '19-24', '25-34', '35-44', '45-54', '55-64', '65-74',
           '75-84', '85-94', '95+']
ages_ecdc = {'Age0_4': 1, 'Age5_9': 2, 'Age10_14': 3, 'Age15_17': 4,
             'Age18_24': 5, 'Age25_49': 6, 'Age50_59': 7, 'Age60_69': 8,
             'Age70_79': 9, 'Age80+': 10}
rev_bin_mz = {'b.d.': 0, '0-18': 1, '19-24': 2, '25-34': 3, '35-44': 4, '45-54': 5,
              '55-64': 6, '65-74': 7, '75-84': 8, '85-94': 9, '95+': 10
              }
age_bins = {
    'bin5':
        {
            0: 'b.d.', 1: '0-4', 2: '5-9', 3: '10-14', 4: '15-19', 5: '20-24', 6: '25-29', 7: '30-34',
            8: '35-39', 9: '40-44', 10: '45-49', 11: '50-54', 12: '55-59', 13: '60-64', 14: '65-69',
            15: '70-74', 16: '75-79', 17: '80-84', 18: '85-89', 19: '90-94', 20: '95-99', 21: '100+'
        },
    'bin':
        {
            0: 'b.d.', 1: '0-11', 2: '12-19', 3: '20-39', 4: '40-59', 5: '60-69', 6: '70+'
        },
    'bin10':
        {
            0: 'b.d.', 1: '0-9', 2: '10-19', 3: '20-29', 4: '30-39', 5: '40-49', 6: '50-59',
            7: '60-69', 8: '70-79', 9: '80-89', 10: '90-99', 11: '100+'
        },
    'bin_ecdc':
        {
            0: 'b.d.', 1: '0-4', 2: '5-9', 3: '10-14', 4: '15-17', 5: '18-24', 6: '25-49', 7: '50-59',
            8: '60-69', 9: '70-79', 10: '80+'
        },
    'bin_mz':
        {
            0: 'b.d.', 1: '0-18', 2: '19-24', 3: '25-34', 4: '35-44', 5: '45-54',
            6: '55-64', 7: '65-74', 8: '75-94', 9: '85-94', 10: '95+'
        },
    'bin_eurostat':
        {
            0: 'b.d.', 1: '0-19', 2: '20-39', 3: '40-59', 4: '60-79', 5: '80+'
        }
}
ages_ecdc_order = {
    'Age0_4': 1,
    'Age5_9': 2,
    'Age10_14': 3,
    'Age15_17': 4,
    'Age18_24': 5,
    'Age25_49': 6,
    'Age50_59': 7,
    'Age60_69': 8,
    'Age70_79': 9,
    'Age80+': 10
}
