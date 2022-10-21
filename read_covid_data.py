import pandas as pd
import numpy as np
import urllib.request
import datetime
import time
# from datetime import datetime
import constants

from constants import wojewodzkie_list, eu_countries_list, arab_countries_list, subgroups, demoludy_list, \
    skandynawskie_list, powiat_translation_table, eurostat_nuts, wojew_cap, data_files, fields, \
    nuts_countries, terc_woj, eurostat_nuts_reverse, rev_bin_mz
from datetime import datetime as dt

from util import xlate_array, save_github


def correct_data(df):
    columns_needed = set(fields.keys())
    columns_exists = set(df.columns)
    if columns_exists - columns_needed:
        df.drop(columns=list(columns_exists - columns_needed), axis=1, inplace=True)
    if columns_needed - columns_exists:
        cols = list(columns_needed - columns_exists)
        for col in cols:
            df[col] = 0
    for field in fields:
        if df[field].dtype != fields[field]['dtype']:
            if not field in ['mort', 'new_excess']:
                df[field].fillna(0, inplace=True)
                df[field] = df[field].astype(fields[field]['dtype'])
    if df.empty:
        raise RuntimeError('Wykryto błędy strukturalne danych')
    return df


def load_data_cities():
    df = pd.read_csv(data_files['cities_cases']['src_fn'])
    df0 = df.copy()

    tab = []
    indexes = df.index[df['Unnamed: 0'].isnull()]
    start = 1
    for i in range(len(indexes)):
        dfx = df0.loc[start:indexes[i] - 1].copy()
        if len(dfx) > 0:
            if pd.isnull(dfx.iloc[0]['Nazwa']):
                dfx = dfx[1:]
            dfx.dropna(axis=0, how='all', inplace=True)
            wojew = dfx.iloc[0]['Nazwa'].lower()
            dfx = dfx[1:]
            dfx.drop(['Unnamed: 0'], axis=1, inplace=True)
            dfx.rename(columns={'Unnamed: 1': 'code'}, inplace=True)
            dfx.fillna(0, inplace=True)
            dfxx = dfx.melt(id_vars=['code', 'Nazwa'], var_name='date', value_name='total_cases')
            start = indexes[i] + 1
            for col in ['new_cases', 'new_deaths', 'total_deaths', 'new_recoveries', 'total_recoveries',
                        'total_active', 'new_tests', 'total_tests']:
                dfxx[col] = 0
            dfxx['wojew'] = wojew
            dfxx = dfxx[~dfxx['date'].str.contains('Unnamed', regex=False)]
            dfxx['date'] = ['2020-' + x[3:5] + '-' + x[:2] for x in dfxx['date']]
            dfxx.rename(columns={'Nazwa': 'location'}, inplace=True)
            tab.append(dfxx.iloc[1:])

    df_c = pd.concat([x for x in tab], ignore_index=True, sort=False)
    df_c.sort_values(by=['date', 'location'], inplace=True)

    # pobranie arkusza z liczbą zgonów

    df0 = pd.read_csv(data_files['cities_deaths']['src_fn'])

    # do tab zbieram dane z powiatów z poszczególnych województw
    tab = []
    columns0 = df0.columns[0]
    columns1 = df0.columns[1]
    indexes = df0.index[df0[columns0].isnull()]
    start = 1
    for i in range(len(indexes)):
        dfx = df0.loc[start:indexes[i] - 1].copy()
        if len(dfx) > 0:
            if pd.isnull(dfx.iloc[0]['Nazwa']):
                dfx = dfx[1:]
            dfx.dropna(axis=0, how='all', inplace=True)
            wojew = dfx.iloc[0]['Nazwa'].lower()
            dfx = dfx[1:]
            dfx.drop([columns0], axis=1, inplace=True)
            dfx.rename(columns={columns1: 'code'}, inplace=True)
            dfx.fillna(0, inplace=True)
            dfxx = dfx.melt(id_vars=['code', 'Nazwa'], var_name='date', value_name='total_deaths')
            start = indexes[i] + 1
            for col in ['new_cases', 'new_deaths', 'total_cases', 'new_recoveries', 'total_recoveries',
                        'total_active', 'new_tests', 'total_tests']:
                dfxx[col] = 0
            dfxx['wojew'] = wojew
            dfxx = dfxx[~dfxx['date'].str.contains('Unnamed', regex=False)]
            dfxx['date'] = ['2020-' + x[3:5] + '-' + x[:2] for x in dfxx['date']]
            dfxx.rename(columns={'Nazwa': 'location'}, inplace=True)
            tab.append(dfxx.iloc[1:])

    df_d = pd.concat([x for x in tab], ignore_index=True, sort=False)
    df_d.sort_values(by=['date', 'location'], inplace=True)

    # połączenie obu zestawów danych

    df = pd.merge(df_c, df_d[['total_deaths', 'date', 'location', 'wojew']], how='left',
                  left_on=['date', 'location', 'wojew'],
                  right_on=['date', 'location', 'wojew'])
    df['total_deaths'] = df['total_deaths_y']
    del df['total_deaths_x']
    del df['total_deaths_y']
    df['new_deaths'].fillna(0, inplace=True)
    df['total_deaths'].fillna(0, inplace=True)

    df.sort_values(by=['location', 'date'], inplace=True)
    # obliczenie new_deaths
    df['new_deaths'] = df.groupby(['location', 'wojew'])['total_deaths'].diff()
    # eliminacja korekt < 0
    df['new_deaths'] = df['new_deaths'].apply(lambda x: 0 if x < 0 else x)
    # ponowne przeliczenie total_deaths
    df['total_deaths'] = df.groupby(['location', 'wojew'])['new_deaths'].cumsum()
    # uzupełnienie braków poprzednimi wartościami
    df.groupby(['location', 'wojew'])['total_deaths'].ffill()

    # uzupełnienie danych powiatowych RCB

    pow_columns = ['location', 'wojew', 'new_cases', 'new_reinf', 'd_0',
                   'new_deaths', 'new_deaths_c', 'new_deaths_nc', 'new_tests_poz',
                   'new_recoveries', 'total_quarantine', 'new_tests', 'new_tests_plus',
                   'new_tests_minus', 'new_tests_other']
    # Warszawa m. st. zamieniamy na m.
    df['location'] = df['location'].str.replace('m. st. ', 'm.', regex=False)
    df['location'] = df['location'].str.replace('m. St. ', 'm.', regex=False)
    # przygotowanie lastrec do cumsum
    lastrec = df.loc[df.date == '2020-11-23'].copy()
    lastrec.fillna(0, inplace=True)
    # stare dane obcinamy do 23.11
    df = df[df['date'] < '2020-11-23'].copy()
    rcb_pow = pd.read_csv('data/sources/rcb/powiaty.csv')
    rcb_pow = rcb_pow[['date'] + pow_columns].copy()
    rcb_pow = rcb_pow[rcb_pow['location'] != 'Polska'].copy()
    # Normalizacja nazw miast i powiatów
    rcb_pow.loc[rcb_pow['location'].str.istitle(), ['location']] = 'Powiat m.' + rcb_pow.loc[rcb_pow['location'].str.istitle(), ['location']]
    rcb_pow.loc[~rcb_pow['location'].str.contains('Powiat'), ['location']] = 'Powiat ' + rcb_pow.loc[~rcb_pow['location'].str.contains('Powiat'), ['location']]
    # przygotowanie do cumsum
    rcb_pow['total_cases'] = rcb_pow['new_cases']
    rcb_pow['total_deaths'] = rcb_pow['new_deaths']
    rcb_pow['total_recoveries'] = rcb_pow['new_recoveries']
    rcb_pow['total_tests'] = rcb_pow['new_tests']
    # dodanie rekordu startowego
    lastrec.reset_index(drop=True, inplace=True)
    rcb_pow = pd.concat([rcb_pow, lastrec], ignore_index=True)
    rcb_pow.sort_values(by=['location', 'wojew', 'date'], inplace=True)
    rcb_pow.reset_index(drop=True, inplace=True)
    # obliczenie total_ z new_
    rcb_pow['total_cases'] = rcb_pow.groupby(['location', 'wojew'])['total_cases'].cumsum()
    rcb_pow['total_deaths'] = rcb_pow.groupby(['location', 'wojew'])['total_deaths'].cumsum()
    rcb_pow['total_recoveries'] = rcb_pow.groupby(['location', 'wojew'])['total_recoveries'].cumsum()
    rcb_pow['total_tests'] = rcb_pow.groupby(['location', 'wojew'])['total_tests'].cumsum()
    df = pd.concat([df, rcb_pow], ignore_index=True)
    df.sort_values(by=['location', 'wojew', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ustawienie grup powiat i miasto

    df['Continent_Name'] = 'powiat'
    df.loc[df['location'].str.contains(' m.', regex=False), 'Continent_Name'] = 'miasto'

    # skasowanie przedrostków
    df['location'] = df['location'].str.replace('Powiat m.', '', regex=False)
    df['location'] = df['location'].str.replace(' st. ', '', regex=False)
    df['location'] = df['location'].str.replace('Powiat ', '', regex=False)

    # dogranie ludności do miast i powiatów

    pl = pd.read_csv('data/dict/sl_powiaty.csv')

    # zamiana niejednoznacznych nazw powiatów na powiat (województwo)

    df['location_alt'] = df['location']
    for p in powiat_translation_table:
        df.loc[(df.location == p['location']) & (df.wojew == p['wojew']), 'location'] = p['new_location']
        # df.loc[(df.location == p['location']) & (df.wojew == p['wojew']), 'location_alt'] = p['new_location']

    # Obliczenie new_cases, new_deaths na podstawie total_cases, total_deaths

    df.sort_values(by=['date', 'location', 'date'], inplace=True)
    df.sort_values(by=['location', 'date'], inplace=True)
    df = df.drop_duplicates(subset=['location', 'date', 'wojew']).copy()

    df['new_cases'] = df.groupby(['location'])['total_cases'].diff()
    df['new_deaths'] = df.groupby(['location'])['total_deaths'].diff()
    df['new_cases'].fillna(0, inplace=True)
    df['new_deaths'].fillna(0, inplace=True)

    # korekta ostatniego wpisu new_, jeśli total_ = 0

    df.drop(df[(df.new_cases < 0) & (df.total_cases == 0)].index, inplace=True)
    df.drop(df[(df.new_deaths < 0) & (df.total_deaths == 0)].index, inplace=True)

    # dogranie liczby ludności

    df.sort_values(by=['date', 'location'], inplace=True)
    df = pd.merge(df, pl[['nazwa', 'population']], how='left', left_on=['location'], right_on=['nazwa'])
    df['population'].fillna(1, inplace=True)
    df['population'] = df['population'].astype(np.float64)

    # dogranie współrzędnych do miast i powiatów

    df_miasto = df[df['Continent_Name'] == 'miasto'].copy()
    df_powiat = df[df['Continent_Name'] == 'powiat'].copy()
    df_powiat = pd.merge(df_powiat, miasta, how='left', left_on=['location'], right_on=['powiat'])
    # df_powiat = pd.merge(df_powiat, miasta, how='left', left_on=['location_alt'], right_on=['powiat'])
    df_miasto = pd.merge(df_miasto, miasta, how='left', left_on=['location'], right_on=['nazwa'])
    df = pd.concat([df_powiat, df_miasto], ignore_index=True)
    df.sort_values(by=['location', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.sort_values(by=['date', 'location'], inplace=True)

    # dogranie powierzchni

    df_area_m0 = pd.read_csv('data/dict/sl_miasta_obszar.csv')[['location', 'area']]
    df_area_m = df_area_m0[['location', 'area']]
    df_area_p = pd.read_csv('data/dict/sl_powiaty_obszar.csv')[['location', 'area']]
    df_area = pd.concat([df_area_m, df_area_p], ignore_index=True)
    df_area['location'] = df_area['location'].str.strip()
    df = pd.merge(df, df_area, how='left', left_on=['location'], right_on=['location'])
    df['area'].fillna(0, inplace=True)

    agg_rules_cities = {
        'location': 'first',
        'total_cases': 'sum',
        'new_cases': 'sum',
        'total_deaths': 'sum',
        'new_deaths': 'sum',
        'total_recoveries': 'sum',
        'new_recoveries': 'sum',
        'total_tests': 'sum',
        'new_tests': 'sum',
        'new_deaths_c': 'sum',
        'new_deaths_nc': 'sum',
        'new_tests_poz': 'sum',
        'total_quarantine': 'sum',
        'new_tests_plus': 'sum',
        'new_tests_minus': 'sum',
        'new_tests_other': 'sum',
        'population': 'sum',
        'area': 'sum',
    }
    # dane dzienne dla wszystkich miast
    df_cities = df[df['Continent_Name'] == 'miasto'].groupby(['date']).agg(agg_rules_cities).reset_index()
    df_cities['location'] = 'Wszystkie miasta'
    df_cities['Continent_Name'] = 'Grupa'

    # dane dzienne dla wszystkich powiatów ziemskich
    df_powiaty = df[df['Continent_Name'] == 'powiat'].groupby(['date']).agg(agg_rules_cities).reset_index()
    df_powiaty['location'] = 'Powiaty ziemskie'
    df_powiaty['Continent_Name'] = 'Grupa'

    # dane dzienne dla miast wojewódzkich
    df_wojew = df[df['location'].isin(wojewodzkie_list)].groupby(['date']).agg(agg_rules_cities).reset_index()
    df_wojew['location'] = 'Miasta wojewódzkie'
    df_wojew['Continent_Name'] = 'Grupa'

    # dane dzienne dla Polski
    df_polska = df.groupby(['date']).agg(agg_rules_cities).reset_index()
    df_polska['location'] = 'Wszystkie powiaty'
    df_polska['Continent_Name'] = 'Grupa'

    # # dogranie podsumowanych grup do df

    df = pd.concat([df, df_cities], ignore_index=True)
    df = pd.concat([df, df_powiaty], ignore_index=True)
    df = pd.concat([df, df_wojew], ignore_index=True)
    df = pd.concat([df, df_polska], ignore_index=True)

    # dane dzienne dla województw

    for wojew in df['wojew'].unique():
        df_ww = df[df['wojew'] == wojew].groupby(['date']).agg(agg_rules_cities).reset_index()
        df_ww['location'] = wojew
        df_ww['Continent_Name'] = 'Grupa'
        df = pd.concat([df, df_ww], ignore_index=True)

    # dogranie podsumowanych grup do df

    df.sort_values(by=['date', 'location'], inplace=True)

    df = pd.concat([df, pd.read_csv('data/tmp/temp_suma_polska.csv')])

    df = df.drop_duplicates(subset=['location', 'date', 'wojew']).copy()

    # obliczenie wskaźników

    df.reset_index(drop=True, inplace=True)
    df['total_cases'].fillna(0, inplace=True)
    df['total_deaths'].fillna(0, inplace=True)
    df['total_active'] = df['total_cases'] - df['total_deaths'] - df['total_recoveries']
    df['zapadalnosc'] = round(df['total_cases'] / df['population'] * 100000, 1)
    df['zapadalnosc'].fillna(0, inplace=True)
    df['umieralnosc'] = round(df['total_deaths'] / df['population'] * 100000, 1)
    df['umieralnosc'].fillna(0, inplace=True)
    df['smiertelnosc'] = round(df['total_deaths'] / df['total_cases'] * 100, 1)
    df['smiertelnosc'].fillna(0, inplace=True)
    df['wyzdrawialnosc'] = round(df['total_recoveries'] / df['total_cases'] * 100, 1)
    df['wyzdrawialnosc'].fillna(0, inplace=True)
    # df['CFR'] = df['total_deaths'] / (df['total_deaths'] + df['total_recoveries']) * 100
    df['dynamikaD'] = 1
    df['positive_rate'] = round(df['new_cases'] / df['new_tests'] * 100, 1)
    df['positive_rate'] = df['positive_rate'].replace(np.inf, np.nan)
    df['reproduction_rate'] = 0
    df['hosp_patients'] = 0
    df['icu_patients'] = 0
    df['total_nadzor'] = 0
    df.sort_values(by=['location', 'date'], inplace=True, ascending=True)
    df['kw'] = (df['new_cases']/df['population']*100000)
    xtemp = df.groupby(['location'])['kw'].rolling(7, min_periods=7).mean()
    df['kwarantanna'] = list(xtemp)
    df['kwarantanna'] = df['kwarantanna'].round(2)
    df.dropna(axis='rows', subset=['kwarantanna'], inplace=True)
    df.index = range(len(df))

    df.loc[df['Continent_Name'] == 'powiat', 'location'] = df.loc[df['Continent_Name'] == 'powiat', 'location'].str.lower()

    # skasowanie zerowych danych z następnego dnia

    df = df[~(df['date'] > str(dt.now())[:10])].copy()

    result = correct_data(df)
    if result.empty:
        raise RuntimeError('Wykryto błędy strukturalne danych Powiaty')
    else:
        df = result.copy()
    df = correct_data(df).copy()
    df['double_days'] = calc_dd_all(df)

    df.to_csv(data_files['cities_cases']['data_fn'])
    print("  wygenerowano " + data_files['cities_cases']['data_fn'])

    return df


# Load data for Poland

def load_data_poland():

    def wojew_testy():
        df = pd.read_csv(data_files['wojew_tests']['src_fn'])
        df0 = df.copy()

        def convert_df(dfx, colname):
            dfx.index = dfx['Województwo']
            del dfx['Województwo']
            dfx = dfx[dfx.columns[1:]].copy()
            # zamiana dziwnej spacji na puste
            dfx = dfx.applymap(lambda x: x.replace("\xa0", "")).copy()
            dfxd = dfx.to_dict()
            l_data = []
            l_wojew = []
            l_value = []
            for key1 in dfxd:
                for key2 in dfxd[key1]:
                    l_data.append(key1)
                    l_wojew.append(key2)
                    l_value.append(int(dfxd[key1][key2]))
            df = pd.DataFrame({'date': l_data, 'location': l_wojew, colname: l_value})
            df['date'] += '.2020'
            df.date = pd.to_datetime(df['date'], format='%d.%m.%Y')
            df.index = df.date
            del df['date']
            df = df.groupby(['location']).resample('D').mean()
            df['total_tests'] = df['total_tests'].interpolate().copy()
            df['total_tests'] = round(df['total_tests'], 0)
            df.reset_index(inplace=True)
            return df

        # pobieram tylko pierwszą podtabelę

        dfx = df0.iloc[:17].copy()
        dfx.dropna(axis=1, how='all', inplace=True)
        dfx.dropna(axis=1, how='any', inplace=True)
        dfx.columns = dfx.iloc[0]
        dfx = dfx.iloc[1:]

        df = convert_df(dfx, 'total_tests')

        df['date'] = df.date.astype(str)
        df.sort_values(by=['date', 'location'], inplace=True)

        return df


    def convert_df(dfx, colname):
        dfx.index = dfx['Województwo']
        dfx.drop('Województwo', axis=1, inplace=True)
        cols = []
        for c in dfx.columns:
            if (c + '.2020') in cols:
                cols.append(c + '.2021')
            else:
                cols.append(c + '.2020')
        dfx.columns = cols
        dfxd = dfx.to_dict()
        l_data = []
        l_wojew = []
        l_value = []
        for key1 in dfxd:
            for key2 in dfxd[key1]:
                l_data.append(key1)
                l_wojew.append(key2)
                l_value.append(int(dfxd[key1][key2]))
        df = pd.DataFrame({'date': l_data, 'location': l_wojew, colname: l_value})
        df.date = pd.to_datetime(df['date'], format='%d.%m.%Y')
        return df

    ########################################
    print('>>> Polska - dane wojewódzkie <<<')
    df = pd.read_csv(data_files['poland_all']['src_fn'], dtype=object)
    df0 = df.copy()
    tab = []
    pocz = df0.loc[df0.iloc[:, 0] == 'Województwo'].index
    for i in range(7):
        dfx = df0.iloc[pocz[i]:pocz[i] + 17].copy()
        dfx.dropna(axis=1, how='all', inplace=True)
        dfx.dropna(axis=1, how='any', inplace=True)
        dfx.columns = dfx.iloc[0]
        dfx = dfx.iloc[1:]
        tab.append(dfx)

    columns = [
        'new_cases', 'total_cases', 'new_deaths', 'total_deaths', 'new_recoveries', 'total_recoveries', 'total_active'
    ]

    dfs = []
    for i in range(len(tab)):
        dfs.append(convert_df(tab[i], columns[i]))

    df = pd.merge(dfs[0], dfs[1], left_on=['date', 'location'], right_on=['date', 'location'], how='outer')
    for d in dfs[2:]:
        df = pd.merge(df, d, left_on=['date', 'location'], right_on=['date', 'location'], how='outer')
    df['date'] = df.date.astype(str)

    # zmiana na rok 2021 od 01.01
    ind = df.loc[df.date == '2020-01-01'].index[0]
    df.loc[ind:, 'date'] = df.loc[ind:]['date'].apply(lambda x: '2021-' + x[5:]).copy()
    ind = df.loc[df.date == '2021-12-31'].index[0] + 16
    df.loc[ind:, 'date'] = df.loc[ind:]['date'].apply(lambda x: '2022-' + x[5:]).copy()
    df = df.copy()
    df.sort_values(by=['date', 'location'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # uzupełnienie 'total_tests'

    df_testy = wojew_testy()
    df_testy['location'] = df_testy['location'].str.replace(' *', '', regex=False)
    df = pd.merge(df, df_testy[['location', 'date', 'total_tests']], how='left', left_on=['location', 'date'], right_on=['location', 'date'])

    # uzupełnienie 'new_tests_*' na podstawie 'total_tests*'

    df.sort_values(by=['location', 'date'], inplace=True)
    df['new_tests'] = df.groupby(['location'])['total_tests'].diff()
    df.loc[df.new_tests < 0, 'new_tests'] = 0

    # uzupełnienie danych RCB

    woj_columns = ['location', 'new_cases', 'new_reinf', 'd_0',
                   'new_deaths', 'new_deaths_c', 'new_deaths_nc', 'new_tests_poz',
                   'new_recoveries', 'total_quarantine', 'new_tests', 'new_tests_plus',
                   'new_tests_minus', 'new_tests_other']
    # przygotowanie lastrec do cumsum
    lastrec = df.loc[df.date == '2020-11-23'].copy()
    lastrec.fillna(0, inplace=True)
    # stare dane obcinamy do 23.11
    df = df[df['date'] < '2020-11-23'].copy()

    rcb_woj = pd.read_csv('data/sources/rcb/wojewodztwa.csv')
    rcb_woj = rcb_woj[['date'] + woj_columns].copy()
    rcb_polska = rcb_woj[rcb_woj['location'] == 'Polska'].copy()
    rcb_woj = rcb_woj[rcb_woj['location'] != 'Polska'].copy()
    # przygotowanie do cumsum
    rcb_woj['total_cases'] = rcb_woj['new_cases']
    rcb_woj['total_deaths'] = rcb_woj['new_deaths']
    rcb_woj['total_recoveries'] = rcb_woj['new_recoveries']
    rcb_woj['total_tests'] = rcb_woj['new_tests']
    # dodanie rekordu startowego
    lastrec.reset_index(drop=True, inplace=True)
    rcb_woj = pd.concat([rcb_woj, lastrec], ignore_index=True)
    rcb_woj.sort_values(by=['location', 'date'], inplace=True)
    rcb_woj.reset_index(drop=True, inplace=True)
    # obliczenie total_ z new_
    rcb_woj['total_cases'] = rcb_woj.groupby(['location'])['total_cases'].cumsum()
    rcb_woj['total_deaths'] = rcb_woj.groupby(['location'])['total_deaths'].cumsum()
    rcb_woj['total_recoveries'] = rcb_woj.groupby(['location'])['total_recoveries'].cumsum()
    rcb_woj['total_tests'] = rcb_woj.groupby(['location'])['total_tests'].cumsum()
    rcb_woj.update(rcb_woj.groupby(['location'])['total_active'].ffill())
    df = pd.concat([df, rcb_woj], ignore_index=True)
    df.sort_values(by=['location', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # uzupełnienie informacji o szczepieniach ECDC

    # 'total_vaccinations', 'new_vaccinations', 'people_vaccinated',  'people_fully_vaccinated'
    # total_vaccinations  <-   SZCZEPIENIA_SUMA
    # people_vaccinated  <-   SZCZEPIENIA_SUMA  -  DAWKA_2_SUMA
    # people_fully_vaccinated  <-  DAWKA_2_SUMA

    df_vacc = pd.read_csv(data_files['ecdc_vacc']['data_fn'])
    df_vacc = df_vacc[df_vacc['grupa'] == 'ALL']
    df_vacc = df_vacc[['date', 'location', 'dawka_1', 'dawka_2', 'marka']]
    def f(r):
        if r.marka == 'JANSS':
            r['fully'] = r['dawka_1']
        else:
            r['fully'] = r['dawka_2']
        return r
    df_vacc = df_vacc.apply(f, axis=1)
    df_vacc.reset_index(drop=True, inplace=True)
    df_vacc.sort_values(by=['location', 'date'], inplace=True)
    df_vacc['dawka_12'] = df_vacc['dawka_1'] + df_vacc['dawka_2']

    # sumowanie rodzajów szczepionek

    df_vacc['new_vaccinations'] = df_vacc.groupby(['location', 'date'])['dawka_12'].transform('sum')
    df_vacc['new_people_vaccinated'] = df_vacc.groupby(['location', 'date'])['dawka_1'].transform('sum')
    df_vacc['new_people_fully_vaccinated'] = df_vacc.groupby(['location', 'date'])['fully'].transform('sum')
    df_vacc = df_vacc.drop_duplicates(subset=['location', 'date']).copy()

    # obliczenie wskaźników

    df_vacc['total_vaccinations'] = df_vacc.groupby(['location'])['new_vaccinations'].cumsum()
    df_vacc['people_vaccinated'] = df_vacc.groupby(['location'])['new_people_vaccinated'].cumsum()
    df_vacc['people_fully_vaccinated'] = df_vacc.groupby(['location'])['new_people_fully_vaccinated'].cumsum()

    df_vacc = df_vacc[['date', 'location', 'new_vaccinations', 'total_vaccinations',
                       'people_fully_vaccinated', 'people_vaccinated']]

    # resampling na pojedyncze dni.

    df_vacc['date'] = pd.to_datetime(df_vacc['date'], format='%Y-%m-%d')
    df_vacc.sort_values(by=['location', 'date'], inplace=True)
    df_vacc.index = df_vacc['date']
    df_vacc = df_vacc.groupby(['location']).resample('D').mean()
    df_vacc['new_vaccinations'] = round(df_vacc['new_vaccinations'] / 7, 0)
    df_vacc['total_vaccinations'] = round(df_vacc['total_vaccinations'], 0)
    df_vacc['people_fully_vaccinated'] = round(df_vacc['people_fully_vaccinated'], 0)
    df_vacc['people_vaccinated'] = round(df_vacc['people_vaccinated'], 0)
    df_vacc = df_vacc.ffill().copy()
    df_vacc.reset_index(inplace=True)
    df_vacc['date'] = df_vacc.date.astype('str').str.slice(0, 10)
    df = pd.merge(df, df_vacc, how='left', left_on=['location', 'date'], right_on=['location', 'date'])

    # uzupełnienie "area', 'population', 'population_density'

    df1 = pd.merge(df, stat_wojew[['wojew', 'km2', 'ltotal', 'lkm2']], how='left', left_on=['location'],
                   right_on=['wojew'])

    df1.drop(columns=['wojew'])
    df1.rename(columns={'km2': 'area', 'ltotal': 'population', 'lkm2': 'population_density'}, inplace=True)
    df = df1.copy()
    df['Continent_Name'] = 'Polska'

    # uzupełnienie 'Lat', 'Long'

    df['wojew'] = df['location'].str.lower()
    m_cond = (miasta['funkcja'].str.contains('siedziba wojewody')) | (miasta['funkcja'].str.contains('miasto stołeczne'))
    miasta_woj = miasta.loc[m_cond, ['województwo', 'Lat', 'Long']]
    df = pd.merge(df, miasta_woj, how='left', left_on=['wojew'], right_on=['województwo'])
    df.sort_values(by=['date', 'location'], inplace=True)

    # dane dzienne dla 'Polska'

    agg_rules_polska = {
        'location': 'first',
        'total_cases': 'sum',
        'new_cases': 'sum',
        'total_deaths': 'sum',
        'new_deaths': 'sum',
        'total_recoveries': 'sum',
        'new_recoveries': 'sum',
        'total_tests': 'sum',
        'new_tests': 'sum',
        'new_deaths_c': 'sum',
        'new_deaths_nc': 'sum',
        'new_tests_poz': 'sum',
        'total_quarantine': 'sum',
        'population': 'sum',
        'area': 'sum',
    }
    df_polska = df.groupby(['date']).agg(agg_rules_polska).reset_index()
    df_polska['location'] = 'Polska'
    df_polska['Continent_Name'] = 'Grupa'

    # Dane Polska
    df_polska = df_polska[df_polska['date'] < '2020-11-24'].copy()
    df = pd.concat([df, df_polska], ignore_index=True)

    rcb_polska['total_cases'] = rcb_polska['new_cases'].cumsum()
    rcb_polska['total_deaths'] = rcb_polska['new_deaths'].cumsum()
    rcb_polska['total_recoveries'] = rcb_polska['new_recoveries'].cumsum()
    rcb_polska['total_tests'] = rcb_polska['new_tests'].cumsum()
    rcb_polska['population'] = 38162224
    rcb_polska['area'] = 312705
    rcb_polska['location'] = 'Polska'
    rcb_polska['Continent_Name'] = 'Grupa'
    rcb_polska = rcb_polska[rcb_polska['date'] >= '2020-11-24'].copy()
    df = pd.concat([df, rcb_polska], ignore_index=True)

    # uzupełnienie 'icu_patients', 'hosp_patients' dla województw i Polski

    df_rt = pd.read_csv(data_files['poland_resources']['data_fn'])
    df_rt = df_rt[['date', 'location', 'icu_patients', 'hosp_patients']]
    df_reso_pl = df_rt[df_rt['location'] == 'polska'].copy()
    df_reso_woj = df_rt[df_rt['location'] != 'polska'].copy()
    df_reso_pl['location'] = 'Polska'
    df_polska = pd.merge(df[df['location'] == 'Polska'], df_reso_pl,
                         how='left', left_on=['location', 'date'],
                         right_on=['location', 'date'])
    df_wojew = pd.merge(df[df['location'] != 'Polska'], df_reso_woj,
                        how='left',
                        left_on=['wojew', 'date'], right_on=['location', 'date'])
    df_wojew['location'] = df_wojew['location_x']
    df_wojew['hosp_patients'].fillna(0, inplace=True)
    df_wojew['icu_patients'].fillna(0, inplace=True)
    del df_wojew['location_x']
    df = pd.concat([df_polska, df_wojew], ignore_index=True)

    # uzupełnienie R(t) dla województw i Polski

    df_rt_woj = pd.read_csv(data_files['woj_rt']['data_fn'])
    df_rt_pl = pd.read_csv(data_files['poland_rt']['data_fn'])
    df_rt = pd.concat([df_rt_woj, df_rt_pl], ignore_index=True)
    df['wojew'].fillna('Polska', inplace=True)
    df = pd.merge(df, df_rt, how='left',
                  left_on=['wojew', 'date'],
                  right_on=['location', 'date'])

    df['location'] = df['location_x']
    df[df['location'] == 'Polska'].to_csv('data/tmp/temp_suma_polska.csv', index=False)

    # uzupełnienie 'total_nadzor', 'total_quarantine' dla Polski

    balance = pd.read_csv(data_files['poland_balance']['data_fn'])
    df = pd.merge(df, balance[['date', 'hosp_patients', 'icu_patients', 'location', 'total_nadzor']],
                  how='left',
                  left_on=['date', 'location'],
                  right_on=['date', 'location'])

    hp = list(df.loc[df.location == 'Polska', 'hosp_patients_y'])
    ip = list(df.loc[df.location == 'Polska', 'icu_patients_y'])
    df.loc[df.location == 'Polska', 'hosp_patients_x'] = hp
    df.loc[df.location == 'Polska', 'icu_patients_x'] = ip
    del df['hosp_patients_y']
    del df['icu_patients_y']
    df.rename(columns={'hosp_patients_x': 'hosp_patients'}, inplace=True)
    df.rename(columns={'icu_patients_x': 'icu_patients'}, inplace=True)

    # obliczenie wskaźników

    df['double_days'] = -1
    df['total_active'] = df['total_cases'] - df['total_deaths'] - df['total_recoveries']
    df['zapadalnosc'] = round(df['total_cases'] / df['population'] * 100000, 1)
    df['umieralnosc'] = round(df['total_deaths'] / df['population'] * 100000, 1)
    df['smiertelnosc'] = round(df['total_deaths'] / df['total_cases'] * 100, 1)
    df['wyzdrawialnosc'] = round(df['total_recoveries'] / df['total_cases'] * 100, 1)
    df['positive_rate'] = round(df['new_cases'] / df['new_tests'] * 100, 1)
    df['positive_rate'] = df['positive_rate'].replace(np.inf, np.nan)
    # df['CFR'] = df['total_deaths'] / (df['total_deaths'] + df['total_recoveries']) * 100
    df['dynamikaD'] = 1
    df.sort_values(by=['location', 'date'], inplace=True, ascending=True)

    # wskaźnik kwarantanny narodowej

    df['kw'] = (df['new_cases']/df['population']*100000)
    xtemp = df.groupby(['location'])['kw'].rolling(7, min_periods=7).mean()
    df['kwarantanna'] = list(xtemp)
    df['kwarantanna'] = df['kwarantanna'].round(2)
    df.dropna(axis='rows', subset=['kwarantanna'], inplace=True)
    df.index = range(len(df))

    # uzupełnienie excess mortality

    df_mort = pd.read_csv(data_files['mortality_eurostat']['data_fn'])
    df_mort = df_mort[df_mort['age'] == 'TOTAL'].copy()

    # df_podlaskie = df_mort[df_mort['total'] <= 0].copy()
    df_mort = df_mort[df_mort['total'] > 0].copy()
    means = df_mort[(df_mort['year'] >= 2015) & (df_mort['year'] <= 2019)].groupby('location')['total'].mean()
    means = means.to_dict()
    means['polska'] = means['Polska']

    df_mort = df_mort[['date', 'location', 'total']].copy()
    df_mort.sort_values(by=['location', 'date'], inplace=True)
    df_mort['date'] = pd.to_datetime(df_mort['date'], format='%Y-%m-%d')
    df_mort['location_cap'] = df_mort['location'].apply(lambda x: wojew_cap.get(x))
    df_mort.sort_values(by=['location_cap', 'date'], inplace=True)
    df_mort.index = df_mort.date
    del df_mort['date']
    df_mort['total'] = df_mort['total'].replace(0, np.nan)
    df_mort = df_mort.groupby(['location_cap']).resample('D').mean()
    df_mort = df_mort.interpolate().copy()
    df_mort['total'] = df_mort['total'] / 7
    df_mort.reset_index(level=0, inplace=True)
    df_mort['date'] = df_mort.index.astype('str').str.slice(0, 10)
    df_mort.reset_index(drop=True, inplace=True)
    df = pd.merge(df, df_mort[['date', 'location_cap', 'total']], how='left', left_on=['location', 'date'], right_on=['location_cap', 'date'])
    del df['location_cap']
    df.rename(columns={'total': 'mort'}, inplace=True)
    df['mort'] = df['mort'].replace(0, np.nan)
    df['mean'] = df['location'].str.lower().map(means)
    df.loc[df['mort'].isna(), 'mean'] = np.nan
    df['new_excess'] = df['mort'] - df['mean'] / 7
    df.reset_index(drop=True, inplace=True)
    df.sort_values(by=['location', 'date'], inplace=True)
    df['total_excess'] = df.groupby(['location'])['new_excess'].cumsum()

    # Obliczenie 'double_days' dla wszystkich województw

    df['double_days'] = calc_dd_all(df)
    df['hosp_patients'] = df['hosp_patients'].fillna(0)
    df['icu_patients'] = df['icu_patients'].fillna(0)
    # df['hosp_patients'].fillna(0, inplace=True)
    # df['icu_patients'].fillna(0, inplace=True)
    df['new_cases_total'] = df['new_cases'] + df['new_reinf']

    result = correct_data(df)
    if result.empty:
        raise RuntimeError('Wykryto błędy strukturalne danych Poland')
    else:
        df = result.copy()
    df.to_csv(data_files['poland_all']['data_fn'])
    print("  wygenerowano", data_files['poland_all']['data_fn'])

    return df


def load_data_world():
    print('>>> Dane światowe <<<')
    exclude_df_r = ['Cabo Verde', 'Diamond Princess', 'Eswatini', 'Holy See', 'MS Zaandam', 'Taiwan',
                    'West Bank and Gaza']
    exclude_df = ['Anguilla', 'Aruba', 'Bermuda', 'Bonaire Sint Eustatius and Saba', 'British Virgin Islands',
                  'Cape Verde', 'Cayman Islands', 'Curacao', 'Faeroe Islands', 'Falkland Islands', 'French Polynesia',
                  'Gibraltar', 'Greenland', 'Guam', 'Guernsey', 'Isle of Man', 'Jersey', 'Kosovo',
                  'Montserrat', 'New Caledonia', 'Northern Mariana Islands', 'Palestine', 'Puerto Rico',
                  'Swaziland', 'Taiwan', 'United States Virgin Islands', 'Vatican']

    df = pd.read_csv(data_files['world']['src_fn'])

    fields_all = ['iso_code', 'continent', 'location', 'date', 'total_cases', 'new_cases',
     'new_cases_smoothed', 'total_deaths', 'new_deaths',
     'new_deaths_smoothed', 'total_cases_per_million',
     'new_cases_per_million', 'new_cases_smoothed_per_million',
     'total_deaths_per_million', 'new_deaths_per_million',
     'new_deaths_smoothed_per_million', 'reproduction_rate', 'icu_patients',
     'icu_patients_per_million', 'hosp_patients',
     'hosp_patients_per_million', 'weekly_icu_admissions',
     'weekly_icu_admissions_per_million', 'weekly_hosp_admissions',
     'weekly_hosp_admissions_per_million', 'new_tests', 'total_tests',
     'total_tests_per_thousand', 'new_tests_per_thousand',
     'new_tests_smoothed', 'new_tests_smoothed_per_thousand',
     'positive_rate', 'tests_per_case', 'tests_units', 'total_vaccinations',
     'people_vaccinated', 'people_fully_vaccinated', 'total_boosters',
     'new_vaccinations', 'new_vaccinations_smoothed',
     'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
     'people_fully_vaccinated_per_hundred', 'total_boosters_per_hundred',
     'new_vaccinations_smoothed_per_million',
     'new_people_vaccinated_smoothed',
     'new_people_vaccinated_smoothed_per_hundred', 'stringency_index',
     'population', 'population_density', 'median_age', 'aged_65_older',
     'aged_70_older', 'gdp_per_capita', 'extreme_poverty',
     'cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers',
     'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand',
     'life_expectancy', 'human_development_index',
     'excess_mortality_cumulative_absolute', 'excess_mortality_cumulative',
     'excess_mortality', 'excess_mortality_cumulative_per_million']

    df_world_basic = df[df['location'] == 'World'].copy()

    # skasowanie terytoriów podległych
    df = df[~df['location'].isin(exclude_df)].copy()

    def read_csse(which):
        if which == 'confirmed':
            csse_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
            csse_fn = 'data/tmp/csse_confirmed.csv'
            field_total = 'total_cases'
            field_new = 'new_cases'
        elif which == 'deaths':
            csse_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
            csse_fn = 'data/tmp/csse_deaths.csv'
            field_total = 'total_deaths'
            field_new = 'new_deaths'
        else:
            csse_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
            csse_fn = 'data/tmp/csse_recovered.csv'
            field_total = 'total_recoveries'
            field_new = 'new_recoveries'

        urllib.request.urlretrieve(csse_url, csse_fn)
        df_r0 = pd.read_csv(csse_fn)
        df_r = df_r0.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='date',
                          value_name=field_total)
        df_r['date'] = df_r['date'].astype('datetime64[ns]').astype(str)

        # sumowanie dla krajów z wieloma rekordami

        df_r['Province/State'].fillna('capital', inplace=True)

        agg_rules = {
            'Province/State': 'first',
            'Lat': 'first',
            'Long': 'first',
            field_total: 'sum',
        }
        capital_dict = {
            'Australia': 'Australian Capital Territory',
            'China': 'Beijing',
            'Denmark': 'capital',
            'France': 'capital',
            'Netherlands': 'capital',
            'United Kingdom': 'capital',
        }

        df_r_new = df_r.groupby(['date', 'Country/Region']).agg(agg_rules).reset_index()

        for key in capital_dict:
            new_val_lat = df_r[(df_r['Country/Region'] == key) &
                               (df_r['Province/State'] == capital_dict[key])]['Lat'].iloc[0]
            new_val_long = df_r[(df_r['Country/Region'] == key) &
                                (df_r['Province/State'] == capital_dict[key])]['Long'].iloc[0]
            for index, row in df_r_new[df_r_new['Country/Region'] == key].iterrows():
                df_r_new.at[index, 'Lat'] = new_val_lat
                df_r_new.at[index, 'Long'] = new_val_long

        df_r = df_r_new.copy()

        # uzupełnienie 'new_*' na podstawie 'total_*'

        df_r.sort_values(by=['Country/Region', 'date'], inplace=True)
        df_r = df_r.drop_duplicates(subset=['Country/Region', 'date']).copy()
        df_r[field_new] = df_r.groupby(['Country/Region'])[field_total].diff()
        df_r[field_new].fillna(0, inplace=True)
        df_r.drop(df_r[(df_r[field_new] < 0) & (df_r[field_total] == 0)].index, inplace=True)

        # dodanie rekordu dziesiejszego z powtórzonymi wartościami total z dnia poprzedniego

        df_r.sort_values(by=['date', 'Country/Region'], inplace=True)
        df_r.reset_index(drop=True, inplace=True)
        dmax = df_r.date.max()
        df_rest = df_r[df_r.date == dmax].copy()
        d = datetime.datetime.strptime(dmax, '%Y-%m-%d') + datetime.timedelta(days=1)
        date_str = d.strftime('%Y-%m-%d')
        df_rest['date'] = date_str
        df_rest['new_recoveries'] = 0
        df_r = pd.concat([df_r, df_rest], ignore_index=True)

        # df_r = df_r[df_r['marker'] == '+'].copy()
        replace_values = {
            'Congo (Brazzaville)': 'Democratic Republic of Congo',
            'Czechia': 'Czech Republic',
            'North Macedonia': 'Macedonia',
            'Burma': 'Myanmar',
            'Korea, South': 'South Korea',
            'Taiwan*': 'Taiwan',
            'Timor-Leste': 'Timor',
            'US': 'United States',
            'Congo (Kinshasa)': 'Congo'
        }
        df_r = df_r.replace({'Country/Region': replace_values})

        # skasowanie lokalizacji nieistotnych
        df_r = df_r[~df_r['Country/Region'].isin(exclude_df_r)].copy()

        return df_r

    pola = ['iso_code', 'continent', 'location', 'date', 'total_cases', 'new_cases',
            'total_deaths', 'new_deaths', 'total_tests', 'new_tests', 'population', 'population_density',
           'reproduction_rate', 'hosp_patients', 'icu_patients', 'positive_rate',
            'total_vaccinations', 'new_vaccinations',  'people_vaccinated', 'people_fully_vaccinated']
    df = df[pola].copy()

    # Dane "recovered" z CSSE

    df_csse_recovered = read_csse('recovered')
    df_csse_recovered['Country/Region'] = df_csse_recovered['Country/Region'].str.replace('Czech Republic', 'Czechia', regex=False)
    df = pd.merge(df, df_csse_recovered, how='left', left_on=['location', 'date'], right_on=['Country/Region', 'date'])

    # wypełnienie brakujących danych total_ poprzednimi

    df.sort_values(by=['location', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['total_tests'] = df.groupby(['location'])['total_tests'].ffill()
    df['total_vaccinations'] = df.groupby(['location'])['total_vaccinations'].ffill()

    df['total_active'] = df['total_cases'] - df['total_deaths'] - df['total_recoveries']
    df['total_cases'].fillna(0, inplace=True)
    df['new_cases'].fillna(0, inplace=True)
    df['total_deaths'].fillna(0, inplace=True)
    df['new_deaths'].fillna(0, inplace=True)
    df['total_recoveries'].fillna(0, inplace=True)
    df['new_recoveries'].fillna(0, inplace=True)
    df['new_tests'] = df.groupby(['location'])['total_tests'].diff()
    # df = df[df['location'] != 'Hong Kong'].copy()

    df.sort_values(by=['date', 'location'], inplace=True)
    df.sort_values(by=['location', 'date'], inplace=True)

    df.index = pd.to_datetime(df['date'])

    # Uzupełnienie "Continent_Name" na podstawie "iso_code"

    df = df.dropna(axis='rows', subset=['iso_code'])

    # Obliczenie powierzchni

    df['area'] = df['population'] / df['population_density']

    # dodanie nazwy kontynentu

    df1 = pd.merge(df, countries[['Three_Letter_Country_Code', 'Continent_Name']], how='left', left_on=['iso_code'],
                   right_on=['Three_Letter_Country_Code'])
    df = df1.drop('Three_Letter_Country_Code', axis=1)

    # Skasowanie jednego wystąpienia krajów należących do dwóch kontynentów np. Europe, Turkey

    df.sort_values(by=['location', 'date'], inplace=True)
    df.sort_values(by=['Continent_Name', 'date'], inplace=True)

    # Tłumaczenie na polski

    df['location'] = xlate_array(df['location'])
    df['Continent_Name'] = xlate_array(df['Continent_Name'])

    agg_rules = {
        'location': 'first',
        'total_cases': 'sum',
        'new_cases': 'sum',
        'total_deaths': 'sum',
        'new_deaths': 'sum',
        'total_recoveries': 'sum',
        'total_vaccinations': 'sum',
        'new_recoveries': 'sum',
        'total_tests': 'sum',
        'new_tests': 'sum',
        'population': 'sum',
        'area': 'sum',
    }

    # dane dzienne dla krajów Ameryki Łacińskiej

    la_list = list(df[df['Continent_Name'].isin(['Ameryka Południowa', 'Ameryka Północna'])]['location'].unique())
    la_list.remove('Stany Zjednoczone')
    la_list.remove('Kanada')
    df_la = df[df['location'].isin(la_list)].groupby(['date']).agg(agg_rules).reset_index()
    df_la['location'] = 'Ameryka Łacińska'
    df_la['Continent_Name'] = 'Grupa'

    # dane dzienne dla krajów arabskich

    la_list = list(df[df['location'].isin(arab_countries_list)]['location'].unique())
    df_arab = df[df['location'].isin(la_list)].groupby(['date']).agg(agg_rules).reset_index()
    df_arab['location'] = 'Kraje arabskie'
    df_arab['Continent_Name'] = 'Grupa'

    # dane dzienne dla demoludów

    la_list = list(df[df['location'].isin(demoludy_list)]['location'].unique())
    df_demoludy = df[df['location'].isin(la_list)].groupby(['date']).agg(agg_rules).reset_index()
    df_demoludy['location'] = 'Dawne demoludy'
    df_demoludy['Continent_Name'] = 'Grupa'

    # dane dzienne dla krajów skandynawskich

    la_list = list(df[df['location'].isin(skandynawskie_list)]['location'].unique())
    df_skand = df[df['location'].isin(la_list)].groupby(['date']).agg(agg_rules).reset_index()
    df_skand['location'] = 'Kraje skandynawskie'
    df_skand['Continent_Name'] = 'Grupa'

    # dane dzienne dla krajów Unii Europejskiej

    df_eu = df[df['location'].isin(eu_countries_list)].groupby(['date']).agg(agg_rules).reset_index()
    df_eu['location'] = 'Unia Europejska'
    df_eu['Continent_Name'] = 'Grupa'

    # dane dzienne dla krajów spoza Unii Europejskiej

    df_nie_eu = df[(df['Continent_Name'] == 'Europa') & (~df['location'].isin(eu_countries_list))] \
        .groupby(['date']).agg(agg_rules).reset_index()
    df_nie_eu['location'] = 'Europa spoza Unii Europejskiej'
    df_nie_eu['Continent_Name'] = 'Grupa'

    # dane dzienne dla grup krajów subgroups

    for group in subgroups:
        df_sub = df[(df['iso_code'].isin(subgroups[group]))].groupby(['date']).agg(agg_rules).reset_index()
        df_sub['location'] = group
        df_sub['Continent_Name'] = 'Grupa'
        df = pd.concat([df, df_sub], ignore_index=True)

    # dogranie krajów Unii Europeskiej do df
    # dogranie krajów spoza Unii Europeskiej do df
    # dogranie krajów Ameryki Łacińskiej do df
    # dogranie krajów arabskich do df
    # dogranie demoludów do df
    # dogranie krajów skandynawskich do df

    df = pd.concat([df, df_eu], ignore_index=True)
    df = pd.concat([df, df_nie_eu], ignore_index=True)
    df = pd.concat([df, df_la], ignore_index=True)
    df = pd.concat([df, df_arab], ignore_index=True)
    df = pd.concat([df, df_demoludy], ignore_index=True)
    df = pd.concat([df, df_skand], ignore_index=True)

    # obliczenie wskaźników

    df['total_active'] = df['total_cases'] - df['total_deaths'] - df['total_recoveries']
    df['new_new_cases'] = df.groupby(['location'])['new_cases'].diff()
    df['new_new_cases'].fillna(0, inplace=True)
    df['zapadalnosc'] = round(df['total_cases'] / df['population'] * 100000, 1)
    df['zapadalnosc'].fillna(0, inplace=True)
    df['umieralnosc'] = round(df['total_deaths'] / df['population'] * 100000, 1)
    df['umieralnosc'].fillna(0, inplace=True)
    df['smiertelnosc'] = round(df['total_deaths'] / df['total_cases'] * 100, 1)
    df['smiertelnosc'].fillna(0, inplace=True)
    df['wyzdrawialnosc'] = round(df['total_recoveries'] / df['total_cases'] * 100, 1)
    df['wyzdrawialnosc'].fillna(0, inplace=True)
    # df['CFR'] = df['total_deaths'] / (df['total_deaths'] + df['total_recoveries']) * 100
    df['dynamikaD'] = 1
    df.sort_values(by=['location', 'date'], inplace=True, ascending=True)
    # kwarantanna narodowa
    df['kw'] = (df['new_cases']/df['population']*100000)
    xtemp = df.groupby(['location'])['kw'].rolling(7, min_periods=7).mean()
    df['kwarantanna'] = list(xtemp)
    df['kwarantanna'] = df['kwarantanna'].round(2)

    df['total_quarantine'] = 0
    df['total_nadzor'] = 0
    df.dropna(axis='rows', subset=['kwarantanna'], inplace=True)
    df.index = range(len(df))

    # # uzupełnienie icu_patients i hosp_patients dla Polski
    #
    # df_reso = pd.read_csv(data_files['poland_resources']['data_fn'])
    # df_reso_pl = df_reso[df_reso['location'] == 'polska'].copy()
    # df_reso_pl['location'] = 'Polska'
    # df_pl = df[df['location'] == 'Polska'].copy()
    # del df_pl['icu_patients']
    # del df_pl['hosp_patients']
    # df_nonpl = df[df['location'] != 'Polska'].copy()
    # df_pl = pd.merge(df_pl, df_reso_pl[['date', 'location', 'icu_patients', 'hosp_patients']], how='left', left_on=['location', 'date'], right_on=['location', 'date'])
    # df = df_pl.append(df_nonpl, ignore_index=True)

    # uzupełnienie excess mortality

    df_mort = pd.read_csv(data_files['mortality_eurostat']['data_fn'])
    df_mort = df_mort[df_mort['age'] == 'TOTAL'].copy()

    means = df_mort[df_mort['year']==2019].groupby('location')['total'].mean()
    means = means.to_dict()

    df_mort = df_mort[['date', 'location', 'total']].copy()
    df_mort.sort_values(by=['location', 'date'], inplace=True)
    df_mort['date'] = pd.to_datetime(df_mort['date'], format='%Y-%m-%d')
    df_mort.sort_values(by=['location', 'date'], inplace=True)
    df_mort.index = df_mort.date
    del df_mort['date']
    df_mort['total'] = df_mort['total'].replace(0, np.nan)
    df_mort = df_mort.groupby(['location']).resample('D').mean()
    df_mort = df_mort.interpolate().copy()
    df_mort['total'] = df_mort['total'] / 7
    df_mort.reset_index(level=0, inplace=True)
    df_mort['date'] = df_mort.index.astype('str').str.slice(0, 10)
    # df_mort = df_mort[df_mort['date'] < '2021-02-08'].copy()
    df_mort.reset_index(drop=True, inplace=True)
    df = pd.merge(df, df_mort[['date', 'location', 'total']], how='left', left_on=['location', 'date'], right_on=['location', 'date'])
    df.rename(columns={'total': 'mort'}, inplace=True)
    df['mort'] = df['mort'].replace(0, np.nan)
    df['mean'] = df['location'].map(means)
    df.loc[df['mort'].isna(), 'mean'] = np.nan
    df['new_excess'] = df['mort'] - df['mean'] / 7


    df_mort['total'] = df_mort['total'] / 7

    result = correct_data(df)
    if result.empty:
        raise RuntimeError('Wykryto błędy strukturalne danych Poland')
    else:
        df = result.copy()
    df = correct_data(df).copy()
    df['double_days'] = calc_dd_all(df)
    df.to_csv(data_files['world']['data_fn'])
    print("  wygenerowano " + data_files['world']['data_fn'])
    return df


# Dane Polska bilans

def load_data_balance():
    df = pd.read_csv(data_files['poland_balance']['src_fn'], dtype=str)
    columns = list(df.columns)
    cc = ['Data',
     'Liczba osób hospitalizowanych ',
     'zmiana (d/d)',
     'Liczba łóżek dla pacjentów',
     '% zajęte',
     'Liczba zajętych respiratorów (stan ciężki)',
     'zmiana (d/d).1',
     '% osób pod respiratorem w aktualnej liczbie osób hospitalizowanych',
     'Liczba dostępnych respiratorów',
     '% zajęte.1',
     'Liczba osób objętych kwarantanną',
     'Liczba objętych kwarantanną lokalnie',
     'Obywatele wracający zza granicy',
     'Liczba osób objętych nadzorem epidemiologicznym do 25.01.2021',
     'Unnamed: 14',
     'Unnamed: 15',
     'Unnamed: 16',
     'Unnamed: 17',
     'Unnamed: 18',
     'Unnamed: 19',
     'Unnamed: 20',
     'Unnamed: 21',
     'Unnamed: 22',
     'Unnamed: 23',
     'Przybliżenie od 01.05.2021']
    df = df[[columns[0], columns[1], columns[3], columns[5], columns[8], columns[10], columns[13]]].copy()
    df.columns = ['date', 'hosp_patients', 'total_beds', 'icu_patients', 'total_resp', 'total_quarantine', 'total_nadzor']
    i = df.index[df.date == '1.01'].tolist()[0]
    df.loc[:i, 'year'] = '.2020'
    df.loc[i:, 'year'] = '.2021'
    df['date'] = df['date'].str.replace('\.1$', '.10', regex=True)
    df['date'] = df['date'] + df['year']
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y').astype(str)
    df.fillna(0, inplace=True)
    df['location'] = 'Polska'
    df = df[['date', 'location', 'hosp_patients', 'total_beds', 'icu_patients', 'total_resp', 'total_quarantine', 'total_nadzor']].copy()
    # df.sort_values(by=['date'], inplace=True)
    save_github(data_files['poland_balance']['data_fn'], str(dt.now())[:10] + ' kwarantanna i nadzór')
    df.to_csv(data_files['poland_balance']['data_fn'], index=False)

# Load data for Poland - resources

def load_data_resources():
    df = pd.read_csv(data_files['poland_resources']['src_fn'])
    df0 = df.copy()
    tab = []
    columns = list(df.columns)[1:]
    column0 = list(df.columns)[0]
    for i in range(17):
        subset = [column0] + columns[i*8: (i+1)*8]
        sub_df = df[subset].copy()

        loc = subset[1].split()[0]
        if loc == 'Żródło:':
            loc = 'DOLNOŚLĄSKIE'
        kolumny = ['date', 'hosp_patients', 'new_hosp', 'total_beds', 'used_beds', 'icu_patients', 'new_in_resp', 'total_resp', 'used_resp']
        sub_df.columns = kolumny
        sub_df.dropna(subset=['date'], inplace=True)
        # sub_df['d'] = sub_df['date'] + 0.001
        sub_df['d'] = sub_df['date'].astype(str)
        sub_df['d'] = sub_df['d'] + '.2020'
        sub_df['d'] = sub_df['d'].str.replace('.1.', '.10.', regex=False)
        sub_df['dd'] = pd.to_datetime(sub_df['d'], format='%d.%m.%Y')
        sub_df['date'] = sub_df['dd'].astype(str)
        sub_df = sub_df[kolumny].copy()
        sub_df['location'] = loc.lower()
        # zmiana na rok 2021 od 01.01
        ind = sub_df.loc[sub_df.date == '2020-01-01'].index[0]
        sub_df.loc[ind:, 'date'] = sub_df.loc[ind:]['date'].apply(lambda x: '2021-' + x[5:]).copy()
        tab.append(sub_df)

    df = pd.concat([x for x in tab], ignore_index=True, sort=False)
    df['used_beds'] = df['used_beds'].str.replace('%', '', regex=False).str.replace(',', '.', regex=False).astype('float')
    df['used_resp'] = df['used_resp'].str.replace('%', '', regex=False).str.replace(',', '.', regex=False).astype('float')
    df['hosp_patients'] = df['total_beds'] * df['used_beds'] / 100
    df = df[['date', 'location', 'hosp_patients', 'total_beds', 'icu_patients', 'total_resp', 'used_beds', 'used_resp']].copy()

    df.fillna(0, inplace=True)
    df = df.copy()
    df.sort_values(by=['date', 'location'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # dodatkowy CSV
    dfd = pd.read_csv('data/sources/hospi_respi.csv')
    dfd = dfd[['location', 'date', 'hosp_patients', 'icu_patients']].copy()
    dfd['location'] = dfd['location'].str.lower()
    df = pd.concat([df, dfd], ignore_index=True)
    df.fillna(0, inplace=True)
    df.to_csv(data_files['poland_resources']['data_fn'], index=False)
    print("  wygenerowano", data_files['poland_resources']['data_fn'])

    return df

# Dane testy/szczepienia/infekcje/zgony (Łukasz)

def read_testy_infekcje():
    print('>>> testy/szczepienia/infekcje/zgony (Łukasz) <<<')
    df = pd.read_csv('data/testy i szczepienia/Testy i szczepienia zestawienie.csv', sep=';')
    df.columns = ['data', 'wynik', 'd1', 'd2', 'd3', 'marka', 'dz', 'wiek', 'plec', 'teryt', 'ile']
    map1 = {'Brak': '-------'}
    map2 = {'Brak danych': '-------'}
    map3 = {'nieznana': '-'}
    map4 = {'Brak': '-'}
    df.replace({'d1': map1,
                'd2': map1,
                'd3': map1,
                'dz': map2,
                'marka': map4,
                'wiek': map2,
                'plec': map3,
                'teryt': map1}, inplace=True)
    df['test_rok'] = df['data'].str.slice(0, 4)
    df['test_miesiac'] = df['data'].str.slice(5, 7)
    df['d1_rok'] = df['d1'].str.slice(0, 4)
    df['d1_miesiac'] = df['d1'].str.slice(5, 7)
    df['d2_rok'] = df['d2'].str.slice(0, 4)
    df['d2_miesiac'] = df['d2'].str.slice(5, 7)
    df['d3_rok'] = df['d3'].str.slice(0, 4)
    df['d3_miesiac'] = df['d3'].str.slice(5, 7)
    df['zgon_rok'] = df['dz'].str.slice(0, 4)
    df['zgon_miesiac'] = df['dz'].str.slice(5, 7)
    df['wynik'] = df['wynik'].map({'POZYTYWNY': '+', 'NEGATYWNY': '-', 'NIEROZSTRZYGAJACY': 'nr', 'NIEDIAGNOSTYCZNY': 'nd'})

    map1 = {'----': np.nan}
    map2 = {'--': np.nan}
    df.replace({'test_rok': map1,
                'd1_rok': map1,
                'd2_rok': map1,
                'd3_rok': map1,
                'zgon_rok': map1,
                'test_miesiac': map2,
                'd1_miesiac': map2,
                'd2_miesiac': map2,
                'd3_miesiac': map2,
                'zgon_miesiac': map2}, inplace=True)
    del df['data']
    del df['d1']
    del df['d2']
    del df['d3']
    del df['dz']
    df.to_csv('data/testy i szczepienia/out_Testy i szczepienia zestawienie.csv', index=False)
    chunk_size = 1000000
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunk = df[i * chunk_size:(i + 1) * chunk_size]
        chunk.to_csv('data/testy i szczepienia/chunk_' + str(i) + '.csv', index=False)
    print('  wygenerowano', 'data/testy i szczepienia/out_Testy i szczepienia zestawienie.csv')
    return df


def stat_testy_infekcje():
    df = pd.read_csv('data/testy i szczepienia/out_Testy i szczepienia zestawienie.csv')
    df.fillna(0, inplace=True)
    df['test_n'] = 52 * (df.test_rok - 2020) + df.test_miesiac
    df['test_n'] = df.test_n.replace(-105040, 0)
    df['d1_n'] = 52 * (df.d1_rok - 2020) + df.d1_miesiac
    df['d1_n'] = df.d1_n.replace(-105040, 0)
    df['d2_n'] = 52 * (df.d2_rok - 2020) + df.d2_miesiac
    df['d2_n'] = df.d2_n.replace(-105040, 0)
    df['d3_n'] = 52 * (df.d3_rok - 2020) + df.d3_miesiac
    df['d3_n'] = df.d3_n.replace(-105040, 0)
    df['zgon_n'] = 52 * (df.zgon_rok - 2020) + df.zgon_miesiac
    df['zgon_n'] = df.zgon_n.replace(-105040, 0)
    # delty
    df['test_d1'] = df.test_n - df.d1_n
    df.loc[df.test_d1 < 0, 'test_d1'] = -1
    df['test_d2'] = df.test_n - df.d2_n
    df.loc[df.test_d2 < 0, 'test_d2'] = -1
    df['test_d3'] = df.test_n - df.d3_n
    df.loc[df.test_d3 < 0, 'test_d3'] = -1

    xx = pd.pivot_table(df[df.wynik.isin(['+', '-'])], index=["test_d3", "wynik"], values=["ile"], aggfunc=np.sum)
    xx.reset_index(inplace=True)
    x1 = xx[xx.wynik == '+'].copy()
    x2 = xx[xx.wynik == '-'].copy()
    yy = pd.merge(x1, x2, how='left', left_on=['test_d3'], right_on=['test_d3'])
    pass


# Szczepienia Polska ECDC

def load_data_ecdc_vac_pl():
    df0 = pd.read_csv(data_files['ecdc_vacc']['src_fn'])
    df = df0[df0['ReportingCountry'] == 'PL'].copy()
    del df['FirstDoseRefused']
    df.rename(columns={'DoseAdditional1': 'ThirdDose'}, inplace=True)
    df.rename(columns={'DoseAdditional2': 'FourthDose'}, inplace=True)

    df['Region'] = df['Region'].str.replace('PL92X', 'PL9', regex=False)
    # uzupełnienie brakującej populacji mazowieckiego
    df.loc[(df['Region'] == 'PL9') & (df['TargetGroup'] == 'ALL'), 'Denominator'] = 4381866
    df.loc[(df['Region'] == 'PL9') & (df['TargetGroup'] == 'Age18_24'), 'Denominator'] = 363843
    df.loc[(df['Region'] == 'PL9') & (df['TargetGroup'] == 'Age25_49'), 'Denominator'] = 2044878
    df.loc[(df['Region'] == 'PL9') & (df['TargetGroup'] == 'Age50_59'), 'Denominator'] = 611862
    df.loc[(df['Region'] == 'PL9') & (df['TargetGroup'] == 'Age60_69'), 'Denominator'] = 706729
    df.loc[(df['Region'] == 'PL9') & (df['TargetGroup'] == 'Age70_79'), 'Denominator'] = 398337
    df.loc[(df['Region'] == 'PL9') & (df['TargetGroup'] == 'Age80+'), 'Denominator'] = 256217
    # df.loc[df['Region'] == 'PL9', 'Denominator'] = 5423000
    df['location'] = df['Region'].map(eurostat_nuts)
    df['location'] = df['location'].replace(np.nan, 'Polska')
    df['location'] = df['location'].apply(lambda x: wojew_cap[x])
    df.columns = ['year_week', '_1', 'denom', 'dostawa', 'eksport', 'dawka_1', 'dawka_2',
                  'dawka_3', 'dawka_4', '_2', '_3', '_4', 'grupa', 'marka', 'population', 'location']
    del df['_1']
    del df['_2']
    del df['_3']
    df['dostawa'].fillna(0, inplace=True)
    df['eksport'].fillna(0, inplace=True)

    df['week2'] = df['year_week'].str.slice(6, 8)
    df['year2'] = df['year_week'].str.slice(0, 4)
    df['year2'] = df['year2'].astype(int)

    def f(par):
        week = par['week2']
        year = par['year2']
        if int(year) == 2020:
            ret_val = time.asctime(time.strptime('{} {} 1'.format(int(year), int(week) - 1), '%Y %W %w'))
        else:
            ret_val = time.asctime(time.strptime('{} {} 1'.format(int(year), int(week)), '%Y %W %w'))
        return ret_val
    df['date'] = df[['week2', 'year2']].apply(f, axis=1)
    df['date'] = df['date'].astype('datetime64[ns]')
    del df['year_week']

    df.to_csv(data_files['ecdc_vacc']['data_fn'])
    print("  wygenerowano", data_files['ecdc_vacc']['data_fn'])

    return df


# Szczepienia ECDC Europa

def load_data_ecdc_vac_eu():
    df0 = pd.read_csv(data_files['ecdc_vacc_eu']['src_fn'])
    df = df0[df0['ReportingCountry'] == df0['Region']].copy()
    df['location'] = df['ReportingCountry'].map(nuts_countries)
    df.columns = ['year_week', '_1', 'denom', 'dostawa', '_3', 'dawka_1', '_4', 'dawka_2', '_5', 'Dawka_3', '_6', 'grupa', 'marka', 'population', 'location']
    del df['_1']
    del df['_3']
    del df['_4']
    del df['_5']
    del df['_6']
    df['dostawa'].fillna(0, inplace=True)

    df['week2'] = df['year_week'].str.slice(6, 8)
    df['year2'] = df['year_week'].str.slice(0, 4)
    df['year2'] = df['year2'].astype(int)

    def f(par):
        week = par['week2']
        year = par['year2']
        if int(year) == 2020:
            ret_val = time.asctime(time.strptime('{} {} 1'.format(int(year), int(week) - 1), '%Y %W %w'))
        else:
            ret_val = time.asctime(time.strptime('{} {} 1'.format(int(year), int(week)), '%Y %W %w'))
        return ret_val
    df['date'] = df[['week2', 'year2']].apply(f, axis=1)
    df['date'] = df['date'].astype('datetime64[ns]')
    del df['year_week']
    df.to_csv(data_files['ecdc_vacc_eu']['data_fn'])
    print("  wygenerowano", data_files['ecdc_vacc_eu']['data_fn'])

    return df


# Dane R(t) dla Polski od Adama Gapińskiego

def load_data_Rt():
    df = pd.read_csv(data_files['poland_rt']['src_fn'])
    columns = list(df.columns)
    df = df[[columns[0], columns[21], columns[24]]].copy()
    # df = df[[columns[0], columns[19], columns[22]]].copy()
    df.columns = ['date', 'reproduction_rate', 'epiestim']
    df = df.loc[1:]
    df.fillna(0, inplace=True)
    df['reproduction_rate'] = df['reproduction_rate'].str.replace(',', '.', regex=False)
    df['reproduction_rate'] = df['reproduction_rate'].astype(float)
    df['epiestim'] = df['epiestim'].str.replace(',', '.', regex=False)
    df['epiestim'] = df['epiestim'].astype(float)
    df = df[df['reproduction_rate'] > 0].copy()
    df = df[df['epiestim'] > 0].copy()
    df['location'] = 'Polska'
    df['date'] = df['date'].str.slice(6, 10)+'-'+df['date'].str.slice(3, 5)+'-'+df['date'].str.slice(0, 2)
    df.dropna(axis='rows', subset=['date'], inplace=True)
    df.sort_values(by=['date'], inplace=True)

    df.to_csv(data_files['poland_rt']['data_fn'], index=False)
    print("  wygenerowano", data_files['poland_rt']['data_fn'])

    return df


# Dane historyczne MZ sprzed 23.11.2020
# nieużywany

def load_data_mz_hist():
    print('>>> Dane historyczne MZ <<<')
    df = pd.read_csv(data_files['mz_hist']['src_fn'], sep=';', encoding='cp1250')
    columns = ['dow',
               'date',
               'new_cases',
               'total_cases',
               'new_deaths',
               'total_deaths',
               'new_recoveries',
               'total_recoveries',
               'total_active',
               'total_quarantine',
               'total_nadzor']
    df.columns = columns
    df = df[columns[1:]].copy()
    df['location'] = 'Polska'
    df['date'] = df['date'].str.slice(6, 10)+'-'+df['date'].str.slice(3, 5)+'-'+df['date'].str.slice(0, 2)
    df.sort_values(by=['date'], inplace=True)

    df.to_csv(data_files['mz_hist']['data_fn'], index=False)
    print("  wygenerowano", data_files['mz_hist']['data_fn'])


# Dane R(t) dla województw

def load_data_Rt_woj():
    print('>>> R(t) dla województw Adama Gapińskiego')
    df = pd.read_csv(data_files['woj_rt']['src_fn'])
    columns = list(df.columns)
    df = df[columns[:116]].copy()
    tab = []
    for i in range(16):
        i1 = 4+i*7
        i2 = 4+(i+1)*7
        sub_columns = [columns[0]] + columns[i1:i2]
        dfx = df[sub_columns].copy()
        dfx = dfx.iloc[1:]
        dfx_columns = list(dfx.columns)
        dfx = dfx[[dfx_columns[0], dfx_columns[7]]]
        dfx.columns = ['date', 'reproduction_rate']
        dfx['location'] = dfx_columns[1]
        dfx.fillna(0, inplace=True)
        tab.append(dfx)

    df = pd.concat([x for x in tab], ignore_index=True, sort=False)

    df['date'] = df['date'].str.slice(6, 10)+'-'+df['date'].str.slice(3, 5)+'-'+df['date'].str.slice(0, 2)
    df.sort_values(by=['location', 'date'], inplace=True)
    df['reproduction_rate'] = df['reproduction_rate'].str.replace(',', '.', regex=False)
    df['reproduction_rate'] = df['reproduction_rate'].astype(float)
    df = df[df['reproduction_rate'] > 0].copy()

    df.to_csv(data_files['woj_rt']['data_fn'], index=False)
    print("  wygenerowano", data_files['woj_rt']['data_fn'])

    return df

# Dane GUS mortality

def read_gus():
    print('>>> Zgony wg GUS <<<')
    def gus1(df, year):
        columns = list(df.columns)
        columns[0] = 'grupa'
        columns[1] = 'region'
        columns[2] = 'nazwa'
        # nazwy kolumn z 5 wiersza
        df.columns = ['grupa', 'region', 'nazwa'] + list(df.iloc[5])[3:]
        df = df[df['grupa'] == 'Ogółem'].copy()
        df = df[df['region'].str.len() == 4].copy()
        del df['grupa']
        del df['region']
        dfx = df.melt(id_vars=['nazwa'], var_name='week', value_name='total')
        dfx['week'] = dfx['week'].str.replace('T', '')
        # zamiana dziwnej spacji
        dfx['total'] = dfx['total'].str.replace("\xa0", "")
        dfx['year'] = str(year)
        dfx = dfx[dfx['week'] < '53'].copy()
        f = lambda x, y: time.asctime(time.strptime('{} {} 1'.format(int(y), int(x)), '%Y %W %w'))
        dfx['date'] = dfx[['week', 'year']].apply(lambda x: f(*x), axis=1)
        dfx['date'] = dfx['date'].astype('datetime64[ns]')
        dfx['year'] = pd.DatetimeIndex(dfx['date']).year
        dfx = dfx[['date', 'year', 'week', 'nazwa', 'total']].copy()
        dfx.columns = ['date', 'year', 'week', 'location', 'total']
        dfx['location'] = dfx['location'].str.replace("Mazowiecki regionalny", "Mazowieckie")
        dfx['location'] = dfx['location'].str.replace("Warszawski stołeczny", "Mazowieckie")
        dfx['total'] = dfx['total'].astype(float)
        dfx['total'] = dfx.groupby(['date', 'location'])['total'].transform('sum')
        dfx.drop_duplicates(inplace=True)
        dfx.sort_values(by=['date'], inplace=True)
        return dfx

    df2020 = gus1(pd.read_csv('data/sources/Zgony_wg_tygodni_w_Polsce_2020.csv'), 2020)
    df2021 = gus1(pd.read_csv('data/sources/Zgony_wg_tygodni_w_Polsce_2021.csv'), 2021)
    df2022 = gus1(pd.read_csv('data/sources/Zgony_wg_tygodni_w_Polsce_2022.csv'), 2022)

    df = pd.DataFrame()
    df = pd.concat([df, df2020], ignore_index=True)
    df = pd.concat([df, df2021], ignore_index=True)
    df = pd.concat([df, df2022], ignore_index=True)
    df['location'] = df['location'].str.lower()
    df['short'] = df['location'].map(eurostat_nuts_reverse)
    df_pl = df.copy()
    df_pl['total_pl'] = df.groupby(['date'])['total'].transform('sum')
    df_pl['total'] = df_pl['total_pl']
    del df_pl['total_pl']
    df_pl = df_pl.drop_duplicates(subset=['date']).copy()
    df_pl['location'] = 'Polska'
    df_pl['short'] = 'PL'
    df = pd.concat([df, df_pl], ignore_index=True)
    df['age'] = 'TOTAL'
    df['date'] = df['date'].astype(str)
    df['year'] = df['date'].str.slice(0, 4)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['date', 'location'], inplace=True)

    df.to_csv(data_files['mortality_gus_woj']['data_fn'], index=False)
    print('  wygenerowano', data_files['mortality_gus_woj']['data_fn'])
    return df

# Dane MZ hosp, icu, deaths
# Dudzińska nieużywane

def load_data_mz_hosp():
    print('>>> Dane z MZ <<<')
    df_d = pd.read_csv('data/sources/mz/deaths.csv')
    df_i = pd.read_csv('data/sources/mz/icu.csv')
    df_h = pd.read_csv('data/sources/mz/hosp.csv')
    df_d.columns = ['date', 'age', 'wojew', 'pers_male', 'perc_female']
    df_h.columns = ['date', 'age', 'wojew', 'pers_male', 'perc_female']
    df_i.columns = ['date', 'age', 'wojew', 'pers_male', 'perc_female']
    df_d['wojew'] = df_d['wojew'].str.lower()
    df_h['wojew'] = df_h['wojew'].str.lower()
    df_i['wojew'] = df_i['wojew'].str.lower()
    df_d['type'] = 'D'
    df_h['type'] = 'H'
    df_i['type'] = 'I'
    df = df_d.append(df_h, ignore_index=True)
    df = df.append(df_i, ignore_index=True)
    df = df[['date', 'wojew', 'age', 'type']]
    bins = [0, 18, 29, 39, 49, 59, 64, 1000]
    # bins = [0, 17, 29, 39, 49, 59, 1000]

    def f(x):
        for i in range(1, len(bins)):
            if x >= bins[i-1] and x < bins[i]:
                return i
        return 0

    df['bin'] = df['age'].apply(f)
    x = df.groupby(['type', 'date'])['bin'].value_counts()
    xx = x.reset_index(level=0)
    xxx = xx.reset_index(level=0)
    xxx.columns = ['date', 'type', 'count']
    df_age = xxx.reset_index(level=0)

    df_age.to_csv(data_files['mz_age_d']['data_fn'], index=False)
    print('  wygenerowano', data_files['mz_age_d']['data_fn'])
    return df


# Dane MZ zgony - szczepienia

def load_data_zgony_szczepienia():
    print('>>> Dane z MZ <<<')
    df = pd.read_csv('data/sources/mz_zgony_szczepienia.csv', sep=';')
    # wynik testu
    _wynik = df.rok_data_wyniku * 100 + df.tydzien_data_wyniku
    df['date_wynik'] = pd.to_datetime(_wynik.astype(str) + '0', format='%Y%W%w')
    # pierwsza dawka
    df['rok_pierwsza_dawka'] = df['rok_pierwsza_dawka'].str.replace('1020', '2020')
    df['rok_pierwsza_dawka'] = df['rok_pierwsza_dawka'].str.replace('NULL', '2000')
    df['tydzien_pierwsza_dawka'] = df['tydzien_pierwsza_dawka'].str.replace('NULL', '1')
    _dawka_1 = df.rok_pierwsza_dawka.astype(int) * 100 + df.tydzien_pierwsza_dawka.astype(int)
    df['date_dawka_1'] = pd.to_datetime(_dawka_1.astype(str) + '0', format='%Y%W%w')
    # pelna dawka
    df['rok_pelna_dawka'] = df['rok_pelna_dawka'].str.replace('NULL', '2000')
    df['tydzien_pelna_dawka'] = df['tydzien_pelna_dawka'].str.replace('NULL', '1')
    _dawka_full = df.rok_pelna_dawka.astype(int) * 100 + df.tydzien_pelna_dawka.astype(int)
    df['date_dawka_full'] = pd.to_datetime(_dawka_full.astype(str) + '0', format='%Y%W%w')
    # data zgonu
    df['rok_zgon'] = df['rok_zgon'].str.replace('NULL', '2000')
    df['tydzien_zgon'] = df['tydzien_zgon'].str.replace('NULL', '1')
    _zgon = df.rok_zgon.astype(int) * 100 + df.tydzien_zgon.astype(int)
    df['date_death'] = pd.to_datetime(_zgon.astype(str) + '0', format='%Y%W%w')
    # konwersja na datę
    df['date_wynik'] = df['date_wynik'].astype(str).str.slice(0, 10)
    df['date_dawka_1'] = df['date_dawka_1'].astype(str).str.slice(0, 10)
    df['date_dawka_full'] = df['date_dawka_full'].astype(str).str.slice(0, 10)
    df['date_death'] = df['date_death'].astype(str).str.slice(0, 10)
    df = df[['wynik', 'producent', 'teryt_woj', 'Liczność', 'date_wynik', 'date_dawka_1',
             'date_dawka_full', 'date_death']].copy()
    df['date_wynik'] = df['date_wynik'].str.replace('2000-01-09', '')
    df['date_dawka_1'] = df['date_dawka_1'].str.replace('2000-01-09', '')
    df['date_dawka_full'] = df['date_dawka_full'].str.replace('2000-01-09', '')
    df['date_death'] = df['date_death'].str.replace('2000-01-09', '')
    df.to_csv('data/last/last_mz_zgony_szczepienia.csv', index=False)
    return df


# Dane MZ basiw zgony w przedziałach wiekowych, status szczepienia

def load_data_basiw_d():
    print('>>> Dane o zgonach z BASIW <<<')
    df = pd.read_csv(data_files['basiw_d']['src_fn'], sep=';', encoding='cp1250')
    df_teryt = pd.read_csv('data/dict/teryt.csv')
    df_teryt = df_teryt[df_teryt['NAZWA_DOD'].isin(['powiat', 'miasto na prawach powiatu',
                                                    'miasto stołeczne, na prawach powiatu'])].copy()
    df_teryt['teryt_woj'] = df_teryt['teryt_woj'].astype(int)
    df_teryt['teryt_pow'] = df_teryt['teryt_pow'].astype(int)
    df.columns = ['date', 'teryt_w', 'teryt_p', 'sex', 'age_group', 'ch_wspol',
                  'marka', 'dawka', 'delta', 'd_all']
    df['wpelni'] = df['dawka'].map({'pelna_dawka': 'T', 'przypominajaca': 'T'})
    df['wpelni'] = df['wpelni'].map({np.nan: 'N', 'T': 'T'})
    df['bin_mz'] = df['age_group'].map(rev_bin_mz)
    df['bin_mz'] = df['bin_mz'].fillna(0).astype(int)
    df['date'] = df['date'].str.slice(6, 10) + '-'+df['date'].str.slice(3, 5)+'-' + df['date'].str.slice(0, 2)
    df = pd.merge(df, df_teryt[['teryt_pow', 'powiat', 'wojew']], how='left', left_on=['teryt_p'], right_on=['teryt_pow']).copy()
    df['d_all'] = df['d_all'].astype(np.float64)

    df.to_csv(data_files['basiw_d']['data_fn'], index=False)
    print('  wygenerowano', data_files['basiw_d']['data_fn'])
    return df

# Dane MZ basiw infekcje w przedziałach wiekowych, status szczepienia

def load_data_basiw_i():
    print('>>> Dane o infekcjach z BASIW <<<')
    df = pd.read_csv(data_files['basiw_i']['src_fn'], sep=';', encoding='cp1250')
    df_teryt = pd.read_csv('data/dict/teryt.csv')
    df_teryt = df_teryt[df_teryt['NAZWA_DOD'].isin(['powiat', 'miasto na prawach powiatu',
                                                    'miasto stołeczne, na prawach powiatu'])].copy()
    df_teryt['teryt_woj'] = df_teryt['teryt_woj'].astype(int)
    df_teryt['teryt_pow'] = df_teryt['teryt_pow'].astype(int)
    df.columns = ['date', 'teryt_w', 'teryt_p', 'sex', 'age', 'age_group',
                  'marka', 'dawka', 'nr_inf', 'i_all']
    df['wpelni'] = df['dawka'].map({'pelna_dawka': 'T', 'przypominajaca': 'T'})
    df['wpelni'] = df['wpelni'].map({np.nan: 'N', 'T': 'T'})
    df = pd.merge(df, df_teryt[['teryt_pow', 'powiat', 'wojew']], how='left', left_on=['teryt_p'], right_on=['teryt_pow']).copy()
    df['i_all'] = df['i_all'].astype(np.float64)
    df['age'].fillna(-1, inplace=True)
    df.to_csv(data_files['basiw_i']['data_raw_fn'], index=False)

    # dfxs['bin'] = pd.cut(dfxs[chart_type], bins=bins, labels=bin_labels)

    bins = constants.age_bins_0
    bin_labels = constants.age_bins['bin']
    df['bin'] = pd.cut(df['age'], bins=bins, labels=bin_labels)
    df['bin'].fillna(0, inplace=True)

    bins = constants.age_bins_5
    bin_labels = constants.age_bins['bin5']
    df['bin5'] = pd.cut(df['age'], bins=bins, labels=bin_labels)
    df['bin5'].fillna(0, inplace=True)

    bins = constants.age_bins_10
    bin_labels = constants.age_bins['bin10']
    df['bin10'] = pd.cut(df['age'], bins=bins, labels=bin_labels)
    df['bin10'].fillna(0, inplace=True)

    bins = constants.age_bins_ecdc
    bin_labels = constants.age_bins['bin_ecdc']
    df['bin_ecdc'] = pd.cut(df['age'], bins=bins, labels=bin_labels)
    df['bin_ecdc'].fillna(0, inplace=True)

    df.to_csv(data_files['basiw_i']['data_fn'], index=False)
    print('  wygenerowano', data_files['basiw_i']['data_fn'])
    return df

# Dane MZ age deaths codzienne

def load_data_mz_api_age_d():
    return
    print('>>> Dane z API MZ deaths age <<<')
    df = pd.read_csv(data_files['mz_api_age_d']['src_fn'], sep=';')
    # timestamp -> data
    def f(x):
        ret_val = datetime.datetime.utcfromtimestamp(int(str(x)[:-3])).strftime('%Y-%m-%d %H:%M:%S')
        return ret_val
    df['date'] = df['Data'].apply(f)
    df['date'] = df['date'].astype(str).str.slice(0, 10)
    del df['Unnamed: 18']
    df.to_csv(data_files['mz_api_age_d']['data_fn'], index=False)
    save_github(data_files['mz_api_age_d']['data_fn'], str(dt.now())[:10] + ' MZ API zgony')
    print('  wygenerowano', data_files['mz_api_age_d']['data_fn'])
    return df

# Dane MZ reinfekcje wśród zaszczepionych

def load_data_reinf():
    print('>>> Dane o reinfekcjach <<<')
    df = pd.read_csv(data_files['reinf']['src_fn'], dtype=object, sep=';')
    # df = pd.read_csv(data_files['reinf']['src_fn'], sep=';')
    del df['teryt_wojewodztwo']

    # zamiana na numeric
    df['rok_pierwsza_dawka'] = df['rok_pierwsza_dawka'].str.replace('210', '2021')
    df['licznosc'] = df['licznosc'].str.replace(' ', '')
    df['licznosc'] = df['licznosc'].astype(float)
    df['rok_data_wyniku'].fillna('0', inplace=True)
    df['rok_data_wyniku'] = df['rok_data_wyniku'].astype(int)
    df['tydzien_data_wyniku'].fillna('0', inplace=True)
    df['tydzien_data_wyniku'] = df['tydzien_data_wyniku'].astype(int)
    df['rok_pierwsza_dawka'].fillna('0', inplace=True)
    df['rok_pierwsza_dawka'] = df['rok_pierwsza_dawka'].astype(int)
    df['tydzien_pierwsza_dawka'].fillna('0', inplace=True)
    df['tydzien_pierwsza_dawka'] = df['tydzien_pierwsza_dawka'].astype(int)
    df['rok_druga_dawka'].fillna('0', inplace=True)
    df['rok_druga_dawka'] = df['rok_druga_dawka'].astype(int)
    df['tydzien_druga_dawka'].fillna('0', inplace=True)
    df['tydzien_druga_dawka'] = df['tydzien_druga_dawka'].astype(int)

    # korekta 53 i 53 tygodnia 2020
    df.loc[(df['rok_data_wyniku'] == 2020) & (df['tydzien_data_wyniku'] == 52), 'tydzien_data_wyniku'] = 51
    df.loc[(df['rok_data_wyniku'] == 2020) & (df['tydzien_data_wyniku'] == 53), 'tydzien_data_wyniku'] = 52
    df.loc[(df['rok_pierwsza_dawka'] == 2020) & (df['tydzien_pierwsza_dawka'] == 52), 'tydzien_pierwsza_dawka'] = 51
    df.loc[(df['rok_pierwsza_dawka'] == 2020) & (df['tydzien_pierwsza_dawka'] == 53), 'tydzien_pierwsza_dawka'] = 52
    df.loc[(df['rok_druga_dawka'] == 2020) & (df['tydzien_druga_dawka'] == 52), 'tydzien_druga_dawka'] = 51
    df.loc[(df['rok_druga_dawka'] == 2020) & (df['tydzien_druga_dawka'] == 53), 'tydzien_druga_dawka'] = 52

    # skasowanie nieprawidłowych dat badania
    df = df[~((df['rok_data_wyniku'] == 2021) & (df['tydzien_data_wyniku'] > 39))].copy()
    df = df[~(df['rok_data_wyniku'] > 2021)].copy()

    # zamiana nan na 0 w polach numerycznych
    df['licznosc'].fillna(0, inplace=True)

    # Zamiana par rok/tydzień na daty
    dates = df['rok_data_wyniku'] * 100 + df['tydzien_data_wyniku']
    df['datetime_wynik'] = pd.to_datetime(dates.astype(str) + '0', format='%Y%W%w', errors='coerce')
    dates = df['rok_pierwsza_dawka'] * 100 + df['tydzien_pierwsza_dawka']
    df['datetime_1_dawka'] = pd.to_datetime(dates.astype(str) + '0', format='%Y%W%w', errors='coerce')
    dates = df['rok_druga_dawka'] * 100 + df['tydzien_druga_dawka']
    df['datetime_2_dawka'] = pd.to_datetime(dates.astype(str) + '0', format='%Y%W%w', errors='coerce')

    # Daty niekonwertowalne -> daty puste
    df['date_wynik'] = df['datetime_wynik'].astype(str)
    df['date_1_dawka'] = df['datetime_1_dawka'].astype(str)
    df['date_2_dawka'] = df['datetime_2_dawka'].astype(str)
    df['date_wynik'] = df['date_wynik'].str.replace('NaT', '')
    df['date_1_dawka'] = df['date_1_dawka'].str.replace('NaT', '')
    df['date_2_dawka'] = df['date_2_dawka'].str.replace('NaT', '')

    # Odstępy między datami w tygodniach
    df['tyg_1_dawka'] = (df['datetime_wynik'] - df['datetime_1_dawka']).dt.days / 7
    df['tyg_2_dawka'] = (df['datetime_wynik'] - df['datetime_2_dawka']).dt.days / 7

    # Del niepotrzebne pola
    del df['rok_data_wyniku']
    del df['tydzien_data_wyniku']
    del df['rok_pierwsza_dawka']
    del df['tydzien_pierwsza_dawka']
    del df['rok_druga_dawka']
    del df['tydzien_druga_dawka']
    del df['datetime_wynik']
    del df['datetime_1_dawka']
    del df['datetime_2_dawka']
    # df.rename(columns={'licznosc': 'ile'}, inplace=True)
    df.rename(columns={'test_wynik': 'wynik'}, inplace=True)
    df['ile'] = df['licznosc'].astype(float)

    # test
    dfx = df.loc[~df['producent'].isin(['Pfizer', 'Johnson&Johnson', 'Moderna', 'Astra Zeneca'])].copy()

    #######

    df['tyg_1_dawka'].fillna(-1, inplace=True)
    df['tyg_1_dawka'].astype(int)
    df.loc[df['tyg_1_dawka'] < 0, 'tyg_1_dawka'] = -1
    df['tyg_2_dawka'].fillna(-1, inplace=True)
    df['tyg_2_dawka'].astype(int)
    df.loc[df['tyg_2_dawka'] < 0, 'tyg_2_dawka'] = -1

    df.loc[~df['producent'].isin(['Pfizer', 'Johnson&Johnson', 'Moderna', 'Astra Zeneca']), 'producent'] = 'Niezaszczepiony'

    # aktywny: ['brak szczepienia', '1 dawka', 'wpelni']

    df['aktywny'] = 'brak szczepienia'
    df.loc[(df['producent'] == 'Johnson&Johnson') & (df['tyg_1_dawka'] >= 0), 'aktywny'] = 'w pelni'
    df.loc[(df['producent'].isin(['Pfizer', 'Moderna', 'Astra Zeneca'])) &
           (df['tyg_2_dawka'] >= 0), 'aktywny'] = 'wpelni'
    df.loc[(df['producent'].isin(['Pfizer', 'Moderna', 'Astra Zeneca'])) &
           (df['tyg_1_dawka'] >= 0) & (df['tyg_2_dawka'] == -1), 'aktywny'] = '1 dawka'

    # status:
    # negatywny
    # pozytywny tylko 1 dawka
    # pozytywny po obu dawkach
    # pozytywny między dawkami

    df.reset_index(drop=True, inplace=True)

    df['status'] = 'negatywny'
    df.loc[(df['wynik'] == 'POZYTYWNY') &
           (df['tyg_1_dawka'] >= 0) &
           (df['tyg_2_dawka'] == -1), 'status'] = '1d'
    df.loc[(df['wynik'] == 'POZYTYWNY') &
           (df['tyg_1_dawka'] >= 0) &
           (df['date_2_dawka'] != ''), 'status'] = '12d'
    df.loc[(df['wynik'] == 'POZYTYWNY') & (df['tyg_2_dawka'] != -1), 'status'] = '2d'

    df.to_csv(data_files['reinf']['data_fn'], index=False)

    # ['wynik', 'producent', 'teryt', 'ile', 'date_wynik', 'date_1_dawka',
    #  'date_2_dawka', 'tyg_1_dawka', 'tyg_2_dawka']

    print('  wygenerowano', data_files['reinf']['data_fn'])
    return df

# Dane MZ reinfekcje  - statystyki

def stat_reinf():
    # return
    print('>>> Statystyki reinfekcji <<<')
    df = pd.read_csv(data_files['reinf']['data_fn'])
    df.loc[df['aktywny'] == 'nieaktywny', 'producent'] = 'Niezaszczepiony'
    df.sort_values(by=['producent', 'wynik'], inplace=True)
    df['producent_wynik_count'] = df.groupby(['producent', 'wynik'])['ile'].transform('sum')
    df['count'] = df['ile'].sum()
    df['producent_count'] = df.groupby(['producent'])['ile'].transform('sum')
    df['producent_wynik_percent'] = df['producent_wynik_count'] / df['producent_count']
    df = df.drop_duplicates(subset=['producent', 'wynik']).copy()
    df_stat = df[['producent', 'wynik', 'count', 'producent_count', 'producent_wynik_count', 'producent_wynik_percent']]

    df = pd.read_csv(data_files['reinf']['data_fn'])
    df.sort_values(by=['producent', 'wynik'], inplace=True)
    df['producent_aktywny_count'] = df.groupby(['producent', 'aktywny'])['ile'].transform('sum')
    df['count'] = df['ile'].sum()
    df['producent_count'] = df.groupby(['producent'])['ile'].transform('sum')
    df['producent_aktywny_percent'] = df['producent_aktywny_count'] / df['producent_count']
    df = df.drop_duplicates(subset=['producent', 'aktywny']).copy()
    df_stat2 = df[['producent', 'aktywny', 'count', 'producent_count', 'producent_aktywny_count', 'producent_aktywny_percent']]

    # ['wynik', 'producent', 'teryt', 'ile', 'date_wynik', 'date_1_dawka',
    #  'date_2_dawka', 'tyg_1_dawka', 'tyg_2_dawka']

    return df

# Dane MZ age vacc codzienne

def load_data_mz_api_age_v():
    print('>>> Dane z API MZ vacc age <<<')
    df = pd.read_csv(data_files['mz_api_age_v']['src_fn'], sep=';')
    # timestamp -> data
    def f(x):
        ret_val = datetime.datetime.utcfromtimestamp(int(str(x)[:-3])).strftime('%Y-%m-%d %H:%M:%S')
        return ret_val
    df['date'] = df['Data'].apply(f)
    df['date'] = df['date'].astype(str).str.slice(0, 10)
    df['SZCZEPIENIA_SUMA'].fillna(0, inplace=True)
    del df['Unnamed: 26']
    del df['OBJECTID']
    del df['Data']
    del df['DATA_SHOW']
    df = df.drop_duplicates(subset=['date', 'SZCZEPIENIA_SUMA']).copy()
    df.to_csv(data_files['mz_api_age_v']['data_fn'], index=False)
    save_github(data_files['mz_api_age_v']['data_fn'], str(dt.now())[:10] + ' MZ API szczepienia')
    print('  wygenerowano', data_files['mz_api_age_v']['data_fn'])
    return df

# Dane MZ vacc powiaty

def load_data_mz_api_vacc_powiaty():
    print('>>> Dane z API MZ vacc powiaty <<<')
    df = pd.read_csv(data_files['mz_api_vacc_powiaty']['src_fn'], sep=';')
    # timestamp -> data
    def f(x):
        ret_val = datetime.datetime.utcfromtimestamp(int(str(x)[:-3])).strftime('%Y-%m-%d %H:%M:%S')
        return ret_val
    df['date'] = df['Data'].apply(f)
    df['date'] = df['date'].astype(str).str.slice(0, 10)
    del df['Unnamed: 16']
    del df['OBJECTID']
    del df['Data']

    # normalizacja nazw powiatów i województw

    df = df[['JPT_NAZWA_', 'JPT_KJ_I_1', 'JPT_KJ_I_2', 'POPULACJA', 'SZCZEPIENIA_SUMA', 'SZCZEPIENIA_DZIENNIE',
                   'DAWKA_2_SUMA', 'DAWKA_2_DZIENNIE', 'nast_7_dni_agg', 'nast_30_dni_agg', 'date']].copy()
    df.columns = ['location0', 'JPT1', 'JPT2', 'population', 'total_vacc', 'new_vacc',
                     'total_vacc_2', 'new_vacc_2', 'slots_7', 'slots_30', 'date']

    df['kod'] = df['JPT1'].astype(str)
    df['woj_kod'] = df['JPT2'].str.slice(0, 3)

    # nazwy powiatów

    df_powiaty = pd.read_csv('data/dict/sl_powiaty.csv', sep=',')
    df_powiaty.columns = ['location', 'kod', 'population']
    df_powiaty['kod'] = df_powiaty['kod'].astype(str)
    df = pd.merge(df, df_powiaty[['location', 'kod']], how='left', left_on=['kod'], right_on=['kod'])
    df.sort_values(by=['location'], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # nazwy województw

    df_wojew = pd.read_csv('data/last/last_mz_api_vacc_wojew.csv', sep=',')
    df_wojew = df_wojew[['jpt_nazwa_', 'JPT_KJ_I_2']]
    df_wojew.columns = ['wojew', 'JPT_KJ_I_2']
    df = pd.merge(df, df_wojew, how='left', left_on=['woj_kod'], right_on=['JPT_KJ_I_2'])
    df.rename(columns={'location': 'powiat'}, inplace=True)
    for p in powiat_translation_table:
        df.loc[(df.powiat == p['location']) & (df.wojew == p['wojew']), 'powiat'] = p['new_location']
    df.sort_values(by=['powiat'], inplace=True)

    del df['location0']
    del df['JPT1']
    del df['JPT2']
    del df['kod']
    del df['woj_kod']
    del df['JPT_KJ_I_2']
    df.to_csv(data_files['mz_api_vacc_powiaty']['data_fn'], index=False)
    save_github(data_files['mz_api_vacc_powiaty']['data_fn'], str(dt.now())[:10] + ' MZ API szczepienia powiaty')
    print('  wygenerowano', data_files['mz_api_vacc_powiaty']['data_fn'])
    return df

# Dane MZ punkty szczepień

def load_data_mz_psz():
    print('>>> Dane MZ punkty szczepień <<<')
    df = pd.read_csv(data_files['mz_psz']['src_fn'], sep=',')
    df.rename(columns={'voivodeship': 'wojew'}, inplace=True)
    df.rename(columns={'county': 'powiat'}, inplace=True)
    df['wojew'] = df['wojew'].str.lower()
    df.sort_values(by=['powiat'], inplace=True)
    for p in powiat_translation_table:
        df.loc[(df.powiat == p['location']) & (df.wojew == p['wojew']), 'powiat'] = p['new_location']

    df.to_csv(data_files['mz_psz']['data_fn'], index=False)
    # save_github(data_files['mz_psz']['data_fn'], str(dt.now())[:10] + ' MZ API szczepienia powiaty')
    print('  wygenerowano', data_files['mz_psz']['data_fn'])
    df_g = df.copy()
    df_g['title'] = df_g.facilityName+' '+df_g.zipCode+' '+df_g.address+' '+df_g.place
    df_g = df_g[['lon', 'lat', 'title']]
    df_g.iloc[:2000].to_csv('data/psz_google_1.csv', index=False)
    df_g.iloc[2000:4000].to_csv('data/psz_google_2.csv', index=False)
    df_g.iloc[4000:6000].to_csv('data/psz_google_3.csv', index=False)
    df_g.iloc[6000:].to_csv('data/psz_google_4.csv', index=False)
    return df

# Dane MZ vacc gminy

def load_data_mz_api_vacc_gminy():
    print('>>> Dane z API MZ vacc gminy <<<')
    df = pd.read_csv(data_files['mz_api_vacc_gminy']['src_fn'], sep=',')
    # ['wszystkie gminy', 'gminy miejskie', 'gminy miejsko-wiejskie', 'miasta', 'do 20 tys.', '20-50 tys.',
    #  '50-100 tys', '> 100 tys.']
    df['gmina_nazwa'].fillna('', inplace=True)
    df['gmina_nazwa'] = df['gmina_nazwa'].str.replace('gm.w ', 'gm.w. ', regex=False)
    df['gmina_nazwa'] = df.gmina_nazwa.str.replace(' \(.*\)', '', regex=True)

    df_jpt = pd.read_csv('data/dict/jpt.csv', sep=',')

    df.sort_values(by=['gmina_nazwa', 'powiat_nazwa'], inplace=True)
    df_jpt.sort_values(by=['nazwa', 'powiat'], inplace=True)

    df = pd.merge(df, df_jpt[['nazwa', 'wojew', 'powiat', 'JPT', 'razem']], how='left',
                  left_on=['gmina_nazwa', 'powiat_nazwa', 'liczba_ludnosci'],
                  right_on=['nazwa', 'powiat', 'razem'])
    # df = df.drop_duplicates(subset=['gmina_nazwa', 'powiat_nazwa']).copy()
    # del df['powiat']
    def f(r):
        # wielkość
        if r['liczba_ludnosci'] < 20000:
            r['typ_p'] = '<20k'
        elif 20000 <= r['liczba_ludnosci'] <= 50000:
            r['typ_p'] = '20k-50k'
        elif 50000 <= r['liczba_ludnosci'] <= 100000:
            r['typ_p'] = '50k-100k'
        elif r['liczba_ludnosci'] >= 100000:
            r['typ_p'] = '>100k'
        else:
            r['typ_p'] = 'unk.'
        return r
    df = df.apply(f, axis=1)
    del df['powiat']
    df.columns = ['percent full', '12-19 lat', '20-39 lat', '40-59 lat', '60-69 lat', '70+ lat',
                  'gmina', 'powiat', 'grupa_wiekowa', 'population', 'Województwo',
                  'total_1d', 'total_full', 'dynamika1', 'nazwa', 'wojew', 'JPT', 'razem', 'typ_p']
    df = df[['JPT', 'powiat', '70+ lat', '12-19 lat', '20-39 lat', '40-59 lat', '60-69 lat', 'population',
             'total_full', 'percent full', 'total_1d', 'nazwa', 'wojew', 'typ_p']].copy()

    for p in powiat_translation_table:
        df.loc[(df.powiat == p['location']) & (df.wojew == p['wojew']), 'powiat'] = p['new_location']

    df.dropna(axis='rows', subset=['JPT'], inplace=True)
    # df['JPT'] = 'JPT_' + df['JPT']

    df.to_csv(data_files['mz_api_vacc_gminy']['data_fn'], index=False)
    save_github(data_files['mz_api_vacc_gminy']['data_fn'], str(dt.now())[:10] + ' MZ API szczepienia gminy')
    print('  wygenerowano', data_files['mz_api_vacc_gminy']['data_fn'])
    return df

# Dane MZ vacc gminy

def load_data_mz_api_vacc_gminy_0():
    print('>>> Dane z API MZ vacc gminy <<<')
    df = pd.read_csv(data_files['mz_api_vacc_gminy']['src_fn'], sep=',')
    # ['wszystkie gminy', 'gminy miejskie', 'gminy miejsko-wiejskie', 'miasta', 'do 20 tys.', '20-50 tys.',
    #  '50-100 tys', '> 100 tys.']
    df['gmina_nazwa'].fillna('', inplace=True)
    df['gmina_nazwa'] = df['gmina_nazwa'].str.replace('gm.w ', 'gm.w. ', regex=False)
    def f(r):
        # wielkość

        if r['liczba_ludnosci'] < 20000:
            r['typ_p'] = '<20k'
        elif 20000 <= r['liczba_ludnosci'] <= 50000:
            r['typ_p'] = '20k-50k'
        elif 50000 <= r['liczba_ludnosci'] <= 100000:
            r['typ_p'] = '50k-100k'
        elif r['liczba_ludnosci'] >= 100000:
            r['typ_p'] = '>100k'
        else:
            r['typ_p'] = 'unk.'

        return r
    df = df.apply(f, axis=1)

    df_terc = pd.read_csv('data/dict/TERC_Urzedowy_2020-12-31.csv', sep=';')
    df_terc['POW'].fillna(0, inplace=True)
    df_terc['GMI'].fillna(0, inplace=True)
    df_terc['RODZ'].fillna(0, inplace=True)
    df_terc['WOJ'] = df_terc['WOJ'].astype(int)
    df_terc['POW'] = df_terc['POW'].astype(int)
    df_terc['GMI'] = df_terc['GMI'].astype(int)
    df_terc['RODZ'] = df_terc['RODZ'].astype(int)
    df_terc.sort_values(by=['WOJ', 'NAZWA'], inplace=True)
    df_terc['JPT'] = df_terc['WOJ'].map('{:0>2}'.format)+df_terc['POW'].map('{:0>2}'.format)+df_terc['GMI'].map('{:0>2}'.format)+df_terc['RODZ'].map('{:0>1}'.format)
    df_powiat = df_terc[df_terc['NAZWA_DOD'] == 'powiat'].copy()
    df_terc = pd.merge(df_terc, df_powiat[['NAZWA', 'WOJ', 'POW']], how='left', left_on=['WOJ', 'POW'], right_on=['WOJ', 'POW'])
    df_terc.rename(columns={'NAZWA_x': 'NAZWA'}, inplace=True)
    df_terc.rename(columns={'NAZWA_y': 'powiat'}, inplace=True)
    df_terc['powiat'].fillna('', inplace=True)
    df_terc['nazwa'] = df_terc['NAZWA']

    # ['wszystkie gminy', 'gminy miejskie', 'gminy miejsko-wiejskie', 'miasta', 'do 20 tys.', '20-50 tys.',
    #  '50-100 tys', '> 100 tys.']
    def f(r):
        r['wojew'] = terc_woj[r['WOJ']]
        if r['powiat'] == '':
            r['powiat'] = r['NAZWA']
        return r
    df_terc = df_terc.apply(f, axis=1)
    df_terc['nazwa'].fillna('', inplace=True)

    df.sort_values(by=['gmina_nazwa', 'powiat_nazwa'], inplace=True)

    df = pd.merge(df, df_terc[['nazwa', 'wojew', 'powiat', 'JPT']], how='left', left_on=['gmina_nazwa', 'powiat_nazwa'], right_on=['nazwa', 'powiat'])
    df = df.drop_duplicates(subset=['gmina_nazwa', 'powiat_nazwa']).copy()
    del df['powiat']
    df.columns = ['percent full', '12-19 lat', '20-39 lat', '40-59 lat', '60-69 lat', '70+ lat',
                  'gmina', 'powiat', 'grupa_wiekowa', 'population', 'Województwo',
                  'total_1d', 'total_full', 'typ_p', 'nazwa', 'wojew', 'JPT']
    df = df[['JPT', 'powiat', '70+ lat', '12-19 lat', '20-39 lat', '40-59 lat', '60-69 lat', 'population',
             'total_full', 'percent full', 'total_1d', 'nazwa', 'wojew', 'typ_p']].copy()

    for p in powiat_translation_table:
        df.loc[(df.powiat == p['location']) & (df.wojew == p['wojew']), 'powiat'] = p['new_location']

    df.dropna(axis='rows', subset=['JPT'], inplace=True)
    df['JPT'] = 'JPT_' + df['JPT']

    df.to_csv(data_files['mz_api_vacc_gminy']['data_fn'], index=False)
    save_github(data_files['mz_api_vacc_gminy']['data_fn'], str(dt.now())[:10] + ' MZ API szczepienia gminy')
    print('  wygenerowano', data_files['mz_api_vacc_gminy']['data_fn'])
    return df

# Dane MZ vacc wojew

def load_data_mz_api_vacc_wojew():
    print('>>> Dane z API MZ vacc wojew <<<')
    df = pd.read_csv(data_files['mz_api_vacc_wojew']['src_fn'], sep=';')
    # timestamp -> data
    # def f(x):
    #     ret_val = datetime.datetime.utcfromtimestamp(int(str(x)[:-3])).strftime('%Y-%m-%d %H:%M:%S')
    #     return ret_val
    # df['date'] = df['Data'].apply(f)
    # df['date'] = df['date'].astype(str).str.slice(0, 10)
    del df['Unnamed: 13']
    del df['Shape__Area']
    del df['Shape__Length']
    del df['Shape__Area_2']
    del df['Shape__Length_2']
    del df['Nazwa_UA']
    del df['OBJECTID']
    # del df['Data']
    # del df['DATA_SHOW']
    df.to_csv(data_files['mz_api_vacc_wojew']['data_fn'], index=False)
    save_github(data_files['mz_api_vacc_wojew']['data_fn'], str(dt.now())[:10] + ' MZ API szczepienia województwa')
    print('  wygenerowano', data_files['mz_api_vacc_wojew']['data_fn'])
    return df

# Dane Eurostat mortality age (kraje Europy, wiek co 10 lat)

def load_data_mortality_eurostat():
    print('>>> mortality Eurostat pełny zbiór <<<')
    df = pd.read_csv(data_files['mortality_eurostat']['src_fn'])
    df_gus = pd.read_csv(data_files['mortality_gus_woj']['data_fn'])
    df = df[df['sex'] == 'T'].copy()
    col = list(df.columns)[3]
    df.rename(columns={col: 'short'}, inplace=True)
    del df['sex']
    del df['unit']

    # tylko Polska i województwa

    df = df[df['short'].str.startswith('PL')].copy()
    dfx = df.melt(id_vars=['age', 'short'], var_name='week', value_name='total')

    dfx['total'].fillna(0, inplace=True)
    dfx['week2'] = dfx['week'].str.slice(5, 7)
    dfx['year2'] = dfx['week'].str.slice(0, 4)
    dfx['year2'] = dfx['year2'].astype(int)
    dfx = dfx[~(dfx['week2'] > '52')].copy()
    # dfx = dfx[dfx['year2'] > 2010].copy()
    f = lambda x, y: time.asctime(time.strptime('{} {} 1'.format(int(y), int(x)), '%Y %W %w'))
    dfx['date'] = dfx[['week2', 'year2']].apply(lambda x: f(*x), axis=1)
    dfx['date'] = dfx['date'].astype('datetime64[ns]')
    dfx['year'] = pd.DatetimeIndex(dfx['date']).year
    del dfx['week']
    dfx.rename(columns={'week2': 'week'}, inplace=True)
    del dfx['year2']
    dfx['week'] = dfx['week'].astype(int)
    dfx = dfx[['date', 'year', 'week', 'total', 'age', 'short']].copy()

    # województwa

    dfx['location'] = dfx['short'].map(eurostat_nuts)
    dfx.dropna(subset=['location'], inplace=True)
    # ############
    # # tylko do 2021, bez 2022
    dfx = dfx[dfx['year'] < 2022].copy()
    dfx = dfx[dfx['age'] == 'TOTAL'].copy()
    # # dodanie brakujących danych z GUS
    df_gus['date'] = pd.to_datetime(df_gus['date'])
    set_df = set(dfx['date'].unique())
    set_gus = set(df_gus['date'].unique())
    diff = list(set_gus - set_df)
    df_diff = df_gus[df_gus['date'].isin(diff)]
    dfx = pd.concat([dfx, df_diff], ignore_index=True)
    # ############
    # dfx.sort_values(by=['date'], inplace=True)
    dfx.to_csv(data_files['mortality_eurostat']['data_fn'], index=False)

    print("  wygenerowano", data_files['mortality_eurostat']['data_fn'])

    return dfx


def load_data_rcb():
    print('>>> RCB od 24.12.2020 <<<')
    fn_woj_total = 'data/sources/rcb/wojewodztwa.csv'
    fn_pow_total = 'data/sources/rcb/powiaty.csv'
    fn_xls_woj = 'data/sources/rcb/rcb_woj.xlsx'
    fn_xls_pow = 'data/sources/rcb/rcb_pow.xlsx'

    # utworzenie tabeli woj_total

    import os
    # from app import server
    total_woj = pd.DataFrame()
    total_pow = pd.DataFrame()
    files = os.listdir("data/sources/rcb")
    dates = []
    woj_sum = 0
    woj_woj = 0
    print('  województwa')
    woj_columns_old = ['location', 'new_cases', 'd_0',
                   'new_deaths', 'new_deaths_c', 'new_deaths_nc', 'new_tests_poz',
                   'new_recoveries', 'total_quarantine', 'new_tests', 'new_tests_plus',
                   'new_tests_minus', 'new_tests_other']
    woj_columns_new = ['location', 'd_3', 'new_reinf', 'new_cases',
                   'd_0', 'd_1', 'd_2',
                   'new_deaths', 'new_deaths_c', 'new_deaths_nc', 'new_tests_poz',
                   'new_recoveries', 'total_quarantine', 'new_tests', 'new_tests_plus',
                   'new_tests_minus', 'new_tests_other']
    data_test_i = []
    for file in sorted(files):
        if "rap_rcb_woj_eksport" in file:
            filepath = os.getcwd() + '/data/sources/rcb/' + file
            filedate = file[:8]
            if filedate >= '20201124' and filedate <= '20201223':
                encoding = 'utf8'
                filetype = 11
            else:
                encoding = 'cp1250'
                filetype = 12
            woj_columns = woj_columns_old
            if filedate >= '20220208':
                woj_columns = woj_columns_new
            if filedate == '20210503':
                encoding = 'utf8'
            df_woj = pd.read_csv(filepath, sep=';', encoding=encoding)
            if filetype == 11:
                df_woj.insert(loc=6, column='new_recoveries', value=[0]*len(df_woj))
            if len(df_woj.columns) > len(woj_columns + ['teryt']):
                df_woj.columns = woj_columns + ['teryt', 'data']
            else:
                df_woj.columns = woj_columns + ['teryt']
            newdate = file[:4] + '-' + file[4:6] + '-' + file[6:8]
            if newdate not in dates:
                dates.append(newdate)
                df_woj['date'] = newdate
                fields = ['date'] + woj_columns
                df_woj = df_woj[fields].copy()
                df_woj.at[0, 'location'] = 'Polska'
                n1 = df_woj.iloc[0]['new_cases']
                n2 = df_woj.new_cases.sum() - n1
                woj_sum += n1
                woj_woj += n2
                if n1 != n2:
                    data_test_i.append([newdate, n1, n2, n1-n2])
                if len(total_woj) == 0:
                    total_woj = df_woj.copy()
                else:
                    total_woj = pd.concat([total_woj, df_woj], ignore_index=True)
    data_test_i.append(['RAZEM', woj_sum, woj_woj, woj_sum-woj_woj])
    total_woj.sort_values(by=['date', 'location'], inplace=True)
    total_woj.reset_index(drop=True, inplace=True)
    total_woj.fillna(0, inplace=True)
    # total_woj['new_deaths'].fillna(0, inplace=True)
    for col in woj_columns[1:]:
        total_woj[col] = total_woj[col].astype(np.float64)

    total_woj['location'] = total_woj['location'].apply(lambda x: wojew_cap[x])
    total_woj.drop_duplicates(subset=['date', 'location'])
    total_woj.sort_values(by=['location', 'date'], inplace=True)
    total_woj.reset_index(drop=True, inplace=True)
    total_woj.to_csv(fn_woj_total, index=False)
    print('  zapisano', fn_woj_total)

    # arkusz testów

    cols = total_woj.columns
    tw1 = total_woj[total_woj['location'] != 'Polska'].groupby('date').sum()
    tw1['location'] = 'Suma z województw'
    tw1['date'] = tw1.index
    tw1.reset_index(drop=True, inplace=True)
    tw = pd.concat([tw1, total_woj[total_woj['location'] == 'Polska']], ignore_index=True)
    tw.reset_index(drop=True, inplace=True)
    tw.sort_values(by=['date', 'location'], inplace=True)
    tw = tw[cols].copy()
    del tw['d_0']
    del tw['d_1']
    del tw['d_2']
    del tw['d_3']
    df_test_i = pd.DataFrame(data_test_i, columns=['data', 'z arkusza', 'z województw', 'różnica'])

    del total_woj['d_0']
    del total_woj['d_1']
    del total_woj['d_2']
    del total_woj['d_3']
    with pd.ExcelWriter(fn_xls_woj) as writer:
        total_woj.to_excel(writer, sheet_name='arkusz województw', index=False)
        tw.to_excel(writer, sheet_name='porównanie danych', index=False)
        df_test_i.to_excel(writer, sheet_name='suma infekcji - raport błędów', index=False)
    save_github(fn_xls_woj, max(dates) + ' Dane MZ przypadki województwa (XLSX)')
    print('  zapisano XLSX do github', fn_xls_woj)

    print('  powiaty')
    dates = []
    pow_sum = 0
    pow_pow = 0
    pow_columns_old = ['wojew', 'location', 'new_cases', 'd_0',
                   'new_deaths', 'new_deaths_c', 'new_deaths_nc', 'new_tests_poz',
                   'new_recoveries', 'total_quarantine', 'new_tests', 'new_tests_plus',
                   'new_tests_minus', 'new_tests_other']
    pow_columns_new = ['wojew', 'location', 'd_3', 'new_reinf', 'new_cases',
                   'd_0', 'd_1', 'd_2',
                   'new_deaths', 'new_deaths_c', 'new_deaths_nc', 'new_tests_poz',
                   'new_recoveries', 'total_quarantine', 'new_tests', 'new_tests_plus',
                   'new_tests_minus', 'new_tests_other']
    data_test = []
    for file in sorted(files):
        # print(file)
        if "rap_rcb_pow_eksport" in file:
            filepath = os.getcwd() + '/data/sources/rcb/' + file
            filedate = file[:8]
            pow_columns = pow_columns_old
            if filedate >= '20220208':
                pow_columns = pow_columns_new
            if filedate >= '20201124' and filedate <= '20201223':
                encoding = 'utf8'
                filetype = 11
            else:
                encoding = 'cp1250'
                filetype = 12
            if filedate == '20210503':
                encoding = 'utf8'
            df_pow = pd.read_csv(filepath, sep=';', encoding=encoding)
            if filetype == 11:
                df_pow.insert(loc=7, column='new_recoveries', value=[0] * len(df_pow))
            if len(df_pow.columns) > len(pow_columns + ['teryt']):
                df_pow.columns =  pow_columns + ['teryt', 'data']
            else:
                df_pow.columns = pow_columns + ['teryt']
            newdate = file[:4] + '-' + file[4:6] + '-' + file[6:8]
            if newdate not in dates:
                dates.append(newdate)
                df_pow['date'] = newdate
                fields = ['date'] + pow_columns
                df_pow = df_pow[fields].copy()
                df_pow.at[0, 'location'] = 'Polska'
                n1 = df_pow.iloc[0]['new_cases']
                n2 = df_pow.new_cases.sum() - n1
                pow_sum += n1
                pow_pow += n2
                if n1 != n2:
                    data_test.append([newdate, n1, n2, n1-n2])
                if len(total_pow) == 0:
                    total_pow = df_pow.copy()
                else:
                    total_pow = pd.concat([total_pow, df_pow], ignore_index=True)
    data_test.append(['RAZEM', pow_sum, pow_pow, pow_sum-pow_pow])
    total_pow.sort_values(by=['date', 'location'], inplace=True)
    total_pow.reset_index(drop=True, inplace=True)
    total_pow.fillna(0, inplace=True)
    for col in pow_columns[2:]:
        total_pow[col] = total_pow[col].astype(np.float64)
    total_pow.drop_duplicates(subset=['date', 'location'])
    total_pow.sort_values(by=['location', 'date'], inplace=True)
    total_pow.reset_index(drop=True, inplace=True)
    total_pow.to_csv(fn_pow_total, index=False)
    print('  zapisano', fn_pow_total)

    # arkusz testów

    cols = total_pow.columns
    tw1 = total_pow[total_pow['location'] != 'Polska'].groupby('date').sum()
    tw1['location'] = 'Suma z powiatów'
    tw1['date'] = tw1.index
    tw1.reset_index(drop=True, inplace=True)
    tw = pd.concat([tw1, total_pow[total_pow['location'] == 'Polska']], ignore_index=True)
    tw.reset_index(drop=True, inplace=True)
    tw.sort_values(by=['date', 'wojew', 'location'], inplace=True)
    tw = tw[cols].copy()
    del tw['d_0']
    del tw['d_1']
    del tw['d_2']
    del tw['d_3']
    del tw['wojew']
    df_test = pd.DataFrame(data_test, columns=['data', 'z arkusza', 'suma z powiatów', 'różnica'])

    del total_pow['d_0']
    del total_pow['d_1']
    del total_pow['d_2']
    del total_pow['d_3']
    with pd.ExcelWriter(fn_xls_pow) as writer:
        total_pow.to_excel(writer, sheet_name='arkusz powiatów', index=False)
        tw.to_excel(writer, sheet_name='porównanie danych', index=False)
        df_test.to_excel(writer, sheet_name='suma infekcji - raport błędów', index=False)
    save_github(fn_xls_pow, max(dates) + ' Dane MZ przypadki powiaty (XLSX)')
    print('  zapisano XLSX do github', fn_xls_pow)
    return


def load_data_jpt():
    print('>>> JPT 31.12.2020 <<<')
    fn_jpt_total = 'data/dict/jpt.csv'

    # utworzenie tabeli gus

    import os
    from app import server
    files = os.listdir("data/dict/gus")
    tabs = []
    for file in files:
        filepath = os.getcwd() + '/data/dict/gus/' + file
        df_woj = pd.read_csv(filepath, sep=',', encoding='utf-8', dtype=str)
        df_woj = df_woj.iloc[9:].copy()
        df_woj = df_woj[df_woj.columns[:5]].copy()
        df_woj.columns = ['gmina', 'JPT', 'razem', 'mężczyźni', 'kobiety']
        df_woj.dropna(axis='rows', subset=['JPT'], inplace=True)
        df_woj.reset_index(drop=True, inplace=True)
        df_woj['powiat'] = np.nan
        df_woj['nazwa'] = np.nan
        df_woj['wojew'] = file.split(sep='.')[0]
        df_woj['gmina'] = df_woj['gmina'].str.replace('M. st. ', '', regex=False)
        def f(r):
            r['JPT'] = 'JPT_' + r['JPT']
            if r['gmina'].strip().startswith('Powiat'):     # powiat
                r['powiat'] = r['gmina'].strip().replace('Powiat ', '')
                r['gmina'] = r['gmina'].strip()
                r['nazwa'] = r['gmina']
            elif r['gmina'].startswith(' '):                # zwykła gmina
                r['gmina'] = r['gmina'].strip()
                gm = r['gmina']
                r['nazwa'] = gm.replace('gm.m-w. ', '').replace('gm.w. ', '').replace('m. ', '').replace('M. st. ', '')
            else:                                           # miasto na prawach powiatu
                r['powiat'] = r['gmina']
                r['nazwa'] = r['gmina']
            return r

        df_woj = df_woj.apply(f, axis=1)
        df_woj['powiat'] = df_woj['powiat'].ffill()
        df_woj['gmina'] = df_woj['gmina'].str.replace('Powiat ', '', regex=False)
        tabs.append(df_woj)
    df_all = pd.concat([x for x in tabs], ignore_index=True, sort=False)
    df_all.to_csv(fn_jpt_total, index=False)
    print('  zapisano', fn_jpt_total)
    return


def load_data_szczepienia():
    print('>>> szczepienia <<<')
    fn_woj_total = 'data/sources/szczepienia/wojewodztwa.csv'
    fn_pow_total = 'data/sources/szczepienia/powiaty.csv'
    fn_xls_woj = 'data/sources/szczepienia/szczepienia_woj.xlsx'
    fn_xls_pow = 'data/sources/szczepienia/szczepienia_pow.xlsx'

    # utworzenie tabeli woj_total

    import os
    from app import server
    total_woj = pd.DataFrame()
    total_pow = pd.DataFrame()
    files = os.listdir("data/sources/szczepienia")
    dates = []
    print('  województwa')
    woj_columns = ['location', 'ogolem', 'dziennie', 'dawka_2_og', 'dawka_2_dz', 'teryt']
    data_test_i = []
    for file in sorted(files):
        if "_szczepienia_woj" in file:
            filepath = os.getcwd() + '/data/sources/szczepienia/' + file
            df_woj = pd.read_csv(filepath, sep=';', encoding='utf8')
            if len(df_woj.columns) > len(woj_columns):
                df_woj.columns = woj_columns + ['_1']
            else:
                df_woj.columns = woj_columns
            newdate = file[:4] + '-' + file[4:6] + '-' + file[6:8]
            if newdate not in dates:
                dates.append(newdate)
                df_woj['date'] = newdate
                fields = woj_columns + ['date']
                df_woj = df_woj[fields].copy()
                if len(total_woj) == 0:
                    total_woj = df_woj.copy()
                else:
                    total_woj = pd.concat([total_woj, df_woj], ignore_index=True)
    total_woj['location'] = total_woj['location'].str.replace('cały kraj', 'Polska', regex=False)
    total_woj.sort_values(by=['date', 'location'], inplace=True)
    total_woj.reset_index(drop=True, inplace=True)
    total_woj.fillna(0, inplace=True)
    del total_woj['teryt']
    for col in ['ogolem', 'dziennie', 'dawka_2_og', 'dawka_2_dz']:
        total_woj[col] = total_woj[col].astype(np.float64)
    def f(x):
        if x in wojew_cap:
            return wojew_cap[x]
        else:
            return 'nieznane'
    total_woj['location'] = total_woj['location'].apply(f)
    total_woj.drop_duplicates(subset=['date', 'location'])
    total_woj.sort_values(by=['location', 'date'], inplace=True)
    total_woj.reset_index(drop=True, inplace=True)
    total_woj.to_csv(fn_woj_total, index=False)
    print('  zapisano', fn_woj_total)

    with pd.ExcelWriter(fn_xls_woj) as writer:
        total_woj.to_excel(writer, sheet_name='arkusz województw', index=False)
    save_github(fn_xls_woj, max(dates) + ' Dane MZ szczepienia województwa (XLSX)')
    print('  zapisano XLSX do github', fn_xls_woj)

    print('  powiaty')
    dates = []
    pow_columns = ['wojew', 'location', 'ogolem', 'dziennie', 'dawka_2_og', 'dawka_2_dz', 'teryt']
    data_test = []
    for file in sorted(files):
        if "_szczepienia_pow" in file:
            filepath = os.getcwd() + '/data/sources/szczepienia/' + file
            df_pow = pd.read_csv(filepath, sep=';', encoding='utf8')
            if len(df_pow.columns) > len(pow_columns):
                df_pow.columns = pow_columns + ['_1']
            else:
                df_pow.columns = pow_columns
            newdate = file[:4] + '-' + file[4:6] + '-' + file[6:8]
            if newdate not in dates:
                dates.append(newdate)
                df_pow['date'] = newdate
                fields = ['date'] + pow_columns
                df_pow = df_pow[fields].copy()
                if len(total_pow) == 0:
                    total_pow = df_pow.copy()
                else:
                    total_pow = pd.concat([total_pow, df_pow], ignore_index=True)
    total_pow['wojew'] = total_pow['wojew'].str.replace('Cały kraj', 'Polska', regex=False)
    total_pow.sort_values(by=['date', 'wojew', 'location'], inplace=True)
    total_pow.reset_index(drop=True, inplace=True)
    total_pow['location'].fillna('Polska', inplace=True)
    for col in pow_columns[2:-1]:
        total_pow[col] = total_pow[col].astype(np.float64)
    total_pow.drop_duplicates(subset=['date', 'wojew', 'location'])
    total_pow.sort_values(by=['wojew', 'location', 'date'], inplace=True)
    total_pow.reset_index(drop=True, inplace=True)
    total_pow.to_csv(fn_pow_total, index=False)
    print('  zapisano', fn_pow_total)

    with pd.ExcelWriter(fn_xls_pow) as writer:
        total_pow.to_excel(writer, sheet_name='arkusz powiatów', index=False)
    save_github(fn_xls_pow, max(dates) + ' Dane MZ szczepienia powiaty (XLSX)')
    print('  zapisano XLSX do github', fn_xls_pow)
    return


# Dane BDL GUS

def load_data_bdl():
    print('>>> Dane z BDL GUS <<<')
    df = pd.read_csv('data/dict/bdl_src.csv', sep=';')

    columns = ['kod', 'nazwa', 'razem', '_0_4', '_5_9', '_10_14', '_15-19', '_20_24', '_25_29', '_30_34',
               '_35_39', '_40_44', '_45_49', '_50_54', '_55_59', '_60_64', '_65_69', '_70+', '70_74', '_75_79',
               '_80_84', '_85+', '_0_14', '_']
    df.columns = columns
    df = df[df['nazwa'].str.contains('Powiat')].copy()
    df['_80+'] = df['_80_84'] + df['_85+']
    df['_20_39'] = df['_20_24'] + df['_25_29'] + df['_30_34'] + df['_35_39']
    df['_40_59'] = df['_40_44'] + df['_45_49'] + df['_50_54'] + df['_55_59']
    df['_60_69'] = df['_60_64'] + df['_65_69']
    # df['kod'] = df['kod'].astype(str)
    df['woj'] = df['kod'].astype(str).apply(lambda x: x if len(x) == 7 else '0' + x).str.slice(0, 2).astype(int)
    df['wojew'] = df['woj'].apply(lambda x: terc_woj[x])

    # df['nazwa0'] = df['nazwa']
    df['nazwa'] = df['nazwa'].str.replace('Powiat m.', '', regex=False)
    df['nazwa'] = df['nazwa'].str.replace('Powiat ', '', regex=False)
    df['nazwa'] = df['nazwa'].str.replace(' od 2013', '', regex=False)
    df['nazwa'] = df['nazwa'].str.replace(' st. ', '', regex=False)
    for p in powiat_translation_table:
        df.loc[(df.nazwa == p['location']) & (df.wojew == p['wojew']), 'nazwa'] = p['new_location']
    df.to_csv('data/dict/bdl_last.csv', index=False)
    return df


def make_teryt():
    df_terc = pd.read_csv('data/dict/TERC_Urzedowy_2020-12-31.csv', sep=';')
    df_terc['POW'].fillna(0, inplace=True)
    df_terc['GMI'].fillna(0, inplace=True)
    df_terc['RODZ'].fillna(0, inplace=True)
    df_terc['WOJ'] = df_terc['WOJ'].astype(int)
    df_terc['POW'] = df_terc['POW'].astype(int)
    df_terc['GMI'] = df_terc['GMI'].astype(int)
    df_terc['RODZ'] = df_terc['RODZ'].astype(int)
    df_terc.sort_values(by=['WOJ', 'NAZWA'], inplace=True)
    df_terc['JPT'] = df_terc['WOJ'].map('{:0>2}'.format)+df_terc['POW'].map('{:0>2}'.format)+df_terc['GMI'].map('{:0>2}'.format)+df_terc['RODZ'].map('{:0>1}'.format)
    df_powiat = df_terc[df_terc['NAZWA_DOD'].isin(['powiat', 'miasto na prawach powiatu',
                                                   'miasto stołeczne, na prawach powiatu'])].copy()
    df_terc = pd.merge(df_terc, df_powiat[['NAZWA', 'WOJ', 'POW']], how='left', left_on=['WOJ', 'POW'], right_on=['WOJ', 'POW'])
    df_terc.rename(columns={'NAZWA_x': 'NAZWA'}, inplace=True)
    df_terc.rename(columns={'NAZWA_y': 'powiat'}, inplace=True)
    df_wojew = df_terc[df_terc['NAZWA_DOD'] == 'województwo'].copy()
    df_terc = pd.merge(df_terc, df_wojew[['NAZWA', 'WOJ']], how='left', left_on=['WOJ'], right_on=['WOJ'])
    df_terc.rename(columns={'NAZWA_x': 'NAZWA'}, inplace=True)
    df_terc.rename(columns={'NAZWA_y': 'wojew'}, inplace=True)
    df_terc['powiat'].fillna('', inplace=True)
    df_terc['wojew'].fillna('', inplace=True)
    df_terc['wojew'] = df_terc['wojew'].str.lower()
    df_terc['nazwa'] = df_terc['NAZWA']
    df_terc['teryt_woj'] = df_terc['JPT'].str.slice(0, 2)
    df_terc['teryt_pow'] = df_terc['JPT'].str.slice(0, 4)
    df_terc.to_csv('data/dict/teryt.csv', index=False)
    # df_terc['teryt_pow'] = df_terc['JPT'].str.slice(0, 4).astype(int).astype(str) + df_terc['JPT'].str.slice(2, 4)
    pass


def calc_dd_all(df):
    # calculate double_days for whole database
    # t_start = datetime.datetime.now()

    df0 = df.sort_values(by=['location', 'date'])
    df0.reset_index(drop=True, inplace=True)
    ret_val = []
    for loc in df0['location'].unique():
        df1 = df0[df0.location == loc]
        cases = list(df1['total_cases'])
        for index, item in enumerate(cases):
            v = item / 2
            index_ge = [i for i, j in enumerate(cases) if j >= (item / 2)]
            if len(index_ge) > 0:
                x4 = index_ge[0]
            else:
                x4 = index
            if item > 0:
                days = max(index - x4, 0)
            else:
                days = 0
            ret_val.append(days)
    return ret_val

countries = pd.read_csv("data/dict/sl_country_table.csv")
stat_wojew = pd.read_csv("data/dict/sl_wojew.csv")
miasta = pd.read_csv('data/dict/sl_miasta.csv')

stat_wojew.drop(columns=['ha', 'p1', 'p2', 'p3', 'p4', 'p5'], inplace=True)

# load_data_basiw_d()
# load_data_basiw_i()

# load_data_rcb()
# load_data_poland()
# read_gus()
# load_data_mortality_eurostat()
# load_data_mz_api_age_v()
# load_data_mz_api_vacc_wojew()
# read_testy_infekcje()
# stat_testy_infekcje()
# strefa testów
# load_data_zgony_szczepienia()
# read_gus()
# make_teryt()
# exit(0)
# load_data_poland()
# read_gus()
# load_data_ewp_age()
# load_data_vac_eu()
# load_data_resources()
# load_data_bdl()
# load_data_vac_pl()
# load_data_reinf()
# stat_reinf()
# load_data_ewp_age()
# load_data_mz_cases()
# load_data_balance()
# load_data_resources()

# tabele pomocnicze
# load_data_jpt()
# load_data_basiw_d()
# load_data_basiw_i()
# load_data_zgony_szczepienia()
load_data_ecdc_vac_pl()
load_data_jpt()
load_data_szczepienia()
# load_data_ecdc_vac_pl()
load_data_ecdc_vac_eu()
load_data_mz_api_age_d()
load_data_mz_api_age_v()
load_data_mz_psz()
# load_data_mz_api_vacc_gminy()
load_data_mz_api_vacc_wojew()
load_data_mz_api_vacc_powiaty()
load_data_rcb()
load_data_Rt_woj()
load_data_Rt()
load_data_resources()
# load_data_mortality_eurostat()

# dane pomocnicze poland

# wyłączone z powodu braku aktualizacji przez micalrg
# load_data_balance()

# tabele podstawowe

load_data_poland()
load_data_cities()
load_data_world()

# tabele opcjonalne, aktualizowane na zyczenie

read_gus()
