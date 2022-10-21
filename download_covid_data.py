import pandas as pd
from datetime import datetime as dt

import requests
import json

import constants
from constants import data_files


def add_log(msg):
    with open("log/read_data.log", "a") as file:
        file.write(str(dt.now()) + '  ' + str(msg))
        file.write('\n')


def download_data0(key):
    print(key, '=', data_files[key]['name'], end='. ')
    if data_files[key]['priority'] > 2:
        print(key, constants.bcolors.OKBLUE, '*** pominięte ***', constants.bcolors.ENDC)
        return
    if key == 'mortality_eurostat':
        # print(constants.bcolors.WARNING, '*** POBIERZ RĘCZNIE', data_files[key]['src_fn'], constants.bcolors.ENDC)
        # return
        import eurostat
        df = eurostat.get_data_df('demo_r_mwk2_20')
        df.to_csv(data_files[key]['src_fn'], index=False)
        print(data_files[key]['name'] + ' from ' + data_files[key]['url'], constants.bcolors.OKGREEN, '*** OK ***', constants.bcolors.ENDC)
    elif key == 'mz_api_age_d':
        URL = 'https://services-eu1.arcgis.com/zk7YlClTgerl62BY/arcgis/rest/services/global_corona_widok2/FeatureServer/0/query?f=json&where=Data%3Etimestamp%20%272020-12-23%2022%3A59%3A59%27&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&orderByFields=Data%20asc&outSR=102100&resultOffset=0&resultRecordCount=32000&resultType=standard&cacheHint=true'
        data = requests.get(URL).text
        main_dict = json.loads(data)
        if list(main_dict.keys())[0] == 'error':
            print(constants.bcolors.FAIL, main_dict['error']['message'], constants.bcolors.ENDC)
            return
        def partial_export(file, field):
            for column in field:
                file.write(str(column) + ';')
            file.write('\n')
        with open(data_files[key]['src_fn'], 'w', encoding='utf-8') as f:
            for i, row in enumerate(main_dict['features']):
                if i == 0:
                    partial_export(f, row['attributes'])
                partial_export(f, row['attributes'].values())
        print(data_files[key]['name'] + ' from script', constants.bcolors.OKGREEN, '*** OK ***', constants.bcolors.ENDC)
    elif key == 'mz_api_vacc_powiaty':
        URL = 'https://services-eu1.arcgis.com/zk7YlClTgerl62BY/ArcGIS/rest/services/powiaty_szczepienia_widok2/FeatureServer/0/query?where=1%3D1&objectIds=&time=&resultType=none&outFields=*&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token='
        data = requests.get(URL).text
        main_dict = json.loads(data)
        if list(main_dict.keys())[0] == 'error':
            print(constants.bcolors.FAIL, main_dict['error']['message'], constants.bcolors.ENDC)
            return
        def partial_export(file, field):
            for column in field:
                file.write(str(column) + ';')
            file.write('\n')
        with open(data_files[key]['src_fn'], 'w', encoding='utf-8') as f:
            for i, row in enumerate(main_dict['features']):
                if i == 0:
                    partial_export(f, row['attributes'])
                partial_export(f, row['attributes'].values())
        print(data_files[key]['name'] + ' from script', constants.bcolors.OKGREEN, '*** OK ***', constants.bcolors.ENDC)
    elif key == 'mz_api_vacc_wojew':
        URL = 'https://services-eu1.arcgis.com/zk7YlClTgerl62BY/ArcGIS/rest/services/wojewodztwa_szczepienia_widok3/FeatureServer/0/query?where=1%3D1&objectIds=&time=&resultType=none&outFields=*&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token='
        data = requests.get(URL).text
        main_dict = json.loads(data)
        if list(main_dict.keys())[0] == 'error':
            print(constants.bcolors.FAIL, main_dict['error']['message'], constants.bcolors.ENDC)
            return
        def partial_export(file, field):
            for column in field:
                file.write(str(column) + ';')
            file.write('\n')
        with open(data_files[key]['src_fn'], 'w', encoding='utf-8') as f:
            for i, row in enumerate(main_dict['features']):
                if i == 0:
                    partial_export(f, row['attributes'])
                partial_export(f, row['attributes'].values())
        print(data_files[key]['name'] + ' from script', constants.bcolors.OKGREEN, '*** OK ***', constants.bcolors.ENDC)
    elif key == 'mz_api_age_v':
        URL = 'https://services-eu1.arcgis.com/zk7YlClTgerl62BY/ArcGIS/rest/services/global_szczepienia_widok2/FeatureServer/0/query?where=1%3D1&objectIds=&time=&resultType=none&outFields=*&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token='
        try:
            data = requests.get(URL, timeout=10).text
        except:
            print('*** Błąd request')
        else:
            main_dict = json.loads(data)
            if list(main_dict.keys())[0] == 'error':
                print(constants.bcolors.FAIL, main_dict['error']['message'], constants.bcolors.ENDC)
                return

            def partial_export(file, field):
                for column in field:
                    file.write(str(column) + ';')
                file.write('\n')
            if not 'features' in main_dict.keys():
                print(constants.bcolors.FAIL, '*** Błąd API ***', constants.bcolors.ENDC)
                return
            with open(data_files[key]['src_fn'], 'w', encoding='utf-8') as f:
                for i, row in enumerate(main_dict['features']):
                    if i == 0:
                        partial_export(f, row['attributes'])
                    partial_export(f, row['attributes'].values())
            print(data_files[key]['name'] + ' from script', constants.bcolors.OKGREEN, '*** OK ***', constants.bcolors.ENDC)
    elif key == 'mz_api_age_v3':
        URL = 'https://services-eu1.arcgis.com/zk7YlClTgerl62BY/ArcGIS/rest/services/widok_global_szczepienia_actual/FeatureServer/0/query?where=1%3D1&objectIds=&time=&resultType=none&outFields=*&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token='
        try:
            data = requests.get(URL, timeout=10).text
        except:
            print('*** Błąd request')
        else:
            main_dict = json.loads(data)
            if list(main_dict.keys())[0] == 'error':
                print(constants.bcolors.FAIL, main_dict['error']['message'], constants.bcolors.ENDC)
                return


            def partial_export(file, field):
                for column in field:
                    file.write(str(column) + ';')
                file.write('\n')
            if not 'features' in main_dict.keys():
                print(constants.bcolors.FAIL, '*** Błąd API ***', constants.bcolors.ENDC)
                return
            with open(data_files[key]['src_fn'], 'w', encoding='utf-8') as f:
                for i, row in enumerate(main_dict['features']):
                    if i == 0:
                        partial_export(f, row['attributes'])
                    partial_export(f, row['attributes'].values())
            print(data_files[key]['name'] + ' from script', constants.bcolors.OKGREEN, '*** OK ***', constants.bcolors.ENDC)
    else:
        print(constants.bcolors.WARNING, '*** POBIERZ RĘCZNIE', data_files[key]['src_fn'], constants.bcolors.ENDC)


def download_data(key):
    keys = {
        'mz_api_age_d': 'https://services-eu1.arcgis.com/zk7YlClTgerl62BY/arcgis/rest/services/global_corona_widok2/FeatureServer/0/query?f=json&where=Data%3Etimestamp%20%272020-12-23%2022%3A59%3A59%27&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&orderByFields=Data%20asc&outSR=102100&resultOffset=0&resultRecordCount=32000&resultType=standard&cacheHint=true',
        'mz_api_vacc_powiaty': 'https://services-eu1.arcgis.com/zk7YlClTgerl62BY/ArcGIS/rest/services/powiaty_szczepienia_widok2/FeatureServer/0/query?where=1%3D1&objectIds=&time=&resultType=none&outFields=*&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token=',
        'mz_api_vacc_wojew': 'https://services-eu1.arcgis.com/zk7YlClTgerl62BY/ArcGIS/rest/services/wojewodztwa_szczepienia_widok3/FeatureServer/0/query?where=1%3D1&objectIds=&time=&resultType=none&outFields=*&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token=',
        'mz_api_age_v': 'https://services-eu1.arcgis.com/zk7YlClTgerl62BY/ArcGIS/rest/services/global_szczepienia_widok2/FeatureServer/0/query?where=1%3D1&objectIds=&time=&resultType=none&outFields=*&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token=',
        'mz_api_age_v3': 'https://services-eu1.arcgis.com/zk7YlClTgerl62BY/ArcGIS/rest/services/widok_global_szczepienia_actual/FeatureServer/0/query?where=1%3D1&objectIds=&time=&resultType=none&outFields=*&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token=',

    }
    def get_data(key, URL):
        def partial_export(file, field):
            for column in field:
                file.write(str(column) + ';')
            file.write('\n')
        while True:
            try:
                data = requests.get(URL).text
            except:
                print(key, 'Error loading URL')
            else:
                main_dict = json.loads(data)
                if list(main_dict.keys())[0] == 'error':
                    print(constants.bcolors.FAIL, key, main_dict['error']['message'], constants.bcolors.ENDC)
                    continue
                with open(data_files[key]['src_fn'], 'w', encoding='utf-8') as f:
                    for i, row in enumerate(main_dict['features']):
                        if i == 0:
                            partial_export(f, row['attributes'])
                        partial_export(f, row['attributes'].values())
                print(key + ' (API)', constants.bcolors.OKGREEN, '*** OK ***', constants.bcolors.ENDC)
                break

    # ignorowane

    if data_files[key]['priority'] > 2:
        print(key, constants.bcolors.OKBLUE, '*** pominięte ***', constants.bcolors.ENDC)
        return

    # Eurostat

    elif key == 'mortality_eurostat':
        # print(constants.bcolors.WARNING, '*** POBIERZ RĘCZNIE', data_files[key]['src_fn'], constants.bcolors.ENDC)
        # return
        import eurostat
        df = eurostat.get_data_df('demo_r_mwk2_20')
        df.to_csv(data_files[key]['src_fn'], index=False)
        print(key, constants.bcolors.OKGREEN, '*** OK ***', constants.bcolors.ENDC)

    # API MZ

    elif key in keys:
        get_data(key, keys[key])

    # proste pobranie pliku CSV, JSON, ZIP

    elif data_files[key]['type'] in ['csv', 'json', 'zip']:
        url = data_files[key]['url']
        try:
            if data_files[key]['type'] == 'csv':
                df = pd.read_csv(url)
            elif data_files[key]['type'] == 'json':
                df = pd.read_json(url)
            elif data_files[key]['type'] == 'zip':
                r = requests.get(url, allow_redirects=True)
                open('temp', 'wb').write(r.content)
                from zipfile import ZipFile
                z = ZipFile('temp')
                z.extract('2020_PL_Region_Mobility_Report.csv', 'data/sources/download')
        except:
            print(constants.bcolors.FAIL, key, '*** Błąd ***', constants.bcolors.ENDC)
            return
        else:
            if data_files[key]['type'] in ['csv', 'json']:
                df.to_csv(data_files[key]['src_fn'], index=False)
            print(key, constants.bcolors.OKGREEN, '*** OK ***', constants.bcolors.ENDC)

    # do pobrania ręcznego

    else:
        print(constants.bcolors.WARNING, '*** POBIERZ RĘCZNIE', data_files[key]['src_fn'], constants.bcolors.ENDC)


def test():
    def partial_export(file, field):
        for column in field:
            file.write(str(column) + ';')
        file.write('\n')
    URL = 'https://services-eu1.arcgis.com/zk7YlClTgerl62BY/ArcGIS/rest/services/wojewodztwa_szczepienia_widok3/FeatureServer/0/query?where=1%3D1&objectIds=&time=&resultType=none&outFields=*&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token='
    while True:
        try:
            data = requests.get(URL).text
        except:
            print('Error loading URL')
        else:
            main_dict = json.loads(data)
            if list(main_dict.keys())[0] == 'error':
                print(constants.bcolors.FAIL, main_dict['error']['message'], constants.bcolors.ENDC)
                continue
            with open('/home/docent/test.tmp', 'w', encoding='utf-8') as f:
                for i, row in enumerate(main_dict['features']):
                    if i == 0:
                        partial_export(f, row['attributes'])
                    partial_export(f, row['attributes'].values())
            break

# test()
# exit(0)

download_data('mz_api_age_v')
for key in data_files:
    download_data(key)

print('===== END =====')
