from os import path
import sys, json, time
import pandas as pd

mode = 'apply'

# Read input data
print('Reading Input Data')
if mode == 'build':
    with open('../../data/model_build_inputs/route_data.json', newline='') as in_file:
        actual_routes = json.load(in_file)
    
    route_data = {'route_id':[],'station_code':[],'date':[],'departure_time':[],'exe_cap_cm3':[],'route_score':[],'stops':[],'lat':[],'lng':[],'type':[],'zone_id':[]}
    
    for route_id in actual_routes:
        route = actual_routes[route_id]
        for stops in route['stops']:
            route_data['route_id'].append(route_id)
            route_data['station_code'].append(route['station_code'])
            route_data['date'].append(route['date_YYYY_MM_DD'])
            route_data['departure_time'].append(route['departure_time_utc'])
            route_data['exe_cap_cm3'].append(route['executor_capacity_cm3'])
            route_data['route_score'].append(route['route_score'])
            route_data['stops'].append(stops)
            route_data['lat'].append(route['stops'][stops]['lat'])
            route_data['lng'].append(route['stops'][stops]['lng'])
            route_data['type'].append(route['stops'][stops]['type'])
            route_data['zone_id'].append(route['stops'][stops]['zone_id'])
    
    route_data = pd.DataFrame(route_data)
    route_data['stops'] = route_data['stops'].astype(str)
    
    route_data.to_csv('../data/'+mode+'_route_df.csv',index=False)

if mode == 'apply':
    with open('../../data/model_'+mode+'_inputs/new_route_data.json', newline='') as in_file:
        actual_routes = json.load(in_file)
    
    route_data = {'route_id':[],'station_code':[],'date':[],'departure_time':[],
                  'exe_cap_cm3':[],'stops':[],'lat':[],'lng':[],'type':[],'zone_id':[]}
    
    for route_id in actual_routes:
        route = actual_routes[route_id]
        for stops in route['stops']:
            route_data['route_id'].append(route_id)
            route_data['station_code'].append(route['station_code'])
            route_data['date'].append(route['date_YYYY_MM_DD'])
            route_data['departure_time'].append(route['departure_time_utc'])
            route_data['exe_cap_cm3'].append(route['executor_capacity_cm3'])
            route_data['stops'].append(stops)
            route_data['lat'].append(route['stops'][stops]['lat'])
            route_data['lng'].append(route['stops'][stops]['lng'])
            route_data['type'].append(route['stops'][stops]['type'])
            route_data['zone_id'].append(route['stops'][stops]['zone_id'])
    
    route_data = pd.DataFrame(route_data)
    route_data['stops'] = route_data['stops'].astype(str)

    route_data.to_csv('../data/'+mode+'_route_df.csv',index=False)
