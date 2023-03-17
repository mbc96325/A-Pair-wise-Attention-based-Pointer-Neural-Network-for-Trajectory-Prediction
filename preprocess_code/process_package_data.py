from os import path
import sys, json, time
import pandas as pd

mode = 'apply'

# Read input data
print('Reading Input Data')
if mode == 'build':
    with open('../../data/model_'+mode+'_inputs/package_data.json', newline='') as in_file:
        package_data = json.load(in_file)
    
    package_df = {'route_id':[],'stops':[],'pack_ID':[],'scan_status':[],'time_window_start':[],'time_window_end':[],'planned_service_time':[],'depth_cm':[],'height_cm':[],'width_cm':[]}
    
    for route_id in package_data:
        route = package_data[route_id]
        for stops in route:
            for pack in route[stops]:
                package_df['route_id'].append(route_id)
                package_df['stops'].append(stops)
                package_df['pack_ID'].append(pack)
                package_df['scan_status'].append(route[stops][pack]['scan_status'])
                package_df['time_window_start'].append(route[stops][pack]['time_window']['start_time_utc'])
                package_df['time_window_end'].append(route[stops][pack]['time_window']['end_time_utc'])
                package_df['planned_service_time'].append(route[stops][pack]['planned_service_time_seconds'])
                package_df['depth_cm'].append(route[stops][pack]['dimensions']['depth_cm'])
                package_df['height_cm'].append(route[stops][pack]['dimensions']['height_cm'])
                package_df['width_cm'].append(route[stops][pack]['dimensions']['width_cm'])
    
    package_df = pd.DataFrame(package_df)
    package_df['stops'] = package_df['stops'].astype(str)

    package_df.to_csv('../data/'+mode+'_package_df.csv',index=False)
    
if mode == 'apply':
    
    with open('../../data/model_'+mode+'_inputs/new_package_data.json', newline='') as in_file:
        package_data = json.load(in_file)
    
    package_df = {'route_id':[],'stops':[],'pack_ID':[],'time_window_start':[],'time_window_end':[],'planned_service_time':[],'depth_cm':[],'height_cm':[],'width_cm':[]}
    
    for route_id in package_data:
        route = package_data[route_id]
        for stops in route:
            for pack in route[stops]:
                package_df['route_id'].append(route_id)
                package_df['stops'].append(stops)
                package_df['pack_ID'].append(pack)
                package_df['time_window_start'].append(route[stops][pack]['time_window']['start_time_utc'])
                package_df['time_window_end'].append(route[stops][pack]['time_window']['end_time_utc'])
                package_df['planned_service_time'].append(route[stops][pack]['planned_service_time_seconds'])
                package_df['depth_cm'].append(route[stops][pack]['dimensions']['depth_cm'])
                package_df['height_cm'].append(route[stops][pack]['dimensions']['height_cm'])
                package_df['width_cm'].append(route[stops][pack]['dimensions']['width_cm'])
    
    package_df = pd.DataFrame(package_df)
    
    package_df['stops'] = package_df['stops'].astype(str)
    package_df.to_csv('../data/'+mode+'_package_df.csv',index=False)
   
