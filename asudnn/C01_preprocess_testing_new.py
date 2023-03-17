# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle as pkl
import sys
import json
import time
import _constants

data_raw = pd.read_csv("../data/zone_data.csv")


#%%
# with open('../../data/zone_travel_time.pkl', 'rb') as in_file:
#     tt_avg = pkl.load(in_file)
#     tt_min = pkl.load(in_file)

with open('../data/zone_mean_travel_times.json') as f:
    tt_avg = json.load(f)
# with open('../../data/zone_min_travel_times.json') as f:
#     tt_min = json.load(f)

x = {}
y = {}
zone_id = {}

# with open('../../tsp_xiaotong/mean_dist/opt_zone_seq.p', 'rb') as f:
#     tsp_zone_seq_mean = pkl.load(f)
#
# with open('../../tsp_xiaotong/min_dist/opt_zone_seq.p', 'rb') as f:
#     tsp_zone_seq_min = pkl.load(f)

with open('../data/mean_dist/opt_zone_seq.json', 'rb') as f:
    tsp_zone_seq_mean = json.load(f)
with open('../data/min_dist/opt_zone_seq.json', 'rb') as f:
    tsp_zone_seq_min = json.load(f)

with open('testing_routes.pkl', 'rb') as f:
    test_routes = pkl.load(f)
data_test = data_raw.loc[data_raw['route_id'].isin(test_routes)]




tic = time.time()

route_list_all = []
zone_list = []
for route in tsp_zone_seq_mean:
    route_list_all +=  [route] * len(tsp_zone_seq_mean[route])
    zone_list += tsp_zone_seq_mean[route]

tsp_zone_seq_mean_df = pd.DataFrame({'route_id': route_list_all, 'zone_id': zone_list})
tsp_zone_seq_mean_df['zone_seq_tsp'] = tsp_zone_seq_mean_df.groupby(['route_id']).cumcount() + 1
tsp_zone_seq_mean_df = tsp_zone_seq_mean_df.sort_values(['route_id','zone_seq_tsp'])
tsp_zone_seq_mean_df_add_end =tsp_zone_seq_mean_df.sort_values(['zone_seq_tsp']).drop_duplicates(['route_id'], keep = 'last')
tsp_zone_seq_mean_df_add_end['zone_seq_tsp'] += 1
tsp_zone_seq_mean_df_add_end['zone_id'] = 'END'
tsp_zone_seq_mean_df = pd.concat([tsp_zone_seq_mean_df, tsp_zone_seq_mean_df_add_end])
tsp_zone_seq_mean_df = tsp_zone_seq_mean_df.sort_values(['route_id','zone_seq_tsp'])

route_list_all = []
zone_list = []


for route in tsp_zone_seq_min:

    route_list_all +=  [route] * len(tsp_zone_seq_min[route])
    zone_list += tsp_zone_seq_min[route]

tsp_zone_seq_min_df = pd.DataFrame({'route_id': route_list_all, 'zone_id': zone_list})
tsp_zone_seq_min_df['zone_seq_tsp'] = tsp_zone_seq_min_df.groupby(['route_id']).cumcount() + 1
tsp_zone_seq_min_df = tsp_zone_seq_min_df.sort_values(['route_id','zone_seq_tsp'])
tsp_zone_seq_min_df_add_end =tsp_zone_seq_min_df.sort_values(['zone_seq_tsp']).drop_duplicates(['route_id'], keep = 'last')
tsp_zone_seq_min_df_add_end['zone_seq_tsp'] += 1
tsp_zone_seq_min_df_add_end['zone_id'] = 'END'
tsp_zone_seq_min_df = pd.concat([tsp_zone_seq_min_df, tsp_zone_seq_min_df_add_end])
tsp_zone_seq_min_df = tsp_zone_seq_min_df.sort_values(['route_id','zone_seq_tsp'])



tsp_zone_seq_mean_df['next_route'] = tsp_zone_seq_mean_df['route_id'].shift(-1)
tsp_zone_seq_mean_df['tsp_mean_next_zone'] = tsp_zone_seq_mean_df['zone_id'].shift(-1)

tsp_zone_seq_mean_df = tsp_zone_seq_mean_df.loc[tsp_zone_seq_mean_df['next_route'] == tsp_zone_seq_mean_df['route_id']]

tsp_zone_seq_min_df['next_route'] = tsp_zone_seq_min_df['route_id'].shift(-1)
tsp_zone_seq_min_df['tsp_min_next_zone'] = tsp_zone_seq_min_df['zone_id'].shift(-1)

tsp_zone_seq_min_df = tsp_zone_seq_min_df.loc[tsp_zone_seq_min_df['next_route'] == tsp_zone_seq_min_df['route_id']]


#
#
# def get_angle(p0, p1=np.array([0, 0]), p2=None):
#     ''' compute angle (in degrees) for p0p1p2 corner
#     Inputs:
#         p0,p1,p2 - points in the form of [x,y]
#     '''
#     if p2 is None:
#         p2 = p1 + np.array([1, 0])
#     v0 = np.array(p0) - np.array(p1)
#     v1 = np.array(p2) - np.array(p1)
#
#     angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
#     return np.degrees(angle)
#
#
#
# def angle_between_three_points(a: np.array, b: np.array, c: np.array):
#     ba_original = a - b
#     bc_original = c - b
#     ba = np.squeeze(np.asarray(ba_original))
#     bc = np.squeeze(np.asarray(bc_original))
#     cosine_angle = np.sum(ba * bc,axis = 1) / (np.linalg.norm(ba,axis = 1) * np.linalg.norm(bc, axis = 1))
#     #print(cosine_angle[-1])
#     angle = np.arccos(cosine_angle)
#     return angle, cosine_angle

# worst_value = np.array([[5000,5000,0,0,600,99999,99999,99999]])

data = data_test.copy()

data = data.sort_values(['route_id','zone_seq'])
data['total_num_zones'] = data.groupby(['route_id'])['zone_id'].transform('count')
data['current_zone_percentage'] = data['zone_seq'] / data['total_num_zones']
data['total_num_zones'] = data['total_num_zones']/np.max(data['total_num_zones']) # normalize
data['last_route_id'] = data['route_id'].shift(1)
data['last_zone_id'] = data['zone_id'].shift(1)
data['last_lat'] = data['lat_mean'].shift(1)
data['last_lng'] = data['lng_mean'].shift(1)

data['next_route_id'] = data['route_id'].shift(-1)
data['next_zone_id'] = data['zone_id'].shift(-1)
data.loc[data['next_route_id'] != data['route_id'], 'next_zone_id'] = 'END'

data['weekends'] = 0
data.loc[data['day_of_week'] >= 5, 'weekends'] = 1

data['before_7am'] = 0
data.loc[data['hour'] <= 7,'before_7am'] =1

data['after_10am'] = 0
data.loc[data['hour'] >= 10, 'after_10am'] = 1





data_copy = data.copy()
data_copy['zone_id_avail'] = data_copy['zone_id']
data_copy['zone_seq_avail'] = data_copy['zone_seq']
data_copy['lat_avail'] = data_copy['lat_mean']
data_copy['lng_avail'] = data_copy['lng_mean']
data_copy['num_tra_sig_avail'] = data_copy['num_tra_sig']
data_copy['num_stops_avail'] = data_copy['total_num_stops_per_zone']
data_copy['n_pkg_avail'] = data_copy['n_pkg']
data_copy['weekends_avail'] = data_copy['weekends']
data_copy['before_7am_avail'] = data_copy['before_7am']
data_copy['after_10am_avail'] = data_copy['after_10am']


# data_copy['total_num_stops_avail'] = data_copy['total_num_stops_per_zone']
#data_copy['num_tra_sig_avail'] = data_copy['num_tra_sig']

data_copy['key'] = 1
data['key'] = 1

data_pc = data.merge(data_copy[['route_id','zone_id_avail','zone_seq_avail','lat_avail','lng_avail','key','num_stops_avail','n_pkg_avail',
                                'num_tra_sig_avail','weekends_avail','before_7am_avail','after_10am_avail']], on = ['route_id','key'])
#data_pc = data_pc.loc[data_pc['zone_seq'] < data_pc['zone_seq_avail']]



with open('../../data/zone_tt.pkl', 'rb') as f:
    zone_min_tt_df_out = pkl.load(f)
    zone_mean_tt_df_out = pkl.load(f)


data_pc = data_pc.merge(zone_min_tt_df_out, left_on = ['route_id','zone_id','zone_id_avail'], right_on = ['route_id','from_zone','to_zone'])
data_pc = data_pc.drop(columns = ['from_zone','to_zone'])
data_pc = data_pc.merge(zone_mean_tt_df_out, left_on = ['route_id','zone_id','zone_id_avail'], right_on = ['route_id','from_zone','to_zone'])
data_pc = data_pc.drop(columns = ['from_zone','to_zone'])




data_pc = data_pc.sort_values(['route_id','zone_id','mean_travel_time'])



data_pc = data_pc.merge(tsp_zone_seq_mean_df[['route_id','zone_id','tsp_mean_next_zone']], on = ['route_id','zone_id'])
data_pc = data_pc.merge(tsp_zone_seq_min_df[['route_id','zone_id','tsp_min_next_zone']], on = ['route_id','zone_id'])


data_pc['tsp_next_mean'] = 0
data_pc['tsp_next_min'] = 0

data_pc.loc[data_pc['tsp_mean_next_zone'] == data_pc['zone_id_avail'], 'tsp_next_mean'] = 1
data_pc.loc[data_pc['tsp_min_next_zone'] == data_pc['zone_id_avail'], 'tsp_next_min'] = 1

data_pc.loc[data_pc['min_travel_time'] < 1, 'min_travel_time'] = 1
data_pc.loc[data_pc['mean_travel_time'] < 1, 'mean_travel_time'] = 1

data_pc['zone_avail_seq'] = data_pc.groupby(['route_id','zone_id']).cumcount()

temp1 = data_pc['zone_id_avail'].str.split('.',expand = True)
data_pc['zone_id_avail_BIG'] = temp1.iloc[:,0]
temp2 = data_pc['zone_id'].str.split('.',expand = True)
data_pc['zone_id_BIG'] = temp2.iloc[:,0]

temp1.iloc[:,1] = temp1.iloc[:,1].fillna('99X')
temp1['zone3'] = temp1.iloc[:,1].apply(lambda x:x[:-1])
temp1['zone4'] = temp1.iloc[:,1].apply(lambda x:x[-1])

temp2.iloc[:,1] = temp2.iloc[:,1].fillna('99X')
temp2['zone3'] = temp2.iloc[:,1].apply(lambda x:x[:-1])
temp2['zone4'] = temp2.iloc[:,1].apply(lambda x:x[-1])

data_pc['zone_id_avail_SMALL1'] = temp1['zone3'].astype('int')
data_pc['zone_id_avail_SMALL2'] = temp1['zone4'].apply(lambda x: ord(x)) - 65
data_pc['zone_id_SMALL1'] = temp2['zone3'].astype('int')
data_pc['zone_id_SMALL2'] = temp2['zone4'].apply(lambda x: ord(x)) - 65

data_pc['SMALL_zone_diff'] = np.abs(data_pc['zone_id_avail_SMALL1'] - data_pc['zone_id_SMALL1']) + np.abs(data_pc['zone_id_avail_SMALL2'] - data_pc['zone_id_SMALL2'])

data_pc['SMALL_zone_diff_eq_1'] = data_pc['SMALL_zone_diff'] == 1
data_pc['SMALL_zone_diff_eq_1'] = data_pc['SMALL_zone_diff_eq_1'].astype('int')


data_pc['BIG_zone_id_same'] = data_pc['zone_id_avail_BIG'] == data_pc['zone_id_BIG']
data_pc['BIG_zone_id_same'] = data_pc['BIG_zone_id_same'].astype('int')





#check = data_pc.loc[data_pc['min_travel_time'] == 0]

feature_col = _constants.feature_col

print('process time', time.time() - tic)

tic = time.time()

# route_zone_ = data_pc[['route_id','zone_id','zone_seq']].drop_duplicates()
# route_zone_['key'] = 1
# temp = pd.DataFrame({'key': [1]* num_neighbor, 'zone_avail_seq': np.arange(num_neighbor)})
# route_zone_ = route_zone_.merge(temp, on = ['key'])
#
# route_zone_ = route_zone_.drop(columns = ['key'])
# data_pc = data_pc.merge(route_zone_, on = ['route_id','zone_id','zone_avail_seq','zone_seq'], how = 'right')

# assign worst value to nan

data_pc.loc[data_pc['num_tra_sig_avail'].isna(),'num_tra_sig_avail'] = 0
data_pc.loc[data_pc['num_stops_avail'].isna(),'num_stops_avail'] = 0
data_pc.loc[data_pc['n_pkg_avail'].isna(),'n_pkg_avail'] = 0
data_pc['total_num_zones'] = data_pc['total_num_zones'].ffill()
data_pc['current_zone_percentage'] = data_pc['current_zone_percentage'].ffill()
data_pc['before_7am_avail'] = data_pc['before_7am_avail'].ffill()
data_pc['after_10am_avail'] = data_pc['after_10am_avail'].ffill()
data_pc['weekends_avail'] = data_pc['weekends_avail'].ffill()

data_pc['max_min_tt'] = data_pc.groupby(['route_id','zone_id'])['min_travel_time'].transform('max')
data_pc['max_mean_tt'] = data_pc.groupby(['route_id','zone_id'])['mean_travel_time'].transform('max')

data_pc.loc[data_pc['min_travel_time'].isna(),'min_travel_time'] = data_pc.loc[data_pc['min_travel_time'].isna(),'max_min_tt'] * 1.5
data_pc.loc[data_pc['mean_travel_time'].isna(),'mean_travel_time'] = data_pc.loc[data_pc['mean_travel_time'].isna(),'max_mean_tt'] * 1.5

data_pc = data_pc.ffill()

data_pc = data_pc.sort_values(['route_id','zone_seq','mean_travel_time'])



print(len(pd.unique(data_pc['route_id'])))

x = {}
y = {}
zone_id = {}


tic = time.time()

feature_col_test = _constants.feature_col_test

save_data_pc = data_pc.loc[:,['route_id','zone_id','next_zone_id','zone_id_avail'] + feature_col_test]

with open('data/processed_test_zone_seq_neighbour_all_df' '.pkl', 'wb') as f:
    pkl.dump(save_data_pc,f)


for idx_, info in data_pc.groupby(['route_id','zone_id']):
    r = idx_[0]
    z1 = idx_[1]

    if r not in x:
        x[r] = {}
    if r not in y:
        y[r] = {}
    if r not in zone_id:
        zone_id[r] = {}

    ##
    #info_shuffle = info.sample(frac = 1)
    ##
    attr = info[feature_col_test].copy()
    x[r][z1] = attr

    next_zone = info['next_zone_id'].iloc[0]
    y[r][z1] = next_zone


    zone_id[r][z1] = list(info['zone_id_avail'])
    a=1

print('for loop time', time.time() - tic)
# test_route = 'RouteID_7c3418fd-da5f-4732-b5d8-75a7f8fe1158'
# z_test = 'B-10.2B'
# print(zone_id[test_route][z_test])
# check = data_pc.loc[(data_pc['route_id'] == test_route)&(data_pc['zone_id'] == z_test)]

with open('data/processed_test_zone_seq_neighbour_all' '.pkl', 'wb') as f:
    pkl.dump(x,f)
    pkl.dump(y,f)
    pkl.dump(zone_id, f)

