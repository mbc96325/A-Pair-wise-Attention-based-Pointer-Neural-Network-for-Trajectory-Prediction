# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 18:46:24 2021

@author: qingyi
"""

import numpy as np
import pandas as pd
import pickle as pkl
import sys
import json
import time

import _constants

n_neighbour = _constants.n_neighbour
num_neighbor = n_neighbour
data = pd.read_csv("../data/zone_data.csv")

FILTER_PASSED = True



# %%
with open('../data/zone_mean_travel_times.json') as f:
    tt_avg = json.load(f)
# with open('../../data/zone_min_travel_times.json') as f:
#     tt_min = json.load(f)

x = []
y = []
# nearest_neighbor = {'route_id':[],'zone_id':[],'nearest_neighbor':[]}
nearest_neighbor = {}
route_id = []
zone_id = []
x_lstm = []

# with open('../../tsp_xiaotong/mean_dist/opt_zone_seq.p', 'rb') as f:
#     tsp_zone_seq_mean = pkl.load(f)
#
# with open('../../tsp_xiaotong/min_dist/opt_zone_seq.p', 'rb') as f:
#     tsp_zone_seq_min = pkl.load(f)

with open('../data/mean_dist/opt_zone_seq.json', 'rb') as f:
    tsp_zone_seq_mean = json.load(f)

with open('../data/min_dist/opt_zone_seq.json', 'rb') as f:
    tsp_zone_seq_min = json.load(f)


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



def angle_between_three_points(a: np.array, b: np.array, c: np.array):
    ba_original = a - b
    bc_original = c - b
    ba = np.squeeze(np.asarray(ba_original))
    bc = np.squeeze(np.asarray(bc_original))
    cosine_angle = np.sum(ba * bc,axis = 1) / (np.linalg.norm(ba,axis = 1) * np.linalg.norm(bc, axis = 1))
    cosine_angle[cosine_angle>1] = 1
    cosine_angle[cosine_angle < -1] = -1
    angle = np.arccos(cosine_angle)
    return angle, cosine_angle

# worst_value = np.array([[5000,5000,0,0,600,99999,99999,99999]])

data = data.sort_values(['route_id','zone_seq'])
data['total_num_zones'] = data.groupby(['route_id'])['zone_id'].transform('count')

data['current_zone_percentage'] = data['zone_seq'] / data['total_num_zones']


data['total_num_zones'] = data['total_num_zones']/np.max(data['total_num_zones']) # normalize


#
# data['max_num_tra_sig'] = data.groupby(['route_id'])['num_tra_sig'].transform("max")
# data['max_n_pkg'] = data.groupby(['route_id'])['n_pkg'].transform("max")
#
# data['num_tra_sig'] /= data['max_num_tra_sig'] # normalize
# data['n_pkg'] /= data['max_n_pkg'] # normalize

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
data_copy['n_pkg_avail'] = data_copy['n_pkg']
data_copy['num_stops_avail'] = data_copy['total_num_stops_per_zone']
data_copy['weekends_avail'] = data_copy['weekends']
data_copy['before_7am_avail'] = data_copy['before_7am']
data_copy['after_10am_avail'] = data_copy['after_10am']


# data_copy['total_num_stops_avail'] = data_copy['total_num_stops_per_zone']
#data_copy['num_tra_sig_avail'] = data_copy['num_tra_sig']

data_copy['key'] = 1
data['key'] = 1

data_pc = data.merge(data_copy[['route_id','zone_id_avail','zone_seq_avail','lat_avail','lng_avail','key','num_stops_avail','n_pkg_avail',
                                'num_tra_sig_avail','weekends_avail','before_7am_avail','after_10am_avail']], on = ['route_id','key'])



if FILTER_PASSED:
    data_pc = data_pc.loc[data_pc['zone_seq'] < data_pc['zone_seq_avail']]
    #
    #a=1

#data_pc = data_pc.loc[data_pc['zone_seq'] < data_pc['zone_seq_avail']]



with open('../data/zone_tt.pkl', 'rb') as f:
    zone_min_tt_df_out = pkl.load(f)
    zone_mean_tt_df_out = pkl.load(f)


data_pc = data_pc.merge(zone_min_tt_df_out, left_on = ['route_id','zone_id','zone_id_avail'], right_on = ['route_id','from_zone','to_zone'])
data_pc = data_pc.drop(columns = ['from_zone','to_zone'])
data_pc = data_pc.merge(zone_mean_tt_df_out, left_on = ['route_id','zone_id','zone_id_avail'], right_on = ['route_id','from_zone','to_zone'])
data_pc = data_pc.drop(columns = ['from_zone','to_zone'])


data_pc = data_pc.sort_values(['route_id','zone_seq','mean_travel_time'])
data_pc = data_pc.groupby(['route_id','zone_seq']).head(num_neighbor).reset_index(drop=True)


if FILTER_PASSED:
    add_last = data.groupby(['route_id']).last().reset_index()
    data_pc = pd.concat([data_pc, add_last],sort=False)
    data_pc = data_pc.sort_values(['route_id', 'zone_seq','mean_travel_time'])
    data_pc['min_travel_time'] = data_pc['min_travel_time'].fillna(3000)
    data_pc['mean_travel_time'] = data_pc['mean_travel_time'].fillna(3000)
    data_pc['zone_id_avail'] =  data_pc['zone_id_avail'].fillna('END')
    data_pc = data_pc.fillna(0)
    a=1

data_pc['next_in_neighbor'] = 0
data_pc.loc[data_pc['next_zone_id'] == data_pc['zone_id_avail'], 'next_in_neighbor'] = 1

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








data_pc = data_pc.fillna(0)
angle, cosine_angle = angle_between_three_points(data_pc[['last_lat','last_lng']].values,
                                                 data_pc[['lat_mean','lng_mean']].values,
                                                 data_pc[['lat_avail','lng_avail']].values)
data_pc['degree'] = angle/np.pi * 180
data_pc['cosine_angle'] = cosine_angle
# min_ = np.min(data_pc['degree'])
# max_ = np.max(data_pc['degree'])


data_pc.loc[data_pc['zone_id'] == 'INIT','degree'] = 180
data_pc['forward'] = 0
data_pc.loc[(data_pc['degree'] >= 135), 'forward'] = 1

# data_pc['left_side'] = 0
# data_pc['right_side'] = 0
# data_pc['third_y_minus_first_y'] = data_pc['lng_avail'] - data_pc['last_lng']
# data_pc.loc[(data_pc['degree'] >= 45) & (data_pc['degree'] <= 135) &  (data_pc['third_y_minus_first_y'] >= 0), 'left_side'] = 1 # & (data_pc['cosine_angle']  135)
# data_pc.loc[(data_pc['degree'] >= 45) & (data_pc['degree'] <= 135) &  (data_pc['third_y_minus_first_y'] < 0), 'right_side'] = 1

data_pc.loc[(data_pc['degree'] >= 45) & (data_pc['degree'] < 135), 'side'] = 1

data_pc['backward'] = 0
data_pc.loc[(data_pc['degree'] < 45), 'backward'] = 1



#tsp_zone_seq_mean_df = tsp_zone_seq_mean_df.sort_values(['route_id'])

data_pc = data_pc.merge(tsp_zone_seq_mean_df[['route_id','zone_id','tsp_mean_next_zone']], on = ['route_id','zone_id'])
data_pc = data_pc.merge(tsp_zone_seq_min_df[['route_id','zone_id','tsp_min_next_zone']], on = ['route_id','zone_id'])



data_pc['tsp_next_mean'] = 0
data_pc['tsp_next_min'] = 0

data_pc.loc[data_pc['tsp_mean_next_zone'] == data_pc['zone_id_avail'], 'tsp_next_mean'] = 1
data_pc.loc[data_pc['tsp_min_next_zone'] == data_pc['zone_id_avail'], 'tsp_next_min'] = 1

data_pc.loc[data_pc['min_travel_time'] < 1, 'min_travel_time'] = 1
data_pc.loc[data_pc['mean_travel_time'] < 1, 'mean_travel_time'] = 1

data_pc['zone_avail_seq'] = data_pc.groupby(['route_id','zone_id']).cumcount()






print('process time', time.time() - tic)

tic = time.time()

route_zone_ = data_pc[['route_id','zone_id','zone_seq']].drop_duplicates()
route_zone_['key'] = 1
temp = pd.DataFrame({'key': [1]* num_neighbor, 'zone_avail_seq': np.arange(num_neighbor)})
route_zone_ = route_zone_.merge(temp, on = ['key'])

route_zone_ = route_zone_.drop(columns = ['key'])
data_pc = data_pc.merge(route_zone_, on = ['route_id','zone_id','zone_avail_seq','zone_seq'], how = 'right')

data_pc['FILL_EMPTY'] = data_pc['station_code'].isna().astype('int')

# assign worst value to nan

data_pc.loc[data_pc['degree'].isna(),'degree'] = 0
data_pc.loc[data_pc['forward'].isna(),'forward'] = 0
# data_pc.loc[data_pc['left_side'].isna(),'left_side'] = 0
# data_pc.loc[data_pc['right_side'].isna(),'right_side'] = 0

data_pc.loc[data_pc['side'].isna(),'side'] = 0

data_pc.loc[data_pc['backward'].isna(),'backward'] = 1
data_pc.loc[data_pc['tsp_next_mean'].isna(),'tsp_next_mean'] = 0
data_pc.loc[data_pc['tsp_next_min'].isna(),'tsp_next_min'] = 0
data_pc.loc[data_pc['num_tra_sig_avail'].isna(),'num_tra_sig_avail'] = 0
data_pc.loc[data_pc['num_stops_avail'].isna(),'num_stops_avail'] = 0
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

#################### LSTM embeeding data###########
r_z_unique = data[['route_id','zone_id','zone_seq']].drop_duplicates()
z_att = _constants.z_attr
data_to_merge = data.loc[:,['route_id'] + z_att]
data_to_merge['zone_seq_passed'] = data['zone_seq']
data_to_merge['zone_id_passed'] = data['zone_id']

lstm_data = r_z_unique.merge(data_to_merge, on = ['route_id'])
#lstm_data = lstm_data.sort_values(['route_id','zone_seq'])

#lstm_data = lstm_data

lstm_data = lstm_data.loc[lstm_data['zone_seq'] > lstm_data['zone_seq_passed']]

add_first = r_z_unique.loc[r_z_unique['zone_seq'] == 1]
lstm_data = pd.concat([add_first, lstm_data], sort=False)
lstm_data = lstm_data.sort_values(['route_id','zone_seq'])

lstm_data.loc[lstm_data['zone_seq'] == 1, 'zone_id_passed'] = 'NO_PASSED'

lstm_data = lstm_data.fillna(0)

lstm_data = lstm_data.sort_values(['route_id','zone_id','zone_seq_passed'])
lstm_data['zone_passed_idx'] =lstm_data.groupby(['route_id','zone_id']).cumcount() + 1

max_zone_len = r_z_unique.groupby(['route_id'])['zone_id'].count().max() - 1

lstm_length = lstm_data.groupby(['route_id','zone_id','zone_seq'],sort=False)['zone_seq_passed'].count().reset_index()
lstm_length = lstm_length.sort_values(['route_id','zone_seq'])
lstm_length.loc[lstm_length['zone_id'] == 'INIT', 'zone_seq_passed'] = 0
#lstm_length = lstm_length.

# Padding all sequences to zone_len_max
route_zone_passed_zone = r_z_unique[['route_id','zone_id']].drop_duplicates()
route_zone_passed_zone['key'] = 1
temp = pd.DataFrame({'key': [1] * max_zone_len, 'zone_passed_idx_new': np.arange(1, max_zone_len + 1)})
route_zone_passed_zone = route_zone_passed_zone.merge(temp, on=['key'])
route_zone_passed_zone = route_zone_passed_zone.drop(columns=['key'])

lstm_data_extend = lstm_data.merge(route_zone_passed_zone, left_on = ['route_id','zone_id','zone_passed_idx'], right_on = ['route_id','zone_id','zone_passed_idx_new'], how = 'right')


#lstm_data_extend = lstm_data_extend.sort_values(['route_id','zone_seq'])
lstm_data_extend = lstm_data_extend.ffill()

a=1
################




if FILTER_PASSED:
    tail = 'filter_passed'
else:
    tail = ''
with open('data/processed_zone_info_neighbour_'+ str(num_neighbor) + '_' + tail + '.pkl', 'wb') as f:
    pkl.dump(data_pc, f)

with open('data/LSTM_zone_info_neighbour_'+ str(num_neighbor) + '_' + tail + '.pkl', 'wb') as f:
    pkl.dump(lstm_data_extend, f)

print('fill na and save time', time.time() - tic)


# count = 0
# # total_num = pd.unique()
# for idx_, info in data_pc.groupby(['route_id','zone_id']):
#     r = idx_[0]
#     z1 = idx_[1]
#     count += 1
#     nearest_neighbor[idx_[0]] = {}
#     nearest_neighbor[idx_[0]][idx_[1]] = list(info['zone_id_avail'])
#
#     route_id.append(r)
#     zone_id.append(z1)
#
#     if len(info) ==  n_neighbour:
#         attr = info.loc[:, feature_col].values
#         actual_next = info['next_zone_id'].iloc[0]
#         if sum(info['next_in_neighbor']) == 0:
#             y.append(-1)
#         else:
#             y.append(np.argmax(np.array(info['next_in_neighbor']) == 1))
#         x.append(attr)
#     else:
#         attr = info.loc[:, feature_col].values
#         actual_next = info['next_zone_id'].iloc[0]
#         if sum(info['next_in_neighbor']) == 0:
#             y.append(-1)
#         else:
#             y.append(np.argmax(np.array(info['next_in_neighbor']) == 1))
#
#         max_tt = attr[:, 0].max() * 1.5
#         max_ttm = attr[:, 1].max() * 1.5
#         # feature_col = ['min_travel_time', 'mean_travel_time', 'degree', 'forward', 'side', 'backward', 'tsp_next_mean',
#         #                'tsp_next_min', 'num_tra_sig_avail', 'before_7am_avail', 'after_10am_avail', 'weekends_avail']
#         worst_value = np.array([[max_tt, max_ttm, 0, 0, 0, 1, 0,
#                                  0, 0, 0, 0, 0]])
#         attr = np.vstack([attr, np.repeat(worst_value, n_neighbour - attr.shape[0], axis=0)])
#         x.append(attr)
#
# print('for loop to np array time', time.time() - tic)
#
# x = np.array(x)
# y = np.array(y)
#
# with open('../processed_zone_seq_neighbour_5.pkl', 'wb') as f:
#     pkl.dump(x, f)
#     pkl.dump(y, f)
#
# with open('../route_id_seq.pkl', 'wb') as f:
#     pkl.dump(route_id, f)
#
# with open('../zone_id_seq.pkl', 'wb') as f:
#     pkl.dump(zone_id, f)
#
# with open('../nearest_neighbor_5.pkl', 'wb') as f:
#     pkl.dump(nearest_neighbor, f)


