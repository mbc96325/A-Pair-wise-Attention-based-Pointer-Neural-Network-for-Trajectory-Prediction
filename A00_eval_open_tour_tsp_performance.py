import json
import numpy as np
import pandas as pd
import pickle as pkl
import sys
from datetime import datetime
import os

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from pointer_net import Encoder, Decoder, PointerNetwork
from cython_score_evaluate import evaluate_simple
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor


class Data(Dataset):
    def __init__(self, X, Y, zone_len, route_train, init_dist_seq, device):
        self.x = torch.FloatTensor(X).to(device)
        self.y = torch.LongTensor(Y).to(device)
        self.zone_len = torch.LongTensor(zone_len).reshape(-1).to(device)
        self.route_train = route_train
        self.len = self.x.shape[0]
        self.init_dist_seq = torch.FloatTensor(init_dist_seq).to(device)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.zone_len[index], self.route_train[index], self.init_dist_seq[
            index], index

    def __len__(self):
        return self.len


def zone_id_feature_to_zone_id(zone_id_feature):
    zone1 = chr(int(zone_id_feature[0]) + 65)
    zone2 = str(int(zone_id_feature[1]))
    zone3 = str(int(zone_id_feature[2]))
    zone4 = chr(int(zone_id_feature[3]) + 65)
    return zone1 + '-' + zone2 + '.' + zone3 + zone4


HIDDEN_SIZE_asudnn = 128
HIDDEN_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 30

initial_seq = 'tsp'
Random_seed = 123

LOAD_COST_Mat = True

print('=======Init sequence', initial_seq, '==========')


def process_data():
    if os.path.exists('../data/open_tour_tsp_zone_seq.csv'):
        tsp = pd.read_csv('../data/open_tour_tsp_zone_seq.csv')
        with open('../data/opt_zone_seq_open_tour_pure_tsp.json', 'rb') as f:
            tsp_zone_seq_mean = json.load(f)
    else:
        # Load TSP Sequence
        with open('../data/opt_zone_seq_open_tour_pure_tsp.json', 'rb') as f:
            tsp_zone_seq_mean = json.load(f)
        tsp_list = []
        for r in tsp_zone_seq_mean:
            df = pd.DataFrame(tsp_zone_seq_mean[r], columns=['zone_id'])
            df['route_id'] = r
            df['pred_seq_id'] = df.index + 1
            tsp_list.append(df)
        tsp = pd.concat(tsp_list)

        tsp.sort_values(by=['route_id', 'pred_seq_id'], inplace=True)
        tsp.to_csv('../data/open_tour_tsp_zone_seq.csv', index=False)

    with open('../data/zone_mean_travel_times.json') as f:
        zone_mean_dist = json.load(f)

    if os.path.exists('../data/zone_mean_travel_times.csv'):
        zone_dist_df = pd.read_csv('../data/zone_mean_travel_times.csv')
    else:
        zone_dist_dict = {'route_id': [], 'From_zone': [], 'To_zone': [], 'travel_time': []}
        for r in zone_mean_dist:
            for from_zone in zone_mean_dist[r]:
                for to_zone in zone_mean_dist[r][from_zone]:
                    zone_dist_dict['route_id'].append(r)
                    zone_dist_dict['From_zone'].append(from_zone)
                    zone_dist_dict['To_zone'].append(to_zone)
                    zone_dist_dict['travel_time'].append(zone_mean_dist[r][from_zone][to_zone])

        zone_dist_df = pd.DataFrame(zone_dist_dict)
        zone_dist_df.to_csv('../data/zone_mean_travel_times.csv', index=False)

    a = 1
    # Attributes
    data = pd.read_csv("../data/zone_data.csv")
    data = data.sort_values(by=['route_id', 'zone_seq']).reset_index(drop=True)

    # Node Attributes
    data["first_lat"] = data.groupby('route_id')['lat_mean'].transform("first")
    data["first_lng"] = data.groupby('route_id')['lng_mean'].transform("first")

    data['lat_mean'] -= data['first_lat']
    data['lng_mean'] -= data['first_lng']

    data['before_7'] = (data['hour'] <= 7).astype(int)
    data['after_10'] = (data['hour'] >= 10).astype(int)

    data = data[data['zone_id'] != 'INIT']

    # Number of zones
    zone_len = data.groupby('route_id').count().zone_id
    zone_len_max = zone_len.max()
    zone_len_min = zone_len.min()
    print('max len', zone_len_max, 'min len', zone_len_min)

    zone_len = zone_len.to_numpy()

    temp = data['zone_id'].str.split('-', expand=True)
    temp2 = temp.iloc[:, 1].str.split('.', expand=True)
    temp2['2'] = temp2.iloc[:, 1].apply(lambda x: x[:-1])
    temp2['3'] = temp2.iloc[:, 1].apply(lambda x: x[-1])

    data['zone_id_1'] = temp.iloc[:, 0].apply(lambda x: ord(x)) - 65
    data['zone_id_2'] = np.int32(temp2.iloc[:, 0])
    data['zone_id_3'] = np.int32(temp2['2'])
    data['zone_id_4'] = temp2['3'].apply(lambda x: ord(x)) - 65

    zone_id_feature = ['tsp_seq_id', 'zone_id_1', 'zone_id_2', 'zone_id_3', 'zone_id_4']
    str_info = ['zone_id']
    col_to_encode = ['lat_mean', 'lng_mean', 'n_pkg', 'planned_service_time_sum', 'zone_id_1',
                     'zone_id_2']  # ,'total_num_stops_per_zone','n_pkg','planned_service_time_sum','tsp_seq_id', 'zone_id_1','zone_id_2','zone_id_3','zone_id_4'

    ########

    # data['']
    #########

    # Padding all sequences to zone_len_max
    route_zone_ = data[['route_id']].drop_duplicates()
    route_zone_['key'] = 1
    temp = pd.DataFrame({'key': [1] * zone_len_max, 'zone_seq_new': np.arange(1, zone_len_max + 1) + 1})
    route_zone_ = route_zone_.merge(temp, on=['key'])
    route_zone_ = route_zone_.drop(columns=['key'])

    data_extend = data.merge(route_zone_, left_on=['route_id', 'zone_seq'], right_on=['route_id', 'zone_seq_new'],
                             how='right')

    for key in ['lat_mean', 'lng_mean', 'total_num_stops_per_zone', 'num_tra_sig', 'before_7', 'after_10']:
        data_extend[key] = data_extend[key].ffill()
        data_extend[key] = data_extend[key].bfill()

    data_extend['zone_seq'].fillna(-1, inplace=True)

    if initial_seq == 'random':
        # Shuffle
        data_extend['indicator'] = 1
        data_extend.loc[data_extend['zone_seq'] > 0, 'indicator'] = -1
        np.random.seed(Random_seed)
        data_extend = data_extend.sample(frac=1)
        data_extend.sort_values(by=['route_id', 'indicator'], inplace=True)
        data_extend = data_extend.reset_index(drop=True)
        data_extend['input_seq_id'] = data_extend.groupby('route_id', sort=False).cumcount() + 2
    elif initial_seq == 'tsp':
        # Take TSP Sequence
        tsp['input_seq_id'] = tsp['pred_seq_id']
        data_extend = data_extend.merge(tsp, on=['route_id', 'zone_id'], how='left')
        data_extend.loc[data_extend['input_seq_id'].isna(), 'input_seq_id'] = data_extend.loc[
            data_extend['input_seq_id'].isna(), 'zone_seq_new']
        data_extend = data_extend.sort_values(['route_id', 'input_seq_id'])

    # add tsp seq as feature
    tsp['tsp_seq_id'] = tsp['pred_seq_id']
    print('num tsp route', len(pd.unique(tsp['route_id'])))
    data_extend = data_extend.merge(tsp[['zone_id', 'route_id', 'tsp_seq_id']], on=['route_id', 'zone_id'], how='left')
    data_extend.loc[data_extend['tsp_seq_id'].isna(), 'tsp_seq_id'] = data_extend.loc[
        data_extend['tsp_seq_id'].isna(), 'input_seq_id']

    for key in ['zone_id_1', 'zone_id_2', 'zone_id_3', 'zone_id_4', 'n_pkg', 'planned_service_time_sum']:
        data_extend[key] = data_extend[key].fillna(0)

    # Get y values: the position of element in the input sequence (x-index)
    data_extend['y_not_used'] = data_extend['zone_id'].isna()
    data_extend['x_index'] = data_extend.groupby('route_id', sort=False).cumcount()
    data_extend.sort_values(by=['route_id', 'zone_seq_new'], inplace=True)  # sort to get the y

    data_extend['y'] = data_extend['x_index']
    data_extend.loc[data_extend['y_not_used'], 'y'] = -1
    y = data_extend['y'].values.reshape((-1, zone_len_max))
    # y[0,:]

    data_extend.sort_values(by=['route_id', 'input_seq_id'], inplace=True)

    zone_dist_df = zone_dist_df.merge(data_extend[['route_id', 'zone_id', 'input_seq_id']],
                                      left_on=['route_id', 'To_zone'], right_on=['route_id', 'zone_id'])

    zone_dist_df = zone_dist_df.rename(columns={'input_seq_id': 'To_zone_seq'})

    zone_dist_df['To_zone_seq'] -= 2
    zone_dist_df_pivot = zone_dist_df.pivot_table('travel_time', ['route_id', 'From_zone'], 'To_zone_seq').reset_index(
        drop=False)

    travel_time_col_list = []
    for i in range(zone_len_max):
        zone_dist_df_pivot = zone_dist_df_pivot.rename(columns={i: 'tt_to_input_zone_seq_' + str(i)})
        travel_time_col_list.append('tt_to_input_zone_seq_' + str(i))

    # normalize and fillna
    zone_dist_df_pivot['max_tt_to'] = np.nanmax(zone_dist_df_pivot[travel_time_col_list].values, axis=1)

    for key in travel_time_col_list:
        zone_dist_df_pivot[key] /= zone_dist_df_pivot['max_tt_to']

    zone_dist_df_pivot = zone_dist_df_pivot.fillna(1.5)  # max = 1, not available = 1.5

    ini_to_first_zone = zone_dist_df_pivot.loc[zone_dist_df_pivot['From_zone'] == 'INIT']

    a = 1
    # col_temp = list(zone_dist_df_pivot.columns)

    # zone_dist_df = zone_dist_df.sort_values(['route_id','From_zone'])

    # zone_dist_df_extend = zone_dist_df.merge(route_zone_, left_on=['route_id', 'To_zone_seq'], right_on=['route_id', 'zone_seq_new'], how='right')

    # print(len1, len(zone_dist_df))
    # assert len1 == len(zone_dist_df)
    # first sort to_zone by input seq id
    # then

    data_extend = data_extend.merge(zone_dist_df_pivot[['route_id', 'From_zone'] + travel_time_col_list],
                                    left_on=['route_id', 'zone_id'], right_on=['route_id', 'From_zone'], how='left')
    # data_extend.sort_values(by=['route_id','input_seq_id','To_zone_seq'], inplace=True)
    # a=1
    for key in travel_time_col_list:
        data_extend[key] = data_extend[key].fillna(1.5)

    data_extend['zone_id'] = data_extend['zone_id'].fillna('NoDef')

    feature_cols = col_to_encode + travel_time_col_list + zone_id_feature  # 'total_num_stops_per_zone

    # feature_cols = ['lat_mean','lng_mean','zone_id_1','zone_id_2','zone_id_3','zone_id_4'] + zone_id_feature

    num_features = len(feature_cols)
    num_features_without_zone = len(feature_cols) - len(zone_id_feature)
    num_norm_feature = len(col_to_encode)
    data_extend['zone_id'] = data_extend['zone_id'].fillna('NoDef')

    data_extend.sort_values(by=['route_id', 'input_seq_id'], inplace=True)
    routes = list(pd.unique(data_extend['route_id']))
    x = data_extend[feature_cols].to_numpy().reshape(-1, zone_len_max, num_features)

    # print(x.shape)
    # print(x[0,:,:])
    # for i in range(num_features):
    #     print(x[0,:,i])

    x = x.astype(float)
    if np.isnan(x[0, :, :]).any():
        print('*****exist nan in x')
        exit()
    else:
        print('===No nan in x===')

    str_info_np = data_extend[str_info].to_numpy().reshape(-1, zone_len_max)

    # test1 = data_extend.loc[data_extend['route_id'] == routes_test[1]]

    # Zone_id look up in x: input_seq[(route_id,index in x)] = zone_id
    input_seq = data_extend[['route_id', 'x_index', 'zone_id']].reset_index()
    input_seq = input_seq.set_index(['route_id', 'x_index'])
    input_seq_dict = input_seq['zone_id'].to_dict()

    # Normalization
    num_samples, _, _ = x.shape
    for i in range(num_samples):
        for j in range(num_norm_feature):
            if np.max(np.abs(x[i, :, j])) > 0:
                x[i, :, j] /= np.max(np.abs(x[i, :, j]))

    device = torch.device("cpu")

    # Train Test Split
    with open("../data/train_routes.pkl", "rb") as f:
        routes_train = pkl.load(f)
    with open("../data/testing_routes.pkl", "rb") as f:
        routes_test = pkl.load(f)

    #
    if LOAD_COST_Mat:
        with open('../data/cost_mtx_array.pkl', 'rb') as f:
            cost_mat_array = pkl.load(f)
        with open('../data/stop_idx_map.pkl', 'rb') as f:
            name_map_all = pkl.load(f)
    else:
        cost_mat_array = None
        name_map_all = None

    X_train = x[np.isin(routes, routes_train), :, :]
    Y_train = y[np.isin(routes, routes_train), :]
    str_info_np_train = str_info_np[np.isin(routes, routes_train), :]
    # print(X_train.shape)
    route_train_seq = np.array(routes)[np.isin(routes, routes_train)]
    init_dist_seq = ini_to_first_zone[travel_time_col_list].values[np.isin(routes, routes_train), :]

    zone_len_train = zone_len[np.isin(routes, routes_train)]

    X_test = x[np.isin(routes, routes_test), :, :]
    Y_test = y[np.isin(routes, routes_test), :]
    str_info_np_test = str_info_np[np.isin(routes, routes_test), :]
    zone_len_test = zone_len[np.isin(routes, routes_test)]

    route_test_seq = np.array(routes)[np.isin(routes, routes_test)]

    Y_train_t = torch.FloatTensor(Y_train).to(device)
    X_train_t = torch.FloatTensor(X_train).to(device)
    Y_test_t = torch.FloatTensor(Y_test).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    zone_len_train_t = torch.LongTensor(zone_len_train).to(device)
    zone_len_test_t = torch.LongTensor(zone_len_test).to(device)

    init_dist_seq_test = ini_to_first_zone[travel_time_col_list].values[np.isin(routes, routes_test), :]
    init_dist_seq_test_t = torch.FloatTensor(init_dist_seq_test).to(device)
    return X_train, Y_train, zone_len_train, route_train_seq, device, init_dist_seq, \
           zone_len_max, input_seq_dict, num_features_without_zone, str_info_np_train, route_test_seq, \
           X_test_t, Y_test_t, zone_len_test_t, init_dist_seq_test_t, zone_mean_dist, zone_id_feature, \
           str_info_np_test, Y_test, cost_mat_array, name_map_all, zone_len_test, tsp_zone_seq_mean



def generate_seq_with_multiple_first_zone(Net,max_len,X_test_all,Y_test_all, zone_len_test_all, route_test_seq_all,
                                          init_dist_seq_test_all, str_info_np_test_all, zone_mean_dist):
    # results_all = {''}
    PATTERN_FORCE = False

    results_all = []
    for t in range(max_len):
        # result_temp = {'route_id':[],'start_zone':[],'pred_zone_seq':[],'pred_zone_id':[]}
        result_temp = {}
        valid = zone_len_test_all > t
        X_test_t = X_test_all[valid,:,:]
        Y_test_t = Y_test_all[valid,:]
        zone_len_test_t = zone_len_test_all[valid]
        route_test_seq = route_test_seq_all[valid]
        init_dist_seq_test_t = init_dist_seq_test_all[valid,:]
        str_info_np_test = str_info_np_test_all[valid,:]
        # use t-th zone as the initial
        Net.eval()
        if sum(valid) > 0:
            out, loss, weights = Net(X_test_t, Y_test_t, zone_len_test_t, route_test_seq, init_dist_seq_test_t,
                                         str_info_np_test, zone_mean_dist,
                                         mask_prob=True, teacher_force=False, Print_pred=False, No_dyn_dist=True,
                                         model_apply=True, Calculate_loss=False, initial_pred_id = t, pattern_force = PATTERN_FORCE)
        else:
            break

        first_zone_prob = weights[:, 0, :] # torch.gather(weights[:, 0,:], -1, Y_test_t[:,t].long())

        first_zone_prob_ = torch.gather(first_zone_prob, 1, Y_test_t[:,t].long().view(-1,1)).squeeze()
        out = out.permute(1, 0)
        pred_seq_list = []
        for r, i in zip(route_test_seq, range(len(route_test_seq))):
            # Predicted sequence
            predict_seq = ['INIT']
            for j in out[i, :zone_len_test_t[i]]:
                # try:
                predict_seq.append(input_seq_dict[(r, int(j))])
                # except:
                #     a=1
            pred_seq_list.append(predict_seq)

        result_temp['route_id'] = route_test_seq
        result_temp['start_zone'] = Y_test_t[:,t]
        result_temp['start_zone_prob'] = first_zone_prob_.detach().numpy()
        result_temp['zone_len'] = np.array(zone_len_test_t)
        result_temp['pred_zone_seq'] = pred_seq_list
        result_temp_df = pd.DataFrame(result_temp)
        results_all.append(result_temp_df)

    results_all = pd.concat(results_all)
    if PATTERN_FORCE:
        results_all.to_csv('generated_seq_diff_first_zone_PATTERN_FORCE.csv',index=False)
    else:
        results_all.to_csv('generated_seq_diff_first_zone.csv', index=False)


if __name__ == '__main__':

    X_train, Y_train, zone_len_train, route_train_seq, device, init_dist_seq, zone_len_max, \
    input_seq_dict, num_features_without_zone, str_info_np_train, route_test_seq, \
    X_test_t, Y_test_t, zone_len_test_t, init_dist_seq_test_t, zone_mean_dist, zone_id_feature, \
    str_info_np_test, Y_test, cost_mat_array, name_map_all, zone_len_test, tsp_zone_seq_mean = process_data()


    s_ptr = 0
    s_tsp = []
    route_score = []
    pred_seq_list = []
    actual_seq_list = []

    data_stops = pd.read_csv('../data/build_route_with_seq.csv')
    data_stops['stops'] = data_stops['stops'].fillna('NA')

    first_zone = 0
    second_zone = 0
    third_zone = 0
    forth_zone = 0

    for r, i in zip(route_test_seq, range(len(route_test_seq))):

        data_route = data_stops.loc[data_stops['route_id'] == r]

        if LOAD_COST_Mat:
            cost_mat_route = cost_mat_array[r]
            name_map = name_map_all[r]

        # Actual sequence
        actual_zone_seq = ['INIT']
        for j in Y_test[i, :zone_len_test[i]]:
            actual_zone_seq.append(input_seq_dict[(r, int(j))])

        # actual_zone_s = list(zone_data.loc[zone_data['route_id'] == r, 'zone_id'])
        # print(X_test_t[i,:,:].shape)

        act_zone_df = pd.DataFrame(
            {'zone_id': actual_zone_seq, 'pred_seq_id': np.arange(zone_len_test[i] + 1) + 1})
        act_zone_df = act_zone_df.merge(data_route[['stops', 'zone_id', 'seq_ID']], on=['zone_id'])
        act_zone_df = act_zone_df.sort_values(['pred_seq_id', 'seq_ID'])

        # TSP score - 0.04425974236637206
        # TSP score - 0.03318142779198891

        tsp_zone_seq = tsp_zone_seq_mean[r]


        if tsp_zone_seq[1] == actual_zone_seq[1]:
            first_zone += 1
        if tsp_zone_seq[2] == actual_zone_seq[2]:
            second_zone += 1
        if tsp_zone_seq[3] == actual_zone_seq[3]:
            third_zone += 1
        if tsp_zone_seq[4] == actual_zone_seq[4]:
            forth_zone += 1

        pred_zone_df = pd.DataFrame({'zone_id':tsp_zone_seq_mean[r], 'pred_seq_id':np.arange(zone_len_test[i]+1) + 1})


        pred_zone_df = pred_zone_df.merge(data_route[['stops', 'zone_id', 'seq_ID']], on=['zone_id'])
        pred_zone_df = pred_zone_df.sort_values(['pred_seq_id', 'seq_ID'])
        est_seq = np.array([name_map[s] for s in pred_zone_df['stops']])
        actual_seq = np.array([name_map[s] for s in act_zone_df['stops']])
        score, seq_dev, erp_per_edit, total_dist, total_edit_count = evaluate_simple(actual_seq, est_seq, cost_mat_route)

        s_tsp.append(score)

    print('First zone accuracy {:.4}'.format(first_zone / len(Y_test)))
    print('Second zone accuracy {:.4}'.format(second_zone / len(Y_test)))
    print('Third zone accuracy {:.4}'.format(third_zone / len(Y_test)))
    print('Forth zone accuracy {:.4}'.format(forth_zone / len(Y_test)))
    print('tsp mean score', np.mean(s_tsp))
    print('tsp std score', np.std(s_tsp))