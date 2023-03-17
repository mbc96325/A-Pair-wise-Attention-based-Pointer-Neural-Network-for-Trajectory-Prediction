import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
import copy
import sys
import pickle
from torch.utils.data import Dataset, DataLoader
import _constants
from asu_nn import ASU_NN, ASU_NN_RL
import time

class Data(Dataset):
    def __init__(self, X, Y, device):
        self.x = torch.FloatTensor(X).to(device)
        self.y = torch.LongTensor(Y).reshape(-1).to(device)
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


def train(X_train_t, Y_train_t, X_test_t, Y_test_t, N_training_epoch, net, optimizer, loss_func, trainloader, device):
    # torch.manual_seed(1234)  # for reproducibility
    acc_records = []
    loss_rocords = []
    loss_rocords_test = []
    acc_records_test = []
    for t in range(N_training_epoch):
        for x, y in trainloader:
            net.train()
            out = net(x)  # input x and predict based on x
            loss = loss_func(out, y)  # must be (1. nn output, 2. target), the target label is NOT one-hotted
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            # print(loss)
            # print(out.detach().numpy()[:,0])
            print(x.detach().numpy()[:, 1])
            # print(torch.max(out, 1)[1])
            # print(y)
            # print(out.shape)
            break

        '''
        net.eval()
        out_train = net(X_train_t)
        loss_train = loss_func(out_train, Y_train_t)
        loss_rocords.append(float(loss_train.item()))
        out_test = net(X_test_t)
        loss_test = loss_func(out_test, Y_test_t)
        loss_rocords_test.append(float(loss_test.item()))

        if t % 5 == 0:
            # plot and show learning process
            prediction = torch.max(out_train, 1)[1]
            pred_y = prediction.data.numpy()
            target_y = Y_train_t.data.numpy()
            accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
            acc_records.append(accuracy)
            print('current training epoch %d \t train loss %.2f accuracy %.2f' % \
                  (t, loss_train, accuracy))
            prediction = torch.max(out_test, 1)[1]
            pred_y = prediction.data.numpy()
            target_y = Y_test_t.data.numpy()
            accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
            acc_records_test.append(accuracy)
        '''

    return acc_records, loss_rocords, acc_records_test, loss_rocords_test


def data_preprocessing(data, used_attributes, y_att):
    X = []
    att_index = {}
    num_att = 0
    for key in used_attributes:
        val = used_attributes[key]
        num_att_new = num_att + len(val)
        X.append(data.loc[:, val].values)
        att_index[key] = list(range(num_att, num_att_new))
        num_att = copy.deepcopy(num_att_new)
    X = np.hstack(X)
    scaler = preprocessing.MinMaxScaler().fit(X)
    X_norm = scaler.transform(X)
    Y = data.loc[:, y_att].values
    return X_norm, Y, att_index


def switch_labels(x, y, att_index):
    assert len(x) == len(y)
    label_closest = []
    for i in range(len(y)):
        a, b = np.random.choice(np.arange(5), size=2, replace=False)
        att_a = att_index['x_' + str(a + 1)]
        att_b = att_index['x_' + str(b + 1)]
        temp = x[i, att_a]
        x[i, att_a] = x[i, att_b]
        x[i, att_b] = temp
        if y[i] == a:
            y[i] = b
        elif y[i] == b:
            y[i] = a
        if (a != 0) & (b != 0):
            label_closest.append(0)
        else:
            label_closest.append(np.max([a, b]))

    return x, y, label_closest


def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def angle_between_three_points(a: np.array, b: np.array, c: np.array):
    ba_original = a - b
    bc_original = c - b
    ba = np.squeeze(np.asarray(ba_original))
    bc = np.squeeze(np.asarray(bc_original))
    cosine_angle = np.sum(ba * bc,axis = 1) / (np.linalg.norm(ba,axis = 1) * np.linalg.norm(bc, axis = 1))
    #print(cosine_angle[-1])
    cosine_angle[cosine_angle>1] = 1
    cosine_angle[cosine_angle < -1] = -1
    angle = np.arccos(cosine_angle)
    return angle, cosine_angle

def pred_zone_seq_diff_ini(zone_data_test, att_index, n_output, x_test, y_test, zone_id, test_routes, net_para, tail_name, model, all_seq = False):
    #worst_value = np.array([[5000, 5000, 0, 0, 600, 99999, 99999, 99999]])

    lat_mean_idx = feature_col_test.index('lat_mean')
    lng_mean_idx =feature_col_test.index('lng_mean')
    lat_avail_idx =feature_col_test.index('lat_avail')
    lng_avail_idx =feature_col_test.index('lng_avail')

    model.load_state_dict(net_para)
    model.eval()
    n_neighbour = _constants.n_neighbour
    device = torch.device('cpu')
    route_seq = {'route_id':[],'zone_id':[],'ini_zone_idx':[],'pred_seq_id':[]}

    #total_routes =

    #for route_id in test_routes:
    for first_zone_idx in range(_constants.n_neighbour):
        count = 0

        for route_id in test_routes:
            num_zone = len(zone_id[route_id]['INIT'])
            if num_zone <= first_zone_idx:
                continue
            count += 1
            if count % 100 == 0:
                print('current route', count, 'total', len(test_routes))
            used_stop = []
            last_zone = 'INIT'
            zone_seq_id = 1

            for k in range(num_zone + 1):
                used_stop.append(last_zone)
                route_seq['route_id'].append(route_id)
                route_seq['zone_id'].append(last_zone)
                route_seq['pred_seq_id'].append(zone_seq_id)
                route_seq['ini_zone_idx'].append(first_zone_idx)
                zone_seq_id += 1
                # try:
                x_input_all_df = x_test[route_id][last_zone]

                if last_zone == 'INIT':
                    x_input_all_df['degree'] = 180
                    x_input_all_df['forward'] = 1
                    x_input_all_df['side'] = 0
                    x_input_all_df['backward'] = 0
                else:
                    x_input_all_df['last_lat'] = last_lat
                    x_input_all_df['last_lng'] = last_lng
                    # angle, cosine_angle = angle_between_three_points(x_input_all_df[['last_lat', 'last_lng']].values,
                    #                                                  x_input_all_df[['lat_mean', 'lng_mean']].values,
                    #                                                  x_input_all_df[['lat_avail', 'lng_avail']].values)

                    x_input_all_df['degree'] = 0 / np.pi * 180
                    x_input_all_df['forward'] = x_input_all_df['degree'] >= 135
                    x_input_all_df['forward'] = x_input_all_df['forward'].astype('int')
                    x_input_all_df['side'] = (x_input_all_df['degree'] >= 45) & (x_input_all_df['degree'] < 135)
                    x_input_all_df['side'] = x_input_all_df['side'].astype('int')
                    x_input_all_df['backward'] = x_input_all_df['degree'] < 45
                    x_input_all_df['backward'] = x_input_all_df['backward'].astype('int')

                    # data_pc['left_side'] = 0
                    # data_pc['right_side'] = 0
                    # data_pc['third_y_minus_first_y'] = data_pc['lng_avail'] - data_pc['last_lng']
                    # data_pc.loc[(data_pc['degree'] >= 45) & (data_pc['degree'] <= 135) &  (data_pc['third_y_minus_first_y'] >= 0), 'left_side'] = 1 # & (data_pc['cosine_angle']  135)
                    # data_pc.loc[(data_pc['degree'] >= 45) & (data_pc['degree'] <= 135) &  (data_pc['third_y_minus_first_y'] < 0), 'right_side'] = 1


                last_lat = x_input_all_df['lat_mean'].iloc[0]
                last_lng = x_input_all_df['lng_mean'].iloc[0]

                x_input_all = x_input_all_df[feature_col].values
                # except:
                #     print(route_id)
                #     print(last_zone)
                #     exit()
                # a=1
                zone_id_list = zone_id[route_id][last_zone]
                idx = np.where(np.isin(zone_id_list, used_stop))

                x_input_all_remain = np.delete(x_input_all, idx, axis=0)

                if len(x_input_all_remain) == 0:
                    break
                zone_id_remain = np.delete(zone_id_list, idx, axis=0)
                attr =x_input_all_remain[:n_neighbour, :]
                zone_id_remain_used = zone_id_remain[:5]

                if SHUFFLE_ATTR:
                    attr, zone_id_remain_used = unison_shuffled_copies(attr, zone_id_remain_used)

                if attr.shape[0] < n_neighbour:
                    max_tt = attr[:, 0].max() * 1.5
                    max_ttm = attr[:, 1].max() * 1.5
                    worst_value = np.zeros((1, attr.shape[1]))
                    worst_value[0, 0] = max_tt
                    worst_value[0, 1] = max_ttm
                    for key in ffill_col:
                        idx_ = used_feature.index(key)
                        worst_value[0,idx_] = attr[0,idx_]
                    for key in one_col:
                        idx_ = used_feature.index(key)
                        worst_value[0,idx_] = 1
                    a=1
                    # feature_col = ['min_travel_time', 'mean_travel_time', 'num_stops_avail', 'degree', 'forward', 'side',
                    #                'backward',
                    #                'tsp_next_mean',
                    #                'tsp_next_min', 'num_tra_sig_avail', 'before_7am_avail', 'after_10am_avail',
                    #                'weekends_avail',
                    #                'total_num_zones',
                    #                'current_zone_percentage']

                    attr = np.vstack([attr, np.repeat(worst_value, n_neighbour - attr.shape[0], axis=0)])


                    # max_tt = attr[:, 0].max() * 1.5
                    # max_ttm = attr[:, 1].max() * 1.5
                    # planned_service_time_sum = attr[:, 4].max() * 1.5
                    # total_vol_sum = attr[:, 5].max() * 1.5
                    # total_vol_max = attr[:, 6].max() * 1.5
                    # time_window_end_from_departure_sec_min = attr[:, 7].max() * 1.5
                    # worst_value = np.array([[max_tt, max_ttm, 0, 0, 0, 0, planned_service_time_sum, total_vol_sum, total_vol_max,
                    #                          time_window_end_from_departure_sec_min]])
                    #
                    # #attr = np.concatenate([attr, np.zeros((n_neighbour - attr.shape[0], attr.shape[1])) - 99], axis=0)
                    # attr = np.vstack([attr, np.repeat(worst_value, n_neighbour - attr.shape[0], axis=0)])

                # normalization
                for j in need_to_norm_col:
                    if np.max(attr[:, j]) > 0:
                        attr[:, j] /= np.max(attr[:, j])

                attr = attr.reshape(-1)
                attr = np.array([attr])
                X_test_t = torch.FloatTensor(attr).to(device)
                pred_prob = model(X_test_t).detach().cpu().numpy().reshape(-1)
                pred_prob = pred_prob[:len(zone_id_remain_used)]

                if k == 0:
                    prediction = first_zone_idx
                else:
                    prediction = np.argmax(pred_prob)
                zone_pred = zone_id_remain_used[prediction]
                last_zone = zone_pred

            #attr
        #route_seq_df = pd.DataFrame(route_seq)

    route_seq_df = pd.DataFrame(route_seq)
    all_route_zone = zone_data_test[['route_id','zone_id']].drop_duplicates().copy()
    all_route_zone_pred = all_route_zone.merge(route_seq_df, on = ['route_id','zone_id'],how = 'left')
    #t_r = all_route_zone_pred['route_id'].iloc[0]
    check_na = all_route_zone_pred.loc[all_route_zone_pred['pred_seq_id'].isna()]
    assert len(check_na) == 0
    if all_seq:
        route_seq_df.to_csv('pred_zone_seq_'+tail_name+'_all_seq.csv',index=False)
    else:
        route_seq_df.to_csv('pred_zone_seq_' + tail_name + '.csv', index=False)


def pred_zone_seq_vectorize(zone_data_test, att_index, n_output, zone_df, y_test, zone_id, test_routes, net_para, tail_name, model, all_seq = False):
    #worst_value = np.array([[5000, 5000, 0, 0, 600, 99999, 99999, 99999]])

    model.load_state_dict(net_para)
    model.eval()
    n_neighbour = _constants.n_neighbour
    device = torch.device('cpu')
    route_seq = {'route_id':[],'zone_id':[],'pred_seq_id':[]}

    #total_routes =

    #for route_id in test_routes:
    count = 0

    max_zone_len = np.max(zone_data_test['zone_seq'])

    last_zone = pd.DataFrame({'route_id':test_routes, 'last_zone': ['INIT']*len(test_routes)})
    used_zone = pd.DataFrame({'route_id':test_routes, 'used_zone': ['INIT']*len(test_routes)})
    for z_ in range(max_zone_len):
        zone_df_eval =  zone_df.merge(last_zone, left_on = ['route_id','zone_id'],right_on = ['route_id','last_zone'])
        zone_df_eval = zone_df_eval.merge(used_zone, left_on = ['route_id','zone_id_avail'],right_on = ['route_id','used_zone'], how = 'left')
        zone_df_eval = zone_df_eval.loc[zone_df_eval['used_zone'].isna()]
        zone_df_eval = zone_df_eval.loc[zone_df_eval['zone_id_avail'] != zone_df_eval['']]
        zone_df_eval = zone_df_eval.groupby(['route_id']).head(num_neighbor)
        a=1
    for route_id in test_routes:
        count += 1
        if count % 100 == 0:
            print('current route', count, 'total', len(test_routes))
        used_stop = []

        zone_seq_id = 1
        num_zone = len(zone_id[route_id]['INIT'])
        for k in range(num_zone + 1):

            used_stop.append(last_zone)
            route_seq['route_id'].append(route_id)
            route_seq['zone_id'].append(last_zone)
            route_seq['pred_seq_id'].append(zone_seq_id)
            zone_seq_id += 1
            # try:
            x_input_all_df = x_test[route_id][last_zone]

            if last_zone == 'INIT':
                x_input_all_df['degree'] = 180
                x_input_all_df['forward'] = 1
                x_input_all_df['side'] = 0
                x_input_all_df['backward'] = 0
            else:
                x_input_all_df['last_lat'] = last_lat
                x_input_all_df['last_lng'] = last_lng
                angle, cosine_angle = angle_between_three_points(x_input_all_df[['last_lat', 'last_lng']].values,
                                                                 x_input_all_df[['lat_mean', 'lng_mean']].values,
                                                                 x_input_all_df[['lat_avail', 'lng_avail']].values)

                x_input_all_df['degree'] = angle / np.pi * 180
                x_input_all_df['forward'] = x_input_all_df['degree'] >= 135
                x_input_all_df['forward'] = x_input_all_df['forward'].astype('int')
                x_input_all_df['side'] = (x_input_all_df['degree'] >= 45) & (x_input_all_df['degree'] < 135)
                x_input_all_df['side'] = x_input_all_df['side'].astype('int')
                x_input_all_df['backward'] = x_input_all_df['degree'] < 45
                x_input_all_df['backward'] = x_input_all_df['backward'].astype('int')

                # data_pc['left_side'] = 0
                # data_pc['right_side'] = 0
                # data_pc['third_y_minus_first_y'] = data_pc['lng_avail'] - data_pc['last_lng']
                # data_pc.loc[(data_pc['degree'] >= 45) & (data_pc['degree'] <= 135) &  (data_pc['third_y_minus_first_y'] >= 0), 'left_side'] = 1 # & (data_pc['cosine_angle']  135)
                # data_pc.loc[(data_pc['degree'] >= 45) & (data_pc['degree'] <= 135) &  (data_pc['third_y_minus_first_y'] < 0), 'right_side'] = 1


            last_lat = x_input_all_df['lat_mean'].iloc[0]
            last_lng = x_input_all_df['lng_mean'].iloc[0]

            x_input_all = x_input_all_df[feature_col].values
            # except:
            #     print(route_id)
            #     print(last_zone)
            #     exit()
            # a=1
            zone_id_list = zone_id[route_id][last_zone]
            idx = np.where(np.isin(zone_id_list, used_stop))

            x_input_all_remain = np.delete(x_input_all, idx, axis=0)

            if len(x_input_all_remain) == 0:
                break
            zone_id_remain = np.delete(zone_id_list, idx, axis=0)
            attr =x_input_all_remain[:n_neighbour, :]
            zone_id_remain_used = zone_id_remain[:5]

            if SHUFFLE_ATTR:
                attr, zone_id_remain_used = unison_shuffled_copies(attr, zone_id_remain_used)

            if attr.shape[0] < n_neighbour:
                max_tt = attr[:, 0].max() * 1.5
                max_ttm = attr[:, 1].max() * 1.5
                worst_value = np.zeros((1, attr.shape[1]))
                worst_value[0, 0] = max_tt
                worst_value[0, 1] = max_ttm
                for key in ffill_col:
                    idx_ = used_feature.index(key)
                    worst_value[0,idx_] = attr[0,idx_]
                for key in one_col:
                    idx_ = used_feature.index(key)
                    worst_value[0,idx_] = 1
                a=1
                # feature_col = ['min_travel_time', 'mean_travel_time', 'num_stops_avail', 'degree', 'forward', 'side',
                #                'backward',
                #                'tsp_next_mean',
                #                'tsp_next_min', 'num_tra_sig_avail', 'before_7am_avail', 'after_10am_avail',
                #                'weekends_avail',
                #                'total_num_zones',
                #                'current_zone_percentage']

                attr = np.vstack([attr, np.repeat(worst_value, n_neighbour - attr.shape[0], axis=0)])


                # max_tt = attr[:, 0].max() * 1.5
                # max_ttm = attr[:, 1].max() * 1.5
                # planned_service_time_sum = attr[:, 4].max() * 1.5
                # total_vol_sum = attr[:, 5].max() * 1.5
                # total_vol_max = attr[:, 6].max() * 1.5
                # time_window_end_from_departure_sec_min = attr[:, 7].max() * 1.5
                # worst_value = np.array([[max_tt, max_ttm, 0, 0, 0, 0, planned_service_time_sum, total_vol_sum, total_vol_max,
                #                          time_window_end_from_departure_sec_min]])
                #
                # #attr = np.concatenate([attr, np.zeros((n_neighbour - attr.shape[0], attr.shape[1])) - 99], axis=0)
                # attr = np.vstack([attr, np.repeat(worst_value, n_neighbour - attr.shape[0], axis=0)])

            # normalization
            for j in need_to_norm_col:
                if np.max(attr[:, j]) > 0:
                    attr[:, j] /= np.max(attr[:, j])

            attr = attr.reshape(-1)
            attr = np.array([attr])
            X_test_t = torch.FloatTensor(attr).to(device)
            pred_prob = model(X_test_t).detach().cpu().numpy().reshape(-1)
            pred_prob = pred_prob[:len(zone_id_remain_used)]
            prediction = np.argmax(pred_prob)
            zone_pred = zone_id_remain_used[prediction]
            last_zone = zone_pred

        #attr
        #route_seq_df = pd.DataFrame(route_seq)

    route_seq_df = pd.DataFrame(route_seq)
    all_route_zone = zone_data_test[['route_id','zone_id']].drop_duplicates().copy()
    all_route_zone_pred = all_route_zone.merge(route_seq_df, on = ['route_id','zone_id'],how = 'left')
    #t_r = all_route_zone_pred['route_id'].iloc[0]
    check_na = all_route_zone_pred.loc[all_route_zone_pred['pred_seq_id'].isna()]
    assert len(check_na) == 0
    if all_seq:
        route_seq_df.to_csv('pred_zone_seq_'+tail_name+'_all_seq.csv',index=False)
    else:
        route_seq_df.to_csv('pred_zone_seq_' + tail_name + '.csv', index=False)



def pred_zone_seq_sample(zone_data_test, att_index, n_output, x_test, y_test, zone_id, test_routes, net_para, tail_name, model, num_sample, all_seq = False):
    #worst_value = np.array([[5000, 5000, 0, 0, 600, 99999, 99999, 99999]])
    #test_routes = test_routes[:100]
    model.load_state_dict(net_para)
    model.eval()
    n_neighbour = _constants.n_neighbour
    device = torch.device('cpu')
    route_seq = {'route_id':[],'sample_id':[],'zone_id':[],'pred_seq_id':[]}

    #total_routes =

    #for route_id in test_routes:
    count = 0
    for route_id in test_routes:
        count += 1
        if count % 2 == 0:
            print('current route',count, 'total', len(test_routes))
        for i in range(num_sample):
            used_stop = []
            last_zone = 'INIT'
            zone_seq_id = 1
            num_zone = len(zone_id[route_id]['INIT']) + 1
            for k in range(num_zone):
                used_stop.append(last_zone)
                route_seq['route_id'].append(route_id)
                route_seq['sample_id'].append(i + 1)
                route_seq['zone_id'].append(last_zone)
                route_seq['pred_seq_id'].append(zone_seq_id)

                zone_seq_id += 1
                # try:
                x_input_all = x_test[route_id][last_zone]
                # except:
                #     print(route_id)
                #     print(last_zone)
                #     exit()
                # a=1
                zone_id_list = zone_id[route_id][last_zone]
                idx = np.where(np.isin(zone_id_list, used_stop))

                x_input_all_remain = np.delete(x_input_all, idx, axis=0)

                if len(x_input_all_remain) == 0:
                    break
                zone_id_remain = np.delete(zone_id_list, idx, axis=0)
                attr =x_input_all_remain[:n_neighbour, :]
                zone_id_remain_used = zone_id_remain[:5]

                if attr.shape[0] < n_neighbour:
                    max_tt = attr[:, 0].max() * 1.5
                    max_ttm = attr[:, 1].max() * 1.5
                    worst_value = np.zeros((1, attr.shape[1]))
                    worst_value[0, 0] = max_tt
                    worst_value[0, 1] = max_ttm
                    for key in ffill_col:
                        idx_ = used_feature.index(key)
                        worst_value[0, idx_] = attr[0, idx_]
                    for key in one_col:
                        idx_ = used_feature.index(key)
                        worst_value[0, idx_] = 1

                    attr = np.vstack([attr, np.repeat(worst_value, n_neighbour - attr.shape[0], axis=0)])
                    # max_tt = attr[:, 0].max() * 1.5
                    # max_ttm = attr[:, 1].max() * 1.5
                    # planned_service_time_sum = attr[:, 4].max() * 1.5
                    # total_vol_sum = attr[:, 5].max() * 1.5
                    # total_vol_max = attr[:, 6].max() * 1.5
                    # time_window_end_from_departure_sec_min = attr[:, 7].max() * 1.5
                    # worst_value = np.array([[max_tt, max_ttm, 0, 0, 0, 0, planned_service_time_sum, total_vol_sum, total_vol_max,
                    #                          time_window_end_from_departure_sec_min]])
                    #
                    # #attr = np.concatenate([attr, np.zeros((n_neighbour - attr.shape[0], attr.shape[1])) - 99], axis=0)
                    # attr = np.vstack([attr, np.repeat(worst_value, n_neighbour - attr.shape[0], axis=0)])

                # normalization
                for j in need_to_norm_col:
                    if np.max(attr[:, j]) > 0:
                        attr[:, j] /= np.max(attr[:, j])

                attr = attr.reshape(-1)
                attr = np.array([attr])
                X_test_t = torch.FloatTensor(attr).to(device)
                pred_prob = model(X_test_t)
                pred_prob = pred_prob[:len(zone_id_remain_used)]
                pred_prob_np = pred_prob.detach().cpu().numpy().reshape(-1)
                #pred_prob_np =
                prediction = np.random.choice(range(len(pred_prob_np)), p = pred_prob_np)#torch.max(pred_prob, 1)[1]
                zone_pred = zone_id_remain_used[prediction]
                last_zone = zone_pred

        #attr
        #route_seq_df = pd.DataFrame(route_seq)

    route_seq_df = pd.DataFrame(route_seq)
    # all_route_zone = zone_data_test[['route_id','zone_id']].drop_duplicates().copy()
    # all_route_zone_pred = all_route_zone.merge(route_seq_df, on = ['route_id','zone_id'],how = 'left')
    # #t_r = all_route_zone_pred['route_id'].iloc[0]
    # check_na = all_route_zone_pred.loc[all_route_zone_pred['pred_seq_id'].isna()]
    # assert len(check_na) == 0
    if all_seq:
        route_seq_df.to_csv('pred_zone_seq_'+tail_name+'_all_seq_sample.csv',index=False)
    else:
        route_seq_df.to_csv('pred_zone_seq_' + tail_name + '_sample.csv', index=False)





def pred_zone_seq_sample_vect(zone_data_test, att_index, n_output, x_test, y_test, zone_id, test_routes, net_para, tail_name, model, num_sample, all_seq = False):
    #worst_value = np.array([[5000, 5000, 0, 0, 600, 99999, 99999, 99999]])
    #test_routes = test_routes[:100]
    model.load_state_dict(net_para)
    model.eval()
    n_neighbour = _constants.n_neighbour
    device = torch.device('cpu')
    route_seq = {'route_id':[],'sample_id':[],'zone_id':[],'pred_seq_id':[]}

    #total_routes =

    #for route_id in test_routes:
    count = 0
    for route_id in test_routes:
        count += 1
        if count % 2 == 0:
            print('current route',count, 'total', len(test_routes))
        for i in range(num_sample):
            used_stop = []
            last_zone = 'INIT'
            zone_seq_id = 1
            num_zone = len(zone_id[route_id]['INIT']) + 1
            for k in range(num_zone):
                used_stop.append(last_zone)
                route_seq['route_id'].append(route_id)
                route_seq['sample_id'].append(i + 1)
                route_seq['zone_id'].append(last_zone)
                route_seq['pred_seq_id'].append(zone_seq_id)

                zone_seq_id += 1
                # try:
                x_input_all = x_test[route_id][last_zone]
                # except:
                #     print(route_id)
                #     print(last_zone)
                #     exit()
                # a=1
                zone_id_list = zone_id[route_id][last_zone]
                idx = np.where(np.isin(zone_id_list, used_stop))

                x_input_all_remain = np.delete(x_input_all, idx, axis=0)

                if len(x_input_all_remain) == 0:
                    break
                zone_id_remain = np.delete(zone_id_list, idx, axis=0)
                attr =x_input_all_remain[:n_neighbour, :]
                zone_id_remain_used = zone_id_remain[:5]

                if attr.shape[0] < n_neighbour:
                    max_tt = attr[:, 0].max() * 1.5
                    max_ttm = attr[:, 1].max() * 1.5
                    worst_value = np.zeros((1, attr.shape[1]))
                    worst_value[0, 0] = max_tt
                    worst_value[0, 1] = max_ttm
                    for key in ffill_col:
                        idx_ = used_feature.index(key)
                        worst_value[0, idx_] = attr[0, idx_]
                    for key in one_col:
                        idx_ = used_feature.index(key)
                        worst_value[0, idx_] = 1

                    attr = np.vstack([attr, np.repeat(worst_value, n_neighbour - attr.shape[0], axis=0)])
                    # max_tt = attr[:, 0].max() * 1.5
                    # max_ttm = attr[:, 1].max() * 1.5
                    # planned_service_time_sum = attr[:, 4].max() * 1.5
                    # total_vol_sum = attr[:, 5].max() * 1.5
                    # total_vol_max = attr[:, 6].max() * 1.5
                    # time_window_end_from_departure_sec_min = attr[:, 7].max() * 1.5
                    # worst_value = np.array([[max_tt, max_ttm, 0, 0, 0, 0, planned_service_time_sum, total_vol_sum, total_vol_max,
                    #                          time_window_end_from_departure_sec_min]])
                    #
                    # #attr = np.concatenate([attr, np.zeros((n_neighbour - attr.shape[0], attr.shape[1])) - 99], axis=0)
                    # attr = np.vstack([attr, np.repeat(worst_value, n_neighbour - attr.shape[0], axis=0)])

                # normalization
                for j in need_to_norm_col:
                    if np.max(attr[:, j]) > 0:
                        attr[:, j] /= np.max(attr[:, j])

                attr = attr.reshape(-1)
                attr = np.array([attr])
                X_test_t = torch.FloatTensor(attr).to(device)
                pred_prob = model(X_test_t)
                pred_prob = pred_prob[:len(zone_id_remain_used)]
                pred_prob_np = pred_prob.detach().cpu().numpy().reshape(-1)
                #pred_prob_np =
                prediction = np.random.choice(range(len(pred_prob_np)), p = pred_prob_np)#torch.max(pred_prob, 1)[1]
                zone_pred = zone_id_remain_used[prediction]
                last_zone = zone_pred

        #attr
        #route_seq_df = pd.DataFrame(route_seq)

    route_seq_df = pd.DataFrame(route_seq)
    # all_route_zone = zone_data_test[['route_id','zone_id']].drop_duplicates().copy()
    # all_route_zone_pred = all_route_zone.merge(route_seq_df, on = ['route_id','zone_id'],how = 'left')
    # #t_r = all_route_zone_pred['route_id'].iloc[0]
    # check_na = all_route_zone_pred.loc[all_route_zone_pred['pred_seq_id'].isna()]
    # assert len(check_na) == 0
    if all_seq:
        route_seq_df.to_csv('pred_zone_seq_'+tail_name+'_all_seq_sample.csv',index=False)
    else:
        route_seq_df.to_csv('pred_zone_seq_' + tail_name + '_sample.csv', index=False)



def generate_prob(zone_data_test, att_index, n_output, x_test, y_test, zone_id, test_routes, net_para, tail_name, model, all_seq = False):
    #worst_value = np.array([[5000, 5000, 0, 0, 600, 99999, 99999, 99999]])

    model.load_state_dict(net_para)
    model.eval()

    device = torch.device('cpu')
    route_seq = {'route_id':[],'zone_id':[],'pred_seq_id':[]}

    #total_routes =
    prob = {}
    for route_id in test_routes:
        num_zone = len(zone_id[route_id]['INIT'])
        prob[route_id] = {}
        for z1 in x_test[route_id]:
            prob[route_id][z1] = {}
            attr = x_test[route_id][z1]

            # normalization
            for j in [0, 1]:
                if np.max(attr[:, j]) > 0:
                    attr[:, j] /= np.max(attr[:, j])

            attr = attr.reshape(-1)

            attr = np.array([attr])
            X_test_t = torch.FloatTensor(attr).to(device)
            pred_prob = model(X_test_t)
            a=1




if __name__ == '__main__':

    all_seq = False
    num_neighbor = _constants.n_neighbour

    VEC_PRED = True

    tic = time.time()
    if all_seq:
        with open('data/processed_zone_seq_neighbour_all_RL.pkl', 'rb') as f:
            x_test= pickle.load(f)
            # x ={route: {zone: x}}
            y_test = pickle.load(f)
            # #y  ={route: {zone: y}}
            zone_id = pickle.load(f)

    else:
        with open('data/processed_test_zone_seq_neighbour_all_df.pkl', 'rb') as f:
            zone_df = pickle.load(f)

        with open('data/processed_test_zone_seq_neighbour_all.pkl', 'rb') as f:
            x_test = pickle.load(f)
            y_test = pickle.load(f)
            zone_id = pickle.load(f)


    r_temp = list(x_test.keys())[0]
    z_temp = list(x_test[r_temp].keys())[0]
    _, n_feature = x_test[r_temp][z_temp].shape



    print('load data time',time.time() - tic)


    tic = time.time()



    n_output = num_neighbor

    zone_data = pd.read_csv('../data/zone_data.csv')

    if not all_seq:
        with open('testing_routes.pkl', 'rb') as f:
            test_routes = pickle.load(f)
    else:
        test_routes = pd.unique(zone_data['route_id'])

    feature_col = _constants.feature_col
    if all_seq:
        zone_data_test = zone_data.copy()#.loc[zone_data['route_id'].isin(test_routes)]
    else:
        zone_data_test = zone_data.loc[zone_data['route_id'].isin(test_routes)]

    #tail_name = 'DQN'
    SAMPLE = False
    SHUFFLE_ATTR = False

    feature_col_test = _constants.feature_col_test

    print('before pred process time', time.time() - tic)

    test_list = ['ASU_DNN_no_tsp'] # ASU_DNN_no_tsp # DQN #ASU_DNN_with_tsp 'DQN',
    for tail_name in test_list:
        tic = time.time()
        if tail_name == 'DQN':
            net_para = torch.load('model_output/DQN_with_tsp_info.pt')

            used_feature = _constants.used_feature

            ffill_col = _constants.ffill_col
            zero_col = _constants.zero_col
            one_col = _constants.one_col


            norm_feature = _constants.inner_norm_feature

            used_feature_id = []
            for key in used_feature:
                used_feature_id.append(feature_col.index(key))
            used_feature_id = np.array(used_feature_id)

            need_to_norm_col = []
            for key in norm_feature:
                need_to_norm_col.append(feature_col.index(key))

            att_index = {}
            for i in range(num_neighbor):
                att_index['x_' + str(i + 1)] = used_feature_id + i * n_feature

            model = ASU_NN_RL(att_index, n_hidden_1=64, n_hidden_2=128,
                              n_layer_1=1, n_layer_2=None, n_output=n_output, device="cpu")
        elif tail_name == 'ASU_DNN_no_tsp':
            net_para = torch.load('model_output/trained_model_ASU_NN_no_tspinfo.pt')
            used_feature =_constants.used_feature #_constants.used_feature_no_tsp

            ffill_col = _constants.ffill_col
            zero_col = _constants.zero_col_no_tsp
            one_col = _constants.one_col

            norm_feature = _constants.inner_norm_feature

            # used_feature_id = np.array([0, 1, 2, 3, 4, 5])
            used_feature_id = []
            for key in used_feature:
                used_feature_id.append(feature_col.index(key))
            used_feature_id = np.array(used_feature_id)


            att_index = {}
            for i in range(num_neighbor):
                att_index['x_' + str(i + 1)] = used_feature_id + i * n_feature

            need_to_norm_col = []
            for key in norm_feature:
                need_to_norm_col.append(feature_col.index(key))

            model = ASU_NN(att_index, n_hidden_1=64, n_hidden_2=128,
                           n_layer_1=1, n_layer_2=None, n_output=n_output, device="cpu")


        elif tail_name == 'PG':
            net_para = torch.load('model_output/Fullinfo_PG_Base_asudnn_2_full.pt')

            used_feature = _constants.used_feature

            ffill_col = _constants.ffill_col
            zero_col = _constants.zero_col
            one_col = _constants.one_col
            norm_feature = _constants.inner_norm_feature

            used_feature_id = []
            for key in used_feature:
                used_feature_id.append(feature_col.index(key))
            used_feature_id = np.array(used_feature_id)

            need_to_norm_col = []
            for key in norm_feature:
                need_to_norm_col.append(feature_col.index(key))


            att_index = {}
            for i in range(num_neighbor):
                att_index['x_' + str(i + 1)] = used_feature_id + i * n_feature

            model = ASU_NN(att_index, n_hidden_1=64, n_hidden_2=128,
                              n_layer_1=1, n_layer_2=None, n_output=n_output, device="cpu")

        elif tail_name == 'ASU_DNN_with_tsp':

            used_feature = _constants.used_feature

            ffill_col = _constants.ffill_col
            zero_col = _constants.zero_col
            one_col = _constants.one_col


            norm_feature = _constants.inner_norm_feature

            net_para = torch.load('model_output/trained_model_ASU_NN_with_tspinfo.pt')
            # used_feature_id = np.array([0, 1, 2, 3, 4, 5])
            used_feature_id = []
            for key in used_feature:
                used_feature_id.append(feature_col.index(key))
            used_feature_id = np.array(used_feature_id)

            att_index = {}
            for i in range(num_neighbor):
                att_index['x_' + str(i + 1)] = used_feature_id + i * n_feature

            need_to_norm_col = []
            for key in norm_feature:
                need_to_norm_col.append(feature_col.index(key))

            model = ASU_NN(att_index, n_hidden_1=64, n_hidden_2=128,
                              n_layer_1=1, n_layer_2=None, n_output=n_output, device="cpu")

        if not SAMPLE:
            pred_zone_seq_diff_ini(zone_data_test, att_index, n_output, x_test, y_test, zone_id, test_routes, net_para, tail_name, model, all_seq)
            # pred_zone_seq_vectorize(zone_data_test, att_index, n_output, zone_df, y_test, zone_id, test_routes, net_para,
            #               tail_name, model, all_seq)
        else:
            num_sample = 200
            pred_zone_seq_sample(zone_data_test, att_index, n_output, x_test, y_test, zone_id, test_routes, net_para, tail_name,
                          model, num_sample, all_seq)



        # generate_prob(zone_data_test, att_index, n_output, x_test, y_test, zone_id, test_routes, net_para, tail_name,
        #               model, all_seq)
        print("generate seq time",time.time() - tic)
