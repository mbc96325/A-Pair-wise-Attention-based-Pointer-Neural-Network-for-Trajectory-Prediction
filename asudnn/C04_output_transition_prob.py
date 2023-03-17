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
import json
from asu_nn import ASU_NN


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




def pred_zone_seq(zone_data_test, att_index, n_output, x_test, y_test, zone_id, test_routes):
    #worst_value = np.array([[5000, 5000, 0, 0, 600, 99999, 99999, 99999]])
    prob_output = {}
    model = ASU_NN(att_index, n_hidden_1 = 64, n_hidden_2 = 128,
              n_layer_1 = 1, n_layer_2 = None, n_output = n_output, device = 'cpu')
    model.load_state_dict(torch.load('model_output/trained_model_ASU_NN_no_tspinfo.pt'))
    model.eval()
    n_neighbour = 5
    device = torch.device('cpu')
    route_seq = {'route_id':[],'zone_id':[],'pred_seq_id':[]}

    #total_routes =

    for route_id in test_routes:
        prob_output[route_id] = {}
        #used_stop = []
        last_zone = 'INIT'
        zone_seq_id = 1
        num_zone = len(zone_id[route_id]['INIT'])
        for current_zone in zone_id[route_id]:
            #current_zone = zone_id[route_id][last_zone][0]
            used_stop = [current_zone]
            if 'INIT' not in used_stop:
                used_stop.append('INIT')
            prob_output[route_id][current_zone] = {}
            route_seq['route_id'].append(route_id)
            route_seq['zone_id'].append(last_zone)
            route_seq['pred_seq_id'].append(zone_seq_id)
            zone_seq_id += 1
            x_input_all = x_test[route_id][last_zone]
            zone_id_list = zone_id[route_id][last_zone]
            idx = np.where(np.isin(zone_id_list, used_stop))

            x_input_all_remain = np.delete(x_input_all, idx, axis=0)

            if len(x_input_all_remain) == 0:
                break
            zone_id_remain = np.delete(zone_id_list, idx, axis=0)
            attr =x_input_all_remain[:5, :]
            zone_id_remain_used = zone_id_remain[:5]

            if attr.shape[0] < n_neighbour:
                max_tt = attr[:, 0].max() * 1.5
                max_ttm = attr[:, 1].max() * 1.5
                planned_service_time_sum = attr[:, 4].max() * 1.5
                total_vol_sum = attr[:, 5].max() * 1.5
                total_vol_max = attr[:, 6].max() * 1.5
                time_window_end_from_departure_sec_min = attr[:, 7].max() * 1.5
                worst_value = np.array([[max_tt, max_ttm, 0, 0, 0, 0, planned_service_time_sum, total_vol_sum, total_vol_max,
                                         time_window_end_from_departure_sec_min]])

                #attr = np.concatenate([attr, np.zeros((n_neighbour - attr.shape[0], attr.shape[1])) - 99], axis=0)
                attr = np.vstack([attr, np.repeat(worst_value, n_neighbour - attr.shape[0], axis=0)])

            # normalization
            for j in [0, 1]:
                attr[:, j] /= np.max(attr[:, j])

            attr_flat = attr.reshape(-1)
            attr_flat = np.array([attr_flat])
            X_test_t = torch.FloatTensor(attr_flat).to(device)
            pred_prob = model(X_test_t)
            pred_prob = pred_prob[:len(zone_id_remain_used)]
            for prob, z2 in zip(pred_prob[0], zone_id_remain_used):
                prob_output[route_id][current_zone][z2] = float(prob)
            # print(prob_output)
            a=1

        #attr
        #route_seq_df = pd.DataFrame(route_seq)

    # route_seq_df = pd.DataFrame(route_seq)
    # route_seq_df.to_csv('pred_zone_seq.csv',index=False)

    with open("ASU_DNN_pred_prob_no_tsp.json", "w") as outfile:
        json.dump(prob_output, outfile)

if __name__ == '__main__':

    with open('processed_test_zone_seq_neighbour_all.pkl', 'rb') as f:
        x_test = pickle.load(f)
        y_test = pickle.load(f)
        zone_id = pickle.load(f)

    n_output = 5
    n_feature = 10
    used_feature_id = np.array([0, 1, 2, 3])#, 4, 5

    att_index={'x_1':used_feature_id,
               'x_2':used_feature_id + n_feature,
               'x_3':used_feature_id + 2 * n_feature,
               'x_4':used_feature_id + 3 * n_feature,
               'x_5':used_feature_id + 4 * n_feature}

    with open('testing_routes.pkl', 'rb') as f:
        test_routes = pickle.load(f)

    zone_data = pd.read_csv('../../data/zone_data.csv')

    zone_data_test = zone_data.loc[zone_data['route_id'].isin(test_routes)]






    pred_zone_seq(zone_data_test, att_index, n_output, x_test, y_test, zone_id, test_routes)

