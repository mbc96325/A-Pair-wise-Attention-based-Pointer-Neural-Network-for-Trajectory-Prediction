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
import _constants
import time

from asu_nn import ASU_NN, ASU_NN_same_para

class Data(Dataset):
    def __init__(self, X, Y, device):
        self.x = torch.FloatTensor(X).to(device)
        self.y = torch.LongTensor(Y).reshape(-1).to(device)
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

def train(X_train_t, Y_train_t, X_test_t, Y_test_t, N_training_epoch, net, optimizer, loss_func, trainloader,device):
    #torch.manual_seed(1234)  # for reproducibility
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
            #print(loss)
            #print(out.detach().numpy()[:,0])
            print(x.detach().numpy()[:, 1])
            #print(torch.max(out, 1)[1])
            #print(y)
            #print(out.shape)
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
        X.append(data.loc[:,val].values)
        att_index[key] = list(range(num_att, num_att_new))
        num_att = copy.deepcopy(num_att_new)
    X = np.hstack(X)
    scaler = preprocessing.MinMaxScaler().fit(X)
    X_norm = scaler.transform(X)
    Y = data.loc[:,y_att].values
    return X_norm, Y, att_index

def switch_labels(x, y, att_index, num_neighbor):
    assert len(x) == len(y)
    label_closest = []
    for i in range(len(y)):
        a,b= np.random.choice(np.arange(num_neighbor), size=2, replace=False)
        att_a = att_index['x_'+str(a+1)]
        att_b = att_index['x_'+str(b+1)]
        temp = x[i,att_a]
        x[i,att_a] = x[i,att_b]
        x[i,att_b] = temp
        if y[i]==a:
            y[i]=b
        elif y[i]==b:
            y[i]=a
        if (a!=0)&(b!=0):
            label_closest.append(0)
        else:
            label_closest.append(np.max([a,b]))
        

    return x, y, label_closest


def switch_labels_more(x, y, att_index, num_neighbor):
    assert len(x) == len(y)
    label_closest = []
    for i in range(len(y)):
        a, b, c, d = np.random.choice(np.arange(num_neighbor), size=4, replace=False)
        att_a = att_index['x_' + str(a + 1)]
        att_b = att_index['x_' + str(b + 1)]
        att_c = att_index['x_' + str(c + 1)]
        att_d = att_index['x_' + str(d + 1)]
        temp = x[i, att_a]

        x[i, att_a] = x[i, att_b]

        x[i, att_b] = temp

        temp = x[i, att_c]
        x[i, att_c] = x[i, att_d]
        x[i, att_d] = temp

        if y[i] == a:
            y[i] = b
        elif y[i] == b:
            y[i] = a
        elif y[i] == c:
            y[i] = d
        elif y[i] == d:
            y[i] = c

        if a == 0:
            label_closest.append(b)
        elif b == 0:
            label_closest.append(a)
        elif c == 0:
            label_closest.append(d)
        elif d == 0:
            label_closest.append(c)
        else:
            label_closest.append(0)

    return x, y, label_closest


#
def get_tsp_accuracy(route_and_zone_train, test_idx, Y_test, n_neighbor):
    with open('../../tsp_xiaotong/mean_dist/opt_zone_seq.json', 'rb') as f:
        tsp_zone_seq_mean = json.load(f)
    with open('../../tsp_xiaotong/min_dist/opt_zone_seq.json', 'rb') as f:
        tsp_zone_seq_min = json.load(f)

    with open('data/nearest_neighbor_' + str(n_neighbor) + '.pkl', 'rb') as f:
        nearest_neighbor = pickle.load(f)

    tsp_mean_y = np.zeros(len(Y_test))
    tsp_min_y = np.zeros(len(Y_test))


    test_routes = route_and_zone_train.loc[route_and_zone_train['idx'].isin(test_idx)]
    for route_id, zone_id, idx, num_in_y in zip(test_routes['route_id'], test_routes['zone_id'], test_routes['idx'], range(len(Y_test))):
        _idx_route = tsp_zone_seq_mean[route_id].index(zone_id)
        if _idx_route + 1 < len(tsp_zone_seq_mean[route_id]):
            tsp_mean_pred = tsp_zone_seq_mean[route_id][_idx_route + 1]
            if tsp_mean_pred in nearest_neighbor[(route_id,zone_id)]:
                y_pred = nearest_neighbor[(route_id,zone_id)].index(tsp_mean_pred)
            else:
                y_pred = 0
        else:
            y_pred = 0

        tsp_mean_y[num_in_y] = y_pred

        _idx_route = tsp_zone_seq_min[route_id].index(zone_id)
        if _idx_route + 1 < len(tsp_zone_seq_min[route_id]):
            tsp_min_pred = tsp_zone_seq_min[route_id][_idx_route + 1]
            if tsp_min_pred in nearest_neighbor[(route_id,zone_id)]:
                y_pred = nearest_neighbor[(route_id,zone_id)].index(tsp_min_pred)
            else:
                y_pred = 0
        else:
            y_pred = 0

        tsp_min_y[num_in_y] = y_pred
        a=1

    accuracy = sum((tsp_mean_y == Y_test)*1) / float(Y_test.size)
    print('tsp_mean_y acc', accuracy)
    accuracy = sum((tsp_min_y == Y_test)*1) / float(Y_test.size)
    print('tsp_min_y acc', accuracy)

#def customize_loss():
    

if __name__ == '__main__':
    # ['tt', 'ttm', 'lat_dir_same', 'lng_dir_same', 'tsp_next_mean', 'tsp_next_min',
    #  'planned_service_time_sum', 'total_vol_sum', 'total_vol_max',
    #  'time_window_end_from_departure_sec_min']

    num_neighbor = _constants.n_neighbour
    ACTUAL_FILTER = True
    if ACTUAL_FILTER:
        tail = '_filter_passed'
    else:
        tail = ''

    with open('data/processed_zone_seq_neighbour_'+ str(num_neighbor) + tail + '.pkl', 'rb') as f:
        features = np.load(f, allow_pickle = True)
        labels = np.load(f, allow_pickle = True)

    with open('data/route_id_seq_' + str(num_neighbor)  + tail + '.pkl', 'rb') as f:
        route_id_seq = pickle.load(f)


    with open('data/zone_id_seq_'+ str(num_neighbor)  + tail + '.pkl', 'rb') as f:
        zone_id_seq = pickle.load(f)

    route_and_zone = pd.DataFrame({'route_id':route_id_seq,'zone_id':zone_id_seq})
    route_and_zone['idx'] = route_and_zone.index

    #random.seed(2)

    with open('train_routes.pkl', 'rb') as f:
        train_routes = pickle.load(f)

    with open('testing_routes.pkl', 'rb') as f:
        test_routes = pickle.load(f)

    print('num_train', len(train_routes))
    print('num_test', len(test_routes))

    train_routes_df = pd.DataFrame({'train_route':train_routes})
    route_and_zone_train = route_and_zone.merge(train_routes_df,left_on = ['route_id'],right_on=['train_route'], sort=False)

    #print(len(train_routes_df))

    train_idx = list(route_and_zone_train['idx'])
    test_idx = list(set(range(len(zone_id_seq))).difference(set(train_idx)))

    idx_to_delete = np.argwhere(labels < 0).reshape(-1)
    train_idx = list(set(train_idx).difference(set(idx_to_delete)))
    test_idx = list(set(test_idx).difference(set(idx_to_delete)))

    print(features.shape)
    X_train = features[train_idx, :]#.reshape(len(train_idx),-1)
    Y_train = labels[train_idx]
    X_test = features[test_idx,:]#.reshape(len(test_idx),-1)
    X_test_raw = X_test.copy()
    Y_test = labels[test_idx]

    print('X_train_shape',X_train.shape)
    print('X_test_shape',X_test.shape)

    # get_tsp_accuracy(route_and_zone, test_idx, Y_test, num_neighbor)

    features = features[labels>=0,:,:]
    labels= labels[labels>=0]

    nzone, nnode, n_feature = features.shape

    feature_col = _constants.feature_col


    ############
    # used_feature = _constants.used_feature_no_tsp
    used_feature = _constants.used_feature
    ############
    # used_feature = ['min_travel_time','mean_travel_time']


    inner_norm_feature = _constants.inner_norm_feature   #, 'degree',
    outer_norm_feature = []

    used_feature_id = []
    for key in used_feature:
        used_feature_id.append(feature_col.index(key))
    used_feature_id = np.array(used_feature_id)


    need_to_norm_col = []
    for key in inner_norm_feature:
        need_to_norm_col.append(feature_col.index(key))

    need_to_outer_norm_col = []
    for key in outer_norm_feature:
        need_to_outer_norm_col.append(feature_col.index(key))

    # normalization
    for i in range(len(X_train)):
        for j in need_to_norm_col:
            if np.max(X_train[i,:,j]) > 0:
                X_train[i,:,j] /= np.max(X_train[i,:,j])
                #a=1

    # normalization
    for i in range(len(X_test)):
        for j in need_to_norm_col:
            if np.max(X_test[i, :, j]) > 0:
                X_test[i, :, j] /= np.max(X_test[i, :, j])



    X_train = X_train.reshape(len(train_idx),-1)
    X_test = X_test.reshape(len(test_idx), -1)

    for j in need_to_outer_norm_col:
        for alt in range(num_neighbor):
            idx_ = alt*n_feature + j
            X_train[:,idx_] = (X_train[:,idx_] - np.min(X_train[:,idx_]))/(np.max(X_train[:,idx_]) - np.min(X_train[:,idx_]))
            X_test[:,idx_] = (X_test[:, idx_] - np.min(X_test[:, idx_])) / (
                        np.max(X_test[:, idx_]) - np.min(X_test[:, idx_]))

    N_training_epoch = 35

    if ACTUAL_FILTER:
        SWITCH_LABEL = True
    else:
        SWITCH_LABEL = False
    if SWITCH_LABEL:
        ###############Switch label#################

        att_index_for_switch_label = {}
        for i in range(num_neighbor):
            att_index_for_switch_label['x_' + str(i + 1)] = np.arange(n_feature) + n_feature*i


        # X_train, Y_train, label_closest_train = switch_labels(X_train, Y_train, att_index_for_switch_label, num_neighbor)
        # X_test, Y_test, label_closest_test = switch_labels(X_test, Y_test, att_index_for_switch_label, num_neighbor)

        X_train, Y_train, label_closest_train = switch_labels_more(X_train, Y_train, att_index_for_switch_label, num_neighbor)
        X_test, Y_test, label_closest_test = switch_labels_more(X_test, Y_test, att_index_for_switch_label, num_neighbor)
    else:
        label_closest_train = [0] * len(Y_train)
        label_closest_test = [0] * len(Y_test)
    ################################



    compare1 = [a==b for a,b in zip(label_closest_train, Y_train)]
    print("# closest zone == next zone:", np.sum(compare1))
    print('accuracy all choose closest', np.sum(compare1)/len(X_train))
    #

    #%%
    n_output = len(np.unique(Y_train))
    
    # if torch.cuda.is_available():
    #     dev = "cuda:0"
    # else:


    dev = "cpu"
    device = torch.device(dev)

    att_index = {}
    for i in range(num_neighbor):
        att_index['x_' + str(i+1)] = used_feature_id + i*n_feature

    torch.manual_seed(123)
    net = ASU_NN(att_index, n_hidden_1 = 64, n_hidden_2 = 128,
              n_layer_1 = 1, n_layer_2 = None, n_output = n_output, device = device)


    # print(net)  # net architecture
    net.to(device)
    #print(X_train.shape)
    dataset = Data(X_train,Y_train, device)
    trainloader = DataLoader(dataset=dataset, batch_size=64)

    Y_train_t = torch.LongTensor(Y_train).reshape(-1).to(device)
    X_train_t = torch.FloatTensor(X_train).to(device)
    Y_test_t = torch.LongTensor(Y_test).reshape(-1).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-5)#lr=0.005,
    loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted



    acc_records = []
    loss_records = []
    loss_records_test = []
    acc_records_test = []

    total_training_time = 0

    for t in range(N_training_epoch):

        net.eval()
        out_train = net(X_train_t)
        loss_train = loss_func(out_train, Y_train_t)
        loss_records.append(float(loss_train.item()))
        out_test = net(X_test_t)
        loss_test = loss_func(out_test, Y_test_t)
        loss_records_test.append(float(loss_test.item()))
        
        if t % 3 == 0:
            # plot and show learning process
            prediction = torch.max(out_train, 1)[1]
            pred_y = prediction.data.numpy()
            target_y = Y_train_t.data.numpy()
            accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
            acc_records.append(accuracy)
            print('current training epoch %d \t train loss %.4f accuracy %.4f' % \
                  (t, loss_train, accuracy))
            prediction = torch.max(out_test, 1)[1]
            pred_y = prediction.data.numpy()
            target_y = Y_test_t.data.numpy()
            accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
            acc_records_test.append(accuracy)
            print('test accuracy %.4f' % (accuracy))

        start_time = time.time()
        for x, y in trainloader:
            net.train()
            out = net(x)  # input x and predict based on x
            loss = loss_func(out, y)  # must be (1. nn output, 2. target), the target label is NOT one-hotted
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
        total_training_time += time.time() - start_time

    df = pd.DataFrame({"total_training_time":[total_training_time]})
    df.to_csv('model_output/total_training_time_ASU_NN.csv',index=False)

    if 'tsp_next_mean' in used_feature:
        torch.save(net.state_dict(), 'model_output/trained_model_ASU_NN_with_tspinfo_count_time.pt')

    else:
        torch.save(net.state_dict(), 'model_output/trained_model_ASU_NN_no_tspinfo_count_time.pt')