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

# print(torch.__version__)

import os
import json
import time
import sys
#from score_testing_func import evaluate_simple
#from score_testing_func import evaluate_simple, evaluate_simple_sd
from cython_score_evaluate import evaluate_simple, seq_dev
from pointer_net import Encoder, Decoder, PointerNetwork



Random_seed = 123

class Data(Dataset):
    def __init__(self, X, Y, device):
        self.x = torch.FloatTensor(X).to(device)
        self.y = torch.LongTensor(Y).reshape(-1).to(device)
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


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


def partition(list_in, n, random_seed):
    #random.seed(random_seed)
    random.Random(random_seed).shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def pred_zone_seq(zone_data_test, att_index, n_output, x_test, y_test, zone_id, test_routes):
    #worst_value = np.array([[5000, 5000, 0, 0, 600, 99999, 99999, 99999]])

    model = ASU_NN(att_index, n_hidden_1=64, n_hidden_2=128,
                   n_layer_1=2, n_layer_2=3, n_output = n_output, device='cpu')
    model.load_state_dict(torch.load('model_output/trained_model.pt'))
    model.eval()
    n_neighbour = 5
    device = torch.device('cpu')
    route_seq = {'route_id':[],'zone_id':[],'pred_seq_id':[]}

    #total_routes =

    for route_id in test_routes:
        used_stop = []
        last_zone = 'INIT'
        zone_seq_id = 1
        num_zone = len(zone_id[route_id]['INIT'])
        for k in range(num_zone):
            used_stop.append(last_zone)
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
                if np.max(attr[:, j]) > 0:
                    attr[:, j] /= np.max(attr[:, j])

            attr = attr.reshape(-1)
            attr = np.array([attr])
            X_test_t = torch.FloatTensor(attr).to(device)
            pred_prob = model(X_test_t)
            pred_prob = pred_prob[:len(zone_id_remain_used)]
            prediction = torch.max(pred_prob, 1)[1]
            zone_pred = zone_id_remain_used[prediction[0]]
            last_zone = zone_pred

        #attr
        #route_seq_df = pd.DataFrame(route_seq)

    route_seq_df = pd.DataFrame(route_seq)
    route_seq_df.to_csv('pred_zone_seq.csv',index=False)


def discount_rewards(rewards, gamma=0.99):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

# def discount_rewards(rewards, gamma=0.99):
#     # Cumulative discounted sum
#     r = np.array([gamma**i * rewards[i]
#                   for i in range(len(rewards))])
#     r = r[::-1].cumsum()[::-1]
#     # Subtracting the baseline reward
#     # Intuitively this means if the network predicts what it
#     # expects it should not do too much about it
#     # Stabalizes and speeds up the training
#     return r - r.mean()


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def generate_training_samples(x_r,y_r, ini_to_first_r, num_zone, PG, PG_baseline,PG_ini, device, weight_T, data_route,input_seq_dict,route_id,epsilon,
                                              cost_mat_array_r=None, name_map_all_r=None, replica_num = 1, partition_num = 1):




    PG.eval()
    PG_baseline.eval()

    name_map = name_map_all_r
    # Actual Sequence

    actual_seq = list(data_route['stops'])
    actual_seq = np.array([name_map[s] for s in actual_seq])
    x_r_t = torch.FloatTensor(x_r)

    # print(x_r_t)

    ini_to_first_r_t = torch.FloatTensor(ini_to_first_r)
    num_zone_t =torch.FloatTensor(num_zone)
    # Predicted Sequence
    # if route_id == 'RouteID_eab2f503-0f54-4571-bb4a-402b71472c0c' and replica_num == 2:
    #     a=1

    if np.random.rand() < epsilon:
        out, loss_, prob_ = PG(x_r_t, None, num_zone_t, None, ini_to_first_r_t, None,
                                     None, mask_prob=True, teacher_force=False, Print_pred=False,sample=True,
                                     No_dyn_dist=True, model_apply = True, Calculate_loss = False)
        # out, loss_, prob_ = PG(x_r_t, None, num_zone_t, teacher_force_ratio=0., model_apply=True, sample=True, mask_prob=False)
    else:
        out, loss_, prob_ = PG(x_r_t, None, num_zone_t, None, ini_to_first_r_t, None,
                                     None, mask_prob=True, teacher_force=False, Print_pred=False,sample=False,
                                     No_dyn_dist=True, model_apply = True, Calculate_loss = False)


    out = out.permute(1,0)

    # print(out)

    cost_step = 1 - torch.eq(out[0], torch.tensor(y_r[0])).int()


    predict_seq = ['INIT']
    for j in out[0,:num_zone[0]]:
        predict_seq.append(input_seq_dict[(route_id,int(j))])
    assert len(predict_seq) == num_zone+1

    pred_zone_df = pd.DataFrame({'zone_id': predict_seq, 'pred_seq_id': np.arange(num_zone+1) + 1})
    pred_zone_df = pred_zone_df.merge(data_route[['stops', 'zone_id', 'seq_ID']], on=['zone_id'])
    pred_zone_df = pred_zone_df.sort_values(['pred_seq_id', 'seq_ID'])
    est_seq = list(pred_zone_df['stops'])
    est_seq = np.array([name_map[s] for s in est_seq])


    if replica_num == 1:
        # Predicted Baseline Sequence
        # out_baseline, _, _ = PG_baseline(x_r_t, None, num_zone_t, teacher_force_ratio=0., model_apply=True, mask_prob=False)
        out_baseline, _, _ = PG_baseline(x_r_t, None, num_zone_t, None, ini_to_first_r_t, None,
                                     None, mask_prob=True, teacher_force=False, Print_pred=False,sample=False,
                                     No_dyn_dist=True, model_apply = True, Calculate_loss = False)

        out_baseline = out_baseline.permute(1,0)


        cost_step_base = 1 - torch.eq(out_baseline[0], torch.tensor(y_r[0])).int()

        predict_seq_baseline = ['INIT']
        for j in out_baseline[0,:num_zone[0]]:
            predict_seq_baseline.append(input_seq_dict[(route_id,int(j))])
        assert len(predict_seq_baseline) == num_zone+1

        pred_zone_df_baseline = pd.DataFrame({'zone_id': predict_seq_baseline, 'pred_seq_id': np.arange(num_zone+1) + 1})
        pred_zone_df_baseline = pred_zone_df_baseline.merge(data_route[['stops', 'zone_id', 'seq_ID']], on=['zone_id'])
        pred_zone_df_baseline = pred_zone_df_baseline.sort_values(['pred_seq_id', 'seq_ID'])
        est_seq_baseline = list(pred_zone_df_baseline['stops'])
        est_seq_baseline = np.array([name_map[s] for s in est_seq_baseline])

        if PG_ini is not None:
            out_baseline, _, _ = PG_ini(x_r_t, None, num_zone_t, None, ini_to_first_r_t, None,
                                             None, mask_prob=True, teacher_force=False, Print_pred=False, sample=False,
                                             No_dyn_dist=True, model_apply=True, Calculate_loss=False)

            out_baseline = out_baseline.permute(1, 0)

            cost_step_base = 1 - torch.eq(out_baseline[0], torch.tensor(y_r[0])).int()

            predict_seq_baseline = ['INIT']
            for j in out_baseline[0, :num_zone[0]]:
                predict_seq_baseline.append(input_seq_dict[(route_id, int(j))])
            assert len(predict_seq_baseline) == num_zone + 1

            pred_zone_df_baseline = pd.DataFrame(
                {'zone_id': predict_seq_baseline, 'pred_seq_id': np.arange(num_zone + 1) + 1})
            pred_zone_df_baseline = pred_zone_df_baseline.merge(data_route[['stops', 'zone_id', 'seq_ID']],
                                                                on=['zone_id'])
            pred_zone_df_baseline = pred_zone_df_baseline.sort_values(['pred_seq_id', 'seq_ID'])
            est_seq_baseline = list(pred_zone_df_baseline['stops'])
            est_seq_ini = np.array([name_map[s] for s in est_seq_baseline])
        else:
            est_seq_ini = None

    else:
        cost_step_base = None
    if cost_mat_array_r is not None:
        cost_mat_array_route = cost_mat_array_r
        score, seq_dev_value, erp_per_edit, total_dist, total_edit_count = evaluate_simple(actual_seq, est_seq, cost_mat_array_route)

        if score == 1:
            print('PG score = 1', route_id, 'partition_num', partition_num, 'replica_num', replica_num)

            torch.save(PG.state_dict(), 'model_output/Fullinfo_PG_Pot_Net_score_error_point.pt')
            torch.save(PG_baseline.state_dict(),
                       'model_output/Fullinfo_PG_Base_Pot_Net_score_error_point.pt')

            disrupt_info = {'route_id':route_id,'replica_num':replica_num,'partition_num':partition_num}
            with open('model_output/PG_disrupt_info.pkl', 'wb') as f:
                pickle.dump(disrupt_info, f)
            print('pred_seq', predict_seq)
            print('actual_seq', pd.unique(data_route['zone_id']))
            exit()

        if replica_num == 1:
            score_baseline, seq_dev_value, erp_per_edit, total_dist, total_edit_count = evaluate_simple(actual_seq, est_seq_baseline, cost_mat_array_route)
            if score_baseline == 1:
                print('PG baseline score = 1', route_id, 'partition_num',partition_num,'replica_num',replica_num)
                torch.save(PG.state_dict(), 'model_output/Fullinfo_PG_Pot_Net_score_error_point.pt')
                torch.save(PG_baseline.state_dict(),
                           'model_output/Fullinfo_PG_Base_Pot_Net_score_error_point.pt')

                disrupt_info = {'route_id': route_id, 'replica_num': replica_num, 'partition_num': partition_num}
                with open('model_output/PG_base_disrupt_info.pkl', 'wb') as f:
                    pickle.dump(disrupt_info, f)

                exit()
    else:
        score = seq_dev(actual_seq, est_seq)
        if replica_num == 1:
            score_baseline = seq_dev(actual_seq, est_seq_baseline)

    reward = score * weight_T
    if replica_num == 1:
        reward_b = score_baseline * weight_T
    else:
        reward_b = None
    
    return out, reward, reward_b, cost_step, cost_step_base


def train(batch_states,batch_rewards,batch_rewards_b, batch_actions,batch_init_dist, model, device, optimizer, num_zone_batch, weight_cost, batch_cost_step, batch_cost_step_b):
    # Prepare the batches for training
    # Add states, reward and actions to tensor

    #print(state_tensor.shape)

    # Convert the probs by the model to log probabilities

    # _,_,probs = model(state_tensor, y = None, input_lengths = input_lengths, model_apply = True, teacher_force_ratio = 0, sample = True, mask_prob=False)

    state_np = np.array(batch_states)
    state_np = state_np.astype(float)
    #
    #
    # if np.isnan(state_np).any():
    #     print('*****exist nan in state')
    #     state_np = np.nan_to_num(state_np, nan = 0)
    state_tensor = torch.Tensor(state_np)


    batch_rewards_np = np.array(batch_rewards)
    batch_rewards_np = batch_rewards_np.astype(float)
    # if np.isnan(batch_rewards_np).any():
    #     print('*****exist nan in batch_rewards_np')
    #     batch_rewards_np = np.nan_to_num(batch_rewards_np, nan = 0)
    reward_tensor = torch.Tensor(batch_rewards_np)


    batch_rewards_b_np = np.array(batch_rewards_b)
    batch_rewards_b_np = batch_rewards_b_np.astype(float)
    # if np.isnan(batch_rewards_b_np).any():
    #     print('*****exist nan in batch_rewards_b_np')
    #     batch_rewards_b_np = np.nan_to_num(batch_rewards_b_np, nan = 0)

    reward_tensor_b = torch.Tensor(batch_rewards_b_np)

    batch_init_dist_np = np.array(batch_init_dist)
    batch_init_dist_np = batch_init_dist_np.astype(float)
    # if np.isnan(batch_init_dist_np).any():
    #     print('*****exist nan in batch_init_dist_np')
    #     batch_init_dist_np = np.nan_to_num(batch_init_dist_np, nan = 0)
    batch_init_dist_t = torch.Tensor(batch_init_dist_np)

    num_zone_batch_np = np.array(num_zone_batch)
    num_zone_batch_np = num_zone_batch_np.astype(int)
    # if np.isnan(num_zone_batch_np).any():
    #     print('*****exist nan in num_zone_batch_np')
    #     num_zone_batch_np = np.nan_to_num(num_zone_batch_np, nan = 0)
    input_lengths = torch.Tensor(num_zone_batch_np).squeeze()


    model.train()


    _, _, probs = model(state_tensor, None, input_lengths, None, batch_init_dist_t, None,
     None, mask_prob=False, teacher_force=False, Print_pred=False, sample=True,
     No_dyn_dist=True, model_apply = True, Calculate_loss = False) # mask prob = False, otherwise prob is inf

    probs_np = probs.detach().numpy()
    probs_np = probs_np.astype(float)
    if np.isnan(probs_np).any():
        print('*****exist nan in probs_np')



    batch_cost_step_t = torch.stack(batch_cost_step)
    # print(batch_cost_step_t.shape)

    batch_cost_step_b_t = torch.stack(batch_cost_step_b)
    # print(batch_cost_step_b_t.shape)

    #print(probs[0,5,:])
    #

    #cum_prob = torch.cumsum(probs[0,:,:],dim=1)

    #print(cum_prob[4])
    #print(probs[0][4])
    # sum_log_probs = 0
    #print(batch_actions[0][4])
    batch_actions_t = torch.stack(batch_actions)
    along_idx = -1
    batch_actions_t = batch_actions_t.unsqueeze(along_idx)
    #print(batch_actions_t.shape)
    _probs = torch.gather(probs, along_idx, batch_actions_t)
    _probs = _probs.squeeze(along_idx)
    log_probs = torch.log(_probs)



    reward_tensor = reward_tensor.view(-1,1).expand(log_probs.shape) + batch_cost_step_t
    reward_tensor_b = reward_tensor_b.view(-1,1).expand(log_probs.shape) + batch_cost_step_b_t

    #print(batch_cost_step_t.shape)
    #print(reward_tensor.shape)

    # reward_tensor
    # reward_tensor_b


    reward_diff = reward_tensor - reward_tensor_b
    reward_diff = reward_diff * weight_cost

    log_probs_times_cost = reward_diff * log_probs


    log_probs = torch.flatten(log_probs_times_cost)
    log_probs = log_probs[~torch.isinf(log_probs)]
    mean_log_probs = torch.sum(log_probs)/len(batch_actions)
    #a=1
    # print(log_probs.shape)
    # idx_ = 0
    # for idx_ in range(len(batch_actions)):
    #     sum_log_probs += (reward_tensor[idx_] - reward_tensor_b[idx_]) * torch.sum(probs[idx_].gather(1, batch_actions[idx_].view(-1, 1)))
    # print(probs[idx_].gather(1, batch_actions[idx_].view(-1, 1)))
    #a=1
    #print(log_probs[0])
    # Mask the probs of the selected actions

    # Loss is negative of expected policy function J = R * log_prob

    #sum_log_probs = 1
    optimizer.zero_grad()
    loss = mean_log_probs


    #loss = selected_log_probs.mean()
    # Do the update gradient descent(with negative reward hence is gradient ascent)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()



def train_step_wise(batch_states,batch_rewards,batch_rewards_b, batch_actions, model, device, optimizer, num_zone_batch):
    # Prepare the batches for training
    # Add states, reward and actions to tensor

    model.train()

    state_tensor = torch.Tensor(batch_states)
    reward_tensor = torch.Tensor(batch_rewards)
    reward_tensor_b = torch.Tensor(batch_rewards_b)
    action_tensor = torch.Tensor(batch_actions)

    # Convert the probs by the model to log probabilities
    log_probs = torch.log(model(state_tensor))
    # Mask the probs of the selected actions
    selected_log_probs = (reward_tensor - reward_tensor_b) * log_probs[np.arange(len(action_tensor)), action_tensor]
    # Loss is negative of expected policy function J = R * log_prob


    loss = selected_log_probs.mean()
    #loss = selected_log_probs.mean()
    # Do the update gradient descent(with negative reward hence is gradient ascent)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def get_input(initial_seq='tsp'):
    if os.path.exists('../data/tsp_zone_seq.csv'):
        tsp = pd.read_csv('../data/tsp_zone_seq.csv')
        with open('../data/opt_zone_seq_tour.json', 'rb') as f:
            tsp_zone_seq_mean = json.load(f)
    else:
        # Load TSP Sequence
        with open('../data/opt_zone_seq_tour_pure_tsp.json', 'rb') as f:
            tsp_zone_seq_mean = json.load(f)
        tsp_list = []
        for r in tsp_zone_seq_mean:
            df = pd.DataFrame(tsp_zone_seq_mean[r], columns=['zone_id'])
            df['route_id'] = r
            df['pred_seq_id'] = df.index + 1
            tsp_list.append(df)
        tsp = pd.concat(tsp_list)

        tsp.sort_values(by=['route_id', 'pred_seq_id'], inplace=True)
        tsp.to_csv('../data/tsp_zone_seq.csv', index=False)

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


    return x, y, input_seq_dict, routes, zone_len, ini_to_first_zone[travel_time_col_list].values, num_features_without_zone, \
           num_features, zone_id_feature

    
def main():
    # ['tt', 'ttm', 'lat_dir_same', 'lng_dir_same', 'tsp_next_mean', 'tsp_next_min',
    #  'planned_service_time_sum', 'total_vol_sum', 'total_vol_max',
    #  'time_window_end_from_departure_sec_min']

    data = pd.read_csv('../data/build_route_with_seq.csv')
    data['stops'] = data['stops'].fillna('NA')



    LOAD_OLD = False
    if LOAD_OLD:
        filepath = 'D:/Dropbox (MIT)/00_Research/17_Amazon_LM_competition/data/model_build_inputs/travel_times.json'
        with open(filepath, newline='') as in_file:
            cost_mat_build = json.load(in_file)
    else:
        cost_mat_build = None

    LOAD_NEW = True
    if LOAD_NEW:
        with open('../data/cost_mtx_array.pkl', 'rb') as f:
            cost_mat_array = pickle.load(f)
    else:
        cost_mat_array = None

    with open('../data/stop_idx_map.pkl', 'rb') as f:
        name_map_all = pickle.load(f)

    with open('../data/train_routes.pkl', 'rb') as f:
        train_routes = pickle.load(f)

    if cost_mat_array is None:
        print('*************Only SD for score***********')




    PROCESS_DATA = True
    if PROCESS_DATA:
        initial_seq = 'tsp' # 'tsp'
        print('initial seq is ',initial_seq)
        x,y,input_seq_dict,route_list,zone_len, ini_to_first_zone, num_features_without_zone, n_feature, zone_id_feature = get_input(initial_seq)
        zone_len_max = zone_len.max()

        x_dict = {}
        y_dict = {}
        zone_len_dict = {}
        ini_to_first_zone_dict = {}
        for key in route_list:
            x_dict[key] = x[route_list.index(key):route_list.index(key)+1,:,:]
            y_dict[key] = y[route_list.index(key):route_list.index(key)+1,:]
            zone_len_dict[key] = zone_len[route_list.index(key):route_list.index(key)+1]
            ini_to_first_zone_dict[key] = ini_to_first_zone[route_list.index(key):route_list.index(key)+1,:]
        with open('../data/RL_x_dict.pkl', 'wb') as f:
            pickle.dump(x_dict, f)
            pickle.dump(y_dict, f)
            pickle.dump(zone_len_dict, f)
            pickle.dump(input_seq_dict, f)
            pickle.dump(zone_len_max, f)
            pickle.dump(ini_to_first_zone_dict, f)
            pickle.dump(num_features_without_zone, f)
            pickle.dump(zone_id_feature, f)
            pickle.dump(n_feature, f)


        print('=====Finish process data=======', initial_seq)
    else:
        with open('../data/RL_x_dict.pkl', 'rb') as f:
            x_dict = pickle.load(f)
            y_dict = pickle.load(f)
            zone_len_dict = pickle.load(f)
            input_seq_dict = pickle.load(f)
            zone_len_max = pickle.load(f)
            ini_to_first_zone_dict = pickle.load(f)
            num_features_without_zone = pickle.load(f)
            zone_id_feature = pickle.load(f)
            n_feature = pickle.load(f)




    dev = "cpu"
    device = torch.device(dev)

    torch.manual_seed(123)

    HIDDEN_SIZE_asudnn = 128
    HIDDEN_SIZE = 32

    LOCAL_VIEW = 0.9

    encoder = Encoder(num_features_without_zone, HIDDEN_SIZE)
    decoder = Decoder(num_features_without_zone, HIDDEN_SIZE, num_zone_id_feature = len(zone_id_feature), max_len = zone_len_max, local_view = LOCAL_VIEW, HIDDEN_SIZE_asudnn = HIDDEN_SIZE_asudnn)
    PG = PointerNetwork(encoder, decoder, num_zone_id_feature = len(zone_id_feature), input_seq_dict = input_seq_dict, max_zone_len = zone_len_max)



    encoder_base = Encoder(num_features_without_zone, HIDDEN_SIZE)
    decoder_base = Decoder(num_features_without_zone, HIDDEN_SIZE, num_zone_id_feature = len(zone_id_feature), max_len = zone_len_max, local_view = LOCAL_VIEW, HIDDEN_SIZE_asudnn = HIDDEN_SIZE_asudnn)
    PG_baseline = PointerNetwork(encoder_base, decoder_base, num_zone_id_feature = len(zone_id_feature), input_seq_dict = input_seq_dict, max_zone_len = zone_len_max)
    

    with open('../data/opt_zone_seq.json', 'rb') as f:
        TSP_model = json.load(f)




    PG.to(device)
    PG_baseline.to(device)
    optimizer = torch.optim.Adam(PG.parameters())  # lr=0.005,, weight_decay=1e-5

    weight_T = 50
    weight_cost = 10


    num_per_train = 128 # ie batch size
    num_training_route_partition = int(np.round(len(train_routes) / num_per_train))


    epsilon_raw = 0.3

    gamma = 0.9
    N_training_epoch = 10

    max_Num_replication_per_batch = 10
    tic_s = time.time()

    move_to_next_based_on_last_N = 2


    INIT = True
    if INIT:
        # net_para = torch.load('ptr_net_local_view_05.pt')
        # PG.load_state_dict(net_para.state_dict())
        net_para = torch.load('ptr_net_local_view_05.pt')
        PG.load_state_dict(net_para)
        net_para = torch.load('ptr_net_local_view_05.pt')
        PG_baseline.load_state_dict(net_para)

        encoder_ini = Encoder(num_features_without_zone, HIDDEN_SIZE)
        decoder_ini = Decoder(num_features_without_zone, HIDDEN_SIZE, num_zone_id_feature=len(zone_id_feature),
                               max_len=zone_len_max, local_view=LOCAL_VIEW, HIDDEN_SIZE_asudnn=HIDDEN_SIZE_asudnn)
        PG_ini = PointerNetwork(encoder_ini, decoder_ini, num_zone_id_feature=len(zone_id_feature),
                                     input_seq_dict=input_seq_dict, max_zone_len=zone_len_max)
        net_para = torch.load('ptr_net_local_view_05.pt')
        PG_ini.load_state_dict(net_para)
    else:
        PG_ini = None

    for epoch in range(N_training_epoch):
        print('==========current epoch', epoch ,'=======')

        sum_reward_list = []
        count = 0

        overall_score = []
        overall_score_tsp = []

        RL_routes_partition = partition(train_routes, n = num_training_route_partition, random_seed=111 + epoch + 1)
        tic = time.time()
        partition_num = 0
        for env_routes in RL_routes_partition:

            partition_num += 1
            print('===start partition', partition_num)
            # if partition_num < 16:
            #     continue
            # else:
            #     a=1
            score_routes = []
            score_routes_b = []
            replica_num = 0

            if partition_num == np.round(len(RL_routes_partition) / 2):
                torch.save(PG.state_dict(), 'model_output/Fullinfo_PG_Pot_Net_' +  str(epoch+1) +  '_half.pt')
                torch.save(PG_baseline.state_dict(), 'model_output/Fullinfo_PG_Base_Pot_Net_' +  str(epoch+1) +  '_half.pt')

            if partition_num == len(RL_routes_partition):
                torch.save(PG.state_dict(), 'model_output/Fullinfo_PG_Pot_Net'+  str(epoch+1) + '_full.pt')
                torch.save(PG_baseline.state_dict(),
                           'model_output/Fullinfo_PG_Base_Pot_Net_' + str(epoch + 1) + '_full.pt')

            batch_score_min = 100
            best_per_env_r_score = []
            #tsp_batch_score = np.mean(tsp_score.loc[tsp_score['route_id'].isin(env_routes),'score'])

            TSP_score_batch = []

            while True: #for _ in range(Num_replication_per_batch):

                replica_num += 1

                if replica_num == 7:
                    a=1

                epsilon = epsilon_raw * ( max_Num_replication_per_batch - replica_num + 1 ) / max_Num_replication_per_batch
                batch_rewards = []
                batch_rewards_b = []
                batch_states = []
                batch_actions = []
                batch_init_dist = []
                batch_cost_step = []
                batch_cost_step_b = []


                per_env_r_score = []

                print("num replication", replica_num)

                if replica_num > max_Num_replication_per_batch:
                    print('***Reach max rep num, next instance','current_parittion', partition_num, 'total_partition', len(RL_routes_partition))
                    print('best net score', batch_score_min)
                    base_score =  np.mean(score_routes_b[-move_to_next_based_on_last_N*len(env_routes):])
                    base_net = PG_baseline.state_dict()
                    if batch_score_min > 1.3 * base_score:
                        PG.load_state_dict(base_net)
                        overall_score += score_routes_b[-1*len(env_routes):]
                        print('too bad, load base')
                    else:
                        PG.load_state_dict(best_net)
                        overall_score += best_per_env_r_score
                        print('not too bad, load best net')


                    overall_score_tsp += TSP_score_batch
                    print('overall score', np.nanmean(overall_score),len(overall_score),'overall TSP score', np.nanmean(overall_score_tsp),len(overall_score_tsp))
                    with open("fullinfo_results.txt", "a") as f:
                        f.write("%d,%d,%.4f,%.4f,%.4f,%.4f,%d\n" % (epoch,partition_num,np.nanmean(overall_score), np.nanmean(overall_score_tsp),
                                                                    np.mean(score_routes), np.mean(score_routes_b), int(np.round((time.time() - tic_s)/60))))
                    break

                num_zone_batch = []

                for route in env_routes:
                    # route = 'RouteID_0021a2aa-780f-460d-b09a-f301709e2523'
                    # if route == 'RouteID_eab2f503-0f54-4571-bb4a-402b71472c0c':
                    #     a=1
                    count += 1
                    name_map_all_r = name_map_all[route]

                    if LOAD_NEW:
                        cost_mat_array_r = cost_mat_array[route]
                    else:
                        cost_mat_array_r = None
                    
                    x_r = x_dict[route]
                    y_r = y_dict[route]
                    ini_to_first_r = ini_to_first_zone_dict[route]
                    data_route = data.loc[data['route_id'] == route]
                    num_zone = zone_len_dict[route]
                    num_zone_batch.append(num_zone)

                    ############
                    # net_para = torch.load('model_output/Fullinfo_PG_Pot_Net_1_half.pt')
                    # PG.load_state_dict(net_para)

                    #############

                    y, rewards, rewards_b, cost_step, cost_step_base = generate_training_samples(x_r,y_r,ini_to_first_r, num_zone, PG, PG_baseline,PG_ini, device, weight_T, data_route,input_seq_dict,route,epsilon,
                                              cost_mat_array_r=cost_mat_array_r, name_map_all_r=name_map_all_r, replica_num = replica_num, partition_num = partition_num)

                    batch_cost_step.append(cost_step)
                    states = np.squeeze(x_r)
                    #print(states.shape)
                    #actions = y.squeeze()
                    actions = y.squeeze()
                    sum_reward_list.append(rewards)

                    batch_rewards.append(torch.tensor(rewards))
                    batch_states.append(states)
                    batch_init_dist.append(np.squeeze(ini_to_first_r))
                    batch_actions.append(actions)
                    score_routes.append(rewards/weight_T)
                    per_env_r_score.append(rewards/weight_T)

                    if replica_num == 1:
                        batch_cost_step_b.append(cost_step_base)
                        batch_rewards_b.append(torch.tensor(rewards_b))
                        score_routes_b.append(rewards_b/weight_T)


                    if replica_num == 1:
                        actual_seq = list(data_route['stops'])
                        pred_seq = TSP_model[route]
                        try:
                            assert len(pred_seq) == num_zone + 1
                        except:
                            print('=======Error prediction, please check========')
                            print('route_id', route)
                            print('actual_seq', actual_seq)
                            print('pred_seq', pred_seq)
                            TSP_score_batch.append(0.5)
                            continue

                        pred_zone_df = pd.DataFrame({'zone_id': pred_seq, 'pred_seq_id': np.arange(num_zone + 1) + 1})

                        pred_zone_df = pred_zone_df.merge(data_route[['stops', 'zone_id', 'seq_ID']],
                                                          on=['zone_id'])
                        pred_zone_df = pred_zone_df.sort_values(['pred_seq_id', 'seq_ID'])
                        est_seq = list(pred_zone_df['stops'])

                        name_map = name_map_all_r
                        actual_seq = np.array([name_map[s] for s in actual_seq])
                        est_seq = np.array([name_map[s] for s in est_seq])

                        if cost_mat_array_r is not None:
                            cost_mat_array_route = cost_mat_array_r
                            TSP_score, seq_dev_value, erp_per_edit, total_dist, total_edit_count = evaluate_simple(
                                actual_seq,
                                est_seq,
                                cost_mat_array_route)
                        else:
                            TSP_score = seq_dev(actual_seq, est_seq)
                        TSP_score_batch.append(TSP_score)


                print('avg_route_scores', np.mean(score_routes[-move_to_next_based_on_last_N*len(env_routes):]),
                      'avg_route_scores_base', np.mean(score_routes_b[-move_to_next_based_on_last_N*len(env_routes):]),
                      'TSP_score', np.mean(TSP_score_batch),
                      'sum cost', np.mean(sum_reward_list))

                if np.mean(per_env_r_score) < batch_score_min:
                    best_net = PG.state_dict()
                    batch_score_min = np.mean(per_env_r_score)
                    best_per_env_r_score = per_env_r_score

                if replica_num == 1:
                    batch_cost_step_b_record = copy.deepcopy(batch_cost_step_b)
                    batch_rewards_b_record = copy.deepcopy(batch_rewards_b)
                    score_routes_b_record = copy.deepcopy(score_routes_b)
                else:
                    batch_cost_step_b = copy.deepcopy(batch_cost_step_b_record)
                    batch_rewards_b = copy.deepcopy(batch_rewards_b_record)
                    score_routes_b = copy.deepcopy(score_routes_b_record)

                train(batch_states, batch_rewards,batch_rewards_b, batch_actions, batch_init_dist, PG, device, optimizer, num_zone_batch, weight_cost, batch_cost_step, batch_cost_step_b)

                # train_step_wise(batch_states, batch_rewards, batch_rewards_b, batch_actions, PG, device, optimizer,
                #       num_zone_batch)
                
                if np.mean(score_routes[-move_to_next_based_on_last_N*len(env_routes):]) <= np.mean(score_routes_b[-move_to_next_based_on_last_N*len(env_routes):]):
                    print('***Update_base, next instance','current_parittion', partition_num, 'total_partition', len(RL_routes_partition))
                    PG_baseline.train()
                    PG_baseline.load_state_dict(PG.state_dict())
                    overall_score += per_env_r_score
                    overall_score_tsp += TSP_score_batch
                    print('overall score', np.nanmean(overall_score),len(overall_score),'overall TSP score', np.nanmean(overall_score_tsp), len(overall_score_tsp))
                    with open("fullinfo_results.txt", "a") as f:
                        f.write("%d,%d,%.4f,%.4f,%.4f,%.4f,%d\n" % (epoch,partition_num,np.nanmean(overall_score), np.nanmean(overall_score_tsp),
                                                                    np.mean(score_routes), np.mean(score_routes_b), int(np.round((time.time() - tic_s)/60))))
                    break
        print('===finish epoch', epoch, 'time:',time.time() - tic,'===')

    #
    # if INIT:
    #     with_prior_tail = 'with_prior'
    # else:
    with_prior_tail = ''


    torch.save(PG.state_dict(), 'model_output/Fullinfo_PG_Pot_Net_' + with_prior_tail +'.pt')
    torch.save(PG_baseline.state_dict(),
               'model_output/Fullinfo_PG_Base_Pot_Net_' + with_prior_tail +'.pt')

if __name__ == '__main__':
    main()
