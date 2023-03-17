import random
import pickle as pkl
import pandas as pd
import numpy as np
import _constants


def partition(list_in, n, random_seed):
    #random.seed(random_seed)
    random.Random(random_seed).shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

num_neighbor = _constants.n_neighbour

LOAD_TSP_SEQ_AS_ACTUAL = False

LOAD_ACTUAL_FILTER = True

if LOAD_TSP_SEQ_AS_ACTUAL:
    with open('data/processed_zone_info_neighbour_'+ str(num_neighbor) +'_tsp_seq.pkl', 'rb') as f:
        data = pkl.load(f)
else:
    if LOAD_ACTUAL_FILTER:
        with open('data/processed_zone_info_neighbour_'+ str(num_neighbor) +'_filter_passed.pkl', 'rb') as f:
            data = pkl.load(f)
    else:
        with open('data/processed_zone_info_neighbour_'+ str(num_neighbor) +'.pkl', 'rb') as f:
            data = pkl.load(f)

# data_view = data[['route_id','zone_id','next_zone_id','zone_seq','zone_id_avail',
#                   'zone_avail_seq','next_in_neighbor','tsp_mean_next_zone','tsp_next_mean','mean_travel_time']]

num_tsp_mean_in_neighbor = data.groupby(['route_id','zone_id'],sort = False, as_index = False)['tsp_next_mean'].max()
print('prop tsp mean in neighbor',sum(num_tsp_mean_in_neighbor['tsp_next_mean'] == 1)/len(num_tsp_mean_in_neighbor))

num_tsp_min_in_neighbor = data.groupby(['route_id','zone_id'],sort = False, as_index = False)['tsp_next_min'].max()
print('prop tsp min in neighbor',sum(num_tsp_min_in_neighbor['tsp_next_min'] == 1)/len(num_tsp_min_in_neighbor))

nearest_neighbor_5_out = True
if nearest_neighbor_5_out:
    data_neighbor = data[['route_id','zone_id','zone_id_avail']].copy()
    data_neighbor = data_neighbor.dropna()
    nearest_neighbor = data_neighbor.groupby(['route_id','zone_id']).apply(lambda x:list(x['zone_id_avail']))
    nearest_neighbor_dict = nearest_neighbor.to_dict()

    if LOAD_TSP_SEQ_AS_ACTUAL:
        with open('data/nearest_neighbor_' + str(num_neighbor) + '_tsp_seq.pkl', 'wb') as f:
            pkl.dump(nearest_neighbor_dict, f)
    else:
        with open('data/nearest_neighbor_' + str(num_neighbor) + '.pkl', 'wb') as f:
            pkl.dump(nearest_neighbor_dict, f)

route_id_seq = pd.unique(data['route_id'])

num_cv = 5
route_id_seq_unique = sorted(list(set(route_id_seq)))
routes_partition = partition(route_id_seq_unique, n=num_cv, random_seed=234)
train_routes = []
cv_id = 1
for i in range(num_cv):
    if i != cv_id:
        train_routes += routes_partition[i]

test_routes = routes_partition[cv_id]

with open('train_routes.pkl', 'wb') as f:
    pkl.dump(train_routes, f)

with open('testing_routes.pkl', 'wb') as f:
    pkl.dump(test_routes, f)



feature_col = _constants.feature_col
#print(data[feature_col].values.shape)

data_unique_route_zone = data.drop_duplicates(['route_id','zone_id'])
route_id_seq = list(data_unique_route_zone['route_id'])
zone_id_seq = list(data_unique_route_zone['zone_id'])

data_train = data.copy() #data.loc[data['route_id'].isin(train_routes)]
x_train = data_train[feature_col].values.reshape(-1, num_neighbor , len(feature_col))

y_label = data_train.sort_values(['next_in_neighbor'], ascending=False).drop_duplicates(['route_id','zone_id'])

y_label = y_label.sort_values(['route_id','zone_seq'])
y_label.loc[y_label['next_in_neighbor'] == 0, 'zone_avail_seq'] = -1
y_train = y_label['zone_avail_seq'].values

print('prop in neighbor', sum(y_train != -1)/len(y_train))

print(x_train.shape)
print(y_train.shape)

if LOAD_TSP_SEQ_AS_ACTUAL:
    tail_name = '_tsp_seq'
else:
    if LOAD_ACTUAL_FILTER:
        tail_name = '_filter_passed'
    else:
        tail_name = ''

with open('data/processed_zone_seq_neighbour_' + str(num_neighbor) + tail_name + '.pkl', 'wb') as f:
    pkl.dump(x_train,f)
    pkl.dump(y_train,f)

with open('data/route_id_seq_' + str(num_neighbor) + tail_name + '.pkl', 'wb') as f:
    pkl.dump(route_id_seq, f)

with open('data/zone_id_seq_' + str(num_neighbor) + tail_name + '.pkl', 'wb') as f:
    pkl.dump(zone_id_seq, f)
