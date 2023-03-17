def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn # what a elegant way to avoid the fucking warnings!
import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

import xgboost as xgb
import pandas as pd
import random
from sklearn import preprocessing
import json
import sys
import time
import numpy as np
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(0, '../')
# from score_testing_func import evaluate_simple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


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



# pred_zone_seq_TSP = pd.read_csv("pred_zone_seq_TSP.csv")
# actual_seq = pd.read_csv('../../data/zone_data.csv')
#
#
# pred_zone_seq_TSP = pred_zone_seq_TSP.merge(actual_seq[['route_id','zone_id','zone_seq']], on = ['route_id','zone_id'])
# pred_zone_seq_TSP = pred_zone_seq_TSP.sort_values(['route_id','zone_seq'])



score_reverse = pd.read_csv('../../tsp_xiaotong/score_reverse.csv')

stop_lat_lng = pd.read_csv('../../../data_fake/model_apply_outputs/model_apply_output/build_route_with_seq.csv')
zone_lat_lng = pd.read_csv('../../../data_fake/model_apply_outputs/model_apply_output/zone_data.csv')

package_service_time = pd.read_csv('package_service_time.csv')
# print(type(package_service_time['forward'].iloc[0]))


score_reverse_select_by_service_time = score_reverse.merge(package_service_time, on = ['route_id'])

score_reverse_select_by_service_time['s_diff'] = np.abs(score_reverse_select_by_service_time['first15_servicetime']
                                                        - score_reverse_select_by_service_time['last15_servicetime'])

score_reverse_select_by_service_time['s_diff_percentage'] = score_reverse_select_by_service_time['s_diff']/np.minimum(score_reverse_select_by_service_time['first15_servicetime'],
                                                                                                                  score_reverse_select_by_service_time['last15_servicetime'])
#
# threshold = 0.7
#
# change_sig = score_reverse_select_by_service_time.loc[(score_reverse_select_by_service_time['s_diff_percentage'] > threshold)]
# change_sig_last_big = change_sig.loc[change_sig['last15_servicetime'] > change_sig['first15_servicetime']]
# change_sig_last_big['need_reverse'] = 1
# print(len(change_sig_last_big))
#
#
# score_reverse_time = score_reverse.merge(change_sig_last_big[['route_id','need_reverse']], how = 'left')
#
# score_reverse_time['final_score'] = score_reverse_time['score']
# score_reverse_time.loc[~score_reverse_time['need_reverse'].isna(), 'final_score'] = score_reverse_time.loc[~score_reverse_time['need_reverse'].isna(), 'score_rev']
#
# print('all_tsp score', np.mean(score_reverse_time['score']))
# print('change score', np.mean(score_reverse_time['final_score']))
#



a=1

#
#
#
# package_service_time['service_time_small_current'] = package_service_time['forward'].astype('int')
# package_service_time['service_time_small_reverse'] = package_service_time['reverse'].astype('int')


a=1

# with open('../../tsp_xiaotong/opt_complete_seq_tour.json', 'rb') as f:
#     opt_complete_seq_tour = json.load(f)

# df_dict = {'route_id':[],'stops':[],'pred_seq_id':[]}
# for key in opt_complete_seq_tour:
#     pred_seq_id = 1
#     for stop_id in opt_complete_seq_tour[key]:
#         df_dict['route_id'].append(key)
#         df_dict['stops'].append(stop_id)
#         df_dict['pred_seq_id'].append(pred_seq_id)
#         pred_seq_id += 1
# tsp_df = pd.DataFrame(df_dict)
#


with open('../../../data_fake/model_apply_outputs/model_apply_output/mean_dist/opt_zone_seq_tour.json', 'rb') as f:
    opt_complete_seq_tour = json.load(f)

df_dict = {'route_id':[],'zone_id':[],'pred_seq_id':[]}
for key in opt_complete_seq_tour:
    pred_seq_id = 1
    for stop_id in opt_complete_seq_tour[key]:
        df_dict['route_id'].append(key)
        df_dict['zone_id'].append(stop_id)
        df_dict['pred_seq_id'].append(pred_seq_id)
        pred_seq_id += 1

tsp_df_zone = pd.DataFrame(df_dict)

tsp_df_zone_reverse = tsp_df_zone.copy()


tsp_df_zone_reverse.loc[tsp_df_zone_reverse['pred_seq_id'] == 1, 'pred_seq_id'] = 1000
tsp_df_zone_reverse = tsp_df_zone_reverse.sort_values(['route_id', 'pred_seq_id'],ascending=False)
tsp_df_zone_reverse['seq_temp'] = tsp_df_zone_reverse.groupby(['route_id']).cumcount() + 1

tsp_df_zone = tsp_df_zone.sort_values(['route_id', 'pred_seq_id'])
tsp_df_zone['seq_temp'] = tsp_df_zone.groupby(['route_id']).cumcount() + 1

zone_lat_lng['lat'] = zone_lat_lng['lat_mean']
zone_lat_lng['lng'] = zone_lat_lng['lng_mean']
tsp_df_zone = tsp_df_zone.merge(zone_lat_lng[['route_id', 'zone_id', 'lat', 'lng']], on=['route_id', 'zone_id'])

tsp_df_zone = tsp_df_zone.sort_values(['route_id', 'seq_temp'])

tsp_df_zone_reverse = tsp_df_zone_reverse.merge(zone_lat_lng[['route_id', 'zone_id', 'lat', 'lng']], on=['route_id', 'zone_id'])

tsp_df_zone_reverse = tsp_df_zone_reverse.sort_values(['route_id', 'seq_temp'])


tsp_df_zone['zone_mid_point_lat'] = tsp_df_zone.groupby(['route_id'])['lat'].transform('mean')
tsp_df_zone['zone_mid_point_lng'] = tsp_df_zone.groupby(['route_id'])['lng'].transform('mean')

tsp_df_zone_reverse['zone_mid_point_lat'] = tsp_df_zone_reverse.groupby(['route_id'])['lat'].transform('mean')
tsp_df_zone_reverse['zone_mid_point_lng'] = tsp_df_zone_reverse.groupby(['route_id'])['lng'].transform('mean')


tsp_df_zone_first = tsp_df_zone.loc[tsp_df_zone['seq_temp'] == 2]
tsp_df_zone_reverse_first = tsp_df_zone_reverse.loc[tsp_df_zone_reverse['seq_temp'] == 2]

tsp_df_zone_first['dist_to_mid_tsp'] = np.square(tsp_df_zone_first['lat'] - tsp_df_zone_first['zone_mid_point_lat']) \
                                   + np.square(tsp_df_zone_first['lng'] - tsp_df_zone_first['zone_mid_point_lng'])

tsp_df_zone_reverse_first['dist_to_mid_reverse_tsp'] = np.square(tsp_df_zone_reverse_first['lat'] - tsp_df_zone_reverse_first['zone_mid_point_lat']) \
                                   + np.square(tsp_df_zone_reverse_first['lng'] - tsp_df_zone_reverse_first['zone_mid_point_lng'])

first_to_mid_compare = tsp_df_zone_first[['route_id','dist_to_mid_tsp']].merge(
    tsp_df_zone_reverse_first[['route_id','dist_to_mid_reverse_tsp']], on = ['route_id'])


score_first_to_mid = score_reverse.merge(first_to_mid_compare, on = ['route_id'])

# factor = 1
# score_first_to_mid['forward'] = score_first_to_mid['dist_to_mid_tsp'] * factor < score_first_to_mid['dist_to_mid_reverse_tsp']
# #print(1223 - score_test['forward'].sum())
# score_new = np.mean(score_first_to_mid['forward'] * score_first_to_mid['score'] + (1 - score_first_to_mid['forward']) * score_first_to_mid['score_rev'])
# score_old = np.mean(score_first_to_mid['score'])
# print('score_new',score_new,'score_old',score_old)
#

A=1


def obtain_degree_info_zone_level(tsp_df_):
    zone_lat_lng['lat'] = zone_lat_lng['lat_mean']
    zone_lat_lng['lng'] = zone_lat_lng['lng_mean']
    tsp_df_lat_lng = tsp_df_.merge(zone_lat_lng[['route_id','zone_id','lat','lng']], on = ['route_id','zone_id'])

    tsp_df_lat_lng = tsp_df_lat_lng.sort_values(['route_id','seq_temp'])

    tsp_df_lat_lng['last_route_id'] = tsp_df_lat_lng['route_id'].shift(1)
    tsp_df_lat_lng['last_zone'] = tsp_df_lat_lng['zone_id'].shift(1)
    tsp_df_lat_lng['last_lat'] = tsp_df_lat_lng['lat'].shift(1)
    tsp_df_lat_lng['last_lng'] = tsp_df_lat_lng['lng'].shift(1)

    tsp_df_lat_lng = tsp_df_lat_lng.loc[tsp_df_lat_lng['last_route_id'] == tsp_df_lat_lng['route_id']]

    tsp_df_lat_lng['next_route_id'] = tsp_df_lat_lng['route_id'].shift(-1)
    tsp_df_lat_lng['next_zone'] = tsp_df_lat_lng['zone_id'].shift(-1)
    tsp_df_lat_lng['next_lat'] = tsp_df_lat_lng['lat'].shift(-1)
    tsp_df_lat_lng['next_lng'] = tsp_df_lat_lng['lng'].shift(-1)
    tsp_df_lat_lng = tsp_df_lat_lng.loc[tsp_df_lat_lng['next_route_id'] == tsp_df_lat_lng['route_id']]


    angle, cosine_angle = angle_between_three_points(tsp_df_lat_lng[['last_lat','last_lng']].values,
                                                     tsp_df_lat_lng[['lat','lng']].values,
                                                     tsp_df_lat_lng[['next_lat','next_lng']].values)


    tsp_df_lat_lng['degree'] = angle/np.pi * 180
    tsp_df_lat_lng['cosine_angle'] = cosine_angle
    # min_ = np.min(data_pc['degree'])
    # max_ = np.max(data_pc['degree'])


    tsp_df_lat_lng['forward'] = 0
    tsp_df_lat_lng.loc[(tsp_df_lat_lng['degree'] >= 135), 'forward'] = 1

    tsp_df_lat_lng['left_side'] = 0
    tsp_df_lat_lng['right_side'] = 0
    tsp_df_lat_lng['third_y_minus_first_y'] = tsp_df_lat_lng['next_lng'] - tsp_df_lat_lng['last_lng']
    tsp_df_lat_lng.loc[(tsp_df_lat_lng['degree'] >= 45) & (tsp_df_lat_lng['degree'] <= 135) &  (tsp_df_lat_lng['third_y_minus_first_y'] >= 0), 'left_side'] = 1 # & (data_pc['cosine_angle']  135)
    tsp_df_lat_lng.loc[(tsp_df_lat_lng['degree'] >= 45) & (tsp_df_lat_lng['degree'] <= 135) &  (tsp_df_lat_lng['third_y_minus_first_y'] < 0), 'right_side'] = 1

    tsp_df_lat_lng['backward'] = 0
    tsp_df_lat_lng.loc[(tsp_df_lat_lng['degree'] < 45), 'backward'] = 1


    return tsp_df_lat_lng

tsp_df_lat_lng_forward = obtain_degree_info_zone_level(tsp_df_zone)
tsp_df_lat_lng_backward = obtain_degree_info_zone_level(tsp_df_zone_reverse)

tsp_df_lat_lng_forward_num = tsp_df_lat_lng_forward.groupby(['route_id']).agg({'forward': 'sum', 'backward': 'sum', 'left_side':'sum','right_side':'sum'}).reset_index()

tsp_df_lat_lng_backward_num = tsp_df_lat_lng_backward.groupby(['route_id']).agg({'forward': 'sum', 'backward': 'sum', 'left_side':'sum','right_side':'sum'}).reset_index()




feature = tsp_df_lat_lng_forward_num.merge(tsp_df_lat_lng_backward_num, on = ['route_id'])

score_reverse_left_determine = score_reverse.merge(feature[['route_id','left_side_x','left_side_y']])
score_reverse_left_determine['left_tsp_more'] = score_reverse_left_determine['left_side_x'] > score_reverse_left_determine['left_side_y']
score_reverse_left_determine['left_tsp_more'] = score_reverse_left_determine['left_tsp_more'].astype('int')
score_reverse_left_determine['select_model_score'] = score_reverse_left_determine['score_rev']*score_reverse_left_determine['left_tsp_more'] \
                                                     + (1 - score_reverse_left_determine['left_tsp_more']) * score_reverse_left_determine['score']





print('test select by left score', np.mean(score_reverse_left_determine['select_model_score']) )
print('test all direct tsp score', np.mean(score_reverse_left_determine['score']) )


score_reverse['tsp_good'] = score_reverse['score'] < score_reverse['score_rev']
score_reverse['tsp_good'] = score_reverse['tsp_good'].astype('int')

feature_and_y = feature.merge(score_reverse[['route_id','tsp_good']],on = ['route_id'])

print('total samples',len(feature_and_y))

feature_and_y = feature_and_y.reset_index(drop=True)
feature_and_y['click_wise_current'] = feature_and_y['left_side_x']  >  feature_and_y['right_side_x']



train = feature_and_y.sample(frac=0.7, replace=False, random_state=3)

test = feature_and_y.loc[set(feature_and_y.index).difference(train.index)]


all_col = list(feature_and_y.columns)
used_feature = ['click_wise_current']

used_feature_idx = []
for key in used_feature:
    used_feature_idx.append(all_col.index(key))

X_train = train.iloc[:,used_feature_idx]
Y_train = train.iloc[:,9]

X_test = test.iloc[:,used_feature_idx]
Y_test = test.iloc[:,9]

####################################################
model = MLPClassifier()
####################################################


# model = xgb.XGBClassifier(objective = 'binary:logistic', eval_metric = 'logloss')
# model = RandomForestClassifier()
# model = MLPClassifier()
# fit model
model.fit(X_train, Y_train)

yhat_train = model.predict(X_train)
print('train tsp good pred proportion', sum(yhat_train == 1)/len(yhat_train))

yhat_test = model.predict(X_test)
print('test tsp good pred proportion', sum(yhat_test == 1)/len(yhat_test))

print('train prediction acc', sum(yhat_train == Y_train)/len(Y_train))
print('test prediction acc', sum(yhat_test == Y_test)/len(Y_test))

test['tsp_good_pred'] = yhat_test


test_sample_score = score_reverse.merge(test[['route_id', 'tsp_good_pred']], on =['route_id'])




train['tsp_good_pred'] = yhat_train
train_sample_score = score_reverse.merge(train[['route_id', 'tsp_good_pred']], on =['route_id'])

train_sample_score['select_model_score'] = train_sample_score['score']*train_sample_score['tsp_good_pred'] + (1 - train_sample_score['tsp_good_pred']) * train_sample_score['score_rev']
print('****train select by model score', np.mean(train_sample_score['select_model_score']) )
print('****train all direct tsp score', np.mean(train_sample_score['score']) )

test_sample_score['select_model_score'] = test_sample_score['score']*test_sample_score['tsp_good_pred'] + (1 - test_sample_score['tsp_good_pred']) * test_sample_score['score_rev']
print('====test select by model score', np.mean(test_sample_score['select_model_score']) )
print('====test all direct tsp score', np.mean(test_sample_score['score']) )


a=1



# score_reverse['erp'].hist()
# plt.show()

# score_reverse['erp_rev'].hist()
# plt.show()
a=1