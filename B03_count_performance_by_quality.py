import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def score_table(file_name, save_tail):
    score_all = pd.read_csv(file_name + '.csv')
    data_attributes = pd.read_csv('../data/zone_data.csv')
    data_attributes['total_n_pkg'] = data_attributes.groupby(['route_id'])['n_pkg'].transform('sum')
    data_attributes['total_planned_service_time'] = data_attributes.groupby(['route_id'])['planned_service_time_sum'].transform('sum')
    data_attributes['min_time_window_diff'] = data_attributes.groupby(['route_id'])['time_window_end_from_departure_sec_min'].transform('min')
    data_attributes['avg_num_tra_sig'] = data_attributes.groupby(['route_id'])['num_tra_sig'].transform('mean')
    data_attributes['total_num_stops'] = data_attributes.groupby(['route_id'])['total_num_stops_per_zone'].transform('sum')

    used_col = ['total_n_pkg','total_planned_service_time','min_time_window_diff','avg_num_tra_sig',
                'total_num_stops', 'station_code','exe_cap_cm3','route_score','departure_date_time_local','day_of_week','hour']
    route_att = data_attributes[['route_id'] + used_col].drop_duplicates()
    route_att = route_att.merge(score_all, on = ['route_id'])


    assert len(route_att) == len(score_all)
    score_all_high = route_att.loc[route_att['route_score'] == 'High'].copy()
    score_all_nonhigh = route_att.loc[route_att['route_score'] != 'High'].copy()

    mean_score_high = np.mean(score_all_high['score'])
    median_score_high = np.median(score_all_high['score'])

    mean_score_nonhigh = np.mean(score_all_nonhigh['score'])
    median_score_nonhigh = np.median(score_all_nonhigh['score'])

    acc_high = get_pred_acc(score_all_high)
    acc_nonhigh = get_pred_acc(score_all_nonhigh)

    acc_high['mean_score'] = mean_score_high
    acc_high['median_score'] = median_score_high
    acc_high['route_set'] = 'high-quality'

    acc_nonhigh['mean_score'] = mean_score_nonhigh
    acc_nonhigh['median_score'] = median_score_nonhigh
    acc_nonhigh['route_set'] = 'non-high-quality'

    res_final = pd.concat([pd.DataFrame(acc_high), pd.DataFrame(acc_nonhigh)])
    res_final.to_csv("result/score_high_nonhigh_compare_" + save_tail + '.csv',index=False)
    return (mean_score_nonhigh - mean_score_high) /  mean_score_nonhigh * 100

def get_pred_acc(data):
    first_zone_acc = 0
    second_zone_acc = 0
    third_zone_acc = 0
    fourth_zone_acc = 0
    for pred_seq, actual_zone_seq in zip(data['pred_zone_seq'], data['actual_seq']):
        pred_seq = eval(pred_seq)
        actual_zone_seq = eval(actual_zone_seq)
        if pred_seq[1] == actual_zone_seq[1]:
            first_zone_acc += 1
        if pred_seq[2] == actual_zone_seq[2]:
            second_zone_acc += 1
        if pred_seq[3] == actual_zone_seq[3]:
            third_zone_acc += 1
        if pred_seq[4] == actual_zone_seq[4]:
            fourth_zone_acc += 1
    acc_1 = first_zone_acc / len(data)
    acc_2 = second_zone_acc / len(data)
    acc_3 = third_zone_acc / len(data)
    acc_4 = fourth_zone_acc / len(data)
    res = {'first_zone_acc':[acc_1], 'second_zone_acc':[acc_2],
           'third_zone_acc':[acc_3], 'fourth_zone_acc':[acc_4],
           'num_routes':[len(data)]}
    return res

if __name__ == '__main__':
    file_name_list = {'final_selected_route':'ours',
                      'final_selected_route_lstm_en_de':'lstm_en_de',
                      'final_selected_route_pt_net_original':'pnt_net'}
    save_fig = 0
    pct_decrease_list = []
    for file_name, save_tail in file_name_list.items():
        pct_decrease = score_table(file_name, save_tail)
        pct_decrease_list.append(pct_decrease)
    print('avg decrease:',np.mean(pct_decrease_list),'%')