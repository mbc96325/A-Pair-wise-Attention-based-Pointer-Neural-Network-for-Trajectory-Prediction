import pandas as pd
import numpy as np
import pickle as pkl
from cython_score_evaluate import evaluate_simple

def process_eval_route(model_name, greedy):
    # generated_zone_seq = pd.read_csv('generated_seq_diff_first_zone_PATTERN_FORCE.csv')
    generated_zone_seq = pd.read_csv('generated_seq_diff_first_zone_' + model_name + '.csv')
    generated_zone_seq = generated_zone_seq.sort_values(['route_id','start_zone'])

    lst_col = 'pred_zone_seq'
    generated_zone_seq['pred_zone_seq'] = generated_zone_seq[lst_col].apply(lambda x:eval(x))

    ##############
    if greedy:
        first_zone_top_N_prob = 1 # > 47 means no first N
    else:
        first_zone_top_N_prob = 50
    generated_zone_seq = generated_zone_seq.sort_values(['route_id','start_zone_prob'],ascending=False)
    generated_zone_seq_top_N = generated_zone_seq.groupby(['route_id']).head(first_zone_top_N_prob)
    #################
    # generated_zone_seq['median'] = generated_zone_seq.groupby(['route_id'])['start_zone_prob'].transform('median')
    # generated_zone_seq_top_N = generated_zone_seq.loc[generated_zone_seq['start_zone_prob'] > generated_zone_seq['median']]


    new_form_zone_seq = pd.DataFrame({col:np.repeat(generated_zone_seq_top_N[col].values, generated_zone_seq_top_N[lst_col].str.len()) for col in generated_zone_seq_top_N.columns.difference([lst_col])}
                            ).assign(**{lst_col:np.concatenate(generated_zone_seq_top_N[lst_col].values)})[generated_zone_seq_top_N.columns.tolist()]

    # new_form_zone_seq = pd.DataFrame({col:np.repeat(generated_zone_seq[col].values, generated_zone_seq[lst_col].str.len()) for col in generated_zone_seq.columns.difference([lst_col])}
    #                         ).assign(**{lst_col:np.concatenate(generated_zone_seq[lst_col].values)})[generated_zone_seq.columns.tolist()]

    new_form_zone_seq['pred_zone_id'] = new_form_zone_seq.groupby(['route_id','start_zone']).cumcount() + 1

    ##################



    # calculate customized cost
    new_form_zone_seq['zone_next'] = new_form_zone_seq['pred_zone_seq'].shift(-1)
    new_form_zone_seq_no_ini = new_form_zone_seq.loc[new_form_zone_seq['pred_zone_seq']!='INIT'].copy()

    ini_rows = new_form_zone_seq.loc[new_form_zone_seq['pred_zone_seq']=='INIT'].copy()

    temp = new_form_zone_seq_no_ini['pred_zone_seq'].str.split('-', expand=True)
    temp2 = temp.iloc[:, 1].str.split('.', expand=True)
    temp2['2'] = temp2.iloc[:, 1].apply(lambda x: x[:-1])
    temp2['3'] = temp2.iloc[:, 1].apply(lambda x: x[-1])

    new_form_zone_seq_no_ini['zone_id_1'] = temp.iloc[:, 0].apply(lambda x: ord(x)) - 65
    new_form_zone_seq_no_ini['zone_id_2'] = np.int32(temp2.iloc[:, 0])
    new_form_zone_seq_no_ini['zone_id_3'] = np.int32(temp2['2'])
    new_form_zone_seq_no_ini['zone_id_4'] = temp2['3'].apply(lambda x: ord(x)) - 65

    new_form_zone_seq_no_ini['zone_id_1_next'] = new_form_zone_seq_no_ini['zone_id_1'].shift(-1)
    new_form_zone_seq_no_ini['zone_id_2_next'] = new_form_zone_seq_no_ini['zone_id_2'].shift(-1)
    new_form_zone_seq_no_ini['zone_id_3_next'] = new_form_zone_seq_no_ini['zone_id_3'].shift(-1)
    new_form_zone_seq_no_ini['zone_id_4_next'] = new_form_zone_seq_no_ini['zone_id_4'].shift(-1)

    new_form_zone_seq_no_ini['route_next'] = new_form_zone_seq_no_ini['route_id'].shift(-1)
    new_form_zone_seq_no_ini['start_zone_next'] = new_form_zone_seq_no_ini['start_zone'].shift(-1)
    new_form_zone_seq_no_ini = new_form_zone_seq_no_ini.loc[new_form_zone_seq_no_ini['route_next'] == new_form_zone_seq_no_ini['route_id']]
    new_form_zone_seq_no_ini = new_form_zone_seq_no_ini.loc[new_form_zone_seq_no_ini['start_zone_next'] == new_form_zone_seq_no_ini['start_zone']]
    penalty = 0
    #### no mask prob
    # 100:
    # 1500/5000: 0.03622
    # 500:
    # 0: 0.03766
    # 1000:
    # 300: 0.03598
    # Epoch [0] Score: 0.04172
    ########### mask prob
    # penalty 0: 0.03481
    # penalty 500: 0.03481
    # penalty 5000: 0.03481
    ###########

    new_form_zone_seq_no_ini['same_BIG_zone'] = 0
    new_form_zone_seq_no_ini.loc[(new_form_zone_seq_no_ini['zone_id_1'] == new_form_zone_seq_no_ini['zone_id_1_next']) &
                                 (new_form_zone_seq_no_ini['zone_id_2'] == new_form_zone_seq_no_ini['zone_id_2_next']), 'same_BIG_zone'] = 1

    new_form_zone_seq_no_ini['diff_BIG_zone_cost'] = 0
    new_form_zone_seq_no_ini.loc[new_form_zone_seq_no_ini['same_BIG_zone'] == 0,'diff_BIG_zone_cost'] = penalty * 3

    new_form_zone_seq_no_ini['BIG_zone2_pm1_cost'] = 0
    new_form_zone_seq_no_ini.loc[(new_form_zone_seq_no_ini['zone_id_1'] == new_form_zone_seq_no_ini['zone_id_1_next']) &
                                 (new_form_zone_seq_no_ini['zone_id_2'] != new_form_zone_seq_no_ini['zone_id_2_next']) &
                                 (((new_form_zone_seq_no_ini['zone_id_2'] != new_form_zone_seq_no_ini[
                                     'zone_id_2_next']) + 1) &
                                  ((new_form_zone_seq_no_ini['zone_id_2'] != new_form_zone_seq_no_ini[
                                      'zone_id_2_next']) - 1)), 'BIG_zone2_pm1_cost'] = penalty * 2

    new_form_zone_seq_no_ini['diff_SMALL_zone2_cost'] = 0
    new_form_zone_seq_no_ini.loc[(new_form_zone_seq_no_ini['same_BIG_zone'] == 1) & (new_form_zone_seq_no_ini['zone_id_4'] != new_form_zone_seq_no_ini['zone_id_4_next']),'diff_SMALL_zone2_cost'] = penalty * 2


    new_form_zone_seq_no_ini['SMALL_zone1_pm1_cost'] = 0
    new_form_zone_seq_no_ini.loc[(new_form_zone_seq_no_ini['same_BIG_zone'] == 1) & (new_form_zone_seq_no_ini['zone_id_4'] == new_form_zone_seq_no_ini['zone_id_4_next']) &
                                 (((new_form_zone_seq_no_ini['zone_id_3'] != new_form_zone_seq_no_ini['zone_id_3_next']) + 1) &
                                 ((new_form_zone_seq_no_ini['zone_id_3'] != new_form_zone_seq_no_ini['zone_id_3_next']) - 1)),'SMALL_zone1_pm1_cost'] = penalty

    #
    new_form_zone_seq_no_ini['SMALL_zone2_pm1_cost'] = 0
    new_form_zone_seq_no_ini.loc[(new_form_zone_seq_no_ini['same_BIG_zone'] == 1) & (new_form_zone_seq_no_ini['zone_id_3'] == new_form_zone_seq_no_ini['zone_id_3_next']) &
                                 (((new_form_zone_seq_no_ini['zone_id_4'] != new_form_zone_seq_no_ini['zone_id_4_next']) + 1) &
                                 ((new_form_zone_seq_no_ini['zone_id_4'] != new_form_zone_seq_no_ini['zone_id_4_next']) - 1)),'SMALL_zone2_pm1_cost'] = penalty

    #


    zone_mean_travel_times = pd.read_csv('../data/zone_mean_travel_times.csv')
    # new_form_zone_seq_no_ini['travel_time'] =

    final_data = pd.concat([ini_rows, new_form_zone_seq_no_ini])
    final_data = final_data.sort_values(['route_id','start_zone','pred_zone_id'])

    fill_na_0_col = ['SMALL_zone2_pm1_cost','diff_BIG_zone_cost','BIG_zone2_pm1_cost','diff_SMALL_zone2_cost','SMALL_zone1_pm1_cost']
    for key in fill_na_0_col:
        final_data[key] = final_data[key].fillna(0)


    final_data = final_data.merge(zone_mean_travel_times, left_on = ['route_id','pred_zone_seq','zone_next'],right_on=['route_id','From_zone','To_zone'])
    all_cost_col = fill_na_0_col + ['travel_time']
    final_data['all_cost'] = 0
    for key in all_cost_col:
        final_data['all_cost'] += final_data[key]

    final_data = final_data.sort_values(['route_id','start_zone','pred_zone_id'])
    route_eval = final_data.groupby(['route_id','start_zone'])['all_cost'].sum().reset_index()
    route_eval = route_eval.sort_values(['route_id','all_cost'])
    lowest_cost_route = route_eval.drop_duplicates(['route_id'],keep = 'first')

    to_eval_route = generated_zone_seq.merge(lowest_cost_route[['route_id','start_zone']], on = ['route_id','start_zone'])
    to_eval_route = to_eval_route.reset_index(drop=True)
    # to_eval_route.to_csv('final_selected_route.csv',index=False)
    return to_eval_route

def output_score(to_eval_route, model_name):
    with open('../data/cost_mtx_array.pkl', 'rb') as f:
        cost_mat_array = pkl.load(f)
    with open('../data/stop_idx_map.pkl', 'rb') as f:
        name_map_all = pkl.load(f)

    data_stops = pd.read_csv('../data/build_route_with_seq.csv')
    data_stops['stops'] = data_stops['stops'].fillna('NA')

    zone_data = pd.read_csv('../data/zone_data.csv')

    route_score = []
    to_eval_route['actual_seq'] = 0
    to_eval_route['score'] = 0

    first_zone_acc = 0
    second_zone_acc = 0
    third_zone_acc = 0
    fourth_zone_acc = 0
    for idx_, route_id, pred_seq, zone_len in zip(list(to_eval_route.index), to_eval_route['route_id'],to_eval_route['pred_zone_seq'],to_eval_route['zone_len']):

        cost_mat_route = cost_mat_array[route_id]
        name_map = name_map_all[route_id]
        data_route = data_stops.loc[data_stops['route_id'] == route_id]
        pred_zone_df = pd.DataFrame(
            {'zone_id': pred_seq, 'pred_seq_id': np.arange(zone_len + 1) + 1})
        pred_zone_df = pred_zone_df.merge(data_route[['stops', 'zone_id', 'seq_ID']], on=['zone_id'])
        pred_zone_df = pred_zone_df.sort_values(['pred_seq_id', 'seq_ID'])

        actual_zone_seq = list(zone_data.loc[zone_data['route_id'] == route_id, 'zone_id'])
        if pred_seq[1] == actual_zone_seq[1]:
            first_zone_acc += 1
        if pred_seq[2] == actual_zone_seq[2]:
            second_zone_acc += 1
        if pred_seq[3] == actual_zone_seq[3]:
            third_zone_acc += 1
        if pred_seq[4] == actual_zone_seq[4]:
            fourth_zone_acc += 1

        act_zone_df = pd.DataFrame(
            {'zone_id': actual_zone_seq, 'pred_seq_id': np.arange(zone_len + 1) + 1})
        act_zone_df = act_zone_df.merge(data_route[['stops', 'zone_id', 'seq_ID']], on=['zone_id'])
        act_zone_df = act_zone_df.sort_values(['pred_seq_id', 'seq_ID'])

        to_eval_route.loc[idx_, 'actual_seq'] = str(actual_zone_seq)



        actual_seq = np.array([name_map[s] for s in act_zone_df['stops']])
        est_seq = np.array([name_map[s] for s in pred_zone_df['stops']])

        score, seq_dev, erp_per_edit, total_dist, total_edit_count = evaluate_simple(actual_seq,
                                                                                     est_seq,
                                                                                     cost_mat_route)
        route_score.append(score)
        to_eval_route.loc[idx_, 'score'] = score



    print('First zone accuracy {:.4}'.format(first_zone_acc / len(to_eval_route)))
    print('Second zone accuracy {:.4}'.format(second_zone_acc / len(to_eval_route)))
    print('Third zone accuracy {:.4}'.format(third_zone_acc / len(to_eval_route)))
    print('Forth zone accuracy {:.4}'.format(fourth_zone_acc / len(to_eval_route)))

    s_ptr = np.mean(route_score)
    print('num_route',len(route_score))
    print('Score: {:.10}'.format(s_ptr))
    print('Score std',np.std(route_score))
    to_eval_route.to_csv('final_selected_route_' + model_name + '.csv', index=False)


if __name__  == '__main__':
    model_name = 'pnt_net_with_asnn_att_rand_seq' #  lstm_en_de    pt_net_original    pnt_net_with_asnn_att     pt_net_original_rand_seq      lstm_en_de_rand_seq   pnt_net_with_asnn_att
    greedy = False
    seq_g_name = 'Greedy' if greedy else 'Alg 1'
    print('current model', model_name, seq_g_name)
    to_eval_route = process_eval_route(model_name, greedy)
    output_score(to_eval_route, model_name)
