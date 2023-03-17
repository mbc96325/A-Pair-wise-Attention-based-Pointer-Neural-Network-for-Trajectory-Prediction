import numpy as np
import pandas as pd
import pickle
import json
import sys
import time

#import cython_score_evaluate as cse

def get_tsp_result():

    with open('../../../data_fake/model_apply_outputs/model_apply_output/mean_dist/opt_zone_seq_tour.json', 'rb') as f:
        tsp_zone_seq_mean = json.load(f)

    # with open('../../tsp_xiaotong/mean_dist/opt_zone_seq.p', 'rb') as f:
    #     tsp_zone_seq_mean = pickle.load(f)
    # with open('../../tsp_xiaotong/min_dist/opt_zone_seq.p', 'rb') as f:
    #     tsp_zone_seq_min = pickle.load(f)
    a=1
    df_dict = {'route_id':[],'zone_id':[],'pred_seq_id':[]}
    for key in tsp_zone_seq_mean:
        pred_seq_id = 1
        for zone_id in tsp_zone_seq_mean[key]:
            df_dict['route_id'].append(key)
            df_dict['zone_id'].append(zone_id)
            df_dict['pred_seq_id'].append(pred_seq_id)
            pred_seq_id += 1
    tsp_df = pd.DataFrame(df_dict)
    tsp_df.to_csv("pred_zone_seq_TSP.csv")
    return tsp_df


def main(data_test, cost_mat_build, result, tail_name, cost_mat_array, name_map_all):

    build_route_seq = data_test.copy()
    build_route_seq = build_route_seq.sort_values(['route_id','seq_ID'])
    result_stop_opt = result.merge(data_test[['route_id','stops','zone_id','seq_ID']], on = ['route_id','zone_id'], how = 'right')
    assert len(result_stop_opt) == len(result_stop_opt.dropna())
    result_stop_opt = result_stop_opt.sort_values(['route_id','pred_seq_id','seq_ID'])


    count = 0
    route_id_list = test_routes

    score_results = {'route_id':[],'route_length':[],'scenario':[],'score':[],'seq_dev':[],'erp_per_edit':[],'erp_edit_distance':[],'erp_edit_count':[]}



    # first_zone = 0
    # second_zone = 0
    # third_zone = 0
    # forth_zone = 0

    for route_id in route_id_list:


        # if tsp_zone_seq[1] == actual_zone_seq[1]:
        #     first_zone += 1
        # if tsp_zone_seq[2] == actual_zone_seq[2]:
        #     second_zone += 1
        # if tsp_zone_seq[3] == actual_zone_seq[3]:
        #     third_zone += 1
        # if tsp_zone_seq[4] == actual_zone_seq[4]:
        #     forth_zone += 1
        #

        count += 1
        if count%50 == 0:
            score_results_df = pd.DataFrame(score_results)
            if NEW_SCORE:
                score_results_df.to_csv('score_results_' + tail_name + '_NEW.csv', index=False)
            else:
                score_results_df.to_csv('score_results_' + tail_name + '.csv', index=False)
            # score_results_df.to_csv('TSP_score_results.csv', index=False)
            print('Current route num', count, 'total', len(route_id_list))
            print('score avg',np.mean(score_results_df['score']))
            #print('Route length', len(actual_seq_))


        route_data = build_route_seq.loc[build_route_seq['route_id'] == route_id]
        actual_seq_ = list(route_data['stops'])


        est_seq_1 = list(result_stop_opt.loc[result_stop_opt['route_id'] == route_id, 'stops'])


        # cost_mat = generate_rand_cost(actual_seq_)


        score_results['route_id'].append(route_id)
        score_results['route_length'].append(len(actual_seq_))
        score_results['scenario'].append('ASU DNN inner opt')
        # score_results['scenario'].append('TSP inner opt')

        if NEW_SCORE:
            name_map = name_map_all[route_id]
            cost_mat_array_route = cost_mat_array[route_id]

            actual_seq = np.array([name_map[s] for s in actual_seq_])
            est_seq = np.array([name_map[s] for s in est_seq_1])
            score, seq_dev, erp_per_edit, total_dist, total_edit_count = evaluate_simple(actual_seq, est_seq,
                                                                                             cost_mat_array_route)
        else:
            cost_mat = cost_mat_build[route_id]  #
            score, seq_dev, erp_per_edit, total_dist, total_edit_count = evaluate_simple(actual_seq_, est_seq_1, cost_mat)


        score_results['score'].append(score)
        score_results['seq_dev'].append(seq_dev)
        score_results['erp_per_edit'].append(erp_per_edit)
        score_results['erp_edit_distance'].append(total_dist)
        score_results['erp_edit_count'].append(total_edit_count)



    score_results_df = pd.DataFrame(score_results)

    if NEW_SCORE:
        score_results_df.to_csv('score_results_' + tail_name + '_NEW.csv',index=False)
    else:
        score_results_df.to_csv('score_results_' + tail_name + '.csv', index=False)
    # score_results_df.to_csv('TSP_score_results.csv', index=False)
    return score_results_df



def select_min_tt_seq(result):

    new_form_zone_seq = result.copy()
    new_form_zone_seq = new_form_zone_seq.rename(columns = {'ini_zone_idx':'start_zone'})
    new_form_zone_seq = new_form_zone_seq.sort_values(['route_id','start_zone','pred_seq_id'])

    new_form_zone_seq['zone_next'] = new_form_zone_seq['zone_id'].shift(-1)
    new_form_zone_seq['route_next'] = new_form_zone_seq['route_id'].shift(-1)
    new_form_zone_seq['start_zone_next'] = new_form_zone_seq['start_zone'].shift(-1)
    new_form_zone_seq = new_form_zone_seq.loc[new_form_zone_seq['route_next'] == new_form_zone_seq['route_id']]
    new_form_zone_seq = new_form_zone_seq.loc[new_form_zone_seq['start_zone_next'] == new_form_zone_seq['start_zone']]


    zone_mean_travel_times = pd.read_csv('../data/zone_mean_travel_times.csv')
    # new_form_zone_seq_no_ini['travel_time'] =
    final_data = new_form_zone_seq.copy()

    final_data = final_data.merge(zone_mean_travel_times, left_on = ['route_id','zone_id','zone_next'],right_on=['route_id','From_zone','To_zone'])
    all_cost_col = ['travel_time']
    final_data['all_cost'] = 0
    for key in all_cost_col:
        final_data['all_cost'] += final_data[key]

    final_data = final_data.sort_values(['route_id','start_zone','pred_seq_id'])
    route_eval = final_data.groupby(['route_id','start_zone'])['all_cost'].sum().reset_index()
    route_eval = route_eval.sort_values(['route_id','all_cost'])
    lowest_cost_route = route_eval.drop_duplicates(['route_id'],keep = 'first')

    result = result.rename(columns = {'ini_zone_idx':'start_zone'})
    to_eval_route = result.merge(lowest_cost_route[['route_id','start_zone']], on = ['route_id','start_zone'])
    to_eval_route = to_eval_route.reset_index(drop=True)

    to_eval_route = to_eval_route.sort_values(['route_id','start_zone','pred_seq_id'])
    return to_eval_route[['route_id','zone_id','pred_seq_id']].copy()


if __name__ == '__main__':
    with open('testing_routes.pkl', 'rb') as f:
        test_routes = pickle.load(f)
    data = pd.read_csv('../data/build_route_with_seq.csv')
    data_test = data.loc[data['route_id'].isin(test_routes)].copy()
    data_test['stops'] = data_test['stops'].fillna('NA')

    zone_data = pd.read_csv("../data/zone_data.csv")
    with open('testing_routes.pkl', 'rb') as f:
        test_routes = pickle.load(f)

    zone_data = zone_data.loc[zone_data['route_id'].isin(test_routes)]

    NEW_SCORE = True
    if not NEW_SCORE:
        from score_testing_func import evaluate_simple
    else:
        from cython_score_evaluate import evaluate_simple


    LOAD = True
    if LOAD and not NEW_SCORE:
        filepath = 'D:/Dropbox (MIT)/00_Research/17_Amazon_LM_competition/data/model_build_inputs/travel_times.json'
        with open(filepath, newline='') as in_file:
            cost_mat_build = json.load(in_file)
    else:
        cost_mat_build = None

    test_list = ['ASU_DNN_no_tsp']  #['DQN', 'ASU_DNN_no_tsp','ASU_DNN_with_tsp','TSP]

    if NEW_SCORE:
        with open('../data/cost_mtx_array.pkl', 'rb') as f:
            cost_mat_array = pickle.load(f)
    
        with open('../data/stop_idx_map.pkl', 'rb') as f:
            name_map_all = pickle.load(f)
    else:
        cost_mat_array = None
        name_map_all = None


    for tail_name in test_list:
        tic = time.time()
        print('=======current ', tail_name, '==========')
        if tail_name == 'TSP':
            result = get_tsp_result()
        else:
            result = pd.read_csv('pred_zone_seq_' + tail_name + '.csv')



        print('num routes', len(pd.unique(result['route_id'])))

        acc = zone_data[['route_id','zone_id','zone_seq']].merge(result, on = ['route_id','zone_id'], how = 'left')


        check_na = acc[acc['pred_seq_id'].isna()]
        if len(check_na) != 0:
            print('check problem')
            test = result.loc[result['route_id'] == check_na['route_id'].iloc[0]]
            test2 = zone_data.loc[zone_data['route_id'] == check_na['route_id'].iloc[0]]
            exit()


        acc = acc.sort_values(['route_id','zone_seq'])
        print('overall acc',len(acc.loc[acc['zone_seq']==acc['pred_seq_id']])/len(acc))
        acc_last =  acc.drop_duplicates(['route_id'], keep = 'last')
        for k in range(1,5):
            acc_k = acc.loc[acc['zone_seq'] == k+1]
            print("acc",k, len(acc_k.loc[acc_k['zone_seq']==acc_k['pred_seq_id']])/len(test_routes))

        print('acc last', len(acc_last.loc[acc_last['zone_seq']==acc_last['pred_seq_id']])/len(test_routes))


        score_results_df = main(data_test, cost_mat_build, result, tail_name, cost_mat_array, name_map_all)
        print('====final score',np.mean(score_results_df['score']))
        print('score std',np.std(score_results_df['score']))
        #print("====evaluate score time", time.time() - tic,'=====')