import pandas as pd
import numpy as np

test_list = ['ASU_DNN_with_tsp']#['DQN','ASU_DNN_with_tsp','ASU_DNN_no_tsp','PG']  # ASU_DNN_no_tsp # DQN
for tail_name in test_list:

    print('==========current', tail_name, '==========')
    ASU_DNN_score_results = pd.read_csv('score_results_' + tail_name + '_NEW.csv')
    ASU_DNN_score_results['scenario'] = tail_name + ' inner opt'

    TSP_score_results = pd.read_csv('TSP_score_results.csv')

    print(tail_name + ' score', np.mean(ASU_DNN_score_results['score']))
    print(tail_name + ' sd', np.mean(ASU_DNN_score_results['seq_dev']))
    print(tail_name + ' erp_edit_count', np.mean(ASU_DNN_score_results['erp_edit_count']))
    print(tail_name + ' erp_edit_distance', np.mean(ASU_DNN_score_results['erp_edit_distance']))

    print('TSP score', np.mean(TSP_score_results['score']))
    print('TSP sd', np.mean(TSP_score_results['seq_dev']))
    print('TSP erp_edit_count', np.mean(TSP_score_results['erp_edit_count']))
    print('TSP erp_edit_distance', np.mean(TSP_score_results['erp_edit_distance']))

    compare_ = TSP_score_results.merge(ASU_DNN_score_results, on =['route_id'])
    compare_['best'] = np.minimum(compare_['score_y'], compare_['score_x'])
    print('avg best', np.mean(compare_['best'] ))
    compare_['score_diff'] = compare_['score_y'] - compare_['score_x']

    #print()

    compare_ = compare_.sort_values(['score_diff'], ascending=False)

    a=1