import numpy as np
import pandas as pd
import pickle
import os
from math import ceil
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import copy
from scipy.stats import entropy
from matplotlib.ticker import FormatStrFormatter
import statsmodels.api as sm


from sklearn.metrics import r2_score




def generate_data_for_reg():
    data_attributes = pd.read_csv('../data/zone_data.csv')
    data_attributes['total_n_pkg'] = data_attributes.groupby(['route_id'])['n_pkg'].transform('sum')
    data_attributes['total_planned_service_time'] = data_attributes.groupby(['route_id'])['planned_service_time_sum'].transform('sum')
    data_attributes['min_time_window_diff'] = data_attributes.groupby(['route_id'])['time_window_end_from_departure_sec_min'].transform('min')
    data_attributes['avg_num_tra_sig'] = data_attributes.groupby(['route_id'])['num_tra_sig'].transform('mean')
    data_attributes['total_num_stops'] = data_attributes.groupby(['route_id'])['total_num_stops_per_zone'].transform('sum')

    used_col = ['total_n_pkg','total_planned_service_time','min_time_window_diff','avg_num_tra_sig','total_num_stops', 'station_code','exe_cap_cm3','route_score','departure_date_time_local','day_of_week','hour']
    route_att = data_attributes[['route_id'] + used_col].drop_duplicates()

    route_att['if_high_qua_route'] = 0
    route_att.loc[route_att['route_score'].isin(['High','high']),'if_high_qua_route'] = 1

    staion_code = list(pd.unique(route_att['station_code']))
    staion_code.sort()
    LA = ['DLA3','DLA4','DLA5','DLA7','DLA8','DLA9']
    AU = ['DAU1']
    BO = ['DBO1','DBO2','DBO3']
    CH = ['DCH1','DCH2','DCH3','DCH4']
    SE = ['DSE2','DSE4','DSE5']

    route_att['if_in_LA'] = 0
    route_att.loc[route_att['station_code'].isin(LA),'if_in_LA'] = 1

    route_att['if_in_CH'] = 0
    route_att.loc[route_att['station_code'].isin(CH),'if_in_CH'] = 1

    route_att['if_in_BO'] = 0
    route_att.loc[route_att['station_code'].isin(BO),'if_in_BO'] = 1

    route_att['if_in_SE'] = 0
    route_att.loc[route_att['station_code'].isin(SE),'if_in_SE'] = 1

    route_att['if_in_AU'] = 0
    route_att.loc[route_att['station_code'].isin(AU),'if_in_AU'] = 1

    route_att['exe_cap_m3'] = route_att['exe_cap_cm3'] / 1e6 # m3

    route_att['weekends'] = 0
    route_att.loc[route_att['day_of_week'] >= 5, 'weekends'] = 1

    route_att['before_7am'] = 0
    route_att.loc[route_att['hour'] <= 7, 'before_7am'] = 1

    route_att['after_10am'] = 0
    route_att.loc[route_att['hour'] >= 10, 'after_10am'] = 1

    route_att['total_planned_service_time'] /= 3600 # hour

    route_att['min_time_window_diff'] /= 3600 # hour

    col_X = ['total_n_pkg','total_planned_service_time','min_time_window_diff','avg_num_tra_sig','total_num_stops',
             'exe_cap_m3','if_in_LA','if_in_CH','if_in_BO','weekends','before_7am','after_10am']
    return route_att, col_X

def run_linear_reg(data, col_X):

    #############duration
    # col_X = ['num_trip_in_day_mean','num_trip_in_day_std','num_days_with_travel','first_departure_time_std',
    #          'entropy_act_dur','if_student', 'if_senior']
    col_Y = ['if_high_qua_route']

    print('num routes', len(data))
    X = data.loc[:,col_X]
    Y = data.loc[:,col_Y]
    X = sm.add_constant(X)

    print('num of high quality routes', np.sum(Y))

    est = sm.Logit(Y, X)
    est2 = est.fit()
    results_summary = est2.summary()
    print(results_summary)
    results_as_html = results_summary.tables[1].as_html()
    table = pd.read_html(results_as_html, header=0, index_col=0)[0]
    table['Variable'] = ['Intercept'] + col_X

    table.to_csv('table/estimate_para_on_high_qual_route.csv',index=False)

if __name__ == '__main__':

    # data_path = '../data/'

    route_att, col_X = generate_data_for_reg()
    run_linear_reg(route_att, col_X)