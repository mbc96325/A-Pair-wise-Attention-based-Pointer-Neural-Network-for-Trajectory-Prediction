n_neighbour = 5

feature_col = ['min_travel_time', 'mean_travel_time', 'num_stops_avail', 'degree', 'forward','side', 'backward',
               'BIG_zone_id_same','SMALL_zone_diff_eq_1', 'num_tra_sig_avail', 'before_7am_avail', 'after_10am_avail', 'weekends_avail',
               'total_num_zones','n_pkg','n_pkg_avail',
               'current_zone_percentage'] #    'tsp_next_mean', 'tsp_next_min',

feature_col_test = ['min_travel_time', 'mean_travel_time', 'num_stops_avail','lat_mean','lng_mean','lat_avail','lng_avail','BIG_zone_id_same','SMALL_zone_diff_eq_1', 'num_tra_sig_avail', 'before_7am_avail', 'after_10am_avail', 'weekends_avail',
               'total_num_zones','n_pkg','n_pkg_avail',
               'current_zone_percentage'] #   'tsp_next_mean', 'tsp_next_min',

used_feature = ['mean_travel_time','BIG_zone_id_same','SMALL_zone_diff_eq_1',
                'total_num_zones','n_pkg','n_pkg_avail']

logistic_features = ['min_travel_time', 'mean_travel_time', 'num_stops_avail','n_pkg_avail',
                     'BIG_zone_id_same','num_tra_sig_avail']

logistic_features_first = ['min_travel_time', 'mean_travel_time', 'num_stops_avail',
                     'num_tra_sig_avail','n_pkg_avail']

x_attr = ['min_travel_time', 'mean_travel_time', 'num_stops_avail', 'degree', 'tsp_next_mean',  'tsp_next_min', 'num_tra_sig_avail','BIG_zone_id_same','SMALL_zone_diff_eq_1','n_pkg']
z_attr = ['before_7am', 'after_10am','weekends','total_num_zones','current_zone_percentage']

#
# used_feature = ['min_travel_time', 'mean_travel_time',
#                 'tsp_next_mean','tsp_next_min']


used_feature_no_tsp = ['min_travel_time', 'mean_travel_time', 'num_stops_avail', 'degree', 'forward', 'side', 'backward',
                'num_tra_sig_avail', 'before_7am_avail', 'after_10am_avail', 'weekends_avail',
                'total_num_zones','SMALL_zone_diff_eq_1','BIG_zone_id_same',
                'current_zone_percentage']

inner_norm_feature = ['min_travel_time', 'mean_travel_time', 'num_stops_avail', 'degree',
                      'num_tra_sig_avail']

ffill_col = ['before_7am_avail', 'after_10am_avail', 'weekends_avail', 'total_num_zones', 'current_zone_percentage']
ffill_col = [key for key in ffill_col if key in used_feature]
zero_col = ['num_stops_avail', 'degree', 'forward', 'side', 'tsp_next_mean', 'tsp_next_min', 'num_tra_sig_avail']
zero_col_no_tsp = ['num_stops_avail', 'degree', 'forward', 'side', 'num_tra_sig_avail']
zero_col_no_tsp = [key for key in zero_col_no_tsp if key in used_feature]
one_col = ['backward']
one_col = [key for key in one_col if key in used_feature]