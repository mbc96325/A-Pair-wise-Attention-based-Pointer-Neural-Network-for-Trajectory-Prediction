import json
import pandas as pd
import pickle


def process_travel_time():
    with open('../../data/zone_mean_travel_times.json') as f:
        zone_mean_travel_times = json.load(f)

    zone_mean_tt_df = {'route_id':[],'from_zone':[], 'to_zone':[],'mean_travel_time':[]}
    for key in zone_mean_travel_times:
        for from_zone in zone_mean_travel_times[key]:
            for to_zone in zone_mean_travel_times[key][from_zone]:
                if (from_zone != to_zone): #and (from_zone == 'INIT'):
                    zone_mean_tt_df['route_id'].append(key)
                    zone_mean_tt_df['from_zone'].append(from_zone)
                    zone_mean_tt_df['to_zone'].append(to_zone)
                    zone_mean_tt_df['mean_travel_time'].append(zone_mean_travel_times[key][from_zone][to_zone])

    zone_mean_tt_df_out = pd.DataFrame(zone_mean_tt_df)
    #zone_mean_tt_df_out = zone_mean_tt_df_out.sort_values(['route_id','from_zone'])


    with open('../../data/zone_min_travel_times.json') as f:
        zone_mean_travel_times = json.load(f)

    zone_mean_tt_df = {'route_id':[],'from_zone':[], 'to_zone':[],'min_travel_time':[]}
    for key in zone_mean_travel_times:
        for from_zone in zone_mean_travel_times[key]:
            for to_zone in zone_mean_travel_times[key][from_zone]:
                if (from_zone != to_zone): #and (from_zone == 'INIT'):
                    zone_mean_tt_df['route_id'].append(key)
                    zone_mean_tt_df['from_zone'].append(from_zone)
                    zone_mean_tt_df['to_zone'].append(to_zone)
                    zone_mean_tt_df['min_travel_time'].append(zone_mean_travel_times[key][from_zone][to_zone])

    zone_min_tt_df_out = pd.DataFrame(zone_mean_tt_df)



    with open('../../data/zone_tt.pkl', 'wb') as f:
        pickle.dump(zone_min_tt_df_out,f)
        pickle.dump(zone_mean_tt_df_out,f)


    nearest = 8
    zone_mean_tt_df_out = zone_mean_tt_df_out.sort_values(['route_id','from_zone','mean_travel_time'])
    zone_mean_tt_df_out = zone_mean_tt_df_out.groupby(['route_id','from_zone']).head(nearest)
    zone_mean_tt_df_out.to_csv('../../data/zone_mean_tt_df_nearest_' + str(nearest) + '.csv',index=False)

    zone_min_tt_df_out = zone_min_tt_df_out.sort_values(['route_id','from_zone','min_travel_time'])
    zone_min_tt_df_out = zone_min_tt_df_out.groupby(['route_id','from_zone']).head(nearest)
    zone_min_tt_df_out.to_csv('../../data/zone_min_tt_df_nearest_'+ str(nearest) +'.csv',index=False)


if __name__ == '__main__':
    process_travel_time()
