from os import path
import sys, json, time
import pandas as pd

tic = time.time()
route_data = pd.read_csv('../data/build_route_df.csv', keep_default_na=False)
seq_df = pd.read_csv('../data/build_route_seq_df.csv', keep_default_na=False)
package = pd.read_csv('../data/build_package_df.csv', keep_default_na=False)
print('load data time',time.time() - tic)

assert len(route_data.loc[route_data['stops'].isna()]) == 0
assert len(seq_df.loc[seq_df['stops'].isna()]) == 0
assert len(package.loc[package['stops'].isna()]) == 0

#%%
route_seq = pd.merge(route_data, seq_df,on = ['route_id','stops'], how = 'left')
route_seq = route_seq.sort_values(['route_id','seq_ID'])
assert len(route_seq.loc[route_seq['seq_ID'].isna()]) == 0

route_seq.to_csv('../data/build_route_with_seq.csv',index=False)

#%%
route_seq_package = pd.merge(route_seq, package, on = ['route_id','stops'],how = 'left')
route_seq_package = route_seq_package.sort_values(['route_id','seq_ID'])
route_seq_package.to_csv('../data/build_route_with_seq_and_package.csv',index=False)

