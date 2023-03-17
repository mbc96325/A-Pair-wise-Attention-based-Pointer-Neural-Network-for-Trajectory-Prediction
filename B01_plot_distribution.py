import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


save_fig = 1

route_seq = pd.read_csv('../data/build_route_with_seq.csv')
route_seq['stops'] = route_seq['stops'].fillna('NA')
number_stop = route_seq.groupby(['route_id'])['stops'].count().reset_index().rename(columns = {'stops':'num_stops'})



fig, ax = plt.subplots(figsize=(8, 6))


plt.hist(number_stop['num_stops'], density=False, bins=50, facecolor = 'gray', edgecolor='white')  # density=False would make counts
# plt.ylabel('Probability')
# plt.xlabel('Data')

plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# plt.ylim(0.35,0.46)
# # ax.set_yticklabels([str(i) + '%' for i in range(40, 61, 10)])


y_label = 'Counts'
ax.set_ylabel(y_label, fontsize=16)
# # ax.set_xticks(ind + (len(path_scenario) - 1) / 2 * width)
# ax.set_xticklabels(labels_list, fontsize=16)
ax.set_xlabel('Number of stops per route', fontsize=16)
#
# # ax.legend(legends_handle, legend_labels, fontsize=16, loc='upper right')
plt.tight_layout()
if save_fig == 0:
    plt.show()
else:
    plt.savefig('img/num_stops_distribution.jpg', dpi=200)

a=1