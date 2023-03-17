import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

save_fig = 1
#model_name = '_' + 'pnt_net_with_asnn_att'
model_name = ''
score_all = pd.read_csv('final_selected_route' + model_name +'.csv')

fig, ax = plt.subplots(figsize=(8, 6))


plt.hist(score_all['score'], density=False, bins=30, facecolor = 'green', edgecolor='white')  # density=False would make counts

mean_ = np.mean(score_all['score'])
median_ = np.median(score_all['score'])

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
ax.set_xlabel('Disparity scores', fontsize=16)

# # ax.legend(legends_handle, legend_labels, fontsize=16, loc='upper right')
#plt.plot([mean_,mean_],[0,500], 'k--',alpha = 1)
#plt.plot([median_,median_],[0,500], '--',alpha = 1)

plt.text(0.12, 230, 'Mean = ' + str(round(mean_, 4)), fontsize = 16)
plt.text(0.12, 210, 'Median = ' + str(round(median_, 4)), fontsize = 16)
plt.ylim([0,250])
plt.tight_layout()
if save_fig == 0:
    plt.show()
else:
    plt.savefig('img/score_distribution.jpg', dpi=200)

#a=1