import pickle
import json


with open('testing_routes.pkl', 'rb') as f:
    testing_routes = pickle.load(f)


data_path = '../../data/'

data_new_path = '../../data_fake/'

# file_path = data_path + 'model_apply_inputs/new_package_data.json'
# print('Reading Input Data for generate_build_route_seq_df')
# with open(file_path, newline='') as in_file:
#     new_package_data = json.load(in_file)

file_path = data_path + 'model_build_inputs/package_data.json'
print('Reading Input Data for generate_build_route_seq_df')
with open(file_path, newline='') as in_file:
    new_package_data = json.load(in_file)


file_path = data_path + 'model_build_inputs/route_data.json'
print('Reading Input Data for generate_build_route_seq_df')
with open(file_path, newline='') as in_file:
    new_route_data = json.load(in_file)


file_path = data_path + 'model_build_inputs/travel_times.json'
print('Reading Input Data for generate_build_route_seq_df')
with open(file_path, newline='') as in_file:
    new_travel_times = json.load(in_file)

new_package_data_new = {}
new_route_data_new = {}
new_travel_times_new = {}
for key in testing_routes:
    new_package_data_new[key] = new_package_data[key]
    new_route_data_new[key] = new_route_data[key]
    new_travel_times_new[key] = new_travel_times[key]




file_path = data_new_path + 'model_apply_inputs/new_route_data.json'
print('dump Input Data for generate_build_route_seq_df')
with open(file_path, 'w') as in_file:
    json.dump(new_route_data_new, in_file)



file_path = data_new_path + 'model_apply_inputs/new_travel_times.json'
print('dump Input Data for generate_build_route_seq_df')
with open(file_path, 'w') as in_file:
    json.dump(new_travel_times_new, in_file)




file_path = data_new_path + 'model_apply_inputs/new_package_data.json'
print('dump Input Data for generate_build_route_seq_df')
with open(file_path, 'w') as in_file:
    json.dump(new_package_data_new, in_file)
