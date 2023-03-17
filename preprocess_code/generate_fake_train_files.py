import pickle
import json

with open('train_routes.pkl', 'rb') as f:
    train_routes = pickle.load(f)

with open('testing_routes.pkl', 'rb') as f:
    testing_routes = pickle.load(f)


data_path = '../../data/'

data_new_path = '../../data_fake/'

file_path = data_path + 'model_build_inputs/actual_sequences.json'
print('Reading Input Data for generate_build_route_seq_df')
with open(file_path, newline='') as in_file:
    actual_sequences = json.load(in_file)

file_path = data_path + 'model_build_inputs/travel_times.json'
print('Reading Input Data for generate_build_route_seq_df')
with open(file_path, newline='') as in_file:
    travel_times = json.load(in_file)




file_path = data_path + 'model_build_inputs/route_data.json'
print('Reading Input Data for generate_build_route_seq_df')
with open(file_path, newline='') as in_file:
    route_data = json.load(in_file)


file_path = data_path + 'model_build_inputs/invalid_sequence_scores.json'
print('Reading Input Data for generate_build_route_seq_df')
with open(file_path, newline='') as in_file:
    invalid_sequence_scores = json.load(in_file)

file_path = data_path + 'model_build_inputs/package_data.json'
print('Reading Input Data for generate_build_route_seq_df')
with open(file_path, newline='') as in_file:
    package_data = json.load(in_file)



actual_sequences_new = {}
route_data_new = {}
invalid_sequence_scores_new = {}
package_data_new = {}
travel_times_new = {}

for key in train_routes:
    actual_sequences_new[key] = actual_sequences[key]
    route_data_new[key] = route_data[key]
    invalid_sequence_scores_new[key] = invalid_sequence_scores[key]
    package_data_new[key] = package_data[key]
    travel_times_new[key] = travel_times[key]






file_path = data_new_path + 'model_build_inputs/actual_sequences.json'
print('put Input Data for generate_build_route_seq_df')
with open(file_path, "w") as in_file:
    json.dump(actual_sequences_new, in_file)

file_path = data_new_path + 'model_build_inputs/travel_times.json'
print('put Input Data for generate_build_route_seq_df')
with open(file_path,  "w") as in_file:
    json.dump(travel_times_new, in_file)


file_path = data_new_path + 'model_build_inputs/route_data.json'
print('put Input Data for generate_build_route_seq_df')
with open(file_path, "w") as in_file:
    json.dump(route_data_new, in_file)


file_path = data_new_path + 'model_build_inputs/invalid_sequence_scores.json'
print('put Input Data for generate_build_route_seq_df')
with open(file_path, "w") as in_file:
    json.dump(invalid_sequence_scores_new, in_file)

file_path = data_new_path + 'model_build_inputs/package_data.json'
print('put Input Data for generate_build_route_seq_df')
with open(file_path, "w") as in_file:
    json.dump(package_data_new, in_file)


