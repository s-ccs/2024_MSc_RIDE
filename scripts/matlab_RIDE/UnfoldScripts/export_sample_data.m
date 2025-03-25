data_filepath

%WARNING: This will overwrite the old file
output_path = data_path + 'matlab_ride_samp_face.h5';

%export the matlab example data
load("./RIDE_call/example/samp_face.mat")
delete(output_path)
h5create(output_path, '/dataset_data', size(data))
h5write(output_path, '/dataset_data', data)
h5create(output_path, '/dataset_rt', size(rt))
h5write(output_path, '/dataset_rt', rt)

