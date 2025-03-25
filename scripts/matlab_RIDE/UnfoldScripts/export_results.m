data_filepath

%WARNING: This will overwrite the old file
%export RIDE results to an h5 file
output_path = data_path + 'matlab_ride_results_real.h5';

delete(output_path)
h5create(output_path, '/dataset_erp', size(results.erp))
h5write(output_path, '/dataset_erp', results.erp)
h5create(output_path, '/dataset_s', size(results.s))
h5write(output_path, '/dataset_s', results.s)
h5create(output_path, '/dataset_r', size(results.r))
h5write(output_path, '/dataset_r', results.r)
h5create(output_path, '/dataset_c', size(results.c))
h5write(output_path, '/dataset_c', results.c)
h5create(output_path, '/dataset_c_latency', size(results.latency_c))
h5write(output_path, '/dataset_c_latency', results.latency_c)
