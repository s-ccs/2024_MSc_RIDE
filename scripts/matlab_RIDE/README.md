# Matlab Ride Project

This project contains the orginal matlab RIDE implementation with some scripts that were used during the development of UnfoldRIDE. There are scripts to:

<pre>
- "run_import_simulated.m"  import and run simulated data + benchmark       
- "run_matlab_sample.m"     run the real dataset included in the matlab version + benchmark
- "export_results.m"        export algorithm results using hdf5 for plotting in julia
- "export_sample_data.m"    export sample data using hdf5 for use in UnfoldRIDE
- "data_filepath.m"         set the data directory for imports and exports
</pre>

A version of the matlab RIDE_call project is included in this github repository, but also available for download here:

https://cns.hkbu.edu.hk/RIDE_files/Page308.htm

https://github.com/guangouyang/RIDE

There are some comments in the RIDE_call/Ride_call.m starting with "Till:" to mark a couple locations. Namely where Heuristic 1,2,3 are applied and where the mean of all channel latencies is calculated and saved as the new overall latency.