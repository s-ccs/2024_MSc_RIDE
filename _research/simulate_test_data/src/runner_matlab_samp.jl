using Revise
includet("./simulation/simulate_test_data.jl")
includet("./ride/ride.jl")

using .Ride


#import data from hdf5. Created for use with the sample data exported from matlab
function import_data_from_hdf5(file_path)
    import_data = h5read(file_path, "/dataset_data")
    rt = h5read(file_path, "/dataset_rt")

    data = Vector{Float64}()
    #fill the ride_matrix with the data
    offset = 800
    for i in 1:offset
        push!(data, 0)
    end
    for x in axes(import_data, 3)
        for y in axes(import_data, 1)
            push!(data, import_data[y,1,x])
        end
        for i in 1:offset
            push!(data, 0)
        end
    end
    s = Vector(1:length(rt)).* (offset + size(import_data,1)) .- size(import_data,1) .+ 50
    rt = Int.(round.(rt ./ 2))
    rt = rt .+ s
    evts = DataFrame()
    for i in eachindex(rt)
        push!(evts, (event = 'S', latency = s[i]))
        push!(evts, (event = 'R', latency = rt[i]))
    end

    return data, evts
end

#import data
begin 
    data, evts = import_data_from_hdf5("matlab_ride_samp_face.h5")
    plot_first_three_epochs_of_raw_data(data, evts);
end

#run the ride algorithm on the simulated data
begin
    #config for ride algorithm
    cfg = ride_config(
        sfreq = 1,
        s_range = [0, 250],
        r_range = [-150, 150],
        c_range = [-200, 200],
        c_estimation_range = [50, 450],
        epoch_range = [-50,500],
        epoch_event_name = 'S'
    )

    #run the ride algorithm
    c_latencies, s_erp, c_erp, r_erp = ride_algorithm_unfold(data, evts, cfg)
end