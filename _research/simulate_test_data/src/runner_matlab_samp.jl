using Revise
includet("./simulation/simulate_test_data.jl")
includet("./ride/ride.jl")

using .Ride


#import data from hdf5. Created for use with the sample data exported from matlab
function import_data_from_hdf5(file_path, channel)
    #imported data is in matlab format: timestep, channel, epoch
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
            push!(data, import_data[y,channel,x])
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
    data, evts = import_data_from_hdf5("matlab_ride_samp_face.h5", 44)
    #plot_first_three_epochs_of_raw_data(data, evts);


end

#run the ride algorithm on the simulated data
begin
    #config for ride algorithm
    cfg = ride_config(
        sfreq = 1,
        s_range = [0, 250],
        r_range = [-150, 150],
        c_range = [-200, 200],
        c_estimation_range = [0, 400],
        epoch_range = [-49,500],
        epoch_event_name = 'S',
        iteration_limit = 5,
        heuristic1 = true,
        heuristic2 = true,
        heuristic3 = true
    )

    #run the ride algorithm
    c_latencies, s_erp, c_erp, r_erp = ride_algorithm(data, evts, cfg, ride_original)
end


#test that the erp calculated on the original imported data and 
#the erp calculated after concatenation and unfold.epoch() is identical
if true == true
    channel = 44
    file_path = "matlab_ride_samp_face.h5"

    data, evts = import_data_from_hdf5("matlab_ride_samp_face.h5", channel)
    import_data = h5read(file_path, "/dataset_data")

    orig_erp = median(import_data[:,channel,:], dims = 2)
    f = lines(orig_erp[:,1], color = :red, linewidth = 3)

    evts_s = @subset(evts, :event .== 'S')
    new_data_epoched = Unfold.epoch(data = data, tbl = evts_s, τ = [-49,500], sfreq = 1)[1]
    n, new_data_epoched = Unfold.drop_missing_epochs(evts_s, new_data_epoched)
    new_erp = median(new_data_epoched, dims = 3)

    lines!(new_erp[1,:,1], color = :blue)
    display(f)

    @assert(orig_erp[:,1] == new_erp[1,:,1])
end