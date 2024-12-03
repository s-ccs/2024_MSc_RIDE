using Revise
includet("./simulation/simulate_test_data.jl")
includet("./ride/ride.jl")


using .Ride

function plot_first_three_epochs_of_raw_data(data, evts)
    f = Figure()

    evts_s = @subset(evts, :event .== 'S')
    evts_r = @subset(evts, :event .== 'R')

    Axis(f[1,1], title = "First 600 data points")
    graph = lines!(first(vec(data),evts_s.latency[4]); color = "black")
    graph_r = vlines!(first(evts_r.latency,3), color = "blue")
    graph_s = vlines!(first(evts_s.latency,3), color = "red")
    
    Legend(f[1,2]
        , [graph, graph_r, graph_s]
        , ["Data", "Reaction Times", "Stimulus Onsets"]
    )
    display(f)
end

#function simulate_default_run_once()
    #simulate data
    sequence_design, components, multi_onset = default_sequence_design()
    rng = MersenneTwister(1234)
    data, evts = simulate(
        rng,
        sequence_design,
        components,
        multi_onset,
        PinkNoise(),
    )

    plot_first_three_epochs_of_raw_data(data, evts);

    #config for ride algorithm
    cfg = ride_config(
        sfreq = 100,
        s_range = [-0.2, 0.3],
        r_range = [0, 0.8],
        c_range = [-0.4, 0.4], # change to -0.4 , 0.4 or something because it's attached to the latency of C
        c_estimation_range = [0.2, 1.0],
        epoch_range = [-0.3,1.6],
        epoch_event_name = 'S',
        residue_matching = true
    )

    save_to_hdf5_ride_format(data, evts, cfg.epoch_range, cfg.epoch_event_name, 'R', cfg.sfreq)

    #remove the C events from the evts table, these will be estimated by the ride algorithm
    evts_without_c = @subset(evts, :event .!= 'C')

    #run the ride algorithm
    c_latencies, s_erp, c_erp, r_erp = ride_algorithm(data, evts_without_c, cfg)
    
    #plot the results
    #multi_onset.stimulus_onset.width = 0
    #multi_onset.component_to_stimulus_onsets.width .= 0
#
    #clean_data, clean_evts = simulate(
    #    rng,
    #    sequence_design,
    #    components,
    #    multi_onset,
    #)
    #
    #clean_data_epoched = Unfold.epoch(data = clean_data, tbl = clean_evts, τ = cfg.epoch_range, sfreq = cfg.sfreq)[1]
    #n, clean_data_epoched = Unfold.drop_missing_epochs(clean_evts, clean_data_epoched)
    #clean_erp = median(clean_data_epoched, dims = 3)
#
    #f = Figure()
    #Axis(f[1,1], title = "C ERP")
    #lines!(f[1,1], c_erp[1,:])
    #lines!(f[1,2],  clean_data_epoched[1,:,1])

#end



#simulate_default_run_once()