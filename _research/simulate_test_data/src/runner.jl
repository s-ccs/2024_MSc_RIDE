using Revise
includet("./simulation/simulate_test_data.jl")
includet("./ride/ride.jl")


using .Ride

begin
    #simulate data
    sequence_design, components, multi_onset = default_sequence_design()

    rng = MersenneTwister() #1234
    data, evts = simulate(
        rng,
        sequence_design,
        components,
        multi_onset,
        PinkNoise(),
    )
    
    sequence_design_clean, components_clean, multi_onset_clean = default_sequence_design(0, 200, 1, 0, 45, 5, 0, 45, 5)
    data_clean, evts_clean = simulate(
        rng,
        sequence_design_clean,
        components_clean,
        multi_onset_clean,
        #PinkNoise(),
    )

    sequence_design_clean, components_clean, multi_onset_clean = default_sequence_design(0, 200, 1, 0, 45, 0, 0, 45, 0, 1, 0)
    data_clean_s, evts_clean_s = simulate(
        rng,
        sequence_design_clean,
        components_clean,
        multi_onset_clean,
        #PinkNoise(),
    )

    sequence_design_clean, components_clean, multi_onset_clean = default_sequence_design(0, 200, 0, 0, 45, 5, 0, 45, 0, 0, 1)
    data_clean_r, evts_clean_r = simulate(
        rng,
        sequence_design_clean,
        components_clean,
        multi_onset_clean,
        #PinkNoise(),
    )

    sequence_design_clean, components_clean, multi_onset_clean = default_sequence_design(0, 200, 0, 0, 45, 0, 0, 45, 5, 0, 0)
    data_clean_c, evts_clean_c = simulate(
        rng,
        sequence_design_clean,
        components_clean,
        multi_onset_clean,
        #PinkNoise(),
    )

    plot_first_three_epochs_of_raw_data(data, evts);
end

begin
    s_width = 0
    s_offset = 300
    s_beta = 1
    r_width = rand(30:70)
    r_offset = rand(10:70)
    r_beta = rand(vcat(-5:-2,2:5))
    c_width = rand(30:60)
    c_offset = rand(20:40)
    c_beta = rand(vcat(-5:-3,3:5))

    sequence_design, components, multi_onset = default_sequence_design(s_width, s_offset, s_beta, c_width, c_offset, c_beta, r_width, r_offset, r_beta)

    rng = MersenneTwister() #1234
    data, evts = simulate(
        rng,
        sequence_design,
        components,
        multi_onset,
        #PinkNoise(),
    )

    sequence_design_clean, components_clean, multi_onset_clean = default_sequence_design(0, s_offset + round(Int,s_width/2), s_beta, 0, c_offset + round(Int,c_width/2), c_beta, 0, r_offset + round(Int,r_width/2), r_beta)
    data_clean, evts_clean = simulate(
        rng,
        sequence_design_clean,
        components_clean,
        multi_onset_clean,
        #PinkNoise(),
    )

    plot_first_three_epochs_of_raw_data(data, evts);
    plot_first_three_epochs_of_raw_data(data_clean, evts_clean);
end

begin
    #config for ride algorithm
    cfg = ride_config(
        sfreq = 100,
        s_range = [-0.2, 0.4],
        r_range = [0, 0.8],
        c_range = [-0.4, 0.4], # change to -0.4 , 0.4 or something because it's attached to the latency of C
        c_estimation_range = [0.2, 1.2],
        epoch_range = [-0.3,1.6],
        epoch_event_name = 'S'
    )

    save_to_hdf5_ride_format(data, evts, cfg.epoch_range, cfg.epoch_event_name, 'R', cfg.sfreq)

    #remove the C events from the evts table, these will be estimated by the ride algorithm
    evts_without_c = @subset(evts, :event .!= 'C')

    #run the ride algorithm
    c_latencies, s_erp, c_erp, r_erp = ride_algorithm_unfold(data, evts_without_c, cfg)
end



begin
    evts_clean_s = @subset(evts_clean, :event .== 'S')
    data_epoched_clean = Unfold.epoch(data = data_clean, tbl = evts_clean_s, τ = cfg.epoch_range, sfreq = cfg.sfreq)[1]
    n, data_epoched_clean = Unfold.drop_missing_epochs(evts_clean_s, data_epoched_clean)
    erp_clean = mean(data_epoched_clean, dims = 3)

    data_epoched_clean = Unfold.epoch(data = data_clean_s, tbl = evts_clean_s, τ = cfg.epoch_range, sfreq = cfg.sfreq)[1]
    n, data_epoched_clean = Unfold.drop_missing_epochs(evts_clean_s, data_epoched_clean)
    erp_clean_s = mean(data_epoched_clean, dims = 3)

    data_epoched_clean = Unfold.epoch(data = data_clean_c, tbl = evts_clean_s, τ = cfg.epoch_range, sfreq = cfg.sfreq)[1]
    n, data_epoched_clean = Unfold.drop_missing_epochs(evts_clean_s, data_epoched_clean)
    erp_clean_c = mean(data_epoched_clean, dims = 3)

    data_epoched_clean = Unfold.epoch(data = data_clean_r, tbl = evts_clean_s, τ = cfg.epoch_range, sfreq = cfg.sfreq)[1]
    n, data_epoched_clean = Unfold.drop_missing_epochs(evts_clean_s, data_epoched_clean)
    erp_clean_r = mean(data_epoched_clean, dims = 3)

    f = Figure()
    Axis(f[1,1], yticks = -100:100)
    #lines!(f[1,1], c_erp[1,:])
    raw = lines!(f[1,1],  erp_clean[1,:,1], color = "black")
    s = lines!(f[1,1],  erp_clean_s[1,:,1], color = "blue")
    c = lines!(f[1,1],  erp_clean_c[1,:,1], color = "red")
    r = lines!(f[1,1],  erp_clean_r[1,:,1], color = "green")
    Legend(f[1,2]
    , [raw, s, r, c]
    , ["Raw ERP", "S ERP", "R ERP", "C ERP"]
    )
    display(f)
    save("actual_erps.png", f)
end

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

function import_data_from_hdf5(file_path)
    data = h5read(file_path, "/dataset_data")
    rt = h5read(file_path, "/dataset_rt")

    return data, rt
end

#data_import, rt = import_data_from_hdf5("matlab_ride_samp_face.h5")