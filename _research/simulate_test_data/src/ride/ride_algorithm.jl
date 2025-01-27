function ride_algorithm(data, evts, cfg::ride_config, Modus::Type{ride_original})
    @debug "Running RIDE algorithm with cfg: $cfg"
    ## data_preparation
    data_reshaped = reshape(data, (1,:))
    evts_s = @subset(evts, :event .== 'S')
    evts_r = @subset(evts, :event .== 'R')

    #epoch data with the cfg.epoch_range to see how many epochs we have
    #cut evts to match the determined number of epochs
    #the resulting data_epoched is also used for the c latency estimation
    data_epoched, data_epoched_times = Unfold.epoch(data = data_reshaped, tbl = evts_s, τ = cfg.epoch_range, sfreq = cfg.sfreq)
    n ,data_epoched = Unfold.drop_missing_epochs(evts_s, data_epoched)
    number_epochs = size(data_epoched, 3)
    #@assert size(evts) == (number_epochs * 2) "Size of evts is $(size(evts)) but should be $(number_epochs * 2)"
    evts_s = evts_s[1:number_epochs,:]
    evts_r = evts_r[1:number_epochs,:]

    #reduce evts to the number of epochs
    while size(evts,1) > number_epochs*2
        deleteat!(evts, size(evts,1))
    end
    @assert size(evts,1) == number_epochs*2 "Size of evts is $(size(evts,1)) but should be $(number_epochs*2)"
    ##

    ## initial C latency estimation
    c_latencies_df = initial_peak_estimation(data, evts_s, cfg)
    evts_c = build_c_evts_table(c_latencies_df, evts_s)
    c_range_adj = c_range_adjusted(cfg.c_range)
    ##

    ## initial erp calculation
    #calculate initial  erp of S
    data_epoched_s, data_epoched_s_times = Unfold.epoch(data = data_reshaped, tbl = evts_s, τ = cfg.s_range, sfreq = cfg.sfreq)
    n, data_epoched_s = Unfold.drop_missing_epochs(evts_s, data_epoched_s)
    s_erp = median(data_epoched_s, dims = 3)

    #calculate initial erp of R
    data_epoched_r, data_epoched_r_times = Unfold.epoch(data = data_reshaped, tbl = evts_r, τ = cfg.r_range, sfreq = cfg.sfreq)
    n, data_epoched_r = Unfold.drop_missing_epochs(evts_r, data_epoched_r)
    r_erp = median(data_epoched_r, dims = 3)

    #calculate initial erp of C by subtracting S and R from the data
    data_subtracted_s_and_r = subtract_to_data_epoched(data_reshaped, evts_c, c_range_adj, [(evts_s, s_erp, cfg.s_range), (evts_r, r_erp, cfg.r_range)], cfg.sfreq)
    c_erp = median(data_subtracted_s_and_r, dims = 3)
    ##

    ## prepare figure arrays
    plot_first_epoch(cfg, evts_s, evts_r, evts_c, data_reshaped)
    figures_latency = Array{Figure,1}()
    push!(figures_latency, plot_c_latency_estimation_four_epochs(data_epoched, reshape(c_latencies_df.latency,(1,:)), c_erp[1,:,1]))
    figures_erp = Array{Figure,1}()
    push!(figures_erp, plot_data_plus_component_erp(data_epoched, evts_s, evts_r, s_erp, r_erp, c_erp, reshape(c_latencies_df.latency,(1,:)), cfg))
    ##

    c_latencies_df_prev_prev = nothing
    c_latencies_df_prev = nothing


    ## iteration start
    for i in range(1,cfg.iteration_limit) 
        ## decompose data into S, R and C components using the current C latencies
        for j in 1:25
            #calculate erp of C by subtracting S and R from the data
            data_subtracted_s_and_r = subtract_to_data_epoched(data_reshaped, evts_c, c_range_adj, [(evts_s, s_erp, cfg.s_range), (evts_r, r_erp, cfg.r_range)], cfg.sfreq)
            c_erp = median(data_subtracted_s_and_r, dims = 3)
            #calculate erp of S
            data_subtracted_c_and_r = subtract_to_data_epoched(data_reshaped, evts_s, cfg.s_range, [(evts_c, c_erp, c_range_adj), (evts_r, r_erp, cfg.r_range)], cfg.sfreq)
            s_erp = median(data_subtracted_c_and_r, dims = 3)
            #calculate erp of R
            data_subtracted_s_and_c = subtract_to_data_epoched(data_reshaped, evts_r, cfg.r_range, [(evts_s, s_erp, cfg.s_range), (evts_c, c_erp, c_range_adj)], cfg.sfreq)
            r_erp = median(data_subtracted_s_and_c, dims = 3)
        end
        ##

        ## update C latencies via pattern matching
        #perform the pattern matching on the data with the S and R components subtracted
        data_subtracted_s_and_r = subtract_to_data(data_reshaped, [(evts_s, s_erp, cfg.s_range), (evts_r, r_erp, cfg.r_range)], cfg.sfreq)
        data_epoched_subtracted_s_and_r, n = Unfold.epoch(data = data_subtracted_s_and_r, tbl = evts_s, τ = cfg.epoch_range, sfreq = cfg.sfreq)
        n, data_epoched_subtracted_s_and_r = Unfold.drop_missing_epochs(evts_s, data_epoched_subtracted_s_and_r)
        xcorr, m, onset = findxcorrpeak(data_epoched_subtracted_s_and_r[1,:,:],c_erp[1,:,1])
        c_latencies = reshape(m .- round(Int,  (c_range_adj[1] * cfg.sfreq)), (1,:))
        for (i, row) in enumerate(eachrow(latencies_df))
            if(row.fixed) continue end
            row.latency = c_latencies[i]
        end
        ##

        ## heuristics
        if cfg.heuristic1 && !isnothing(c_latencies_df_prev) && !isnothing(c_latencies_df_prev_prev)
            heuristic1(c_latencies_df, c_latencies_df_prev, c_latencies_df_prev_prev)
        end

        if cfg.heuristic2 && !isnothing(c_latencies_df_prev)
            heuristic2(c_latencies_df, c_latencies_df_prev, xcorr, cfg.heuristic2_rng)
        end

        if cfg.heuristic3 && !isnothing(c_latencies_df_prev)
            heuristic3(c_latencies_df, c_latencies_df_prev, xcorr, cfg.heuristic3_threshhold, onset=onset)
        end

        c_latencies_df_prev_prev = deepcopy(c_latencies_df_prev)
        c_latencies_df_prev = deepcopy(c_latencies_df)
        evts_c = build_c_evts_table(c_latencies_df, evts_s)
        ##

        ## add plots
        push!(figures_latency, plot_c_latency_estimation_four_epochs(data_epoched, reshape(c_latencies_df.latency,(1,:)), c_erp[1,:,1]))
        push!(figures_erp, plot_data_plus_component_erp(data_epoched, evts_s, evts_r, s_erp, r_erp, c_erp, reshape(c_latencies_df.latency,(1,:)), cfg))
        ##
    end

    ##last iteration using the mean instead of median
    #calculate erp of C by subtracting S and R from the data
    data_subtracted_s_and_r = subtract_to_data_epoched(data_reshaped, evts_c, c_range_adj, [(evts_s, s_erp, cfg.s_range), (evts_r, r_erp, cfg.r_range)], cfg.sfreq)
    c_erp = mean(data_subtracted_s_and_r, dims = 3)
    #calculate erp of S
    data_subtracted_c_and_r = subtract_to_data_epoched(data_reshaped, evts_s, cfg.s_range, [(evts_c, c_erp, c_range_adj), (evts_r, r_erp, cfg.r_range)], cfg.sfreq)
    s_erp = mean(data_subtracted_c_and_r, dims = 3)
    #calculate erp of R
    data_subtracted_s_and_c = subtract_to_data_epoched(data_reshaped, evts_r, cfg.r_range, [(evts_s, s_erp, cfg.s_range), (evts_c, c_erp, c_range_adj)], cfg.sfreq)
    r_erp = mean(data_subtracted_s_and_c, dims = 3)
    push!(figures_erp, plot_data_plus_component_erp(data_epoched, evts_s, evts_r, s_erp, r_erp, c_erp, reshape(c_latencies_df.latency,(1,:)), cfg))
    ##

    ## plotting
    #plot the estimated c latencies for each iteration
    for (i,f) in enumerate(figures_latency)
        Label(f[0, :], text = "Original Ride: Estimated C latency, Iteration $(i-1)", halign = :center)
        display(f)
    end
    #plot the calculated erp for each iteration
    for (i,f) in enumerate(figures_erp)
        Label(f[0, :], text = "Original Ride: Calculated Erp, Iteration $(i-1)")
        display(f)
    end
    ##

    return reshape(c_latencies_df.latency,(1,:)), s_erp, r_erp, c_erp
end