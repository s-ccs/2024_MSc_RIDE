function unfold_pattern_matching(latencies_df, data_residuals_continous, c_erp, evts, cfg)
    #epoch residue
    evts_s = @subset(evts, :event .== 'S')
    data_residuals_epoched, times = Unfold.epoch(data = data_residuals_continous, tbl = evts_s, τ = cfg.epoch_range, sfreq = cfg.sfreq)
    n, data_residuals_epoched = Unfold.drop_missing_epochs(evts_s, data_residuals_epoched)

    xc, result, onset = findxcorrpeak(data_residuals_epoched[1,:,:],c_erp)
    c_latencies = result .- round(Int,  (c_range_adjusted(cfg.c_range)[1] * cfg.sfreq))

    for (i, row) in enumerate(eachrow(latencies_df))
        if(row.fixed) continue end
        row.latency = c_latencies[i]
    end

    return latencies_df, xc, onset
end


function unfold_decomposition(data, evts_with_c)
    #unfold deconvolution
    m = fit(UnfoldModel,[
        'S' => (@formula(0~1),firbasis(cfg.s_range,cfg.sfreq,"")),
        'R' => (@formula(0~1),firbasis(cfg.r_range,cfg.sfreq,"")),
        'C' => (@formula(0~1),firbasis(c_range_adjusted(cfg.c_range),cfg.sfreq,""))],
        evts_with_c,data)
    c_table = coeftable(m)
    s_erp = c_table[c_table.eventname .== 'S',:estimate]
    r_erp = c_table[c_table.eventname .== 'R',:estimate]
    c_erp = c_table[c_table.eventname .== 'C',:estimate]

    yhat = predict(m,exclude_basis = 'C', overlap = true)
    y = data
    residuals_without_SR = Unfold._residuals(UnfoldModel,yhat,y)

    return s_erp, r_erp, c_erp, residuals_without_SR
end

function ride_algorithm(data, evts, cfg::ride_config, Modus::Type{ride_unfold})
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

    ## initial unfold deconvolution
    m = fit(UnfoldModel,[
        'S' => (@formula(0~1),firbasis(cfg.s_range,cfg.sfreq,"")),
        'R' => (@formula(0~1),firbasis(cfg.r_range,cfg.sfreq,""))],
        evts,data_reshaped)
    c_table = coeftable(m)
    s_erp = c_table[c_table.eventname .== 'S',:estimate]
    r_erp = c_table[c_table.eventname .== 'R',:estimate]
    ##

    ## initial residue calculation (data minus S and R)
    #exclude='C' doesn't seem to do anything. Probably because the model doesn't know what 'C' is. 
    #Evts doesn't contain 'C' at this point.
    yhat = predict(m,exclude_basis = 'C', overlap = true)
    y = data_reshaped
    residuals_without_SR = Unfold._residuals(UnfoldModel,yhat,y)
    ##

    ## initial C latency estimation
    c_latencies_df = initial_peak_estimation(residuals_without_SR, evts_s, cfg)
    evts_c = build_c_evts_table(c_latencies_df, evts_s)
    ##

    ## calculate first c_erp from initial latencies and residue/data
    data_residuals_c_epoched, times = Unfold.epoch(data = residuals_without_SR, tbl = evts_c, τ = c_range_adjusted(cfg.c_range), sfreq = cfg.sfreq)
    n, data_residuals_c_epoched = Unfold.drop_missing_epochs(evts_c, data_residuals_c_epoched)
    c_erp = median(data_residuals_c_epoched, dims = 3) 
    ##

    ## prepare figure arrays
    plot_first_epoch(cfg, evts_s, evts_r, evts_c, data_reshaped)
    figures_latency = Array{Figure,1}()
    push!(figures_latency, plot_c_latency_estimation_four_epochs(data_epoched, reshape(c_latencies_df.latency,(1,:)), c_erp[1,:,1]))
    figures_erp = Array{Figure,1}()
    push!(figures_erp, plot_data_plus_component_erp(data_epoched, evts_s, evts_r, reshape(s_erp,(1,:,1)), reshape(r_erp,(1,:,1)), reshape(c_erp,(1,:,1)), reshape(c_latencies_df.latency,(1,:)), cfg))
    ##

    ## initial pattern matching with the first calculated c_erp
    c_latencies_df_prev_prev = nothing
    c_latencies_df_prev = deepcopy(c_latencies_df)
    c_latencies_df, xcorr = unfold_pattern_matching(c_latencies_df, residuals_without_SR, c_erp[1,:,1], evts_s, cfg)
    evts_c = build_c_evts_table(c_latencies_df, evts_s)
    ##
    
    ## add plots after the first pattern matching
    push!(figures_latency, plot_c_latency_estimation_four_epochs(data_epoched, reshape(c_latencies_df.latency,(1,:)), c_erp[1,:,1]))
    push!(figures_erp, plot_data_plus_component_erp(data_epoched, evts_s, evts_r, reshape(s_erp,(1,:,1)), reshape(r_erp,(1,:,1)), reshape(c_erp,(1,:,1)), reshape(c_latencies_df.latency,(1,:)), cfg))
    ##


    ## iteration start
    for i in range(1,cfg.iteration_limit) 
        ## decompose data into S, R and C components using the current C latencies
        evts_with_c = sort(vcat(evts, evts_c), [:latency])
        s_erp, r_erp, c_erp, residue = unfold_decomposition(data_reshaped, evts_with_c)
        ##

        ## update C latencies via pattern matching
        c_latencies_df, xcorr, onset = unfold_pattern_matching(c_latencies_df, residue, c_erp, evts_s, cfg)
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
        push!(figures_latency, plot_c_latency_estimation_four_epochs(data_epoched, reshape(c_latencies_df.latency,(1,:)), c_erp))
        push!(figures_erp, plot_data_plus_component_erp(data_epoched, evts_s, evts_r, reshape(s_erp,(1,:,1)), reshape(r_erp,(1,:,1)), reshape(c_erp,(1,:,1)), reshape(c_latencies_df.latency,(1,:)), cfg))
        ##
    end

    ## plotting
    #plot the estimated c latencies for each iteration
    for (i,f) in enumerate(figures_latency)
        Label(f[0, :], text = "Estimated C latency, Iteration $(i-1)", halign = :center)
        display(f)
    end
    #plot the calculated erp for each iteration
    for (i,f) in enumerate(figures_erp)
        Label(f[0, :], text = "Calculated Erp, Iteration $(i-1)")
        display(f)
    end
    ##

    return reshape(c_latencies_df.latency,(1,:)), s_erp, r_erp, c_erp
end