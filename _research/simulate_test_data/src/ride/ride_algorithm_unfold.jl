function unfold_pattern_matching(data_continous, c_erp, evts, cfg)
    #epoch residue
    evts_s = @subset(evts, :event .== 'S')
    data_residuals_epoched, times = Unfold.epoch(data = data_continous, tbl = evts_s, τ = cfg.epoch_range, sfreq = cfg.sfreq)
    n, data_residuals_epoched = Unfold.drop_missing_epochs(evts_s, data_residuals_epoched)

    xc, result = findxcorrpeak(data_residuals_epoched[1,:,:],c_erp)
    c_latencies = reshape(result .- round(Int,  (cfg.c_range[1] * cfg.sfreq)), (1,:))
    
    evts_c = copy(evts_s)
    evts_c[!,:latency] .= round.(Int, evts_s[!,:latency] + c_latencies[1,:] .+ (cfg.epoch_range[1]*cfg.sfreq))
    evts_c[!,:event] .= 'C'

    return c_latencies, evts_c
end

function unfold_decomposition(data, evts_with_c)
    #unfold deconvolution
    m = fit(UnfoldModel,[
        'S' => (@formula(0~1),firbasis(cfg.s_range,cfg.sfreq,"")),
        'R' => (@formula(0~1),firbasis(cfg.r_range,cfg.sfreq,"")),
        'C' => (@formula(0~1),firbasis(cfg.c_range,cfg.sfreq,""))],
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

function latency_estimation(d, evts_s, cfg)
    ## initial C latency estimation
    data_residuals_epoched, times = Unfold.epoch(data = d, tbl = evts_s, τ = cfg.epoch_range, sfreq = cfg.sfreq)
    n, data_residuals_epoched = Unfold.drop_missing_epochs(evts_s, data_residuals_epoched)
    #Peak estimation/algorithm for initial c latencies
    c_latencies = Matrix{Float64}(undef, 1, size(data_residuals_epoched, 3))
    for a in (1:size(data_residuals_epoched, 3))
        range = round.(Int, cfg.c_estimation_range[1] * cfg.sfreq) : round(Int, cfg.c_estimation_range[2] * cfg.sfreq)
        c_latencies[1,a] = (findmax(abs.(data_residuals_epoched[1,range,a])) .+ range[1] .- 1)[2]
    end
    #Create C event table by copying S and adding the estimated latency
    evts_c = copy(evts_s)
    evts_c[!,:latency] .= round.(Int, evts_s[!,:latency] + c_latencies[1,:] .+ (cfg.epoch_range[1]*cfg.sfreq))
    evts_c[!,:event] .= 'C'
    ##
    return c_latencies, evts_c
end

function ride_algorithm(data, evts, cfg::ride_config, Modus::Type{ride_unfold})
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
    yhat = predict(m)
    y = data_reshaped
    residuals_without_SR = Unfold._residuals(UnfoldModel,yhat,y)
    ##@show size(yhat)
    ##@show size(y)
    ##@show size(residuals_without_SR)
    ##plot_first_three_epochs_of_raw_data(yhat)
    ##plot_first_three_epochs_of_raw_data(y)
    ##plot_first_three_epochs_of_raw_data(residuals_without_SR)
    ##

    ## initial C latency estimation
    c_latencies, evts_c = latency_estimation(residuals_without_SR, evts_s, cfg)
    ##

    ## calculate first c_erp from initial latencies and residue/data
    data_residuals_c_epoched, times = Unfold.epoch(data = residuals_without_SR, tbl = evts_c, τ = cfg.c_range, sfreq = cfg.sfreq)
    n, data_residuals_c_epoched = Unfold.drop_missing_epochs(evts_c, data_residuals_c_epoched)
    c_erp = median(data_residuals_c_epoched, dims = 3) 
    ##

    ## prepare figure arrays
    plot_first_epoch(cfg, evts_s, evts_r, evts_c, data_reshaped)
    figures_latency = Array{Figure,1}()
    push!(figures_latency, plot_c_latency_estimation_four_epochs(data_epoched, c_latencies))
    figures_erp = Array{Figure,1}()
    push!(figures_erp, plot_data_plus_component_erp(data_epoched, evts_s, evts_r, reshape(s_erp,(1,:,1)), reshape(r_erp,(1,:,1)), reshape(c_erp,(1,:,1)), c_latencies, cfg))
    ##

    ## iteration start
    for i in range(1,cfg.iteration_limit) 
        ## decompose data into S, R and C components using the current C latencies
        evts_with_c = sort(vcat(evts, evts_c), [:latency])
        s_erp, r_erp, c_erp, residue = unfold_decomposition(data_reshaped, evts_with_c)
        ##

        ## update C latencies via pattern matching
        c_latencies, evts_c = unfold_pattern_matching(residue, c_erp, evts_s, cfg)
        ##

        ## add plots
        push!(figures_latency, plot_c_latency_estimation_four_epochs(data_epoched, c_latencies))
        push!(figures_erp, plot_data_plus_component_erp(data_epoched, evts_s, evts_r, reshape(s_erp,(1,:,1)), reshape(r_erp,(1,:,1)), reshape(c_erp,(1,:,1)), c_latencies, cfg))
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

    return c_latencies, s_erp, r_erp, c_erp
end