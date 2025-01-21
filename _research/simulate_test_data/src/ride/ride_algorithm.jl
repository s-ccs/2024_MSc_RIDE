using UnfoldSim
using CairoMakie
using Random
using Unfold
using UnfoldMakie
using StableRNGs
using Parameters
using HDF5
using DataFrames
using DataFramesMeta
using Statistics
using DSP

@with_kw struct ride_config
    sfreq::Int
    s_range::Vector{Float64}
    r_range::Vector{Float64}
    c_range::Vector{Float64}
    c_estimation_range::Vector{Float64}
    epoch_range::Vector{Float64}
    epoch_event_name::Char
    iteration_limit::Int = 5
end

function ride_algorithm(data, evts, cfg::ride_config, Modus::Type{ride_original})

    #todo: define the reaction times as a separate input parameter instead of as part of the evts table
    #todo: change the data input format to somewhat match the ride input format
    #       Matrix Float64 with Channel:TimeStep:Trial


    data_reshaped = reshape(data, (1,:))
    evts_s = @subset(evts, :event .== 'S')
    evts_r = @subset(evts, :event .== 'R')

    #epoch data with the cfg.epoch_range to see how many epochs we have
    #the resulting data_epoched is also used for the c latency estimation
    data_epoched, data_epoched_times = Unfold.epoch(data = data_reshaped, tbl = evts_s, τ = cfg.epoch_range, sfreq = cfg.sfreq)
    n ,data_epoched = Unfold.drop_missing_epochs(evts_s, data_epoched)
    number_epochs = size(data_epoched, 3)
    evts_s = evts_s[1:number_epochs,:]
    evts_r = evts_r[1:number_epochs,:]

    #Peak estimation/algorithm for initial c latencies
    global c_latencies = Matrix{Float64}(undef, 1, size(data_epoched, 3))
    for a in (1:size(data_epoched, 3))
        range = round.(Int, cfg.c_estimation_range[1] * cfg.sfreq) : round(Int, cfg.c_estimation_range[2] * cfg.sfreq)
        c_latencies[1,a] = (findmax(abs.(data_epoched[1,range,a])) .+ range[1] .- 1)[2]
    end
    #testing whether the algorithm works with all c_latencies = 0 
    #c_latencies = zeros(Float64, 1, size(data_epoched, 3))

    #Create C event table by copying S and adding the estimated latency
    global evts_c = copy(evts_s)
    evts_c[!,:latency] .= round.(Int, evts_s[!,:latency] + c_latencies[1,:] .+ (cfg.epoch_range[1]*cfg.sfreq))
    evts_c[!,:event] .= 'C'

    plot_first_epoch(cfg, evts_s, evts_r, evts_c, data_reshaped)

    #calculate initial  erp of S
    data_epoched_s, data_epoched_s_times = Unfold.epoch(data = data_reshaped, tbl = evts_s, τ = cfg.s_range, sfreq = cfg.sfreq)
    n, data_epoched_s = Unfold.drop_missing_epochs(evts_s, data_epoched_s)
    s_erp_temp = median(data_epoched_s, dims = 3)

    #calculate initial erp of R
    data_epoched_r, data_epoched_r_times = Unfold.epoch(data = data_reshaped, tbl = evts_r, τ = cfg.r_range, sfreq = cfg.sfreq)
    n, data_epoched_r = Unfold.drop_missing_epochs(evts_r, data_epoched_r)
    r_erp_temp = median(data_epoched_r, dims = 3)

    #calculate initial erp of C by subtracting S and R from the data
    data_subtracted_s_and_r = subtract_to_data_epoched(data_reshaped, evts_c, cfg.c_range, [(evts_s, s_erp_temp, cfg.s_range), (evts_r, r_erp_temp, cfg.r_range)], cfg.sfreq)
    c_erp_temp = median(data_subtracted_s_and_r, dims = 3)
    
    figures_latency = Array{Figure,1}()
    push!(figures_latency, plot_c_latency_estimation_four_epochs(data_epoched, c_latencies))
    figures_erp = Array{Figure,1}()
    push!(figures_erp, plot_data_plus_component_erp(data_epoched, evts_s, evts_r, s_erp_temp, r_erp_temp, c_erp_temp, c_latencies, cfg))
     
    #outer iteration RIDE
    for i in 1:cfg.iteration_limit
        #inner iteration RIDE
        for j in 1:25
            #calculate erp of C by subtracting S and R from the data
            data_subtracted_s_and_r = subtract_to_data_epoched(data_reshaped, evts_c, cfg.c_range, [(evts_s, s_erp_temp, cfg.s_range), (evts_r, r_erp_temp, cfg.r_range)], cfg.sfreq)
            c_erp_temp = median(data_subtracted_s_and_r, dims = 3)
            #calculate erp of S
            data_subtracted_c_and_r = subtract_to_data_epoched(data_reshaped, evts_s, cfg.s_range, [(evts_c, c_erp_temp, cfg.c_range), (evts_r, r_erp_temp, cfg.r_range)], cfg.sfreq)
            s_erp_temp = median(data_subtracted_c_and_r, dims = 3)
            #calculate erp of R
            data_subtracted_s_and_c = subtract_to_data_epoched(data_reshaped, evts_r, cfg.r_range, [(evts_s, s_erp_temp, cfg.s_range), (evts_c, c_erp_temp, cfg.c_range)], cfg.sfreq)
            r_erp_temp = median(data_subtracted_s_and_c, dims = 3)
        end
        
        push!(figures_erp, plot_data_plus_component_erp(data_epoched, evts_s, evts_r, s_erp_temp, r_erp_temp, c_erp_temp, c_latencies, cfg))


        #perform the pattern matching on the data with the S and R components subtracted
        #this should help avoid false positive matches on the S or R component
        #this is different from the original RIDE algorithm
        data_subtracted_s_and_r = subtract_to_data(data_reshaped, [(evts_s, s_erp_temp, cfg.s_range), (evts_r, r_erp_temp, cfg.r_range)], cfg.sfreq)
        data_epoched_subtracted_s_and_r, n = Unfold.epoch(data = data_subtracted_s_and_r, tbl = evts_s, τ = cfg.epoch_range, sfreq = cfg.sfreq)
        n, data_epoched_subtracted_s_and_r = Unfold.drop_missing_epochs(evts_s, data_epoched_subtracted_s_and_r)
        xc, m = findxcorrpeak(data_epoched_subtracted_s_and_r[1,:,:],c_erp_temp[1,:,1])
        c_latencies = reshape(m .- round(Int,  (cfg.c_range[1] * cfg.sfreq)), (1,:))

        evts_c = copy(evts_s)
        evts_c[!,:latency] .= round.(Int, evts_s[!,:latency] + c_latencies[1,:] .+ (cfg.epoch_range[1]*cfg.sfreq))
        evts_c[!,:event] .= 'C'



        push!(figures_latency, plot_c_latency_estimation_four_epochs(data_epoched, c_latencies))
    end

    #last iteration using the mean instead of median
    #calculate erp of C by subtracting S and R from the data
    data_subtracted_s_and_r = subtract_to_data_epoched(data_reshaped, evts_c, cfg.c_range, [(evts_s, s_erp_temp, cfg.s_range), (evts_r, r_erp_temp, cfg.r_range)], cfg.sfreq)
    c_erp_temp = mean(data_subtracted_s_and_r, dims = 3)
    #calculate erp of S
    data_subtracted_c_and_r = subtract_to_data_epoched(data_reshaped, evts_s, cfg.s_range, [(evts_c, c_erp_temp, cfg.c_range), (evts_r, r_erp_temp, cfg.r_range)], cfg.sfreq)
    s_erp_temp = mean(data_subtracted_c_and_r, dims = 3)
    #calculate erp of R
    data_subtracted_s_and_c = subtract_to_data_epoched(data_reshaped, evts_r, cfg.r_range, [(evts_s, s_erp_temp, cfg.s_range), (evts_c, c_erp_temp, cfg.c_range)], cfg.sfreq)
    r_erp_temp = mean(data_subtracted_s_and_c, dims = 3)
    
    push!(figures_erp, plot_data_plus_component_erp(data_epoched, evts_s, evts_r, s_erp_temp, r_erp_temp, c_erp_temp, c_latencies, cfg))



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
    save("ride_erp.png", figures_erp[length(figures_erp)])

    return c_latencies, s_erp_temp, c_erp_temp, r_erp_temp
    
    #s_erp
    #r_erp
    #c_erp
    #c_latencies
    #results

end