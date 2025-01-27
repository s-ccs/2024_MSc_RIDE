abstract type ride_modus end

struct ride_original <: ride_modus end

struct ride_unfold <: ride_modus end

function subtract_to_data(data, others_evts_erp_tuples, sfreq)
    data_subtracted = copy(data)
    for (evts, erp, range) in others_evts_erp_tuples
        for i in evts.latency
            sub_range = i + round(Int, range[1] * sfreq) : i + round(Int, (range[1] * sfreq)) + size(erp[1,:,1])[1] - 1
            if sub_range[end] > length(data_subtracted)
                continue
            end
            data_subtracted[1,sub_range] -= erp[1,:,1]
        end
    end
    return data_subtracted
end

function subtract_to_data_epoched(data, target_evts, target_range, others_evts_erp_tuples, sfreq)
    data_subtracted = subtract_to_data(data, others_evts_erp_tuples, sfreq)
    data_epoched_subtracted, n = Unfold.epoch(data = data_subtracted, tbl = target_evts, τ = target_range, sfreq = sfreq);
    n, data_epoched_subtracted = Unfold.drop_missing_epochs(target_evts, data_epoched_subtracted)
    #new_erp = median(data_epoched_subtracted, dims = 3)
    return data_epoched_subtracted
end

function findxcorrpeak(d,kernel;window=false)
	#the purpose of this method is to find the peak of the cross correlation between the kernel and the data
    #kernel = C component erp. Hanning is applied to factor the center of the C erp more than the edges.
	weightedkernel = window ? kernel .*  hanning(length(kernel)) : kernel
	xc = xcorr.(eachcol(d),Ref(weightedkernel); padmode = :none)
    #xc_cut = []
    #for x in xc
    #    push!(xc_cut, x[length(kernel)+1:end])
    #end
    onset = length(kernel)
	m = [findmax(x)[2] for x in xc] .- onset
	return xc, m, onset
end


function initial_peak_estimation(data_residuals_continous, evts_s, cfg)
    ## initial C latency estimation
    data_residuals_epoched, times = Unfold.epoch(data = data_residuals_continous, tbl = evts_s, τ = cfg.epoch_range, sfreq = cfg.sfreq)
    n, data_residuals_epoched = Unfold.drop_missing_epochs(evts_s, data_residuals_epoched)
    #Peak estimation/algorithm for initial c latencies
    c_latencies = Matrix{Float64}(undef, 1, size(data_residuals_epoched, 3))
    for a in (1:size(data_residuals_epoched, 3))
        range = round.(Int, (cfg.c_estimation_range[1] - cfg.epoch_range[1]) * cfg.sfreq) : round(Int, (cfg.c_estimation_range[2] - cfg.epoch_range[1]) * cfg.sfreq)
        c_latencies[1,a] = (findmax(abs.(data_residuals_epoched[1,range,a])) .+ range[1] .- 1)[2] + round(Int, cfg.c_range[1] * cfg.sfreq)
    end
    latencies_df = DataFrame(latency = c_latencies[1,:], fixed = false)
    return latencies_df
end

function c_range_adjusted(c_range::Vector{Float64})
    return [0, c_range[2] - c_range[1]]
end

#Create C event table by copying S and adding the estimated latency
function build_c_evts_table(latencies_df, evts)
    evts_s = @subset(evts, :event .== 'S')
    evts_c = copy(evts_s)
    evts_c[!,:latency] .= round.(Int, evts_s[!,:latency] + latencies_df[!,:latency] .+ (cfg.epoch_range[1]*cfg.sfreq))
    evts_c[!,:event] .= 'C'
    return evts_c
end



using Test
using DSP
using UnfoldSim
using CairoMakie


if 1 == 1
    d = UnfoldSim.pad_array(hanning(10),-35,0)
    kernel = hanning(20)

    f = Figure()
    lines(f[1,1],d)
    lines!(kernel)
    xc, m, onset = findxcorrpeak(d,kernel)
    lines(f[1,2],xc[1])

    lines(f[2,1],d)
    vlines!([m[1]])
    lines!(m[1].+(1:length(kernel)),kernel)

    display(f)
    using Test
    @test findxcorrpeak(d,kernel)[2] == [30]
    @test findxcorrpeak(d,kernel;window=true)[2] == [30]
end

function createTestData()
    design = SingleSubjectDesign(;
        conditions = Dict(
            :condA => ["LevelA"],
        ),
    ) |> x -> RepeatDesign(x, 5);
    p1 = LinearModelComponent(;
        basis = hanning(100), 
        formula = @formula(0 ~ 1), 
        β = [1]
    );
    onset = UniformOnset(
        width = 0,
        offset = 200,
    );
    data, evts = simulate(
        MersenneTwister(1),
        design,
        [p1],
        onset,
    );
    return data, evts
end

if 1 == 0
    sfreq = 100

    data = reshape(vcat(zeros(100), hanning(100), zeros(100), hanning(100), zeros(100)), (1,:))
    evts = DataFrame(:event => ['B','B'], :latency => [101,301])
    range_test = [0.0, 1.0]

    erp_to_subtract = reshape(hanning(100),(1,:,1))

    result_zero = subtract_to_data(data, [(evts, erp_to_subtract, range_test)], sfreq)

    f = Figure()
    lines(f[1,1], data[1,:])
    lines(f[1,2], erp_to_subtract[1,:])
    lines(f[2,1], result_zero[1,:])
    display(f)

    @test result_zero[1,:] == zeros(length(result_zero[1,:]))
end

if 1 == 0
    sfreq = 100
    data, evts = createTestData()
    range_test = [0.0, 1.0]
    data = reshape(data, (1,:))

    erp_to_subtract = reshape(hanning(100),(1,:,1))

    result_zero = subtract_to_data(data, [(evts, erp_to_subtract, range_test)], sfreq)

    f = Figure()
    lines(f[1,1], data[1,:])
    lines(f[1,2], erp_to_subtract[1,:])
    lines(f[2,1], result_zero[1,:])
    display(f)
    
    @test result_zero[1,:] == zeros(length(result_zero[1,:]))
end


using Peaks
using Statistics
using Distributions
using DataFrames
using Random
using Parameters
include("./ride_structs.jl")


# check for multiple "competing" peaks in the xcorrelation
# any peak with a value > maximum * equality_threshold is considered a competing peak
# the peak closest to the previous latency is chosen
function heuristic3(latencies_df, latencies_df_old, xcorr, equality_threshold::Float64; onset::Int64 = 0)
    @assert size(latencies_df,1) == size(xcorr,1) "latencies_df and xcorr must have the same size"
    @assert size(latencies_df_old,1) == size(latencies_df,1) "latencies_df and latencies_df_old must have the same size"
    @assert equality_threshold > 0 && equality_threshold <= 1 "equality_threshold must be between 0 and 1"
    for (i,row) in enumerate(eachrow(latencies_df_old))
        if row.fixed continue end
        maxima = findmaxima(xcorr[i])
        if isempty(maxima) continue end
        maximum_value = findmax(maxima.heights)[1]
        competing_peaks = []
        for (j,peak) in enumerate(maxima.indices)
            if xcorr[i][peak] > maximum_value * equality_threshold
                push!(competing_peaks, peak - onset)
            end
        end
        if isempty(competing_peaks) continue end
        closest_peak = argmin(abs.(competing_peaks .- row.latency))
        latencies_df.latency[i] = competing_peaks[closest_peak]
    end
end

# check if the xcorrelation is convex by searching for peaks
# when no peak is found, the xcorrelation is considered convex and 
# the latency is randomized with a gaussian distribution over the previous latencies
function heuristic2(latencies_df, latencies_df_old, xcorr, rng = MersenneTwister(1234))
    @assert size(latencies_df,1) == size(xcorr,1) "latencies_df and xcorr must have the same size"
    ##you cannot calculate a standard deviation with less than 2 values
    standard_deviation = std(latencies_df_old.latency)
    if(isnan(standard_deviation))
        standard_deviation = 1
    end
    normal_distribution = Normal(standard_deviation, mean(latencies_df_old.latency))
    for (i,row) in enumerate(eachrow(latencies_df))
        if row.fixed continue end
        maxima = findmaxima(xcorr[i])
        maxima_length = length(maxima.indices)
        if maxima_length == 0
            row.latency = round(Int, rand(rng, normal_distribution))
            row.fixed = true
            @debug "heuristic2 randomized latency for epoch $i"
        end
    end
end

# make sure the changes in the latencies are monoton
# if a non monoton change is detected, revert the change and set the latency as fixed
function heuristic1(latencies_df, latencies_df_old, latencies_df_old_old)
    for (i,row) in enumerate(eachrow(latencies_df))
        if row.fixed continue end
        prev_change = latencies_df_old.latency[i] - latencies_df_old_old.latency[i]
        new_change = row.latency - latencies_df_old.latency[i]
        if prev_change > 0 && new_change < 0 || prev_change < 0 && new_change > 0
            row.latency = latencies_df_old.latency[i]
            row.fixed = true
            @debug "heuristic1 reverted non-monoton latency change for epoch $i"
        end
    end
end

## test heuristic3
if 1 == 1
    cfg = ride_config(
        sfreq = 100,
        c_range = [-0.4, 0.4],
        s_range = [0, 0],
        r_range = [0, 0],
        c_estimation_range = [0.2, 1.2],
        epoch_range = [-0.3,1.6],
        epoch_event_name = 'S',
    )
    #identical epochs with perfect match at 100 and subpar match at 300
    epoch1 = reshape(vcat(zeros(100), hanning(100), zeros(100), hanning(100).*0.9, zeros(100)), (1,:,1))
    epoch2 = reshape(vcat(zeros(100), hanning(100), zeros(100), hanning(100).*0.9, zeros(100)), (1,:,1))
    data_epoched = cat(epoch1, epoch2, dims=3)

    #same latency for both epochs
    latencies_df = DataFrame(latency = [100,100], fixed = [false, false])
    #201 is closer to the subpar 300 peak in the previous latencies, should trigger heuristic
    latencies_df_old = DataFrame(latency = [201,200], fixed = [false, false])
    xc, xc_values, onset = findxcorrpeak(data_epoched[1,:,:], hanning(100))

    heuristic3(latencies_df, latencies_df_old, xc, 0.8, onset=onset)

    @test latencies_df.latency == [300, 100]
    @test latencies_df.fixed == [false, false]
end

## test heuristic2
if 1 == 1
    epoch1 = reshape(vcat(zeros(100), hanning(100) .* -1), (1,:,1))
    epoch2 = reshape(vcat(zeros(100), hanning(100)), (1,:,1))
    data_epoched = cat(epoch1, epoch2, dims=3)

    xc, xc_values = findxcorrpeak(data_epoched[1,:,:], hanning(100))
    latencies_df_old = copy(latencies_df)
    latencies_df.latency = xc_values

    latencies_df = DataFrame(latency = [0,100], fixed = [false, false])
    latencies_df_old = DataFrame(latency = [50,60], fixed = [false, false])


    xc = [hanning(100) .* -1, hanning(100)]

    rng = MersenneTwister(9876)
    heuristic2(latencies_df, latencies_df_old, xc, rng)

    @test latencies_df.latency == [64,100]
    @test latencies_df.fixed == [true, false]
end

## test heuristic1
if 1 == 1
    latencies_df = DataFrame(latency = [75,33], fixed = [false, false])
    latencies_df_old = DataFrame(latency = [70,30], fixed = [false, false])
    latencies_df_old_old = DataFrame(latency = [50,40], fixed = [false, false])

    heuristic1(latencies_df, latencies_df_old, latencies_df_old_old)

    @test latencies_df.latency == [75,30]
    @test latencies_df.fixed == [false, true]
end

