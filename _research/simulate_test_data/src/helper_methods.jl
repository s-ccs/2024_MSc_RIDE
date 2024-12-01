function pad_erp_to_epoch_size(erp, component_range, median_latency, cfg)
    padding_front = zeros(Float64, 1, max(round(Int, median_latency + (component_range[1] * cfg.sfreq) - (cfg.epoch_range[1] * cfg.sfreq)), 0), 1)
    padding_back = zeros(Float64, 1, max(epoch_length - size(padding_front, 2) - size(erp, 2), 0), 1) 
    return hcat(padding_front, erp, padding_back)
end

function plot_c_latency_estimation_four_epochs()
    #plotting the estimated c latencies on a couple of epochs
    f = Figure()
    for a in 1:2, b in 1:2; 
        i = (a-1)*2 + b
        Axis(f[a,b],title = "Estimated C latency epoch $i")
        lines!(f[a,b],data_epoched[1,:,i]; color = "black")
        vlines!(f[a,b],c_latencies[1,i]; color = "blue")
    end
    return f
end

function plot_first_epoch()
    #plot one epoch of the entire epoch window, then one epoch of S,R and C
    f = Figure()
    Axis(f[1,1],title = "data_epoched")
    epoch_range =  round.(Int, cfg.epoch_range .* cfg.sfreq .+ evts_s.latency[1])
    lines!(f[1,1],data_reshaped[1,epoch_range[1]:epoch_range[2]]; color = "black")
    Axis(f[1,2],title = "data_epoched_s")
    s_range = round.(Int, cfg.s_range .* cfg.sfreq .+ evts_s.latency[1])
    lines!(f[1,2],data_reshaped[1,s_range[1]:s_range[2]]; color = "red")
    Axis(f[2,1],title = "data_epoched_r")
    r_range = round.(Int, cfg.r_range .* cfg.sfreq .+ evts_r.latency[1])
    lines!(f[2,1],data_reshaped[1,r_range[1]:r_range[2]]; color = "blue")
    Axis(f[2,2],title = "data_epoched_c")
    c_range = round.(Int, cfg.c_range .* cfg.sfreq .+ evts_c.latency[1])
    lines!(f[2,2],data_reshaped[1,c_range[1]:c_range[2]]; color = "green")
    display(f)
end

function plot_data_plus_component_erp()
    s_erp_padded = pad_erp_to_epoch_size(s_erp_temp, cfg.s_range, 0, cfg)

    #calculate the median latency of R from S onset
    r_latencies_from_s_onset = zeros(Float64, 1, size(evts_r, 1))
    r_latencies_from_s_onset[1,:] = round.(Int, evts_r.latency[:] - evts_s.latency[:])
    r_median_latency_from_s_onset = round(Int, median(r_latencies_from_s_onset))
    r_erp_padded = pad_erp_to_epoch_size(r_erp_temp, cfg.r_range, r_median_latency_from_s_onset, cfg)
    
    c_median_latency = round(Int, median(c_latencies) + (cfg.epoch_range[1] * cfg.sfreq))
    c_erp_padded = pad_erp_to_epoch_size(c_erp_temp, cfg.c_range, c_median_latency, cfg)
    raw_erp = median(data_epoched, dims = 3)
    
    f = Figure()
    Axis(f[1,0])
    raw = lines!(raw_erp[1,:,1]; color = "black", linewidth = 3)
    s = lines!(s_erp_padded[1,:,1]; color = "red")
    r = lines!(r_erp_padded[1,:,1]; color = "blue")
    c = lines!(c_erp_padded[1,:,1]; color = "green")
    Legend(f[1,1]
        , [raw, s, r, c]
        , ["Raw ERP", "S ERP", "R ERP", "C ERP"]
    )
    return f
end

function subtract_to_data_epoched(data, target_evts, target_range, others_evts_erp_tuples)
    data_subtracted = copy(data)
    for (evts, erp, range) in others_evts_erp_tuples
        for i in evts.latency
            sub_range = i + round(Int, range[1] * sfreq) : round(Int, i + (range[1] * sfreq) + size(erp[1,:,1])[1] - 1)
            data_subtracted[1,sub_range] -= erp[1,:,1]
        end
    end    
    data_epoched_subtracted, n = Unfold.epoch(data = data_subtracted, tbl = target_evts, τ = target_range, sfreq = sfreq)
    n, data_epoched_subtracted = Unfold.drop_missing_epochs(target_evts, data_epoched_subtracted)
    return data_epoched_subtracted
end

function subtract_to_erp(data, target_evts, target_range, others_evts_erp_tuples)
    data_epoched_subtracted = subtract_to_data_epoched(data, target_evts, target_range, others_evts_erp_tuples)
    new_erp = median(data_epoched_subtracted, dims = 3)
    return new_erp
end

function findxcorrpeak(d,kernel;window=true)
	# the purpose of this method is to find the peak of the cross correlation between the kernel and the data

    #kernel = C component erp. Hanning is applied to factor the center of the C erp more than the edges.
	weightedkernel = window ? kernel .*  hanning(length(kernel)) : kernel
    
	xc = xcorr.(eachcol(d),Ref(weightedkernel); padmode = :none)
    #range_start = round(Int, length(weightedkernel)/2  + 1)
    #range_stop = range_start + size(d, 1) - 1 
    #@show size(xc_temp)
    #@show xc = xc_temp[:][range_start:range_stop]
    #@show size(d, 1)
    #m = zeros(size(xc))
    #for (i,x) in enumerate(xc)
    #    m[i] = findmax(abs.(x))[2] - (length(x)+1)/2 - 1
    #end
	m = [findmax(x)[2] for x in xc] .- (length(kernel))
	return xc,m
end


using Test
#----
if 1 == 1
d = UnfoldSim.pad_array(hanning(10),-35,0)
kernel = hanning(20)

f = Figure()
lines(f[1,1],d)
lines!(kernel)
xc,m = findxcorrpeak(d,kernel)
lines(f[1,2],xc[1])

lines(f[2,1],d)
vlines!([m[1]])
lines!(m[1].+(1:length(kernel)),kernel)

display(f)
using Test
@test findxcorrpeak(d,kernel)[2] == [30]
@test findxcorrpeak(d,kernel;window=false)[2] == [30]
end

#=

sdals
AbstractDesign
=#