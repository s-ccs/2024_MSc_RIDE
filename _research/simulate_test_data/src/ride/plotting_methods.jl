function plot_c_latency_estimation_four_epochs(data_epoched, c_latencies)
    #plotting the estimated c latencies on a couple of epochs
    f = Figure()
    #c_latency_offset = round(Int, ((cfg.c_range[2] - cfg.c_range[1]) * cfg.sfreq) / 2);
    for a in 1:2, b in 1:2; 
        i = (a-1)*2 + b
        Axis(f[a,b],title = "Estimated C latency epoch $i")
        lines!(f[a,b],data_epoched[1,:,i]; color = "black")
        vlines!(f[a,b],c_latencies[1,i]; color = "blue")
    end
    return f
end

function plot_first_epoch(cfg, evts_s, evts_r, evts_c, data_reshaped)
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

function plot_data_plus_component_erp(data_epoched, evts_s, evts_r, s_erp_temp, r_erp_temp, c_erp_temp, c_latencies, cfg)
    s_erp_padded = pad_erp_to_epoch_size(s_erp_temp, cfg.s_range, 0, cfg)

    #calculate the median latency of R from S onset
    r_latencies_from_s_onset = zeros(Float64, 1, size(evts_r, 1))
    r_latencies_from_s_onset[1,:] = round.(Int, evts_r.latency[:] - evts_s.latency[:])
    r_median_latency_from_s_onset = round(Int, median(r_latencies_from_s_onset))
    r_erp_padded = pad_erp_to_epoch_size(r_erp_temp, cfg.r_range, r_median_latency_from_s_onset, cfg)
    
    c_median_latency = round(Int, median(c_latencies) + (cfg.epoch_range[1] * cfg.sfreq))
    c_erp_padded = pad_erp_to_epoch_size(c_erp_temp, cfg.c_range, c_median_latency, cfg)
    raw_erp = mean(data_epoched, dims = 3)
    
    f = Figure()
    Axis(f[1,0], yticks = -100:100)
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