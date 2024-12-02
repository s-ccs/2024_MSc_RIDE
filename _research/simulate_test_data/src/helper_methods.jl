function pad_erp_to_epoch_size(erp, component_range, median_latency, cfg)
    padding_front = zeros(Float64, 1, max(round(Int, median_latency + (component_range[1] * cfg.sfreq) - (cfg.epoch_range[1] * cfg.sfreq)), 0), 1)
    padding_back = zeros(Float64, 1, max(epoch_length - size(padding_front, 2) - size(erp, 2), 0), 1) 
    return hcat(padding_front, erp, padding_back)
end

function subtract_to_data(data, others_evts_erp_tuples, sfreq)
    data_subtracted = copy(data)
    for (evts, erp, range) in others_evts_erp_tuples
        for i in evts.latency
            sub_range = i + round(Int, range[1] * sfreq) : i + round(Int, (range[1] * sfreq)) + size(erp[1,:,1])[1] - 1
            data_subtracted[1,sub_range] -= erp[1,:,1]
        end
    end
    return data_subtracted
end

function subtract_to_erp(data, target_evts, target_range, others_evts_erp_tuples, sfreq)
    data_subtracted = subtract_to_data(data, others_evts_erp_tuples, sfreq)
    data_epoched_subtracted, n = Unfold.epoch(data = data_subtracted, tbl = target_evts, τ = target_range, sfreq = sfreq)
    n, data_epoched_subtracted = Unfold.drop_missing_epochs(target_evts, data_epoched_subtracted)
    new_erp = median(data_epoched_subtracted, dims = 3)
    return new_erp
end

function findxcorrpeak(d,kernel;window=false)
	#the purpose of this method is to find the peak of the cross correlation between the kernel and the data
    #kernel = C component erp. Hanning is applied to factor the center of the C erp more than the edges.
	weightedkernel = window ? kernel .*  hanning(length(kernel)) : kernel
	xc = xcorr.(eachcol(d),Ref(weightedkernel); padmode = :none)
	m = [findmax(x)[2] for x in xc] .- (length(kernel))
	return xc,m
end


using Test
using DSP


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

if 1 == 1
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

if 1 == 1
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


#=
sdals
AbstractDesign
=#