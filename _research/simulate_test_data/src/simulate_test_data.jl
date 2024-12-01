
#import Pkg
#Pkg.add("CairoMakie")
#Pkg.add("Random")
#Pkg.add("Unfold")
#Pkg.add("UnfoldMakie")
#Pkg.add("StableRNGs")
#Pkg.add("Parameters")
#Pkg.add("HDF5")
#Pkg.add("DataFrames")
#Pkg.add("DataFramesMeta")
#Pkg.add([
#    Pkg.PackageSpec(url="https://github.com/unfoldtoolbox/UnfoldSim.jl.git", rev="v4.0")
#])

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

include("helper_methods.jl")

sfreq = 100

design =
    SingleSubjectDesign(;
        conditions = Dict(
            :condition => ["car", "face"],
            :continuous => range(0, 1, length = 2),
        ),
    ) |> x -> RepeatDesign(x, 20);

sequenceDesign = SequenceDesign(design, "SCR")

p1 = LinearModelComponent(;
    basis = p100(), 
    formula = @formula(0 ~ 1), 
    β = [1]
);

n1 = LinearModelComponent(;
    basis = n170(),
    formula = @formula(0 ~ 1 + condition),
    β = [1, 1],
);

p3 = LinearModelComponent(;
    basis = p300(),
    formula = @formula(0 ~ 1 + continuous), #+ continuous + continuous^2
    β = [5, 1],
);

n4 = LinearModelComponent(;
    basis = p100(),
    formula = @formula(0 ~ 1), #+ continuous + continuous^2
    β = [-5],
);

@with_kw struct DummySizeDesign <: AbstractDesign
    size = 0
end

function Base.size(design::DummySizeDesign)
    return design.size
end

onsetStimulus = UniformOnset(
    width = 0,
    offset = 200
)

onsetC = UniformOnset(
    width = 30,
    offset = 30,
)

onsetR = UniformOnset(
    width = 30,
    offset = 40, #15
)

@with_kw struct MultiOnset <: AbstractOnset
    stimulus_onset::AbstractOnset
    component_to_stimulus_onsets::Vector{AbstractOnset}
end

onset3 = MultiOnset(
    onsetStimulus,
    [onsetC, onsetR],
)

global stimulus_offset_accumulated = Vector{Int}
global reaction_times = Vector{Int}

function UnfoldSim.simulate_onsets(rng, onset::MultiOnset, simulation::Simulation) 
    design_size = size(simulation.design)
    number_of_components = length(onset.component_to_stimulus_onsets)
    divided_design_size = Int(ceil(design_size / (number_of_components + 1)))

    stimulus_offset = Vector{Int}
    component_offsets = Vector{Vector{Int}}()

    #calculate raw offsets
    stimulus_offset = simulate_interonset_distances(rng, onset.stimulus_onset, DummySizeDesign(divided_design_size))
    stimulus_offset_accumulated = accumulate(+, stimulus_offset, dims = 1, init = 1)
    for obj in onset.component_to_stimulus_onsets
        push!(component_offsets, simulate_interonset_distances(rng, obj, DummySizeDesign(divided_design_size)))
    end

    #combine the stimulus offsets and component offsets into one vector
    result = Vector{Int}()    
    for i in 1:divided_design_size
        current_offset = stimulus_offset_accumulated[i]
        push!(result, current_offset)
        for j in 1:length(component_offsets)
            push!(result, current_offset + component_offsets[j][i])
        end
    end

    #result can be filled with to many items due to rounding errors.
    #remove them until result matches the design size
    while design_size < length(result)
        deleteat!(result, length(result))
    end

    #todo: change the reaction times to point to the peak of the R component
    #grab reaction times
    global reaction_times = component_offsets[2]# .+ stimulus_offset_accumulated
    global reaction_times_graph = component_offsets[2] .+ stimulus_offset_accumulated
    global stimulus_offset_accumulated = stimulus_offset_accumulated

    return result
end

components = Dict('S' => [p1, n1], 'C' => [n4], 'R' => [p3])
onset2 = UniformOnset(
    width = 0,
    offset = 60,
)
data, evts = simulate(
    RandomDevice(),
    sequenceDesign,
    components,
    onset3,
    PinkNoise(),
);


#plot a few data points with the first 2 reaction times and stimulus onsets
begin
    f = Figure()
    Axis(f[1,1], title = "First 600 data points")
    graph = lines!(first(vec(data),600); color = "black")
    graph_events_s = @subset(evts, :event .== 'S')
    graph_rt = vlines!(first(reaction_times_graph,2), color = "blue")
    graph_s = vlines!(first(graph_events_s[!,:latency],2), color = "red")
    
    Legend(f[1,2]
        , [graph, graph_rt, graph_s]
        , ["Data", "Reaction Times", "Stimulus Onsets"]
    )
    display(f)
end


#Ride input format Matrix Int:Int:Int with TimeStep:Channel:Trial
ride_matrix = zeros(Float64, onsetStimulus.offset, 1, length(stimulus_offset_accumulated)) # Vector{Vector{Vector{Float64}}}()


for i in 1:onsetStimulus.offset
    for (n,obj) in enumerate(stimulus_offset_accumulated)
        ride_matrix[i,1,n] = data[min(obj+i,length(data))]
    end
end


h5open("simulated_data.h5", "w") do file
    # Save the 3D array
    write(file, "dataset_data", ride_matrix)
    
    # Save the vector rt
    write(file, "dataset_rt", Float64.(reaction_times))
end

println("data generation: success")

show(first(evts, 6), allcols = true)

#End of data generation ######################################################################################

@with_kw struct ride_config
    sfreq::Int
    s_range::Vector{Float64}
    r_range::Vector{Float64}
    c_range::Vector{Float64}
    c_estimation_range::Vector{Float64}
    epoch_range::Vector{Float64}
    epoch_event_name::Char
end

#todo: define the reaction times as a separate input parameter instead of as part of the evts table
#todo: change the data input format to somewhat match the ride input format
#       Matrix Float64 with Channel:TimeStep:Trial

cfg = ride_config(
    sfreq = 100,
    s_range = [-0.2, 0.5],
    r_range = [0, 0.8],
    c_range = [-0.6, 0.6], # change to -0.4 , 0.4 or something because it's attached to the latency of C
    c_estimation_range = [0.2, 1.0],
    epoch_range = [-0.3,1.6],
    epoch_event_name = 'S',
)

epoch_length = round(Int, (cfg.epoch_range[2] - cfg.epoch_range[1]) * cfg.sfreq)
data_reshaped = reshape(data, (1,:))
evts_s = @subset(evts, :event .== 'S')
evts_r = @subset(evts, :event .== 'R')

#epoch data with the cfg.epoch_range to see how many epochs we have
#the resulting data_epoched is also used for the c latency estimation
data_epoched, data_epoched_times = Unfold.epoch(data = data_reshaped, tbl = evts_s, τ = cfg.epoch_range, sfreq = sfreq)
n ,data_epoched = Unfold.drop_missing_epochs(evts_s, data_epoched)
number_epochs = size(data_epoched, 3)

#epoch data for s
data_epoched_s, data_epoched_s_times = Unfold.epoch(data = data_reshaped, tbl = evts_s, τ = cfg.s_range, sfreq = sfreq)
evts_s, data_epoched_s = Unfold.drop_missing_epochs(evts_s, data_epoched_s)
evts_s = evts_s[1:number_epochs,:]

#epoch data for r
data_epoched_r, data_epoched_r_times = Unfold.epoch(data = data_reshaped, tbl = evts_r, τ = cfg.r_range, sfreq = sfreq)
evts_r, data_epoched_r = Unfold.drop_missing_epochs(evts_r, data_epoched_r)
evts_r = evts_r[1:number_epochs,:]

#Peak estimation/algorithm for initial c latencies
c_latencies = Matrix{Float64}(undef, 1, size(data_epoched, 3))
for a in (1:size(data_epoched, 3))
    range = round.(Int, cfg.c_estimation_range[1] * sfreq) : round(Int, cfg.c_estimation_range[2] * sfreq)
    c_latencies[1,a] = (findmax(abs.(data_epoched[1,range,a])) .+ range[1] .- 1)[2]
end

#Create C event table by copying S and adding the estimated latency
evts_c = copy(evts_s)
evts_c[!,:latency] .= round.(Int, evts_s[!,:latency] + c_latencies[1,:] .+ (cfg.epoch_range[1]*sfreq))
evts_c[!,:event] .= 'C'


plot_first_epoch()

#prepare data for the ride algorithm
s_erp_temp = median(data_epoched_s, dims = 3)
r_erp_temp = median(data_epoched_r, dims = 3)
figures_latency = Array{Figure,1}()
push!(figures_latency, plot_c_latency_estimation_four_epochs())
figures_erp = Array{Figure,1}()

#outer iteration RIDE
for i in 1:5
    #inner iteration RIDE
    for j in 1:25
        #calculate erp of C by subtracting S and R from the data
        global c_erp_temp = subtract_to_erp(data_reshaped, evts_c, cfg.c_range, [(evts_s, s_erp_temp, cfg.s_range), (evts_r, r_erp_temp, cfg.r_range)])
        #calculate erp of S
        global s_erp_temp = subtract_to_erp(data_reshaped, evts_s, cfg.s_range, [(evts_c, c_erp_temp, cfg.c_range), (evts_r, r_erp_temp, cfg.r_range)])
        #calculate erp of R
        global r_erp_temp = subtract_to_erp(data_reshaped, evts_r, cfg.r_range, [(evts_s, s_erp_temp, cfg.s_range), (evts_c, c_erp_temp, cfg.c_range)])
    end
    
    push!(figures_erp, plot_data_plus_component_erp())

    data_epoched_subtracted_s_and_r = data_epoched
    #data_epoched_subtracted_s_and_r = subtract_to_data_epoched(data_reshaped, evts_s, cfg.epoch_range, [(evts_s, s_erp_temp, cfg.s_range), (evts_r, r_erp_temp, cfg.r_range)])
    #new_erp = median(data_epoched_subtracted_s_and_r, dims = 3)
    #pattern matching to update the c latencies
    global xc, m = findxcorrpeak(data_epoched_subtracted_s_and_r[1,:,:],c_erp_temp[1,:,1])
    @show global c_latencies = reshape(m .- round(Int,  (cfg.c_range[1] * cfg.sfreq)), (1,:))
    global evts_c = copy(evts_s)
    global evts_c[!,:latency] .= round.(Int, evts_s[!,:latency] + c_latencies[1,:] .+ (cfg.epoch_range[1]*sfreq))
    global evts_c[!,:event] .= 'C'
    push!(figures_latency, plot_c_latency_estimation_four_epochs())
end



#todo: the final step should use mean instead of median to calculate the erp

#plot the estimated c latencies for each iteration
for (i,f) in enumerate(figures_latency)
    Label(f[0, :], text = "Estimated C latency, Iteration $(i-1)", halign = :center)
    display(f)
end
#plot the calculated erp for each iteration
for (i,f) in enumerate(figures_erp)
    Label(f[0, :], text = "Calculated Erp, Iteration $i")
    display(f)
end

#weightedkernel = hanning(50)
#aa = hanning(100)
#xc_temp = xcorr.(Ref(weightedkernel),Ref(aa); padmode = :none)
#
#range_start = round(Int, length(weightedkernel)/2  + 1)
#range_stop = range_start + length(aa) - 1
#xc = xc_temp[range_start:range_stop]
#
#res = findmax(abs.(xc))