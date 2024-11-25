
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
    width = 20,
    offset = 65,
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
    graph_s = vlines!(first(events_s[!,:latency],2), color = "red")
    
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
#todo: change the data input format to match the somewhat match the ride input format
#       Matrix Float64 with Channel:TimeStep:Trial

cfg = ride_config(
    sfreq = 100,
    s_range = [-0.2, 0.5],
    r_range = [0, 0.8],
    c_range = [-0.4, 0.4], # change to -0.4 , 0.4 or something because it's attached to the latency of C
    c_estimation_range = [0.2, 1.0],
    epoch_range = [-0.3,1.6],
    epoch_event_name = 'S',
)

epoch_length = round(Int, (cfg.epoch_range[2] - cfg.epoch_range[1]) * cfg.sfreq)

data_reshaped = reshape(data, (1,:))

#epoch data with the cfg.epoch_range to see how many epochs we have
#the resulting data_epoched is also used for the c latency estimation
data_epoched, data_epoched_times = Unfold.epoch(data = data_reshaped, tbl = evts_s, τ = cfg.epoch_range, sfreq = sfreq)
n ,data_epoched = Unfold.drop_missing_epochs(evts_s, data_epoched)
number_epochs = size(data_epoched, 3)

#epoch data for s
evts_s = @subset(evts, :event .== 'S')
data_epoched_s, data_epoched_s_times = Unfold.epoch(data = data_reshaped, tbl = evts_s, τ = cfg.s_range, sfreq = sfreq)
evts_s, data_epoched_s = Unfold.drop_missing_epochs(evts_s, data_epoched_s)
evts_s = evts_s[1:number_epochs,:]
#plotting one epoch, just for debugging
lines(data_epoched_s[1,:,1]; color = "black")
size(data_epoched_s)
size(evts_s)
#epoch data for r
evts_r = @subset(evts, :event .== 'R')
data_epoched_r, data_epoched_r_times = Unfold.epoch(data = data_reshaped, tbl = evts_r, τ = cfg.r_range, sfreq = sfreq)
evts_r, data_epoched_r = Unfold.drop_missing_epochs(evts_r, data_epoched_r)
evts_r = evts_r[1:number_epochs,:]
#plotting one epoch, just for debugging
lines(data_epoched_r[1,:,1]; color = "black")
size(data_epoched_r)





#findmax
#Initial Latency estimation of C

c_latencies = Matrix{Float64}(undef, 1, size(data_epoched, 3))
size(c_latencies)
#set start as the center point of the c_range
c_range_center = cfg.c_estimation_range[1] + ((cfg.c_estimation_range[2] - cfg.c_estimation_range[1]) / 2)
c_range_center_index = round(Int, c_range_center * sfreq)
index_max = round(Int, ((cfg.c_estimation_range[2]-cfg.c_estimation_range[1]) / 2) * sfreq)
#Peak estimation/algorithm
for a in (1:size(data_epoched, 3))
    max = 0;
    max_index = c_range_center_index;
    for b in (0:index_max)
        value_upwards = data_epoched[1, c_range_center_index + b, a]
        if(abs(value_upwards) > abs(max)) 
            max = value_upwards
            max_index = c_range_center_index + b
        end

        value_lower = data_epoched[1, c_range_center_index - b, a]
        if(abs(value_lower) > abs(max)) 
            max = value_lower
            max_index = c_range_center_index - b
        end
    end
    c_latencies[1,a] = max_index
end

#todo replacement for latency loop, suggessted by rene:
#(value, index) = findmax(data_epoched_s; 3)

#plotting the estimated c latencies on a couple of epochs
begin
    f = Figure()
    for a in 1:2, b in 1:2; 
        i = (a-1)*2 + b
        lines(f[a,b],data_epoched[1,:,i]; color = "black")
        vlines!(f[a,b],c_latencies[1,i]; color = "blue")
    end
    display(f)
end

#Create C event table by copying S and adding the estimated latency
evts_c = copy(evts_s)
evts_c[!,:latency] .= evts_s[!,:latency] + c_latencies[1,:] .+ (cfg.epoch_range[1]*sfreq)
evts_c[!,:event] .= 'C'

#plot one epoch of the entire epoch window, then one epoch of S,R and C
begin
    f = Figure()
    Axis(f[1,1],title = "data_epoched")
    lines!(f[1,1],data_epoched[1,:,1]; color = "black")
    Axis(f[1,2],title = "data_epoched_s")
    lines!(f[1,2],data_epoched_s[1,:,1]; color = "black")
    Axis(f[2,1],title = "data_epoched_r")
    lines!(f[2,1],data_epoched_r[1,:,1]; color = "black")
    Axis(f[2,2],title = "data_epoched_c")
    lines!(f[2,2],data_epoched_c[1,:,1]; color = "black")
    display(f)
end

#Combine the event tables for S, C and R
evts_s_c_r = vcat(evts_s, evts_c, evts_r)
sort!(evts_s_c_r, :latency)
evts_s_c_r

#exchange every char 'S' in the event column with the String "S"
#and every char 'R' with the String "R"
#this is needed for the fit function for unkown reasons
begin
    function convert(::Type{String}, input::Char)
        if input == 'S'
            return "S"
        elseif input == 'R'
            return "R"
        elseif input == 'C'
            return "C"
        end
    end
    function convert(::Type{String}, input::String)
        return input
    end
    evts_s_c_r[!,:event]  = convert.(String, evts_s_c_r[:,:event])
end

data_epoched_c, data_epoched_c_times = Unfold.epoch(data = data_reshaped, tbl = evts_c, τ = cfg.c_range, sfreq = sfreq)

#Epoch the data for the combined event table
#data_epoched, data_epoched_times = Unfold.epoch(data = data_reshaped, tbl = evts_s_c_r, τ = cfg.epoch_range, sfreq = sfreq)
#evts_s_c_r, data_epoched = Unfold.drop_missing_epochs(evts_s_c_r, data_epoched)

#m = fit(UnfoldModel,[
#    "S" => (@formula(0 ~ 1), data_epoched_times), #firbasis(cfg.s_range,sfreq,"")
#    "C" => (@formula(0 ~ 1), data_epoched_times),
#    "R" => (@formula(0 ~ 1), data_epoched_times) #firbasis(cfg.r_range,sfreq,"")
#    ],
#    evts_s_c_r,data_epoched)



#maybe replace fit with maxlength
#using Statistics
#?mean
#mean( ,dims=3)
#the paper actually uses median. Unfold fit wouldn't be correct anyways.
#the final step should then use mean instead of median


#calculate erp of S
s_erp_temp = median(data_epoched_s, dims = 3)
s_padding_front = zeros(Float64, 1, round(Int, (cfg.s_range[1] - cfg.epoch_range[1]) * sfreq), 1)
s_padding_back = zeros(Float64, 1, round(Int, (cfg.epoch_range[2] - cfg.s_range[2]) * sfreq), 1)
s_erp = hcat(s_padding_front, s_erp_temp, s_padding_back)

#calculate erp of C. This isn't part of the algorithm, should be replaced with c=data-s-r
c_erp_temp = median(data_epoched_c, dims = 3)
c_latencies_from_s_onset = zeros(Float64, 1, size(evts_c, 1))
c_latencies_from_s_onset[1,:] = round.(Int, evts_c.latency[:] - evts_s.latency[:])
c_median_latency_from_s_onset = round(Int, median(c_latencies_from_s_onset))
c_padding_front = zeros(Float64, 1, round(Int, c_median_latency_from_s_onset + (cfg.c_range[1] * cfg.sfreq) - (cfg.epoch_range[1] * cfg.sfreq)), 1)
c_padding_back = zeros(Float64, 1, epoch_length - size(c_padding_front, 2) - size(c_erp_temp, 2), 1)
c_erp = hcat(c_padding_front, c_erp_temp, c_padding_back)

#calculate erp of R
r_erp_temp = median(data_epoched_r, dims = 3)
r_latencies_from_s_onset = zeros(Float64, 1, size(evts_r, 1))
r_latencies_from_s_onset[1,:] = round.(Int, evts_r.latency[:] - evts_s.latency[:])
r_median_latency_from_s_onset = round(Int, median(r_latencies_from_s_onset))
r_padding_front = zeros(Float64, 1, round(Int, r_median_latency_from_s_onset + (cfg.r_range[1] * cfg.sfreq) - (cfg.epoch_range[1] * cfg.sfreq)), 1)
r_padding_back = zeros(Float64, 1, epoch_length - size(r_padding_front, 2) - size(r_erp_temp, 2), 1)
r_erp = hcat(r_padding_front, r_erp_temp, r_padding_back)

#S and R are generated and padded
#todo: calculate C with c=data-s-r
#todo: write inner iteration with convergence Check
#todo: write pattern_matching
#todo: write outer iteration, probably with C latency convergence check



#first(coeftable(m), 6)
#results = coeftable(m)
#s = results[results.eventname .== "S",:estimate]
##cut s to it's estimated range
#s_range_start = max(1, round(Int, (cfg.s_range[1] - cfg.epoch_range[1]) * sfreq))
#s_range_end =  min(round(Int,cfg.epoch_range[2] * sfreq), round(Int, (cfg.s_range[2] - cfg.epoch_range[1]) * sfreq))
#s_range_adjusted = [s_range_start, s_range_end]

#r = results[results.eventname .== "R",:estimate]
##adjust r_range to point to the correct region in the epoch_range
#r_range_start = max(1, round(Int, (cfg.r_range[1] - cfg.epoch_range[1]) * sfreq))
#r_range_end =  min(round(Int,cfg.epoch_range[2] * sfreq), round(Int, (cfg.r_range[2] - cfg.epoch_range[1]) * sfreq))
#r_range_adjusted = [r_range_start, r_range_end]

#c = results[results.eventname .== "C",:estimate]

f = lines(s_erp[1,:,1]; color = "red")
lines!(r_erp[1,:,1]; color = "blue")
lines!(c_erp[1,:,1]; color = "green")
display(f)

#f = plot_erp(results, axis = (title = "ERP after first model fit", xlabel = "Time", ylabel = "Amplitude"))
#display(f)

#pad resulting s c and r with zeros to the same length
#this allows for matrix subtraction in the next step with native julia methods

#evts_s = filter(row -> row.event in ['S'], evts)
#data_residual, data_residual_times = Unfold.epoch(data = data, tbl = evts_s, τ = cfg.epoch_range, sfreq = sfreq)
#evts_s, data_residual = Unfold.drop_missing_epochs(evts_s, data_residual)

#data_residual = copy(data)
#
#evts_s = filter(row -> row.event in ['S'], evts)
#for i in evts_s.latency
#    range = round(Int, (cfg.s_range[1] * cfg.sfreq) + i):round(Int, (cfg.s_range[2] * cfg.sfreq) + i)
#    data_residual[range] -= s[s_range_adjusted]
#end

#for i in 1:size(data_residual, 3)
#    data_residual[1,s_range_adjusted,i] -= s[s_range_adjusted]
#    data_residual[1,r_range_adjusted,i] -= r[r_range_adjusted]
#end
#data_residual = data_epoched.-s
#data_residual = data_residual.-r


#plotting the original data with the calculated residuals in green
#begin
#    f = Figure()
#    for a in 1:2, b in 1:2; 
#        i = (a-1)*2 + b
#        lines(f[a,b],data_epoched[1,:,i]; color = "black")
#        lines!(f[a,b],data_residual[1,:,i]; color = "green")
#    end
#    display(f)
#end


#predict_results = Unfold.predict(m,exclude = "C",epoch_to = "S", eventcolumn = :events, epoch_timewindow=[-0.20, 1.8],  overlap = false)   

#@show typeof(predict_results)
#plot_erp(predict_results, axis = (title = "Residue after first model fit", xlabel = "Time [s]", ylabel = "Amplitude [μV]"))
