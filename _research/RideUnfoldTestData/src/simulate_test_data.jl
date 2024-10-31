using UnfoldSim
using CairoMakie
using Random
using Unfold
using UnfoldMakie
using StableRNGs
using Parameters
using HDF5

design =
    SingleSubjectDesign(;
        conditions = Dict(
            :condition => ["car", "face"],
            :continuous => range(0, 1, length = 2),
        ),
    ) |> x -> RepeatDesign(x, 2);

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
    β = [-4],
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
    #PinkNoise,
);



#plot the graph
f = lines(vec(data); color = "black")
vlines!(reaction_times_graph, color = "blue")
#plot the conditions/components?
#vlines!(evts.latency; color = ["orange", "teal"][1 .+ (evts.condition.=="face")])

display(f)

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

println("success")


#m = fit(
#    UnfoldModel,
#    Dict(
#        Any => (
#            @formula(0 ~ 1 + condition + spl(continuous, 4)),
#            firbasis(τ = [-0.1, 1], sfreq = 100, name = "basis"),
#        ),
#    ),
#    evts,
#    data,
#);
#
#plot_erp(
#    coeftable(m);
#    axis = (
#        title = "Estimated regression parameters",
#        xlabel = "Time [s]",
#        ylabel = "Amplitude [μV]",
#    ),
#)