
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

@with_kw struct MultiOnset <: AbstractOnset
    stimulus_onset::AbstractOnset
    component_to_stimulus_onsets::Vector{AbstractOnset}
end

function default_sequence_design(s_width = 0, s_offset = 200, s_beta = 1 , c_width = 30, c_offset = 30, c_beta = 5, r_width = 60, r_offset = 15, r_beta = 5, continous_s = 1, continous_r = 1)
    # Define the design
    design = SingleSubjectDesign(;
        conditions = Dict(
            :condition => ["car", "face"],
            :continuous => range(0, 1, length = 2),
        ),
    ) |> x -> RepeatDesign(x, 20);

    sequence_design = SequenceDesign(design, "SCR")

    # Define the components
    p1 = LinearModelComponent(;
        basis = p100(), 
        formula = @formula(0 ~ 1), 
        β = [s_beta]
    );

    n1 = LinearModelComponent(;
        basis = n170(),
        formula = @formula(0 ~ 1 + condition),
        β = [s_beta, continous_s],
    );

    p3 = LinearModelComponent(;
        basis = p300(),
        formula = @formula(0 ~ 1 + continuous),
        β = [c_beta, continous_r],
    );

    n4 = LinearModelComponent(;
        basis = p100(),
        formula = @formula(0 ~ 1), 
        β = [r_beta],
    );

    onsetStimulus = UniformOnset(
        width = s_width,
        offset = s_offset,
    )

    onsetC = UniformOnset(
        width = c_width,
        offset = c_offset,
    )

    onsetR = UniformOnset(
        width = r_width,
        offset = r_offset,
    )

    multi_onset = MultiOnset(
        onsetStimulus,
        [onsetC, onsetR],
    )

    components = Dict('S' => [p1, n1], 'C' => [n4], 'R' => [p3])

    return sequence_design, components, multi_onset
end

@with_kw struct DummySizeDesign <: AbstractDesign
    size = 0
end

function Base.size(design::DummySizeDesign)
    return design.size
end

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

    #result can be filled with too many items due to rounding errors.
    #remove them until result matches the design size
    while design_size < length(result)
        deleteat!(result, length(result))
    end

    #todo: change the reaction times to point to the peak of the R component
    return result
end

function save_to_hdf5_ride_format(data, evts, epoch_range, epoch_char, reaction_char, sfreq)
    evts_epoch_temp = @subset(evts, :event .== epoch_char)
    data_epoched_temp, times = Unfold.epoch(data = data, tbl = evts_epoch_temp, τ = epoch_range, sfreq = sfreq)
    evts_epoch, data_epoched = Unfold.drop_missing_epochs(evts_epoch_temp, data_epoched_temp)

    #grab the reaction times from the epoched data
    evts_r = @subset(evts, :event .== reaction_char)[!,:latency][1:size(data_epoched, 3)]
    reaction_times = evts_r - evts_epoch[!,:latency]

    #matlab_ride format: Matrix{Float64} with TimeStep:Channel:Trial
    ride_matrix = zeros(Float64, size(data_epoched, 2), size(data_epoched, 1), size(data_epoched, 3))

    #fill the ride_matrix with the data
    for x in axes(ride_matrix, 1)
        for y in axes(ride_matrix, 2)
            for z in axes(ride_matrix, 3)
                ride_matrix[x,y,z] = data_epoched[y,x,z]
            end
        end
    end

    #todo maybe create chanlocs and save them here
    
    h5open("simulated_data.h5", "w") do file
        # Save the 3D array
        write(file, "dataset_data", ride_matrix)
        
        # Save the vector rt
        write(file, "dataset_rt", Float64.(reaction_times))
    end

end