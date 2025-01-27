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

@with_kw mutable struct simulation_inputs
    rng::AbstractRNG = MersenneTwister(1234)
    noise::AbstractNoise = NoNoise()
    s_width::Int = 0
    s_offset::Int = 200
    s_beta::Int = 1
    s_continous::Int = 1
    c_width::Int = 30
    c_offset::Int = 30
    c_beta::Int = 5
    r_width::Int = 60
    r_offset::Int = 15
    r_beta::Int = 5
    r_continous::Int = 1
end

function simulate_default_plus_clean(simulation_inputs = simulation_inputs())
    data, evts = default_sequence_design(simulation_inputs)

    clean_inputs = deepcopy(simulation_inputs)
    clean_inputs.s_offset = clean_inputs.s_offset + Int(round(clean_inputs.s_width / 2))
    clean_inputs.s_width = 0
    clean_inputs.r_offset = clean_inputs.r_offset + Int(round(clean_inputs.r_width / 2))
    clean_inputs.r_width = 0
    clean_inputs.c_offset = clean_inputs.c_offset + Int(round(clean_inputs.c_width / 2))
    clean_inputs.c_width = 0
    data_clean, evts_clean = default_sequence_design(clean_inputs)

    s_clean_inputs = deepcopy(clean_inputs)
    s_clean_inputs.r_beta = 0
    s_clean_inputs.r_continous = 0
    s_clean_inputs.c_beta = 0
    data_clean_s, n = default_sequence_design(s_clean_inputs)

    r_clean_inputs = deepcopy(clean_inputs)
    r_clean_inputs.s_beta = 0
    r_clean_inputs.s_continous = 0
    r_clean_inputs.c_beta = 0
    data_clean_r, n = default_sequence_design(r_clean_inputs)

    c_clean_inputs = deepcopy(clean_inputs)
    c_clean_inputs.s_beta = 0
    c_clean_inputs.s_continous = 0
    c_clean_inputs.r_beta = 0
    c_clean_inputs.r_continous = 0
    data_clean_c, n = default_sequence_design(c_clean_inputs)

    return data, evts, data_clean, evts_clean, data_clean_s, data_clean_r, data_clean_c
end

function default_sequence_design(simulation_inputs = simulation_inputs())
    # Define the design
    design = SingleSubjectDesign(;
        conditions = Dict(
            :condition => ["car", "face"],
            :continuous => range(0, 1, length = 2),
        ),
    ) |> x -> RepeatDesign(x, 20);

    sequence_design = SequenceDesign(design, "SCR")

    # Define the components
    s_component_p = LinearModelComponent(;
        basis = p100(), 
        formula = @formula(0 ~ 1), 
        β = [simulation_inputs.s_beta]
    );

    s_component_n = LinearModelComponent(;
        basis = n170(),
        formula = @formula(0 ~ 1 + condition),
        β = [simulation_inputs.s_beta, simulation_inputs.s_continous],
    );

    r_component = LinearModelComponent(;
        basis = p300(),
        formula = @formula(0 ~ 1 + continuous),
        β = [simulation_inputs.r_beta, simulation_inputs.r_continous],
    );

    c_component = LinearModelComponent(;
        basis = p100(),
        formula = @formula(0 ~ 1), 
        β = [simulation_inputs.c_beta],
    );

    onsetStimulus = UniformOnset(
        width = simulation_inputs.s_width,
        offset = simulation_inputs.s_offset,
    )

    onsetC = UniformOnset(
        width = simulation_inputs.c_width,
        offset = simulation_inputs.c_offset,
    )

    onsetR = UniformOnset(
        width = simulation_inputs.r_width,
        offset = simulation_inputs.r_offset,
    )

    multi_onset = MultiOnset(
        onsetStimulus,
        [onsetC, onsetR],
    )

    components = Dict('S' => [s_component_p, s_component_n], 'C' => [c_component], 'R' => [r_component])

    data, evts = simulate(
        simulation_inputs.rng,
        sequence_design,
        components,
        multi_onset,
        simulation_inputs.noise,
    )

    return data, evts
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