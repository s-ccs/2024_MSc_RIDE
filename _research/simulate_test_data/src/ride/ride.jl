module Ride

using Revise

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

export ride_algorithm, ride_algorithm_unfold, ride_config, ride_original, ride_unfold
includet("./helper_methods.jl")
includet("./ride_algorithm.jl")
includet("./ride_algorithm_unfold.jl")
includet("./plotting_methods.jl")
includet("./ride_structs.jl")

end