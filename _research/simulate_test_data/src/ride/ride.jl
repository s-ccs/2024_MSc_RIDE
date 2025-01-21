module Ride


using Revise

export ride_algorithm, ride_algorithm_unfold, ride_config, ride_original, ride_unfold
includet("./helper_methods.jl")
includet("./ride_algorithm.jl")
includet("./ride_algorithm_unfold.jl")
includet("./plotting_methods.jl")

end