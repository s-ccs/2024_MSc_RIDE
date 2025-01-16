module Ride


using Revise

export ride_algorithm, ride_algorithm_unfold, ride_config
includet("./ride_algorithm.jl")
includet("./ride_algorithm_unfold.jl")
includet("./plotting_methods.jl")
includet("./helper_methods.jl")

end