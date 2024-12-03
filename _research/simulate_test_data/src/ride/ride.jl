module Ride


using Revise

export ride_algorithm, ride_config
includet("./ride_algorithm.jl")
includet("./plotting_methods.jl")
includet("./helper_methods.jl")

end