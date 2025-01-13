module PassiveAgressive

using OnlineStatsBase
greet() = print("Hello World!")

mutable struct PAClassifier <: OnlineStat{Number}
    weight::Vector{Float64}
    n::Int
end
function OnlineStatsBase._fit!(o::PAClassifier, y)
    o.n += 1
end


mutable struct PARegressor <: OnlineStat{Number}
    weight::Vector{Float64}
    n::Int
end
function OnlineStatsBase._fit!(o::PARegressor, y)
    o.n += 1
end
end # module PassiveAgressive
