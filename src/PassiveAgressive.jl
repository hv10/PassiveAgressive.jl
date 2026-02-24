module PassiveAgressive

using OnlineStatsBase
using LinearAlgebra

export PAClassifier, PARegressor, PAUniclassClassifier
export predict

const Tup{T} = Union{NTuple{N,T} where {N},NamedTuple{names,Tuple{N,<:T} where {N}} where {names}}
const VectorOb{T} = Union{AbstractVector{<:T},Tup{T}}
const XY{T,S} = Union{Tuple{T,S},Pair{T,S},NamedTuple{names,Tuple{T,S}}} where {names,T<:VectorOb{Number},S<:Number}

rule(::Val{:base}, lt, x, C) = lt / norm(x)^2
rule(::Val{:v1}, lt, x, C) = min(C, lt / norm(x)^2)
rule(::Val{:v2}, lt, x, C) = lt / (norm(x)^2 + 1 / 2 * C)

update_rule(tp::Symbol, C::Number) = begin
    if tp == :base
        return (lt, x) -> lt / norm(x)^2
    elseif tp == :v1
        return (lt, x) -> min(C, lt / norm(x)^2)
    elseif tp == :v2
        return (lt, x) -> lt / (norm(x)^2 + 1 / 2 * C)
    end
    throw(ArgumentError("Unsupported PAC Type"))
end

mutable struct PAClassifier{F} <: OnlineStat{Union{Tuple,Vector{Number}}}
    weight::Vector{Float64} # weights
    rule::F
    C::Float64
    n::Int
end
PAClassifier(in_::Int=1; type::Symbol=:base, C=1) = begin
    return PAClassifier(zeros(in_), type, C, 0)
end
predict(o::PAClassifier, y::AbstractArray) = sign(dot(o.weight, y))
OnlineStatsBase._fit!(o::PAClassifier, y::Vector{Number}) = predict(o, y)
OnlineStatsBase._fit!(o::PAClassifier, y::Tuple) = begin
    if !(y[2] isa Nothing)
        yh = predict(o, y[1])
        o.n += 1
        lt = max(0, 1 - y[2] * yh)
        τ = rule(Val(o.rule), lt, y[1], o.C)
        o.weight = o.weight + τ * y[2] * y[1]
        return yh
    else
        return predict(o, y[1])
    end
end

mutable struct PARegressor{F} <: OnlineStat{Union{Tuple,Vector{Number}}}
    weight::Vector{Float64} # weights
    rule::F
    C::Float64
    ϵ::Float64
    n::Int
end
PARegressor(in_::Int=1; type::Symbol=:base, ϵ=0.1, C=1) = begin
    return PARegressor(zeros(in_), type, C, ϵ, 0)
end
predict(o::PARegressor, y::AbstractArray) = dot(o.weight, y)
OnlineStatsBase._fit!(o::PARegressor, y::Vector{Number}) = predict(o, y)
OnlineStatsBase._fit!(o::PARegressor, y::Tuple) = begin
    if !(y[2] isa Nothing)
        yh = predict(o, y[1])
        cond_term = abs(yh − y[2])
        lt = ifelse(cond_term <= o.ϵ, 0, cond_term - o.ϵ)
        τ = rule(Val(o.rule), lt, y[1], o.C)
        o.weight = o.weight + τ * sign(y[2] - yh) * y[1]
        o.n += 1
        return yh
    else
        return predict(o, y[1])
    end
end

rule_uniclass(::Val{:base}, lt, x, C) = lt
rule_uniclass(::Val{:v1}, lt, x, C) = min(C, lt)
rule_uniclass(::Val{:v2}, lt, x, C) = lt / (1 + (1 / 2 * C))

update_rule_uniclass(tp::Symbol, C::Number) = begin
    if tp == :base
        return (lt, x) -> lt
    elseif tp == :v1
        return (lt, x) -> min(C, lt)
    elseif tp == :v2
        return (lt, x) -> lt / (1 + (1 / 2 * C))
    end
    throw(ArgumentError("Unsupported PAC Type"))
end

"""
I think this is not working correctly.
"""
mutable struct PAUniclassClassifier{T,F<:AbstractFloat} <: OnlineStat{AbstractVector{<:Number}}
    weight::Vector{T} # weights
    rule::F
    C::T
    ϵ::T
    adaptive::Bool
    B::T
    n::Int
end
PAUniclassClassifier(in_::Int=1; type::Symbol=:base, ϵ=0.1, B=1e10, C=1, adaptive=true) = begin
    if adaptive
        ϵ = 0.0 # init ϵ to be zero
        weight = zeros(in_ + 1)
        weight[end] = B
    else
        weight = zeros(in_)
    end
    return PAUniclassClassifier(weight, type, C, ϵ, adaptive, B, 0)
end
predict(o::PAUniclassClassifier{T}, y::AbstractVector{T}) where T = o.adaptive ? dot(o.weight[1:end-1], y) : dot(o.weight, y)
OnlineStatsBase._fit!(o::PAUniclassClassifier, y::AbstractVector{<:Number}) = begin
    w = o.weight
    if o.adaptive
        y = vcat(y, 0)
    end
    cond_term = norm(w - y, 2)
    if o.adaptive
        lt = ifelse(cond_term^2 <= o.B^2, 0, cond_term - o.ϵ)
    else
        lt = ifelse(cond_term <= o.ϵ, 0, cond_term - o.ϵ)
    end
    τ = rule_uniclass(Val(o.rule), lt, y, o.C)
    w = w + τ * ((y - w) / norm(y - w, 2))
    if o.adaptive
        # update our bound ϵ
        o.ϵ = sqrt(o.B^2 - w[end]^2)
        o.weight[end] = o.ϵ
    end
    o.weight = w
    o.n += 1
    if o.adaptive
        return predict(o, y[1:end-1])
    else
        return predict(o, y)
    end
end

end # module PassiveAgressive
