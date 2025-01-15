using Test
using TestItems

using PassiveAgressive

@testsnippet Imports begin
    using LinearAlgebra
    using OnlineStats
    using Random
end

@testitem "PAC does fit." setup = [Imports] begin
    rng = Random.MersenneTwister(32)
    X = randn(rng, 100, 10)
    w_true = rand(rng, 10)
    y = sign.(dot.([w_true], eachrow(X)))
    o = PassiveAgressive.PAClassifier(10)
    preds_before = PassiveAgressive.predict.(o, eachrow(X))
    acc_before = sum(y .== preds_before) / size(X, 1)
    fit!(o, zip(eachrow(X), y))
    preds_after = PassiveAgressive.predict.(o, eachrow(X))
    acc_after = sum(y .== preds_after) / size(X, 1)
    @test acc_after > acc_before
    @info "ACC Change:" (acc_before, acc_after)
end

@testitem "PAC V1 does fit." setup = [Imports] begin
    rng = Random.MersenneTwister(32)
    X = randn(rng, 100, 10)
    w_true = rand(rng, 10)
    y = sign.(dot.([w_true], eachrow(X)))
    o = PassiveAgressive.PAClassifier(10; type=:v1)
    preds_before = PassiveAgressive.predict.(o, eachrow(X))
    acc_before = sum(y .== preds_before) / size(X, 1)
    fit!(o, zip(eachrow(X), y))
    preds_after = PassiveAgressive.predict.(o, eachrow(X))
    acc_after = sum(y .== preds_after) / size(X, 1)
    @test acc_after > acc_before
    @info "ACC Change:" (acc_before, acc_after)
end

@testitem "PAC V2 does fit." setup = [Imports] begin
    rng = Random.MersenneTwister(32)
    X = randn(rng, 100, 10)
    w_true = rand(rng, 10)
    y = sign.(dot.([w_true], eachrow(X)))
    o = PassiveAgressive.PAClassifier(10; type=:v2)
    preds_before = PassiveAgressive.predict.(o, eachrow(X))
    acc_before = sum(y .== preds_before) / size(X, 1)
    fit!(o, zip(eachrow(X), y))
    preds_after = PassiveAgressive.predict.(o, eachrow(X))
    acc_after = sum(y .== preds_after) / size(X, 1)
    @test acc_after > acc_before
    @info "ACC Change:" (acc_before, acc_after)
end

@testitem "PAC w/ unsupported Algo Type" begin
    @test_throws ArgumentError PassiveAgressive.PAClassifier(10; type=:unsupp)
end

@testitem "PAR V1 does fit." setup = [Imports] begin
    rng = Random.MersenneTwister(32)
    X = randn(rng, 100, 10)
    w_true = collect(1:10)
    y = dot.([w_true], eachrow(X))
    o = PassiveAgressive.PARegressor(10; type=:v1)
    preds_before = PassiveAgressive.predict.(o, eachrow(X))
    l1_before = sum(abs.(y .- preds_before)) / size(X, 1)
    fit!(o, zip(eachrow(X), y))
    preds_after = PassiveAgressive.predict.(o, eachrow(X))
    l1_after = sum(abs.(y .- preds_after)) / size(X, 1)
    @info "MAE Before" l1_before
    @info "MAE After" l1_after
    @test l1_after < l1_before
end

@testitem "PAUniclass does fit." setup = [Imports] begin
    rng = Random.MersenneTwister(32)
    X = randn(rng, 100, 10)
    X_out = randn(rng, 5, 10) .+ 4 # shift in all dim by 4
    o = PassiveAgressive.PAUniclassClassifier(10; type=:base, ϵ=3.0, adaptive=false)
    fit!(o, eachrow(X))
    preds_in = PassiveAgressive.predict.(o, eachrow(X))
    preds_out = PassiveAgressive.predict.(o, eachrow(X_out))
    @info "Pred.Inside:" sum(preds_in .<= o.ϵ)
    @info "Pred.Outside:" sum(preds_out .> o.ϵ)
end

@testitem "PAUniclass (adaptive) does fit." setup = [Imports] begin
    rng = Random.MersenneTwister(32)
    X = randn(rng, 100, 10)
    X_out = randn(rng, 5, 10) .+ 4 # shift in all dim by 4
    o = PassiveAgressive.PAUniclassClassifier(10; type=:v2, ϵ=0.0, C=2, B=4, adaptive=true)
    fit!(o, eachrow(X))
    preds_in = PassiveAgressive.predict.(o, eachrow(X))
    preds_out = PassiveAgressive.predict.(o, eachrow(X_out))
    @info "Pred.Inside:" sum(preds_in .<= o.ϵ)
    @info "Pred.Outside:" sum(preds_out .> o.ϵ)
end