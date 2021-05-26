#= ===============================================================================
Title: Sythetic Example - Investigating Updates
Updated: 2021-05-12
Description:
- For a given sequence of rewards, how does LIME-UCB update with only intercept?
- How does LIME-UCB update estimation across profiles?
=============================================================================== =#

#--- Packages
using Distributions
using LinearAlgebra
using Plots
using Serialization
using Random
using Interact
using Blink
using LaTeXStrings

include("./aux_function_toy.jl")

#--- Examine LIME-UCB's dynamics: contextual-free environment

# n = 100
# p = 1
# α = 1.0
# reward_mean = 0.04
# fe_mu0 = fill(0, p)
# fe_var0 = 1 * Matrix(I, p, p)
# re_var0 = 1 * Matrix(I, p, p)
# noise_var0 = 1
# slider_lime(reward_mean, n, p, fe_mu0, fe_var0, re_var0, noise_var0, α)

begin

    p = 1

    v = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2]

    range_fe_var0 = [i * Matrix(I, p, p) for i in v]

    range_re_var0 = [i * Matrix(I, p, p) for i in vcat(v, 4e-4)]

    range_α =
        OrderedDict(zip(["0.1", "1", "2", "Est: 3"], [0.1, 1.0, 2.0, 3.0]))

    test =
        @manipulate for T in dropdown(100:100:1000, value = 100, label = "T"),
            σ² in slider(0.01:0.01:1, value = 0.04, label = "σ²"),
            r̄ in slider(0.0:0.01:1, value = 0.04, label = "r̄"),
            μ in slider(0.0:0.01:1, value = 0.04, label = "μᵦ"),
            σ²ᵦ in dropdown(
                range_fe_var0,
                value = 1e-6 * Matrix(I, p, p),
                label = "Ωᵦ",
            ),
            σ²ᵢ in dropdown(
                range_re_var0,
                value = 4e-4 * Matrix(I, p, p),
                label = "Ωᵢ",
            ),
            α in dropdown(range_α, value = 3.0, label = "α")

            slider_lime(r̄, T, p, fill(μ, p), σ²ᵦ, σ²ᵢ, σ², α)
        end


    @layout! test vbox(
        :r̄,
        :μ,
        hbox(
            pad(1em, :σ²ᵦ),
            pad(1em, :σ²ᵢ),
            pad(1em, :σ²),
            pad(1em, :T),
            pad(1em, :α),
        ),
        vskip(2em),
        observe(_),
    )

    w = Window()
    body!(w, test)
end

#--- Examine LIME-UCB's dynamics: intercept + one binary feature

# Set-up
n = 10
p = 2
α = 1.0
fe_mu0 = fill(0, p)
fe_var0 = 1e-6 * Matrix(I, p, p)
#re_var0 = 1 * Matrix(I, p, p)
s₀ = 0.1
s₁ = 0.1
ρ = 0.9
cov = ρ * s₀ * s₁
re_var0 = [s₀^2 cov
           cov s₁^2]
noise_var0 = 0.04
β = [0.02, 0.05]
pf_learn = [1, 0]
pf_infer = [1, 1]
# Plot
slider_profile(β, n, p, pf_learn, pf_infer, fe_mu0, fe_var0, re_var0, noise_var0, α; seed = 112233)

begin
    p = 2

    v = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2]

    range_α =
        OrderedDict(zip(["0.1", "1", "2", "Est: 3"], [0.1, 1.0, 2.0, 3.0]))

    test =
        @manipulate for pf_learn in dropdown([[1,0], [1,1]], value = [1,0], label = "learning profile"),
            pf_infer in dropdown([[1,0], [1,1]], value = [1,1], label = "inferring profile"),
            β₀ in slider(-1:0.01:1, value = 0.02, label = "β₀"),
            β₁ in slider(-1:0.01:1, value = 0.03, label = "β₁"),
            μ₀ in slider(-1:0.01:1, value = 0.00, label = "μ₀"),
            μ₁ in slider(-1:0.01:1, value = 0.00, label = "μ₁"),
            τ₀² in dropdown(v, value = 1e-6, label = "τ₀²"),
            τ₁² in dropdown(v, value = 1e-6, label = "τ₁²"),
            ρᵦ in dropdown(-0.9:0.1:0.9, value = 0.0, label = "ρᵦ"),
            s₀² in dropdown(vcat(0.000177, v), value = 1, label = "s₀²"),
            s₁² in dropdown(vcat(0.000330, v), value = 1, label = "s₁²"),
            ρᵢ in dropdown(-0.9:0.1:0.9, value = 0.0, label = "ρᵢ"),
            T in dropdown([1, 3,5, 10, 30, 50,100,300,500,1000], value = 10, label = "T"),
            σ² in slider(0.01:0.01:1, value = 0.04, label = "σ²"),
            α in dropdown(range_α, value = 1.0, label = "α")

            β = [β₀, β₁]
            μᵦ = [μ₀, μ₁]
            τₐ² = ρᵦ * sqrt(τ₀² * τ₁²)
            Ωᵦ = [
                τ₀² τₐ²
                τₐ² τ₁²
            ]
            sₐ² = ρᵢ * sqrt(s₀² * s₁²)
            Ωᵢ = [
                s₀² sₐ²
                sₐ² s₁²
            ]

            slider_profile(β, T, p, pf_learn, pf_infer, μᵦ, Ωᵦ, Ωᵢ, σ², α; seed = 112233)

        end


    @layout! test vbox(
        hbox(pad(1em, :β₀), pad(1em, :β₁), pad(1em, :μ₀), pad(1em, :μ₁)),
        hbox(
            pad(1em, :τ₀²),
            pad(1em, :τ₁²),
            pad(1em, :ρᵦ),
            hskip(2em),
            pad(1em, :s₀²),
            pad(1em, :s₁²),
            pad(1em, :ρᵢ),
        ),
        hbox(pad(1em, :pf_learn), pad(1em, :pf_infer), pad(1em, :σ²), pad(1em, :T), pad(1em, :α)),
        vskip(2em),
        observe(_),
    )

    w = Window()
    body!(w, test)
end

println("\n ========== Test ends ========== \n")



#= Comment begins
#--- Experiment: One-Armed Bandit Problem Using Different Algorithms (Intercept Only)

# Data
reward_mean = 0.05
n = 100
rewardseq = gen_reward(n, reward_mean)
datastream = [OneInstance(i, [1]) for i in rewardseq]

# Tuning parameters
algnames = (
    beta = "Beta",
    ucb = "Simple UCB",
    lime_fixed = "LIME(Fixed)",
    lime = "LIME",
)
tuneparas = (beta = 0.95, ucb = 1.0, lime_fixed = 1.0, lime = 1.0)
p = 1

# Prior
stat_beta0 = Beta((reward_mean + 0.0001) / ((1 - reward_mean) + 0.0001), 1)

stat_ucb0 = StatUCB(
    0, # nchosen
    0.0, # estmu
    (estmu0 = 0.0, noise_var0 = 1),
)

stat_fe0 = StatLIMEFe(
    1 * Matrix(I, p, p), # fe_var
    fill(0, p), # fe_mu
    (fe_var0 = 1 * Matrix(I, p, p), fe_mu0 = fill(0, p), noise_var0 = 1),
)
stat_re0 = StatLIMERe(
    0, # nchosen
    zeros(0, p), # feature/design matrix
    Bool[], # reward
    1 * Matrix(I, p, p), # re_omg
    zeros(p, p), # re_rho
    zeros(p), # re_bhat
    zeros(p, p), # C_var
    zeros(p), # C_mu
    (re_var0 = 1 * Matrix(I, p, p), noise_var0 = 1),
)

# Run experiment --> Temporary
result_beta = test_beta(datastream, q = tuneparas.beta, stat_beta0 = stat_beta0)
result_ucb = test_ucb(datastream, α = tuneparas.ucb, stat_ucb0 = stat_ucb0)
result_lime_fixed = test_lime_fixed(
    datastream,
    p,
    α = tuneparas.lime_fixed,
    stat_fe0 = stat_fe0,
    stat_re0 = stat_re0,
)
result_lime = test_lime(
    datastream,
    p,
    α = tuneparas.lime,
    stat_fe0 = stat_fe0,
    stat_re0 = stat_re0,
)
Comment ends =#
