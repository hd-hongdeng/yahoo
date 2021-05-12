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

n = 100
p = 1
# α = 1.0
# reward_mean = 0.04
# fe_mu0 = fill(0, p)
# fe_var0 = 1 * Matrix(I, p, p)
# re_var0 = 1 * Matrix(I, p, p)
# noise_var0 = 1
# slider_lime(reward_mean, n, p, fe_mu0, fe_var0, re_var0, noise_var0, α)

begin
    v = [0.001, 0.01, 0.1, 1, 10]

    range_fe_var00 = [i * Matrix(I, p, p) for i in vcat(v, 1e-6)]
    range_fe_var0 = OrderedDict(
        zip(["0.001", "0.01", "0.1", "1", "10", "Est: 1e-6"], range_fe_var00),
    )

    range_re_var00 = [i * Matrix(I, p, p) for i in vcat(v, 4e-4)]
    range_re_var0 = OrderedDict(
        zip(["0.001", "0.01", "0.1", "1", "10", "Est: 4e-4"], range_re_var00),
    )

    range_noise_var0 = OrderedDict(
        zip(["0.001", "0.01", "0.1", "1", "10", "Est: 4e-2"], vcat(v, 0.04)),
    )

    range_α = OrderedDict(zip(["0.1", "1", "2", "Est: 3"], [0.1, 1.0, 2.0, 3.0]))

    test = @manipulate for r̄ in slider(0.0:0.01:1, value = 0.04, label = "r̄"),
        μ in slider(0.0:0.01:1, value = 0.04, label = "μᵦ"),
        σ²ᵦ in togglebuttons(
            range_fe_var0,
            value = 1e-6 * Matrix(I, p, p),
            label = "Ωᵦ",
        ),
        σ²ᵢ in togglebuttons(
            range_re_var0,
            value = 4e-4 * Matrix(I, p, p),
            label = "Ωᵢ",
        ),
        σ² in togglebuttons(range_noise_var0, value = 0.04, label = "σ²"),
        α in togglebuttons(range_α, value = 3.0, label = "α")

        slider_lime(r̄, n, p, fill(μ, p), σ²ᵦ, σ²ᵢ, σ², α)
    end


    @layout! test vbox(
        #hbox(pad(1em, :r̄), pad(1em, :μ)),
        :r̄,
        :μ,
        hbox(pad(1em, :σ²ᵦ), pad(1em, :σ²ᵢ), pad(1em, :σ²), pad(1em, :α)),
        vskip(2em),
        observe(_),
    )

    w = Window()
    body!(w, test)
end

#--- Examine LIME-UCB's dynamics: intercept + one binary feature

# Set-up
n = 100
p = 2
# Prior
# α = 1.0
# fe_mu0 = fill(0, p)
# fe_var0 = 1 * Matrix(I, p, p)
# re_var0 = 1 * Matrix(I, p, p)
# noise_var0 = 1
# # Plot
# slider_profile(β, n, p, fe_mu0, fe_var0, re_var0, noise_var0, α; seed = 112233)

begin
    range_noise_var0 = OrderedDict(
        zip(["0.01", "0.1", "1", "10", "Est: 4e-2"], vcat(v, 0.04)),
    )

    range_α = OrderedDict(zip(["0.1", "1", "2", "Est: 3"], [0.1, 1.0, 2.0, 3.0]))

    test = @manipulate for β₀ in slider(-1:0.01:1, value = 0.02, label = "β₀"),
        β₁ in slider(-1:0.01:1, value = 0.03, label = "β₁"),
        μ₀ in slider(-1:0.01:1, value = 0.02, label = "μ₀"),
        μ₁ in slider(-1:0.01:1, value = 0.03, label = "μ₁"),
        σ²₀ in dropdown([1e-6, 0.01, 0.1, 1, 10], value = 1e-6, label = "σ²₀"),
        σ²₁ in dropdown([1e-6, 0.01, 0.1, 1, 10], value = 1e-6, label = "σ²₁"),
        cᵦ in dropdown([-0.5, -0.1, 0.0, 0.1, 0.5], value = 0.0, label = "cᵦ"),
        s²₀ in dropdown([0.000177, 0.01, 0.1, 1, 10], value = 0.000177, label = "s²₀"),
        s²₁ in dropdown([0.000330, 0.01, 0.1, 1, 10], value = 0.000330, label = "s²₁"),
        cᵢ in dropdown([0.000154, -0.5, -0.1, 0.0, 0.1, 0.5], value = 0.000154, label = "cᵢ"),
        σ² in togglebuttons(range_noise_var0, value = 0.04, label = "σ²"),
        α in togglebuttons(range_α, value = 3.0, label = "α")

        β = [β₀, β₁]
        μᵦ = [μ₀, μ₁]
        Ωᵦ = [σ²₀ cᵦ
              cᵦ σ²₁]
      Ωᵢ = [s²₀ cᵢ
            cᵢ s²₁]

        slider_profile(β, n, p, μᵦ, Ωᵦ, Ωᵢ, σ², α; seed = 112233)

    end


    @layout! test vbox(
        hbox(pad(1em, :β₀), pad(1em, :β₁), pad(1em, :μ₀), pad(1em, :μ₁)),
        hbox(pad(1em, :σ²₀), pad(1em, :σ²₁), pad(1em, :cᵦ), hskip(2em), pad(1em, :s²₀), pad(1em, :s²₁), pad(1em, :cᵢ)),
        hbox(pad(1em, :σ²), pad(1em, :α)),
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
