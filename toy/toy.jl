#= ===============================================================================
Title: Test - Investigating updates
Updated: 2021-04-15
Description:
- For a given sequence of rewards, how each algorithm updates the UCB.
To-do:
- Insert calibration for priors.
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

#--- Examine LIME-UCB's dynamics

n = 100
p = 1
α = 1.0
# reward_mean = 0.24
# fe_mu0 = fill(0, p)
# fe_var0 = 1 * Matrix(I, p, p)
# re_var0 = 1 * Matrix(I, p, p)
# noise_var0 = 1
# slider_lime(reward_mean, n, p, fe_mu0, fe_var0, re_var0, noise_var0, α)

begin
    v = [0.0001, 0.001, 0.01, 0.1, 1]

    range_fe_var00 = [i * Matrix(I, p, p) for i in vcat(v, 1e-6)]
    range_fe_var0 = OrderedDict(
        zip(["0.0001", "0.001", "0.01", "0.1", "1", "Est: 1e-6"], range_fe_var00),
    )

    range_re_var00 = [i * Matrix(I, p, p) for i in vcat(v, 4e-4)]
    range_re_var0 = OrderedDict(
        zip(["0.0001", "0.001", "0.01", "0.1", "1", "Est: 4e-4"], range_re_var00),
    )

    range_noise_var0 = OrderedDict(
        zip(["0.0001", "0.001", "0.01", "0.1", "1", "Est: 4e-2"], vcat(v, 0.04)),
    )

    test = @manipulate for r̄ in slider(
            0.0:0.01:1,
            value = 0.04,
            label = "Mean reward",
        ),
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
        σ² in togglebuttons(
            range_noise_var0,
            value = 0.04,
            label = "σ²",
        )

        slider_lime(r̄, n, p, fill(μ, p), σ²ᵦ, σ²ᵢ, σ², α)
    end


    @layout! test vbox(
        #hbox(pad(1em, :r̄), pad(1em, :μ)),
        :r̄,
        :μ,
        hbox(pad(1em, :σ²ᵦ), pad(1em, :σ²ᵢ), pad(1em, :σ²)),
        vskip(2em),
        observe(_),
    )

    w = Window()
    body!(w, test)
end


println("\n ========== Test ends ========== \n")

#= Comment begins
#--- Experiment: One-Armed Bandit Problem Using Different Algorithms

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
