#= ===============================================================================
Title: Test - Investigating updates
Updated: 2021-04-15
Description:
- For a given sequence of rewards, how each algorithm updates the UCB.
To-do:
- Insert calibration for priors.
=============================================================================== =#

#--- Packages
using Revise
using Distributions
using LinearAlgebra
using Plots
using Serialization
using Random

include("../aux_function_test.jl")

#--- Specification
prior_id = 0
reward_id = 0.5
# Specify the sequence of rewards and fetures
n = 100
#data = OneInstance(0, [1])
#datastream = fill(data, n)
Random.seed!(23345)
setone = sample(1:n, Int(n * reward_id), replace = false)
rewardseq = fill(Float64(0), n)
rewardseq[setone] .= 1
println("CTR = ", mean(rewardseq))
datastream = [OneInstance(i, [1]) for i in rewardseq]

# Specify priors and tuning parameters
algnames = ["Beta", "Simple UCB", "LIME(Fixed)", "LIME"]
tuneparas = (beta = 0.95, ucb = 1.0, lime_fixed = 1.0, lime = 1.0)
p = 1
# Prior
stat_beta0 = Beta((reward_id+0.0001)/((1-reward_id)+0.0001), 1)
stat_ucb0 = StatUCB(
    0, # nchosen
    0, # estmu
    (estmu0 = 0, noise_var0 = 1),
)
stat_fe0 = StatLIMEFe(
    0.1 * Matrix(I, p, p), # fe_var
    fill(0, p), # fe_mu
    (fe_var0 = 0.1 * Matrix(I, p, p), fe_mu0 = fill(0, p), noise_var0 = 1),
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

# # Prior
# result_initial = open(deserialize, "case_p1/result/result_initial.bin")
# p = length(result_initial.fe_mu0)
#
# stat_beta0 = Beta(0.038 / (1 - 0.038), 1)
# mean(stat_beta0)
#
# stat_UCB0 = UCBStat(0,
#                     result_initial.ucb1estmu0,
#                     (estmu0 = result_initial.ucb1estmu0,
#                      noise_var0 = result_initial.ucb1estvar0))
#
# stat_LIME_re0 = LIMEStatRE(0,
#                             Bool[], zeros(0, p),
#                             zeros(p, p), zeros(p), result_initial.re_var0,
#                             (re_var0 = result_initial.re_var0,
#                             noise_var0 = result_initial.noise_var0))
# stat_LIME_fe0 = LIMEStatFE(result_initial.fe_mu0,
#                            result_initial.fe_var0,
#                             (fe_mu0 = result_initial.fe_mu0,
#                              fe_var0 = result_initial.fe_var0,
#                              noise_var0 = result_initial.noise_var0))

#--- Experiment: One-Armed Bandit Problem
# Run experiment
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
# Collect results
results = [result_beta, result_ucb, result_lime_fixed, result_lime]
# Extract ucb-related indices
inds = [getproperty(result, :inds) for result in results]
inds_ub = [getproperty.(ind, :ub) for ind in inds]
# Extract LIME's parameters
re_omg = [getfield(coll, :re_omg)[1] for coll in result_lime.stats.re]
re_rho = [getfield(coll, :re_rho)[1] for coll in result_lime.stats.re]
re_bhat = [getfield(coll, :re_bhat)[1] for coll in result_lime.stats.re]
fe_var = [getfield(coll, :fe_var)[1] for coll in result_lime.stats.fe]
fe_mu = [getfield(coll, :fe_mu)[1] for coll in result_lime.stats.fe]
datastream_add0 = vcat(OneInstance(0, fill(0, p)), datastream)
inds_detail = [
    getucb(
        result_lime.stats.re[i],
        result_lime.stats.fe[i],
        datastream_add0[i].feature,
        α = tuneparas.lime,
        detail = true,
    ) for i = 1:length(result_lime.stats.re)
]
center_fe = [getfield(coll, :center_fe)[1] for coll in inds_detail]
center_re = [getfield(coll, :center_re)[1] for coll in inds_detail]
var_fe = [getfield(coll, :var_fe)[1] for coll in inds_detail]
var_re = [getfield(coll, :var_re)[1] for coll in inds_detail]

#--- Plot: Index
function plotinds(
    inds,
    inds_ub,
    result_beta,
    result_ucb,
    result_lime_fixed,
    result_lime,
)
    # Plotting set-up
    mycolor = [:red, :blue, :tan, :green3]
    mytitle = [
        "Beta's Quantile (q = $(tuneparas.beta))",
        "Simple UCB (α = $(tuneparas.ucb))",
        "LIME (Fixed) (α = $(tuneparas.lime_fixed))",
        "LIME (α = $(tuneparas.lime))",
    ]
    myylims = (
        minimum(minimum(minimum.(i)) for i in inds) - 0.1,
        maximum(
            maximum(maximum.(i)) for i in [
                result_beta.inds,
                result_ucb.inds[begin+1:end],
                result_lime_fixed.inds,
                result_lime.inds,
            ]
        ),
    )
    # Plot upper bounds, centers and widths
    plts_inds = [
        plotindex(
            inds[i];
            α = tuneparas[i],
            linecolor = mycolor[i],
            title = mytitle[i],
        ) for i = 1:length(inds)
    ]
    plt_inds = plot(plts_inds..., ylims = myylims)
    # Plot upper bounds only
    plt_inds_ub = plot(
        range(0, length(inds_ub[1]) - 1, step = 1),
        inds_ub,
        lc = permutedims(mycolor),
        ls = :dash,
        label = permutedims(mytitle),
        xlabel = "Step",
        title = "Bounds",
        leg = :best,
    )
    # Combine pervious plots
    plt_inds_final = plot(
        plt_inds,
        plt_inds_ub,
        layout = grid(1, 2, widths = [0.6, 0.4]),
        guidefontsize = 6,
        legendfontsize = 6,
        tickfontsize = 6,
        titlefontsize = 8,
        lw = 0.8,
        #xticks = range(0, length(inds[1]), step = 1),
        thickness_scaling = 0.8,
        #size = (1200, 800),
    )
    display(plt_inds_final)
    return plt_inds_final
end

plt_inds_final = plotinds(
    inds,
    inds_ub,
    result_beta,
    result_ucb,
    result_lime_fixed,
    result_lime,
)


#--- Plot: LIME-UCB's Parameters

function plotlime(
    fe_mu,
    re_bhat,
    re_rho,
    fe_var,
    re_omg,
    center_fe,
    center_re,
    var_fe,
    var_re,
)
    xs = range(0, length(fe_mu) - 1, step = 1)
    plt1 = plot(
        xs,
        [fe_mu, re_bhat, re_rho],
        label = ["FE-Mean" "RE-Mean" "Corr"],
        ls = [:solid :solid :dash],
        lc = [:blue :red :orange],
        title = "LIME-UCB's Parameters Related to Center",
    )
    plt2 = plot(
        xs,
        [fe_var, re_omg, re_rho],
        label = ["FE-Var" "RE-Var" "Corr"],
        ls = [:solid :solid :dash],
        lc = [:blue :red :orange],
        title = "LIME-UCB's Parameters Related to Width",
    )
    plt3 = plot(
        xs,
        [center_fe, center_re, center_fe + center_re],
        label = ["Center-FE" "Center-RE" "Center"],
        lc = [:blue :red :black],
        title = "Decomposition of LIME-UCB's Center",
    )
    plt4 = plot(
        xs,
        [var_fe, var_re, var_fe + var_re],
        label = ["Raw Width-FE" "Raw Width-RE" "Squared Width"],
        lc = [:blue :red :black],
        title = "Decomposition of LIME-UCB's Width",
    )
    plt = plot(
        plt1,
        plt2,
        plt3,
        plt4,
        xlabel = "Step",
        guidefontsize = 6,
        legendfontsize = 6,
        tickfontsize = 6,
        titlefontsize = 8,
        #lw  = 0.8,
    )
    display(plt)
    return plt
end

function plotlime(
    fe_mu,
    re_bhat,
    re_rho,
    fe_var,
    re_omg
)
    xs = range(0, length(fe_mu) - 1, step = 1)
    adj_re_mu = map(re_bhat, re_rho, fe_mu) do x, y, z
        return x - y * z
    end
    adj_re_var = map(re_omg, re_rho, fe_var) do x,y,z
        return x + y * z * y'
    end
    neg_var = map(re_rho, fe_var) do x,y
        return  (-2) * x*y
    end
    plt1 = plot(
        xs,
        [fe_mu, adj_re_mu],
        label = ["FE-Mean" "Adj. RE-Mean"],
        ls = [:solid :solid],
        lc = [:blue :red],
        title = "LIME-UCB's Parameters Related to Center",
    )
    plt2 = plot(
        xs,
        [fe_var, adj_re_var, neg_var],
        label = ["FE-Var" "Adj. RE-Var" "Neg-Var"],
        ls = [:solid :solid :dash],
        lc = [:blue :red :orange],
        title = "LIME-UCB's Parameters Related to Width",
    )
    plt3 = plot(
        xs[2:end],
        [fe_mu[2:end], adj_re_mu[2:end], fe_mu[2:end] + adj_re_mu[2:end]],
        label = ["Center-FE" "Center-RE" "Center"],
        lc = [:blue :red :black],
        title = "Decomposition of LIME-UCB's Center",
    )
    plt4 = plot(
        xs[2:end],
        [fe_var[2:end], adj_re_var[2:end], neg_var[2:end], fe_var[2:end] + adj_re_var[2:end] + neg_var[2:end]],
        label = ["Raw Width-FE" "Raw Width-RE" "Neg Var" "Squared Width"],
        lc = [:blue :red :orange :black],
        title = "Decomposition of LIME-UCB's Width",
    )
    plt = plot(
        plt1,
        plt2,
        plt3,
        plt4,
        xlabel = "Step",
        guidefontsize = 6,
        legendfontsize = 6,
        tickfontsize = 6,
        titlefontsize = 8,
        #lw  = 0.8,
    )
    display(plt)
    return plt
end

plt_lime = plotlime(
    fe_mu,
    re_bhat,
    re_rho,
    fe_var,
    re_omg,
    center_fe,
    center_re,
    var_fe,
    var_re,
)

plt_lime2 = plotlime(
    fe_mu,
    re_bhat,
    re_rho,
    fe_var,
    re_omg
)

plot(plt_lime, plt_lime2, leg = false, ylim = (-1, 1))

#savefig(plt_inds_final, "figure/fig_test_prior$(prior_id)_reward$(reward_id)_t$(n).pdf")
#savefig(plt_lime, "figure/fig_test_prior$(prior_id)_reward$(reward_id)_t$(n)_lime.pdf")


#=
function asymoptucb(; n = 1000)
    f(t) = 1 + t * log(t)^2
    t = 1:n
    #plt1 = plot(t, f.(t), title = "f(t)", leg = false)
    Tᵢ = 1:n
    g(t) = sqrt(2 * log(f(t)))
    plt2 = plot(t, g.(t), ylabel = "α", leg = false)
    w(t, Tᵢ) = sqrt((2 * log(f(t))) / Tᵢ)
    plt3 = plot(t, w.(t, Tᵢ), ylabel = "Uncertainty", leg = false)

    plt = plot(#plt1,
                plt2, plt3,
                title = "Asympototically Optimal UCB",
                titlefontsize = 8,
                guidefontsize = 8,
                xlabel = "Step")
    display(plt)
end

asymoptucb()
#savefig("figure/fig_test_asymoptucb.pdf")
=#

#--- Compare articles with different CTR

# Generate rewards
n = 100

ctr1 = 0.07
#setone = sample(1:n, Int(ceil(n * ctr1)), replace = false)
setone = Int.(range(1, n * ctr1 * 2, step = 2))
rewardseq = fill(Float64(0), n)
rewardseq[setone] .= 1
println("CTR = ", mean(rewardseq))
datastream1 = [OneInstance(i, [1]) for i in rewardseq]

ctr2 = 0.08
#setone = sample(1:n, Int(ceil(n * ctr2)), replace = false)
setone = Int.(range(1, n * ctr2 * 2, step = 2))
rewardseq = fill(Float64(0), n)
rewardseq[setone] .= 1
println("CTR = ", mean(rewardseq))
datastream2 = [OneInstance(i, [1]) for i in rewardseq]

ctr3 = 0.09
#setone = sample(1:n, Int(ceil(n * ctr3)), replace = false)
setone = Int.(range(1, n * ctr3 * 2, step = 2))
rewardseq = fill(Float64(0), n)
rewardseq[setone] .= 1
println("CTR = ", mean(rewardseq))
datastream3 = [OneInstance(i, [1]) for i in rewardseq]

# Run experiment
result_ucb1 = test_ucb(datastream1, α = tuneparas.ucb, stat_ucb0 = stat_ucb0)
result_ucb2 = test_ucb(datastream2, α = tuneparas.ucb, stat_ucb0 = stat_ucb0)
result_ucb3 = test_ucb(datastream3, α = tuneparas.ucb, stat_ucb0 = stat_ucb0)
result_lime1 = test_lime(
    datastream1,
    p,
    α = tuneparas.lime,
    stat_fe0 = stat_fe0,
    stat_re0 = stat_re0,
)
result_lime2 = test_lime(
    datastream2,
    p,
    α = tuneparas.lime,
    stat_fe0 = stat_fe0,
    stat_re0 = stat_re0,
)
result_lime3 = test_lime(
    datastream3,
    p,
    α = tuneparas.lime,
    stat_fe0 = stat_fe0,
    stat_re0 = stat_re0,
)

# Plot
function plotub(result1, result2, ctr1, ctr2)
    y1 = getfield.(result1.inds, :ub)
    y2 = getfield.(result2.inds, :ub)
    xs = range(0, length(y2) - 1, step = 1)
    plot(
        xs,
        [y1, y2],
        label = ["CTR = $ctr1" "CTR = $ctr2"],
        xlabel = "Step",
        ylabel = "Bound",
    )
end

plt1 = plotub(result_ucb1, result_ucb2, ctr1, ctr2)
insertstep = Int(5)
plot!(range(insertstep, insertstep +  n, step = 1), getfield.(result_ucb3.inds, :ub), label = "CTR = $ctr3")
plt2 = plotub(result_lime1, result_lime2, ctr1, ctr2)
plot!(range(insertstep, insertstep +  n, step = 1), getfield.(result_lime3.inds, :ub), label = "CTR = $ctr3")

plt_final = plot(
    plt1,
    plt2,
    ylims = (0, 2),
    titlefontsize = 8,
    guidefontsize = 8,
    legendfontsize = 6,
    title = ["Simple UCB (α = $(tuneparas.ucb))" "LIME (α = $(tuneparas.lime))"],
)
display(plt_final)

#savefig(plt_final, "figure/fig_test_prior$(prior_id)_reward$(ctr1)vs$(ctr2)_t$(n).pdf")

println("\n ========== Test ends ========== \n")
