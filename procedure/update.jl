#= ===============================================================================
Title: Test - Investigating updates
Updated: 2021-04-15
Description:
- For a given sequence of rewards, how each algorithm updates the UCB.
To-do:
- Insert calibration for priors.
=============================================================================== =#
# Packages
# using Distributions, Plots, Serialization

# Experiment's specification
# Scenario id
id = 2
# Rewards
n_step = 100
#rewards = fill(Bool(1), n_step)
# rewards = convert(Vector{Bool}, rand([0,1], n_step))
rewards = vcat(ones(Int(n_step/2)), zeros(Int(n_step/2)))
features = fill([1], n_step)

# Prior
result_initial = open(deserialize, "case_p1/result/result_initial.bin")
p = length(result_initial.fe_mu0)

#= Case 1

# Uncertainty level
αs = [0.95, 1, 1]

# Prior

stat_beta0 = Beta(0.001, 1)
mean(stat_beta0)

stat_UCB0 = UCBStat(0, 
                    0, 
                    (estmu0 = 0, noise_var0 = 1))

stat_LIME_re0 = LIMEStatRE(0,
                           [], zeros(0, p),
                           zeros(p, p), zeros(p), ones(1, 1),
                           (re_var0 = ones(1, 1), noise_var0 = 1))
stat_LIME_fe0 = LIMEStatFE([0], ones(1, 1),
                           (fe_mu0 = [0], fe_var0 = ones(1, 1), noise_var0 = 1))
=#

# Case 2
# Uncertainty level
αs = [0.95, 3, 2.5]

# Prior
stat_beta0 = Beta(0.038 / (1 - 0.038), 1)
mean(stat_beta0)
                           
stat_UCB0 = UCBStat(0, 
                    result_initial.ucb1estmu0, 
                    (estmu0 = result_initial.ucb1estmu0, 
                     noise_var0 = result_initial.ucb1estvar0))

stat_LIME_re0 = LIMEStatRE(0, 
                            Bool[], zeros(0, p),
                            zeros(p, p), zeros(p), result_initial.re_var0,
                            (re_var0 = result_initial.re_var0,
                            noise_var0 = result_initial.noise_var0))
stat_LIME_fe0 = LIMEStatFE(result_initial.fe_mu0, 
                           result_initial.fe_var0,
                            (fe_mu0 = result_initial.fe_mu0,
                            fe_var0 = result_initial.fe_var0,
                            noise_var0 = result_initial.noise_var0))

# Run Experiment

stat_beta = experiment(n_step, rewards, stat_beta0)
ucb_beta = [getucb_beta(stat; α=αs[1]) for stat in stat_beta]

stat_UCB = experiment(n_step, rewards, stat_UCB0)
ucb_ucb = [getucb_ucb(stat; α=αs[2]) for stat in stat_UCB]

stat_LIME_re = experiment(n_step, rewards, features, stat_LIME_re0)
ucb_lime = [getucb_lime(stat, stat_LIME_fe0, [1]; α=αs[3]) for stat in stat_LIME_re]

# Plot UCBs for all candidates
mydata =  [ucb_beta, ucb_ucb, ucb_lime]
mycolor = [:red, :blue, :green]
mytitle = ["Beta's Quantile (q = $(αs[1]))", "UCB Algorithm (α = $(αs[2]))", "LIME Algorithm (α = $(αs[3]))"]
myylims = (minimum(minimum(minimum.(i)) for i in mydata),
           maximum(maximum(maximum.(i)) for i in [ucb_beta, ucb_ucb[begin + 1:end], ucb_lime]) 
           )
arrplt_ucb = [plot_ucb(mydata[i], αs[i]; linecolor=mycolor[i], title=mytitle[i]) for i in 1:length(mydata)]
plt_ucb = plot(arrplt_ucb...,
           size=(800, 460),
           layout=(1, 3),
           ylims=myylims,
           titlefontsize=8,
           legendfontsize = 8)
savefig(plt_ucb, "figure/fig_update_case$(id)_t$n_step.pdf") 

# Additional plot for bounds
mydata_ucb = [getindex.(i, :ucb) for i in mydata]
plt_ucb2 = plot(mydata_ucb, lc = permutedims(mycolor), label = permutedims(mytitle), xlabel = "step", title = "Bounds")
savefig(plt_ucb2, "figure/fig_update_case$(id)_t$(n_step)_bounds.pdf") 
# Check quantiles
qt_beta = cdf.(stat_beta, getindex.(ucb_beta, :ucb))
println("Quantile of Beta Distribution:\n", round.(qt_beta .* 100))
# Check variance ratio
ratio = stat_LIME_re0.prior.re_var0 / stat_LIME_re0.prior.noise_var0
println("Ratio of variances:\n", (ratio[1]))

println("\n ========== Test ends ========== \n")