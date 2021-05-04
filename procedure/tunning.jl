#===============================================================================
Title: Module - Tunning of algorithms' learning rates
Updated: 2021-04-13
Description:
- Choose the best tunnign parameters for UCB, LinUCB and LIME-UCB.
===============================================================================#

# Number of features
#user_features = [:u1, :u19]
user_features = [:u1]
p = length(user_features)

# Simulation times

# Specify the desired number of steps for the sythetic histories
trystep = 30_000
# Specify the number of MC simulations
n_sm = 10
# Prepare random seeds for multi-threading
Random.seed!(2021)
# Save random seeds for multi-threading
random_seed = rand(1:10^5, n_sm)
open("case_p1/result/tunning_random_seed_$(trystep)_$(n_sm)_$p.bin", "w") do io
    serialize(io, random_seed)
end
# Sepcify the range of learning rates
learning_rate = [1, 1.5, 2, 2.5, 3]
n_α = length(learning_rate)
open("case_p1/result/tunning_learning_rate_$(trystep)_$(n_sm)_$p.bin", "w") do io
    serialize(io, learning_rate)
end

# Load data stream
stream_tun2 = loadjdf("../data/modify3.jdf") |> DataFrame # In VS code, 36 seconds
#propertynames(stream_tun2) |> println

# Extra data for tunning
issorted(stream_tun2, :date_neg6)
filter!(:date_neg6 => (x -> x >= DateTime(2011, 10, 5, 8) && Date(x) < Date(2011, 10, 12)), stream_tun2)
select!(stream_tun2, :time, :date_neg6, :display, :click, user_features, r"col")
transform!(stream_tun2, user_features => ByRow(tuple) => :profile)

#--- Import prior initialization

# Import estimation results for initializing prior parameters.
result_initial = open(deserialize, "case_p1/result/result_initial.bin")
# Construct priors
prior_ucb = Dict(
    "ucb1estmu0" => result_initial.ucb1estmu0,
    "ucb1estvar0" => result_initial.ucb1estvar0,
)
prior_limeucb = Dict(
    "fe_mu0" => result_initial.fe_mu0,
    "fe_var0" => result_initial.fe_var0,
    "re_var0" => result_initial.noise_var0 * Matrix(I, p, p),
    "noise_var0" => result_initial.noise_var0,
)

#--- Simulation

println("\n -- Run: UCB-1 --")
result_ucb = Array{Any}(undef, n_α)
for i = ProgressBar(1:n_α)
        result_ucb[i] = simulator_ucb_mtp(
        n_sm,
        stream_tun2,
        trystep,
        random_seed,
        prior_ucb;
        α = learning_rate[i],
    )
end
open("case_p1/result/tunning_result_ucb_$(trystep)_$(n_sm)_$p.bin", "w") do io
    serialize(io, result_ucb)
end
#result_ucb = open(deserialize, "case_p1/result/tunning_result_ucb_$(trystep)_$(n_sm)_$p.bin")

println("\n -- Run: LinUCB --")
result_linucb = Array{Any}(undef, n_α)
for i = ProgressBar(1:n_α)
    result_linucb[i] =
        simulator_linucb_mtp(n_sm,
                            stream_tun2,
                            trystep,
                            random_seed,
                            p;
                            α = learning_rate[i])
end
open("case_p1/result/tunning_result_linucb_$(trystep)_$(n_sm)_$p.bin", "w") do io
    serialize(io, result_linucb)
end
#result_linucb = open(deserialize, "case_p1/result/tunning_result_linucb_$(trystep)_$(n_sm)_$p.bin")

println("\n -- Run: LIME-UCB (Fixed) --")
result_limeucb = Array{Any}(undef, n_α)
for i = ProgressBar(1:n_α)
    result_limeucb[i] = simulator_limeucb_mtp(
        n_sm,
        stream_tun2,
        trystep,
        random_seed,
        prior_limeucb;
        α = learning_rate[i],
        update_fe = false,
    )
end
open("case_p1/result/tunning_result_limeucb_$(trystep)_$(n_sm)_$p.bin", "w") do io
    serialize(io, result_limeucb)
end

#--- Find the best learning rate

# Check CTR obtained by each algorithm
results = [result_ucb..., result_linucb..., result_limeucb...]

# CTRs over policies
avgctr_policies = Array{Any}(undef, length(results))
for i = 1:length(results)
    # One policy
    alloutput = summary_histories(stream_tun2, results[i])
    avgctr_policies[i] = mean(alloutput.avgctrs_histories)
end

# Select the best ones from UCB-typed algorithms for output
ucbidx = findmax(avgctr_policies[1:length(learning_rate)])
linucbidx = findmax(avgctr_policies[(length(learning_rate)+1):(length(learning_rate)+length(learning_rate))
                                    ]
                                    )
limeucbidx = findmax(avgctr_policies[(length(learning_rate)+length(learning_rate)+1):(length(learning_rate)+length(learning_rate)+length(learning_rate))
                                      ]
                                      )
result_tunning = (ucb = learning_rate[ucbidx[2]],
                  linucb = learning_rate[linucbidx[2]],
                  limeucb = learning_rate[limeucbidx[2]])

# Store tunning rates
open("case_p1/result/result_tunning.bin", "w") do io
    serialize(io, result_tunning)
end

println("\n ========== Tunning ends ========== \n")
