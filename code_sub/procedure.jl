#===============================================================================
Title: Module - Import Yahoo dataset
Updated: 2021-03-29
Description:
- Import data stream for evaluation;
- Import results for prior initialization;
- Run simulator-based off-line evaluation;
To-do:
- Tuning learning rate for UCB-typed algorithms with tunning data.
===============================================================================#

#--- Import data

# Load data stream for evaluation (0.5 seconds).
#stream_evl = loadjdf("data/stream_evl.jdf") # In Atom
stream_evl = loadjdf("data/stream_evl.jdf") |> DataFrame # In VS code
# Extract column names of the user features
user_features = propertynames(stream_evl)[occursin.(r"u", names(stream_evl))]
# Compute feature length.
p = length(user_features)

#--- Prior initialization

# Import estimation results for initializing prior parameters.
result_initial = open(deserialize, "result/result_initial.bin")
# Construct priors
ucb1estmu0 = result_initial.ucb1estmu0
ucb1estvar0 = result_initial.ucb1estvar0
prior_limeucb = Dict(
    "fe_mu0" => result_initial.fe_mu0,
    "fe_var0" => result_initial.fe_var0,
    "re_var0" => result_initial.re_var0,
    "noise_var0" => result_initial.noise_var0,
)

#--- Offline evaluation

# Specify the desired number of steps for the sythetic histories
trystep = 5000
# Specify the number of MC simulations
n_sm = 10
# Prepare random seeds for multi-threading
Random.seed!(2021)
# Save random seeds for multi-threading
random_seed = rand(1:10^5, n_sm)
open("result/random_seed.bin", "w") do io
    serialize(io, random_seed)
end

# Run random policy first
println("\n -- Run: Random --")
result_random = simulator_random_mtp(n_sm, stream_evl, trystep, random_seed)
# Save simulated histories of random policy
open("result_limeucb/result_random.bin", "w") do io
    serialize(io, result_random)
end

# Run optimal policies
include("run_optimal.jl")

# Run UCB-typed algorithms
learning_rate = [0.1, 1, 1.5, 2]
n_α = length(learning_rate)
open("result/learning_rate.bin", "w") do io
    serialize(io, learning_rate)
end

println("\n -- Run: UCB-1 --")
result_ucb = Array{Any}(undef, n_α)
for i = ProgressBar(1:n_α)
    α = learning_rate[i]
    #println("Progress:", round(i / n_α * 100), "%")
    result_ucb[i] = simulator_ucb_mtp(
        n_sm,
        stream_evl,
        trystep,
        random_seed;
        ucb1estmu0 = ucb1estmu0,
        ucb1estvar0 = ucb1estmu0,
        α = α,
    )
end
open("result/result_ucb.bin", "w") do io
    serialize(io, result_ucb)
end


println("\n -- Run: LinUCB --")
result_linucb = Array{Any}(undef, n_α)
for i = ProgressBar(1:n_α)
    α = learning_rate[i]
    #println("Progress:", round(i / n_α * 100), "%")
    result_linucb[i] =
        simulator_linucb_mtp(n_sm, stream_evl, trystep, random_seed, p; α = α)
end
open("result/result_linucb.bin", "w") do io
    serialize(io, result_linucb)
end

println("\n -- Run: LIME-UCB (Fixed) --")
result_limeucb = Array{Any}(undef, n_α)
for i = ProgressBar(1:n_α)
    α = learning_rate[i]
    #println("Progress:", round(i / n_α * 100), "%")
    result_limeucb[i] = simulator_limeucb_mtp(
        n_sm,
        stream_evl,
        trystep,
        random_seed,
        prior_limeucb,
        p;
        α = α,
        update_fe = false,
    )
end
open("result/result_limeucb.bin", "w") do io
    serialize(io, result_limeucb)
end
