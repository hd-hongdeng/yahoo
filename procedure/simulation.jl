#= ==============================================================================
Title: Module - Simulation
Updated: 2021-04-13
Description:
- Import data stream for evaluation;
- Import results for prior initialization;
- Run simulator-based off-line evaluation;
To-do:
- Tuning learning rate for UCB-typed algorithms with tunning data.
============================================================================== =#

# Number of features
#user_features = [:u1, :u19]
user_features = [:u1]
p = length(user_features)

# Simulation times

# Specify the desired number of steps for the sythetic histories
trystep = 50_000
# Specify the number of MC simulations
n_sm = 10
# Prepare random seeds for multi-threading
Random.seed!(2021)
# Save random seeds for multi-threading
random_seed = rand(1:10^5, n_sm)
open("case_p1/result/random_seed_$(trystep)_$(n_sm)_$p.bin", "w") do io
    serialize(io, random_seed)
end

# Load data stream
stream_evl = loadjdf("limeucb_evaluation_2021/data/modify3.jdf") |> DataFrame # In VS code, 36 seconds
#propertynames(stream_evl) |> println

# Extra data for evaluation
issorted(stream_evl, :date_neg6)
filter!(:date_neg6 => (x -> x >= DateTime(2011, 10, 12, 8)), stream_evl)
select!(stream_evl, :time, :date_neg6, :display, :click, user_features, r"col")
transform!(stream_evl, user_features => ByRow(tuple) => :profile)

# --- Import prior initialization

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
    "re_var0" => result_initial.re_var0,
    "noise_var0" => result_initial.noise_var0,
)

# Import estimation results for initializing prior parameters.
result_initial = open(deserialize, "result/result_initial_1feature.bin")
# Construct priors
prior_ucb = Dict(
    "ucb1estmu0" => result_initial.ucb1estmu0,
    "ucb1estvar0" => result_initial.ucb1estvar0,
)
prior_limeucb = Dict(
    "fe_mu0" => result_initial.fe_mu0,
    "fe_var0" => result_initial.fe_var0,
    "re_var0" => result_initial.re_var0,
    "noise_var0" => result_initial.noise_var0,
)


# --- Import best learning rate
result_tunning = open(deserialize, "case_p1/result/result_tunning.bin")
#result_tunning = (ucb = 1, linucb = 1, limeucb = 1)

# --- Simulation

# Run random policy first
println("\n -- Run: Random --")
result_random = simulator_random_mtp(n_sm, stream_evl, trystep, random_seed)

# Save simulated histories of random policy
open("case_p1/result/result_random_$(trystep)_$(n_sm)_$p.bin", "w") do io
    serialize(io, result_random)
end

# Run optimal policies
include("simulation_optimal.jl")

println("\n -- Run: UCB-1 --")

result_ucb = simulator_ucb_mtp(
        n_sm,
        stream_evl,
        trystep,
        random_seed,
        prior_ucb;
        α=result_tunning.ucb,
    )
open("case_p1/result/result_ucb_$(trystep)_$(n_sm)_$p.bin", "w") do io
    serialize(io, result_ucb)
end

println("\n -- Run: LinUCB --")
result_linucb =
        simulator_linucb_mtp(n_sm,
                            stream_evl,
                            trystep,
                            random_seed,
                            p;
                            α=result_tunning.linucb)
open("case_p1/result/result_linucb_$(trystep)_$(n_sm)_$p.bin", "w") do io
    serialize(io, result_linucb)
end

println("\n -- Run: LIME-UCB (Fixed) --")
result_limeucb = simulator_limeucb_mtp(
        n_sm,
        stream_evl,
        trystep,
        random_seed,
        prior_limeucb;
        α=result_tunning.limeucb,
        update_fe=false,
    )
open("case_p1/result/result_limeucb_$(trystep)_$(n_sm)_$p.bin", "w") do io
    serialize(io, result_limeucb)
end

println("\n ========== Simulation ends ========== \n")
