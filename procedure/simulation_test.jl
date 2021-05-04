# --- Functions
function plot_cumctrs_histories(stream_evl, result_one, n_step; title = "")
    cumctrs_histories = summary_histories(stream_evl, result_one).cumctrs_histories
    means = [mean(getindex.(cumctrs_histories, i)) for i in 1:n_step]
    stds = [std(getindex.(cumctrs_histories, i)) for i in 1:n_step]
    plt = plot([means, means .+ stds, means .- stds], 
                lc = [:red :blue :blue], ls = [:solid :dash :dash], 
                label = ["μ" "μ ± σ" nothing],
                title = "$(title) \n μ(t=$(n_step))=$(round(means[end], digits = 2))%"
                )
    return plt    
end

# --- Simualtion

# Specify the desired number of steps for the sythetic histories
trystep = 500
# Specify the number of MC simulations
n_sm = 3
# Prepare random seeds for multi-threading
Random.seed!(2021)
# Save random seeds for multi-threading
random_seed = rand(1:10^5, n_sm)
# Load data stream
stream_evl = loadjdf("../data/modify3.jdf") |> DataFrame # In VS code, 36 seconds
# Number of features
# user_features = [:u1, :u19]
user_features = [:u1, :u19]
p = length(user_features)
# Extra data for evaluation
issorted(stream_evl, :date_neg6)
filter!(:date_neg6 => (x -> x >= DateTime(2011, 10, 12, 8)), stream_evl)
select!(stream_evl, :time, :date_neg6, :display, :click, user_features, r"col")
transform!(stream_evl, user_features => ByRow(tuple) => :profile)
transform!(stream_evl, :profile => ByRow(collect) => :feature)

# Uninformative prior
stat_UCB0 = UCBStat(0, 
                    0, 
                    (estmu0 = 0, noise_var0 = 1))

stat_LIME_re0 = LIMEStatRE(0,
                           Bool[], zeros(0, p),
                           zeros(p, p), zeros(p), Matrix(I, p, p),
                           (re_var0 = Matrix(I, p, p), noise_var0 = 1))
stat_LIME_fe0 = LIMEStatFE(fill(0, p), Matrix(I, p, p),
                           (fe_mu0 = fill(0, p), fe_var0 = Matrix(I, p, p), noise_var0 = 1))                           

#= # Informative prior
# Import estimation results for initializing prior parameters.
result_initial = open(deserialize, "case_p2/result/result_initial.bin")

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
                            noise_var0 = result_initial.noise_var0)) =#                            

# Simulation
println("\n -- Run: Random --")
result_random = simulator_random_mtp(n_sm, stream_evl, trystep, random_seed)
x = plot_cumctrs_histories(stream_evl, result_random, trystep; title = "Random")

println("\n -- Run: LIME-UCB (Fixed) --")
result_limeucb = simulator()
