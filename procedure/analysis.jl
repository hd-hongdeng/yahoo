#===============================================================================
Title: Module - Analysis
Updated: 2021-04-13
Description:
- Summarize the performance of different algorithms
To-do:
- This script is messy, and used for analysing rewards only.
===============================================================================#

# Number of features
#user_features = [:u1, :u19]
user_features = [:u1]
p = length(user_features)

# Simulation times

# Specify the desired number of steps for the sythetic histories
trystep = 50_000
# Specify the number of MC simulations
n_sm = 10

#--- Import simulation results
result_random = open(deserialize, "case_p1/result/result_random_$(trystep)_$(n_sm)_$p.bin")
result_opt = open(deserialize, "case_p1/result/result_opt_$(trystep)_$(n_sm)_$p.bin")
result_ucb = open(deserialize, "case_p1/result/result_ucb_$(trystep)_$(n_sm)_$p.bin")
result_linucb = open(deserialize, "case_p1/result/result_linucb_$(trystep)_$(n_sm)_$p.bin")
result_limeucb = open(deserialize, "case_p1/result/result_limeucb_$(trystep)_$(n_sm)_$p.bin")
random_seed = open(deserialize, "case_p1/result/random_seed_$(trystep)_$(n_sm)_$p.bin")

# --- Import best learning rate 
result_tunning = open(deserialize, "case_p1/result/result_tunning.bin")

# Load data stream
stream_evl = loadjdf("limeucb_evaluation_2021/data/modify3.jdf") |> DataFrame # In VS code, 36 seconds
propertynames(stream_evl) |> println

# Extra data for evaluation
issorted(stream_evl, :date_neg6)
filter!(:date_neg6 => (x -> x >= DateTime(2011, 10, 12, 8)), stream_evl)
select!(stream_evl, :time, :date_neg6, :display, :click, user_features, r"col")
transform!(stream_evl, user_features => ByRow(tuple) => :profile)

# Check the share of events used in the stream
far_random = maximum(maximum.(skipmissing.(result_random)))
share_random = far_random / nrow(stream_evl) * 100
println("Share of used events for random policy: ", round(share_random, digits = 2), "%")

# Subset data stream for optimal policies
stream_4opt_far = Int(ceil(far_random * 1.05))
stream_4opt = @view stream_evl[1:stream_4opt_far, :]

#--- Optimal policies
# Collect optimal policies' names
policynames = ["HomoBest", "BPP", "OLS", "Logit", "LMEM"]
# Summary statistics for the histories
a = Array{Any}(undef, length(result_opt))
for i = 1:length(result_opt)
    println("\n --- Analysis: $(policynames[i]) ---")
    # History of one policy
    selected_events = result_opt[i]
    any(ismissing, selected_events) && error("History is not complete!")
    selected_events = convert(Vector{Int}, selected_events)
    # Analysis
    a[i] = summary_onehistory(stream_4opt, selected_events)
end
avgctrs = getproperty.(a, :avgctr)
cumctrs = getproperty.(a, :cumctr)
countarts = getproperty.(a, :countart)
countprofs = getproperty.(a, :countprof)
lastings = getproperty.(a, :lasting)

# Plot CTRs
ymax = maximum.(cumctrs) |> maximum
labels =
    ["$(policynames[i]): $(round(avgctrs[i], digits = 2))" for i = 1:length(policynames)] |> permutedims
plt_ctr_opt = plot(
    cumctrs,
    label = labels,
    lw = 0.8,
    ylabel = "CTR (%)",
    xlabel = "Step",
    ylims = (0, ymax * 1.05),
    legend = :topright,
    titlefontsize = 8,
    legendfontsize = 8,
    labelfontsize = 8,
    title = "Optimal Policies",
)
#display(plt_ctr_opt)

# To Latex: Count # of times each article being chosen
countarts_opt = outerjoin(countarts...; on = :display, makeunique = true)
rename!(countarts_opt, Pair.(2:2:length(policynames)*2+1, string.(policynames, "(num)")))
rename!(
    countarts_opt,
    Pair.(2+1:2:length(policynames)*2+1+1, string.(policynames, "(ctr)")),
)
sort!(countarts_opt, :display)
#@show countarts_opt
outlatex(countarts_opt, "case_p1/table/table_choice_opt_$(trystep)_$(n_sm)_$p.tex"; format = "%.0f")

# Merge the frequency tables across optimal policies
countprofs_opt = outerjoin(countprofs...; on = :profile, makeunique = true)
sort!(countprofs_opt, :profile)
rename!(countprofs_opt, Pair.(2:2:ncol(countprofs_opt)-1, string.(policynames, "(num)")))
rename!(countprofs_opt, Pair.(2+1:2:ncol(countprofs_opt), string.(policynames, "(share)")))
#@show countprofs_opt

# List actual calender time across optimal policies
lastings_opt = DataFrame(lastings)
insertcols!(lastings_opt, 1, :policy => policynames)
#@show lastings_opt


#--- UCB-typed algorithms

# Check the share of events used in the stream
far_ucb = maximum(maximum(maximum.(v)) for v in result_ucb)
far_linucb = maximum(maximum(maximum.(v)) for v in result_linucb)
far_limeucb = maximum(maximum(maximum.(v)) for v in result_limeucb)
far = maximum([far_random, far_ucb, far_linucb, far_limeucb])
share = far / nrow(stream_evl) * 100
println("\n Share of used events at maximum: $(round(share, digits=2))%")

algorithmnames = ["Random", "UCB($(result_tunning.ucb))",
                            "LinUCB($(result_tunning.linucb))",
                            "LIME($(result_tunning.limeucb))"]
results = [result_random, result_ucb, result_linucb, result_limeucb]

# CTRs over policies
avgctr_policies = Array{Any}(undef, length(results))
cumctr_policies = Array{Any}(undef, length(results))
# Count choices
choice_policies = Array{Any}(undef, length(results))
# Frequency of features over policies
freq_policies = Array{Any}(undef, length(results))
# Actual calender time over policies
lasting_policies = Array{Any}(undef, length(results))

for i = 1:length(results)
    # One policy
    result_one = results[i]
    mc_one = length(result_one)

    avgctr_histories,  cumctr_histories, countarts_histories, countprofs_histories, lastings_histories = summary_histories(stream_evl, result_one)
    avgctr_policies[i] = mean(avgctr_histories)
    cumctr_policies[i] = cumctr_histories

    choice_comb = outerjoin(countarts_histories...; on = :display, makeunique = true)
    nums = Array{Any}(undef, nrow(choice_comb))
    ctrs = Array{Any}(undef, nrow(choice_comb))
    for s = 1:nrow(choice_comb)
        row = choice_comb[s, Not(:display)] # one article
        num_total = sum(skipmissing(row[r"num"])) # total number of times being played
        nums[s] = num_total / mc_one # avg number of times being played over MC simulations
        ctrs[s] = sum(skipmissing(row[2*i-1] * row[2*i] for i = 1:mc_one)) / num_total # weighted avg of ctr over MC simulations
    end
    choice_policies[i] =
        DataFrame(:display => choice_comb.display, :nums => nums, :avgctrs => ctrs)

    freq_comb = @pipe outerjoin(countprofs_histories...; on = :profile, makeunique = true) |>
          coalesce.(_, 0) |> # Change missing to zeros
          select!(_, :profile, r"numobs") |>
          transform!(_, 2:ncol(_) => (+) => :total) |>
          transform!(_, :total => (x -> x / mc_one) => :avgnum) |>
          transform!(_, :total => (x -> x / sum(x) * 100) => :share) |>
          select!(_, :profile, :avgnum, :share)
    freq_policies[i] = freq_comb

    lasting_policies[i] = (
        firstevent = minimum([ntp.firstevent for ntp in lastings_histories]),
        lastevent = maximum([ntp.lastevent for ntp in lastings_histories]),
    )
end

#= Plot CTRs
plts = Array{Plots.Plot}(undef, length(cumctr_policies))
for i = 1:length(cumctr_policies)
    plt = plot(cumctr_policies[i], title = algorithmnames[i], label = nothing)
    hline!(
        linestyle = :dash,
        linecolor = :black,
        [avgctr_policies[i]],
        label = "Mean CTR ($(round(avgctr_policies[i], digits = 2)))",
    )
    plts[i] = plt
end
push!(plts, plt_ctr_opt)
plt_all = plot(
    plts...,
    ylabel = "CTR",
    xlabel = "Trial",
    ylims = (0, ymax),
    legend = :topright,
    titlefontsize = 8,
    legendfontsize = 8,
    labelfontsize = 8,
    size = (1200, 800),
)
display(plt_all)
=#


# Plot mean CTR over multiple histories
plts = Array{Plots.Plot}(undef, length(cumctr_policies))
ymaxs = Array{Any}(undef, length(cumctr_policies))
for i in 1:length(cumctr_policies)
    df = DataFrame(cumctr_policies[i])
    x = [mean(r) for r in eachrow(df)]
    y = [std(r) for r in eachrow(df)]
    y_up = x .+ y
    y_bottom = x .- y
    plts[i] = plot(x, labels = "μ")
    plot!(y_up, lc = :black, ls = :dash, labels = "μ ± σ")
    plot!(y_bottom, lc = :black, ls = :dash, labels = nothing)
    title!("$(algorithmnames[i]) \n μ ± σ (t=$(trystep)) = $(round(x[end],digits=2)) ± $(round(y[end],digits=2))")
    ymaxs[i] = maximum(y_up)
end
push!(plts, plt_ctr_opt)
ymax = maximum(ymaxs)
plt_all = plot(
    plts...,
    ylabel = "CTR",
    xlabel = "Trial",
    ylims = (0, ymax),
    legend = :topright,
    titlefontsize = 10,
    legendfontsize = 8,
    labelfontsize = 8,
    size = (1200, 800),
)
#display(plt_all)
savefig("case_p1/figure/plt_ctrs_$(trystep)_$(n_sm)_$p.pdf")

# Box plot
function get_stepctr(clicks; stepsize = 5000)
    totalstep = length(clicks)
    splitidx = range(1, totalstep; step = stepsize) |> Vector
    push!(splitidx, totalstep+1)
    stepclicks = [mean(clicks[splitidx[i]:(splitidx[i+1]-1)]) * 100 for i in 1:(length(splitidx)-1)]

    # For plots
    splitidx[end] = totalstep
    push!(stepclicks, stepclicks[end])

    return splitidx, stepclicks
end
function get_plotstepctr(stepclicks_mc, policyname)
    vs = [[v[i] for v in stepclicks_mc] for i in 1:length(stepclicks_mc[1])]
    plt = boxplot(vs, leg=false, color = :lightblue,
                  title = "$policyname",
                  titlefontsize = 10,
                  xlabel = "Interval",
                  ylabel = "Mean CTR (%)")
end
result_opt2 = [result_opt[:history_besthete_lmem], result_opt[:history_homobest]]
clicks_opt2 = [stream_evl[selected_events,:click] for selected_events in result_opt2]
stepclicks_opt2 = [tp[2] for tp in get_stepctr.(clicks_opt2)]
plt = plot(stepclicks_opt2, 
     lt = :steppost, 
     labels = permutedims(policynames[[end, begin]]),
     legend = :topleft,
     legendfontsize = 8,
     #title = "Optimal policies CTR per 5000 steps"
     )
savefig("case_p1/figure/plt_stepctrs_opt_$(trystep)_$(n_sm)_$p.pdf")

plts = Array{Plots.Plot}(undef, length(algorithmnames))
for i in 1:length(algorithmnames)
    clicks = [stream_evl[selected_events,:click] for selected_events in results[i]]
    stepclicks_mc = [tp[2] for tp in get_stepctr.(clicks)]
    plts[i] = get_plotstepctr(stepclicks_mc, algorithmnames[i])    
end
plts_all = plot(plts..., ylims = (0,15))
savefig("case_p1/figure/plt_stepctrs_ucb_$(trystep)_$(n_sm)_$p.pdf")

# Count choices
choice_all_alg = outerjoin(choice_policies...; on = :display, makeunique = true)
rename!(choice_all_alg, Pair.(2:2:length(algorithmnames)*2+1, string.(algorithmnames, "(num)")))
rename!(choice_all_alg, Pair.(2+1:2:length(algorithmnames)*2+1+1, string.(algorithmnames, "(ctr)")))
choice_latex =
    outerjoin(select(countarts_opt, :display, r"LMEM"), choice_all_alg; on = :display)
sort!(choice_latex, :display)
#@show choice_latex
outlatex(choice_latex, "case_p1/table/table_choice_$(trystep)_$(n_sm)_$p.tex"; format = "%.0f")

# Frequency of features
freq_all = outerjoin(freq_policies...; on = :profile, makeunique = true)
sort!(freq_all, :profile)
rename!(freq_all, Pair.(2:2:ncol(freq_all)-1, string.(algorithmnames, "(avgnum)")))
rename!(freq_all, Pair.(2+1:2:ncol(freq_all), string.(algorithmnames, "(share)")))
freq_latex = outerjoin(countprofs_opt, freq_all; on = :profile)
freq_latex1 = select(freq_latex, :profile, r"num")
#@show freq_latex1
freq_latex2 = select(freq_latex, :profile, r"share")
#@show freq_latex2
outlatex(freq_latex1, "case_p1/table/table_feature1_$(trystep)_$(n_sm)_$p.tex"; format = "%.0f")
outlatex(freq_latex2, "case_p1/table/table_feature2_$(trystep)_$(n_sm)_$p.tex"; format = "%.0f")

# Calender time
lasting_all_alg = DataFrame(lasting_policies)
#rename!(lasting_all_alg, [:Begin, :End])
insertcols!(lasting_all_alg, 1, :policy => algorithmnames)
lasting_latex = vcat(lastings_opt, lasting_all_alg)
@show lasting_latex
outlatex(lasting_latex, "case_p1/table/table_time_$(trystep)_$(n_sm)_$p.tex"; format = "%.0f")

# plot mean CTR over MC simulations for UCB and LIME only
plt = plot(legend = :bottomright, 
            xlabel = "step", 
            ylabel = "Mean CTR",
            title = "$(algorithmnames[4]) and $(algorithmnames[2])'s mean CTRs over $(n_sm) simualtions",
            titlefontsize = 8)
df = DataFrame(cumctr_policies[4])
x = [mean(r) for r in eachrow(df)]
#y = [std(r) for r in eachrow(df)]
plot!(x, labels = algorithmnames[4], linecolor = :blue)
#plot!(x .+ y, labels = nothing, linecolor = :blue, linestyle = :dash)
#plot!(x .- y, labels = nothing, linecolor = :blue, linestyle = :dash)
df = DataFrame(cumctr_policies[2])
x = [mean(r) for r in eachrow(df)]
#y = [std(r) for r in eachrow(df)]
plot!(x, labels = algorithmnames[2], linecolor = :red)
#plot!(x .+ y, labels = nothing, linecolor = :red, linestyle = :dash)
#plot!(x .- y, labels = nothing, linecolor = :red, linestyle = :dash)
#plot!(cumctrs[begin], labels = policynames[begin])
#plot!(cumctrs[end], labels = policynames[end])
plt2 = plot(cumctrs[[end, begin]], 
            labels = permutedims(policynames[[end,begin]]),
            title = "Optimal policies $(policynames[end]) and $(policynames[begin])'s CTRs",
            titlefontsize = 8)
plts = plot(plt, plt2, layout = (2,1))
savefig("case_p1/figure/plt_ctr_mc_$(trystep)_$(n_sm)_$p.pdf")

#=
# pick one history for ucb and limeucb, respectively

# Find best history of best LIME
idx = findmax([v[end] for v in cumctr_policies[4]])[2]
result_limeucb_best_one = result_limeucb_best[idx]
result_ucb_best_one = result_ucb_best[idx]

x = summary_onehistory(stream_evl,
                   convert(Vector{Int}, result_limeucb_best_one))
y = summary_onehistory(stream_evl,
                    convert(Vector{Int}, result_ucb_best_one))

plt = plot(legend = :bottomright,
            xlabel = "step",
            ylabel = "Cum. CTR",
           title = "The best history of $(algorithmnames[4]) (μ = $(x.avgctr)) \n and one history of $(algorithmnames[2]) with the same random seed (μ = $(y.avgctr)) \n and one history of $(algorithmnames[4]) without x (μ = $(z.avgctr))",
           titlefontsize = 8)
plot!(x.cumctr, labels = "$(algorithmnames[4])")
plot!(y.cumctr, labels = "$(algorithmnames[2])")
#plot!(cumctrs[begin], labels = policynames[begin])
#plot!(cumctrs[end], labels = policynames[end])
#plot!(z.cumctr, labels = "$(algorithmnames[4]) without x")

# Check number of new articles at each step
restream = stream_evl[result_limeucb_best_one, Cols(:date_neg6, r"col")]
oldart = collect(skipmissing(restream[1,r"col"]))
newarts = zeros(Int, nrow(restream))
for i in 1:nrow(restream)
    artpool = collect(skipmissing(restream[i,r"col"]))
    in_oldart = artpool .∈ (oldart,)
    newarts[i] = sum(.!(in_oldart))
    oldart = union(oldart, artpool)
end    
result_limeucb_best_one_newarts = copy(newarts)

restream = stream_evl[result_ucb_best_one, Cols(:date_neg6, r"col")]
oldart = collect(skipmissing(restream[1,r"col"]))
newarts = zeros(Int, nrow(restream))
for i in 1:nrow(restream)
    artpool = collect(skipmissing(restream[i,r"col"]))
    in_oldart = artpool .∈ (oldart,)
    newarts[i] = sum(.!(in_oldart))
    oldart = union(oldart, artpool)
end    
result_ucb_best_one_newarts = copy(newarts)

scat = plot(xlabel = "step",
            ylabel = "Number of new articles",)
plot!([result_limeucb_best_one_newarts,
             result_ucb_best_one_newarts],
             labels = permutedims(algorithmnames[[4,2]]))
plot(plt, scat, layout = (2,1))
#savefig("figure/plt_ctr_one_$(trystep)_$(n_sm).pdf")
# Count choices
countart = outerjoin(x.countart, y.countart, on = :display, makeunique = true)
rename!(countart, ["display", "LIME(num)", "LIME(ctr)", "UCB(num)", "UCB(ctr)"])
#outlatex(countart, "table/table_choice_one_$(trystep)_$(n_sm).tex"; format = "%.0f")
countprof = outerjoin(x.countprof, y.countprof, on = :profile, makeunique = true)
rename!(countprof, ["display", "LIME(num)", "LIME(share)", "UCB(num)", "UCB(share)"])
#outlatex(countprof, "table/table_profile_one_$(trystep)_$(n_sm).tex"; format = "%.0f")
lasting = DataFrame([x.lasting, y.lasting])
insertcols!(lasting, 1, :policy => ["LIME", "UCB"])
#outlatex(lasting, "table/table_time_one_$(trystep)_$(n_sm).tex"; format = "%.0f")

# Test: force limeucb to fit intercept only#--- Prior initialization

# Import estimation results for initializing prior parameters.
result_initial = open(deserialize, "result/result_initial_onefeature.bin")
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

seed = random_seed[idx]
@time stream_test = select(stream_evl, Not(:profile)) # 14 seconds
stream_test[!,:profile] .= Ref((1,0))
result_limeucb_nox = simulator_limeucb(stream_test, trystep, seed, prior_limeucb; α=2.5)

result_limeucb_nox = convert(Vector{Int}, result_limeucb_nox)
z = summary_onehistory(stream_test, result_limeucb_nox)

plot(z.cumctr)
=#

#=
#--- LIME-UCB Investigation

# Pick one history
ctrs_limeucb = [mean(stream_evl[ntp.selected_events, :click]) for ntp in result_limeucb_fixa]
result_limeucb_one = result_limeucb_fixa[findmax(ctrs_limeucb)[2]]
println("highest avg ctr: ", findmax(ctrs_limeucb)[1]*100)

#plt_ucbs = plot_limeprediction(result_limeucb_one, stream_evl, limeucbalpha)
#savefig(plt_ucbs, "figure/plt_ucbs.pdf")

# Check when the best choice entered the pool
ba = "id-603485"
sdf = @view stream_evl[result_limeucb_one.selected_events, :]
df = filter(:display => ==(ba), sdf)
click_ba = df.click
println("click: ", click_ba)
lasting_ba = @pipe getproperty(df,:date_neg6) |>
                (n = length(_), lasting = extrema(_))
println("Number of obs: ", lasting_ba.n, ";\n Duration: ", lasting_ba.lasting)
=#

println("\n ========== Analysis ends ========== \n")