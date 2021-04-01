
#=
#--- Load simulation results
result_random = open(deserialize, "data/result_random.bin")
result_opt = open(deserialize, "data/result_opt.bin")
result_ucb = open(deserialize, "data/result_ucb.bin")
result_linucb = open(deserialize, "data/result_linucb.bin")
result_limeucb = open(deserialize, "data/result_limeucb.bin")
=#

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
display(plt_ctr_opt)

# To Latex: Count # of times each article being chosen
countarts_opt = outerjoin(countarts...; on = :display, makeunique = true)
rename!(countarts_opt, Pair.(2:2:length(policynames)*2+1, string.(policynames, "(num)")))
rename!(
    countarts_opt,
    Pair.(2+1:2:length(policynames)*2+1+1, string.(policynames, "(ctr)")),
)
sort!(countarts_opt, :display)
#@show countarts_opt
outlatex(countarts_opt, "table/table_choice_short.tex"; format = "%.0f")

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
far_limeucb =
    maximum(maximum(maximum.(getproperty.(v, :selected_events))) for v in result_limeucb)
far = maximum([far_opt[2], far_random, far_ucb, far_linucb, far_limeucb])
share = far / nrow(stream_evl) * 100
println("\n Share of used events at maximum: $(round(share, digits=2))%")

# Check CTR obtained by each algorithm
algorithmnames = vcat(["Random"],
                       string.("UCB(", learning_rate, ")"),
                       string.("LinUCB(", learning_rate, ")"),
                       string.("LIME(", learning_rate, ")"))
result_limeucb_event = [getproperty.(v, :selected_events) for v in result_limeucb]
results = [result_random, result_ucb..., result_linucb..., result_limeucb_event...]

# CTRs over policies
avgctr_policies = Array{Any}(undef, length(results))
cumctr_policies = Array{Any}(undef, length(results))
for i = 1:length(results)
    # One policy
    alloutput = summary_histories(stream_evl, results[i])
    avgctr_policies[i] = mean(alloutput.avgctrs_histories)
    cumctr_policies[i] = alloutput.cumctrs_histories
end

# Plot CTRs
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
#push!(plts, plt_ctr_opt)
plt_all = plot(
    plts[Not(1)]...,
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
#savefig(plt_all, "figure/plt_ctrs_ucbs.pdf")


# Select the best ones from UCB-typed algorithms for output
ucbidx = findmax(avgctr_policies[2:5])
result_ucb_best = result_ucb[ucbidx[2]]
linucbidx = findmax(avgctr_policies[6:9])
result_linucb_best = result_linucb[linucbidx[2]]
limeucbidx = findmax(avgctr_policies[10:13])
result_limeucb_event_best = result_limeucb_event[limeucbidx[2]]
algorithmnames = ["Random", "UCB($(learning_rate[ucbidx[2]]))",
                            "LinUCB($(learning_rate[linucbidx[2]]))",
                            "LIME($(learning_rate[limeucbidx[2]]))"]
results = [result_random, result_ucb_best, result_linucb_best, result_limeucb_event_best]

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

# Plot CTRs
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
#savefig(plt_all, "figure/plt_ctrs.pdf")

# Count choices
choice_all_alg = outerjoin(choice_policies...; on = :display, makeunique = true)
rename!(choice_all_alg, Pair.(2:2:length(algorithmnames)*2+1, string.(algorithmnames, "(num)")))
rename!(choice_all_alg, Pair.(2+1:2:length(algorithmnames)*2+1+1, string.(algorithmnames, "(ctr)")))
choice_latex =
    outerjoin(select(countarts_opt, :display, r"LMEM"), choice_all_alg; on = :display)
sort!(choice_latex, :display)
#@show choice_latex
#outlatex(choice_latex, "table/table_choice.tex"; format = "%.0f")

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
#outlatex(freq_latex1, "table/table_feature1.tex"; format = "%.0f")
#outlatex(freq_latex2, "table/table_feature2.tex"; format = "%.0f")

# Calender time
lasting_all_alg = DataFrame(lasting_policies)
#rename!(lasting_all_alg, [:Begin, :End])
insertcols!(lasting_all_alg, 1, :policy => algorithmnames)
lasting_latex = vcat(lastings_opt, lasting_all_alg)
@show lasting_latex
#outlatex(lasting_latex, "table/table_time.tex"; format = "%.0f")

#--- LIME-UCB Investigation

# Pick one learning rate
limeucbalpha = learning_rate[limeucbidx[2]]
result_limeucb_fixa = result_limeucb[limeucbidx[2]]
# Pick one history
ctrs_limeucb = [mean(stream_evl[ntp.selected_events, :click]) for ntp in result_limeucb_fixa]
result_limeucb_one = result_limeucb_fixa[findmax(ctrs_limeucb)[2]]
println("highest avg ctr: ", findmax(ctrs_limeucb)[1]*100)

plt_ucbs = plot_limeprediction(result_limeucb_one, stream_evl, limeucbalpha)
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
