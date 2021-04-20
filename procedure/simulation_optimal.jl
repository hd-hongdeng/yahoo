
# Check the share of events used in the stream
far_random = maximum(maximum.(skipmissing.(result_random)))
share_random = far_random / nrow(stream_evl) * 100
println("Share of used events for random policy: ", round(share_random, digits = 2), "%")

# Subset data stream for optimal policies
stream_4opt_far = Int(ceil(far_random * 1.05))
stream_4opt = @view stream_evl[1:stream_4opt_far, :]

# Prepare for optimal policies
# (i) Homogeneous optimal policy
homobest_ctr = @pipe summary_art(stream_4opt, :display) |>
      select(_, :display, :ctr) |>
      sort(_, [:ctr, :display], rev = true) # Make sure the df is sorted by ctr!!!
# (ii) Best-per-profile policy
stream_4opt_byprofile = groupby(stream_4opt, :profile)
bpp_ctr = Dict{Any,DataFrame}()
for i = 1:stream_4opt_byprofile.ngroups
    key_one = keys(stream_4opt_byprofile)[i].profile
    value_one = @pipe summary_art(values(stream_4opt_byprofile)[i], :display) |>
          select(_, :display, :ctr) |>
          sort(_, [:ctr, :display], rev = true)
    bpp_ctr[key_one] = value_one
end
# (iii) Heterogeneous optimal policies with OLS/Logit/LMEM
# OLS/Logit regression per article
stream_4opt_byart = @pipe select(stream_4opt, :display, :click, user_features[Not(1)]) |>
      groupby(_, :display)
lhs = :click
rhs = Expr(:call, :+, user_features[Not(1)]..., 1)
#rhs = Expr(:call, :+, user_features[Not(1)]...)
regs_ols = Dict{String,Any}()
regs_logit = Dict{String,Any}()
for i = 1:stream_4opt_byart.ngroups

    # Data for one article
    sdf = values(stream_4opt_byart)[i]

    # Article name
    article = keys(stream_4opt_byart)[i].display

    # OLS
    reg_ols = lm(@eval(@formula($(lhs) ~ $(rhs))), sdf)
    regs_ols[article] = reg_ols

    # Logit
    reg_logit = glm(@eval(@formula($(lhs) ~ $(rhs))), sdf, Bernoulli(), LogitLink())
    regs_logit[article] = reg_logit
end
# LMEM regression on all articles
#fm = @formula(click ~ 1 + u19 + u17 + u16 + (1 + u19 + u17 + u16 | display))
#fm = @formula(click ~ 1 + u19 + (1 + u19 | display))
fm = @formula(click ~ 1 + (1 | display))
reg_lmem2 = fit(MixedModel, fm, stream_4opt)
hetebest_lmem_re = DataFrame(only(raneftables(reg_lmem2)))
hetebest_lmem_fe = fixef(reg_lmem2)

# Find the space of the feature vector
space = unique(stream_4opt[!, user_features])
space.profile = [Tuple(r) for r in eachrow(space)]
sort!(space, :profile)

# Arm-selection references by OLS/Logit/LMEM regression
hetebest_ols_strategy = Dict{Any,DataFrame}()
hetebest_logit_strategy = Dict{Any,DataFrame}()
hetebest_lmem_strategy = Dict{Any,DataFrame}()
for i = 1:nrow(space) # Loop over possible profiles

    # Given the current profile
    profile_one = space.profile[i]
    context_one = space[i:i, Not(:profile)]
    𝐱 = Array(space[i, Not(:profile)])

    # Prediction based on OLS
    arts = Array{Any}(undef, length(regs_ols))
    predictions = Array{Any}(undef, length(regs_ols))
    for (i, k) in enumerate(keys(regs_ols))
        if p == 1
            predictions[i] = predict(regs_ols[k])
        else
            predictions[i] = predict(regs_ols[k], context_one)[1]
        end
        arts[i] = k
    end
    prediction_ols = DataFrame(:display => arts, :prediction => predictions)
    sort!(prediction_ols, [:prediction, :display], rev = true)
    hetebest_ols_strategy[profile_one] = prediction_ols

    # Prediction based on logit
    arts = Array{Any}(undef, length(regs_logit))
    predictions = Array{Any}(undef, length(regs_logit))
    for (i, k) in enumerate(keys(regs_logit))
        if p == 1
            predictions[i] = predict(regs_logit[k])
        else
            predictions[i] = predict(regs_logit[k], context_one)[1]
        end
        arts[i] = k
    end
    prediction_logit = DataFrame(:display => arts, :prediction => predictions)
    sort!(prediction_logit, [:prediction, :display], rev = true)
    hetebest_logit_strategy[profile_one] = prediction_logit

    # Prediction based on LMEM
    predictions = map(
        (𝐛 -> 𝐱' * Vector(𝐛) + 𝐱' * hetebest_lmem_fe),
        eachrow(hetebest_lmem_re[!, Not(1)])
    )
    prediction_lmem =
        DataFrame(:display => hetebest_lmem_re.display, :prediction => predictions)
    sort!(prediction_lmem, [:prediction, :display], rev = true)
    hetebest_lmem_strategy[profile_one] = prediction_lmem
end

# (i) Homogeneous optimal policy
println("\n -- Run: Homo. Opt. --")
issorted(homobest_ctr, [:ctr, :display], rev = true) || error("homobest_ctr is not sorted!") # Make sure the df is sorted by ctr!!!
history_homobest = simulator_homobest(stream_4opt, trystep, homobest_ctr)
any(ismissing, history_homobest) && println("The history does not reach desired steps!")
# (ii) Best-per-profile policy
println("\n -- Run: BPP --")
for (k, v) in bpp_ctr
    issorted(v, [:ctr], rev = true) || error("bpp_ctr's profile $k is not sorted!") # Make sure the df is sorted by ctr!!!
end
history_bpp = simulator_hetebest(stream_4opt, trystep, bpp_ctr)
any(ismissing, history_bpp) && println("The history does not reach desired steps!")
# (iii) Heterogeneous optimal policies with OLS/Logit/LMEM
println("\n -- Run: Hete-OLS --")
for (k, v) in hetebest_ols_strategy
    issorted(v, [:prediction], rev = true) ||
        error("hetebest_ols_strategy's profile $k is not sorted!") # Make sure the df is sorted by ctr!!!
end
history_besthete_ols = simulator_hetebest(stream_4opt, trystep, hetebest_ols_strategy)
any(ismissing, history_besthete_ols) && println("The history does not reach desired steps!")
println("\n -- Run: Hete-Logit --")
for (k, v) in hetebest_logit_strategy
    issorted(v, [:prediction], rev = true) ||
        error("hetebest_logit_strategy's profile $k is not sorted!") # Make sure the df is sorted by ctr!!!
end
history_besthete_logit = simulator_hetebest(stream_4opt, trystep, hetebest_logit_strategy)
any(ismissing, history_besthete_logit) && println("The history does not reach desired steps!")
println("\n -- Run: Hete-LMEM --")
for (k, v) in hetebest_lmem_strategy
    issorted(v, [:prediction], rev = true) ||
        error("hetebest_lmem_strategy's profile $k is not sorted!") # Make sure the df is sorted by ctr!!!
end
history_besthete_lmem = simulator_hetebest(stream_4opt, trystep, hetebest_lmem_strategy)
any(ismissing, history_besthete_lmem) && println("The history does not reach desired steps!")

result_opt = (
    history_homobest = history_homobest,
    history_bpp = history_bpp,
    history_besthete_ols = history_besthete_ols,
    history_besthete_logit = history_besthete_logit,
    history_besthete_lmem = history_besthete_lmem,
)

# Save simulated histories of optimal policies
open("case_p1/result/result_opt_$(trystep)_$(n_sm)_$p.bin", "w") do io
    serialize(io, result_opt)
end

# Check the share of events used in the stream
far_opt = extrema([h[end] for h in result_opt])
share_opt = far_opt ./ nrow(stream_4opt) .* 100
println(
    "Share of used events for opt. policy: ",
    round(share_opt[1], digits = 2),
    "%~",
    round(share_opt[2], digits = 2),
    "%",
)
