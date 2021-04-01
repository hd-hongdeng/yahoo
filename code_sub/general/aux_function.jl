"""
Output an object to Latex
"""
function outlatex(object, file; format = "%.2f", environment = :tabular)
    s = latexify(object, env = environment, latex = false, fmt = format)
    open(file, "w") do io
        write(file, s)
    end
end

"""
Transpose a DataFrame with one row
"""
function trandf(df::DataFrame)::DataFrame
    value = collect(df[1, :])
    id = names(df)
    tdf = DataFrame(:id => id, :value => value)
end

"""
Summarize article's infomation: ctr, num. of obs. and duration in hour
"""
function summary_art(df::Union{DataFrame,SubDataFrame}, groupname::Union{Symbol,String})

    # Summarize mean click and number of obs.
    stats = @pipe groupby(df, groupname) |>
          combine(
              _,
              :click => mean => :ctr,
              nrow => :numobs,
              :date_neg6 => minimum => :first,
              :date_neg6 => maximum => :last,
          ) |>
          sort(_, [:ctr, groupname], rev = true)

    # Compute duration in hour
    v = stats.last - stats.first
    d = map((x -> round(Second(x).value / (60 * 60), digits = 2)), v)
    insertcols!(stats, 4, :duration => d)

    return stats
end

"""
Check the frequency of features
"""
function checkfeature(profile_array)
    stat = @pipe DataFrame(:profile => profile_array) |>
          groupby(_, :profile) |>
          combine(_, nrow => :numobs) |>
          sort(_, :profile)
    stat.percent = stat.numobs ./ sum(stat.numobs) .* 100
    return stat
end

"""
Compute the CTR for a created history
"""
function checkreward(click)
    ctr = mean(click)
    ctrs = cumsum(click) ./ (1:length(click))
    return ctr, ctrs
end

"""
Summarize CTR, arm-selection strategy, profile freq and calendar time for one history
"""
function summary_onehistory(
    stream::Union{DataFrame,SubDataFrame},
    selected_events::Vector{Int},
)::NamedTuple
    # Extract selected events from the data stream
    restream = @view stream[selected_events, [:date_neg6, :profile, :display, :click]]
    # Compute CTRs
    avgctr, cumctr = checkreward(restream[!, :click])
    # Count # of times each article being selected and its ctr
    countart = @pipe groupby(restream, :display) |>
          combine(_, nrow => :num, :click => mean => :ctr) |>
          transform(_, :ctr => (x -> x * 100) => :ctr)
    # Count the frequency of each profile of feature
    countprof = checkfeature(restream[!, :profile])
    # Extract the actual calender time
    lasting =
        (firstevent = restream[begin, :date_neg6], lastevent = restream[end, :date_neg6])

    return (
        avgctr = avgctr * 100,
        cumctr = cumctr .* 100,
        countart = countart,
        countprof = countprof,
        lasting = lasting,
    )
end

"""
Summarize CTR, arm-selection strategy, profile freq and calendar time for multiple history
"""
function summary_histories(
    stream::Union{DataFrame,SubDataFrame},
    result_algorithm,
)::NamedTuple
    # Summary statistics for the histories
    algorithm_mc = length(result_algorithm)
    a = Array{Any}(undef, algorithm_mc)
    for i = 1:algorithm_mc
        # One History
        selected_events = result_algorithm[i]
        any(ismissing, selected_events) && error("History is not complete!")
        selected_events = convert(Vector{Int}, selected_events)
        # Analysis
        a[i] = summary_onehistory(stream, selected_events)
    end

    return (
        avgctrs_histories = getproperty.(a, :avgctr),
        cumctrs_histories = getproperty.(a, :cumctr),
        countarts_histories = getproperty.(a, :countart),
        countprofs_histories = getproperty.(a, :countprof),
        lastings_histories = getproperty.(a, :lasting),
    )
end

"""
Tentative: Plot UCB, center and width of LIME-UCB at the last round
"""

function plot_limeprediction(result_limeucb_one, stream, limeucbalpha)
    # To be improved later...

    # Extract the mathched events
    restream = @view stream[result_limeucb_one.selected_events,:]

    # Extract articles shown in the matched events
    restream_art = unique(restream.display)

    # Count the frequency of each profile of the context
    contextseen = checkfeature(restream.profile)

    # Extract LIMEUCB's algorithmic parameters from the last round
    display_limeucb = result_limeucb_one.display_limeucb
    fe_limeucb = result_limeucb_one.fe_limeucb

    # Compute the predictions by limeucb for each profile
    actual_ctrs = Array{Any}(undef, nrow(contextseen)) # Actual CTR per article per context
    ucbs = Array{Any}(undef, nrow(contextseen))
    centers = Array{Any}(undef, nrow(contextseen))
    stds = Array{Any}(undef, nrow(contextseen))
    #A = display_limeucb.display
    for i = 1:nrow(contextseen)
        actual_ctrs[i] = @pipe filter(:profile => ==(contextseen.profile[i]), restream) |>
                    groupby(_, :display) |>
                    combine(_, :click => mean => :ctr)
        feature_t = collect(contextseen.profile[i])
        est = ucbs_limeucb(restream_art, display_limeucb, fe_limeucb, feature_t, limeucbalpha).value
        ucbs[i] = getindex.(est, :ucb)
        centers[i] = getindex.(est, :center)
        stds[i] = getindex.(est, :width)
    end

    # Actual CTR per article per context
    df_ctrs = outerjoin(actual_ctrs...; on = :display, makeunique = true)
    sort!(df_ctrs, :display)

    # UCBs by LIME-UCB
    df_ucbs = hcat(DataFrame(:display => restream_art), DataFrame(ucbs))
    sort!(df_ucbs, :display)
    #@show df_ucbs

    # Estimated mean by LIME-UCB
    df_centers = hcat(DataFrame(:display => restream_art), DataFrame(centers))
    sort!(df_centers, :display)
    #@show df_centers

    # Estimated std by LIME-UCB
    df_stds = hcat(DataFrame(:display => restream_art), DataFrame(stds))
    sort!(df_stds, :display)
    #@show df_stds

    # Check order
    (df_ctrs.display == df_ucbs.display == df_centers.display == df_stds.display) || error("Mismatch in display!")

    # Find the maximum value of y values
    df_4ymax = [df_ctrs, df_ucbs, df_centers, df_stds]
    ymax = [maximum([maximum(skipmissing(c)) for c in eachcol(df[!,Not(:display)])]) for df in df_4ymax] |> maximum

    # Find articles with maximum UCB
    df_optart = mapcols(c -> df_ucbs.display[findmax(c)[2]], df_ucbs[!,Not(:display)])

    # Make plots
    plts = Array{Plots.Plot}(undef, nrow(contextseen))
    for i = 1:nrow(contextseen)

        # Plot ctr
        plts[i] =
        scatter(
            df_ctrs[!, i+1],
            ms = 3,
            mc = :darkgreen,
            ma = 0.6,
            markerstrokewidth = 0,
            leg = false,
            markershape = :diamond,
            title = string("Case $i: x=", contextseen.profile[i],
                            ", nr. = $(contextseen.numobs[i]) \n best = $(df_optart[1,i])")
        )
        # Plot estimated mean by LIME-UCB
        scatter!(
            df_centers[!, i+1],
            ms = 2,
            mc = :Gray24,
            markerstrokewidth = 0,
            leg = false,
            markershape = [:circle],
        )
        # Plot estimated std by LIME-UCB
        scatter!(
            df_centers[!, i+1] + df_stds[!, i+1],
            ms = 2,
            mc = :red,
            markerstrokewidth = 0,
            leg = false,
            markershape = [:hline],
        )
        # Connect the UCB point and estimated mean by LIME-UCB
        for k in 1:nrow(df_ucbs)
            x = [k,k]
            y = [df_ucbs[k, i+1], df_centers[k, i+1]]
            plot!(x, y, lw = 0.65, color = :Gray24)
        end
        # Plot UCB by LIME-UCB
        scatter!(
            df_ucbs[!, i+1],
            ms = 2,
            mc = :blue,
            markerstrokewidth = 0,
            leg = false,
            markershape = [:hline],
        )
        #vline!([length(restream_art)], lw = 2, linecolor = :Gray24,linestyle = :dot)
    end

    plt_all = plot(
        plts...,
        #layout = (2, 2),
        ylabel = "UCB",
        xlabel = "Article",
        ylims = (0, ymax+0.01),
        #legend = :topright,
        titlefontsize = 6,
        #legendfontsize = 6,
        labelfontsize = 6,
    )

    return plt_all
end
