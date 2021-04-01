#===============================================================================
Title: Function - algorithm : ucb
Date: 2020-03-29
Description:
Creating simulated histories for ucb.
===============================================================================#

"""
Multiple Simulator for algorithm: ucb1
"""
function simulator_ucb_mtp(
    n::Int,
    stream::DataFrame,
    maxstep::Int,
    random_seed::Vector{Int};
    ucb1estmu0::Real = 0,
    ucb1estvar0::Real = 1,
    α::Real = sqrt(2)
)::Vector{Array{Union{Missing,Int}}}

    println("Algorithm: UCB, MC: ", n)

    # Initialization-simulator
    mtp_histories = Array{Array{Union{Missing,Int}}}(undef, n)

    # Run n simulations
    Threads.@threads for i = ProgressBar(1:n)
        # Update the storage array
        seed = random_seed[i]
        mtp_histories[i] = simulator_ucb(stream, maxstep, seed;ucb1estmu0 = ucb1estmu0, ucb1estvar0 = ucb1estvar0,  α = α)
    end
    return mtp_histories
end

"""
Simulator for algorithm: ucb1
"""
function simulator_ucb(
    stream::DataFrame,
    maxstep::Int,
    #display_ucb::DataFrame,
    seed::Int;
    ucb1estmu0::Real = 0,
    ucb1estvar0::Real = 1,
    α::Real = sqrt(2)
)::Array{Union{Missing,Int}}

    # Initialization (history)

    selected_events = Array{Union{Missing,Int}}(missing, maxstep)
    Random.seed!(seed)

    # Initialization-ucb1
    display_ucb = DataFrame(
        :display => unique(stream.display),
        :estucb => ucb1estmu0,
        :nchosen => 0,
        :estmu => ucb1estmu0,
        :estvar => ucb1estvar0
    ) # 0.04s

    # Create the history for the algorithm
    j, i = 0, 0
    while (j < maxstep && i < nrow(stream))

        # Select one candidate event
        i += 1         # One event in stream is used
        candidate_t = stream[i, :]

        # Extract the arm set
        armset_t = collect(skipmissing(candidate_t[r"col"]))

        # Run algorithm ucb1
        chosen_t = run_ucb(armset_t, display_ucb)

        # Factual display
        display_t = candidate_t.display

        # Retain the candidate event?
        if chosen_t == display_t

            # One event is added to the history
            j += 1

            # Reveal feedback
            reward_t = candidate_t.click

            # Update history

            selected_events[j] = i

            # Update algorithm ucb1
            # Find the article that was chosen
            idx = findfirst(isequal(chosen_t), display_ucb.display)
            # Update the # of times for which article has been chosen
            display_ucb[idx, :nchosen] += 1
            # Compute the new information
            delta = (reward_t - display_ucb[idx, :estmu]) / display_ucb[idx, :nchosen]
            # Update the estimated mean reward for this article
            display_ucb[idx, :estmu] += delta
            # Update the estimated ucb for this article
            display_ucb[idx, :estucb] =
                display_ucb[idx, :estmu] + α * sqrt(display_ucb[idx, :estvar] / display_ucb[idx, :nchosen])

        end
    end

    return selected_events

end

"""
Run algorithm: ucb1
"""
function run_ucb(armset_t::Vector{String}, display_ucb::DataFrame)::String

    # Extract information for arm set at time t
    display_ucb_t = filter(:display => in(Set(armset_t)), display_ucb)

    # Check whether the articles are new in the dictionary
    if any(iszero, display_ucb_t.nchosen) == false
        # Find the article with max. est. ucb
        estucb_max = maximum(display_ucb_t.estucb)
        chosen_idx = findall(isequal(estucb_max), display_ucb_t.estucb)
        chosen = display_ucb_t.display[rand(chosen_idx, 1)[1]]
    # new artile
    else
        # Choose one new article randomly
        chosen_idx = findall(iszero, display_ucb_t.nchosen)
        chosen = display_ucb_t.display[rand(chosen_idx, 1)[1]]
    end

    return chosen
end
