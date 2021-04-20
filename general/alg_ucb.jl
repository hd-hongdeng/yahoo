#===============================================================================
Title: Function - algorithm : ucb
Date: 2020-03-29
Description:
Creating simulated histories for ucb.
============================================================================== =#

"""
Decision rule at time t
"""
function run_ucb(armset_t::Vector{String}, 
                 display_ucb::DataFrame)::String

    # Extract information for arm set at time t
    display_ucb_t = filter(:display => in(Set(armset_t)), display_ucb)

    # Check whether the articles are new
    if any(iszero, display_ucb_t.nchosen) == false
        # Find the article with max. est. ucb
        estucb_max = maximum(display_ucb_t.estucb)
        chosen_idx = findall(isequal(estucb_max), display_ucb_t.estucb)
    # new artile
    else
        # Choose one new article randomly
        chosen_idx = findall(iszero, display_ucb_t.nchosen)
    end

    chosen = display_ucb_t.display[rand(chosen_idx)]

    return chosen
end


"""
Update algorithm's parameter
"""
function update_ucb(display_ucb_idx::DataFrameRow, 
                    reward_t::Real,
                    α::Real
                    )::DataFrameRow
    # Update the # of times for which article has been chosen
    display_ucb_idx.nchosen += 1
    # Compute the new information
    delta = (reward_t - display_ucb_idx.estmu) / display_ucb_idx.nchosen
    # Update the estimated mean reward for this article
    display_ucb_idx.estmu += delta
    # Update the estimated ucb for this article
    display_ucb_idx.estucb = display_ucb_idx.estmu + α * sqrt(display_ucb_idx.estvar / display_ucb_idx.nchosen)

    return display_ucb_idx
end

"""
Create one simulated history with given steps
"""
function simulator_ucb(
    stream::DataFrame,
    maxstep::Int,
    seed::Int,
    prior_ucb::Dict;
    α::Real=sqrt(2)
)::Array{Union{Missing, Int64},1}

    # Initialization: one simulated history
    selected_events = Array{Union{Missing,Int}}(missing, maxstep)
    # Set random seed
    Random.seed!(seed)

    # Initialization: algorithm's prior
    display_ucb = DataFrame(
        :display => unique(stream.display),
        :estucb => prior_ucb["ucb1estmu0"],
        :nchosen => 0,
        :estmu => prior_ucb["ucb1estmu0"],
        :estvar => prior_ucb["ucb1estvar0"],
    ) # 0.04s

    # Create one history for the algorithm
    j, i = 0, 0
    while (j < maxstep && i < nrow(stream))

        # Select one candidate event
        i += 1         # One event in stream is used
        candidate_t = stream[i, :]

        # Extract the arm set
        armset_t = collect(skipmissing(candidate_t[r"col"]))

        # Run algorithm
        chosen_t = run_ucb(armset_t, display_ucb)

        # Factual display
        display_t = candidate_t.display

        # Retain the candidate event?
        if chosen_t == display_t

            # One event is added to the history
            j += 1

            # Reveal feedback
            reward_t = candidate_t.click

            # Store the index of the retained event
            selected_events[j] = i

            # Update algorithm's parameter
            # Find the chosen article
            idx = findfirst(isequal(chosen_t), display_ucb.display)
            # Update chosen article's parameters
            display_ucb[idx,:] = update_ucb(display_ucb[idx,:], reward_t, α)
        end
    end

    return selected_events

end


"""
Create multiple simulated histories with given steps 
"""
function simulator_ucb_mtp(
    n::Int, # Number of MC simulations 
    stream::DataFrame, # Event stream
    maxstep::Int, # Number of steps per history
    random_seed::Vector{Int}, # Pre-drawn random seeds
    prior_ucb::Dict; # Algorithm's prior
    α::Real=sqrt(2) # Algorithm's learning rate
)::Array{Array{Union{Missing, Int64},1},1}

    # Initialization: `n` histories with length `maxstep`
    mtp_histories = Array{Array{Union{Missing,Int}}}(undef, n)

    # Run n simulations
    Threads.@threads for i = ProgressBar(1:n)
        # Set random seed
        seed = random_seed[i]
        # Create one history
        mtp_histories[i] = simulator_ucb(stream, maxstep, seed, prior_ucb; α=α)
    end
    
    return mtp_histories
end