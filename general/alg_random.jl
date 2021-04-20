#===============================================================================
Title: Function - algorithm : random
Date: 2021-03-29
Description:
Random policy.
===============================================================================#

"""
Decision rule at time t
"""
function run_random(armset_t::Vector{String})::String

    chosen = rand(armset_t)

    return chosen
end


"""
Create one simulated history with given steps
"""
function simulator_random(stream::DataFrame, 
                          maxstep::Int, seed::Int
                          )::Array{Union{Missing, Int64},1}

    # Initialization
    selected_events = Array{Union{Missing,Int}}(missing, maxstep)
    Random.seed!(seed)

    # Create the history for algorithm best
    j, i = 0, 0
    while (j < maxstep && i < nrow(stream))

        # Select one candidate event
        i += 1         # One event in stream is used
        candidate_t = stream[i, :]

        # Extract the arm set
        armset_t = collect(skipmissing(candidate_t[r"col"]))

        # Make a decision
        chosen_t = run_random(armset_t)

        # Factual display
        display_t = candidate_t.display

        # Retain candidate event?
        if chosen_t == display_t

            # One event is added to the history
            j += 1

            # Store the index of the retained event
            selected_events[j] = i

        end
    end

    return selected_events

end



"""
Create multiple simulated histories with given steps
"""
function simulator_random_mtp(n::Int, # Number of MC simulations 
                              stream::DataFrame, # Event stream
                              maxstep::Int,  # Number of steps per history
                              random_seed::Vector{Int} # Pre-drawn random seeds
                              )::Array{Array{Union{Missing, Int64},1},1}

    # Initialization: `n` histories with length `maxstep`
    mtp_histories = Array{Vector{Union{Missing,Int}}}(undef, n)

    # Run n simulations
    Threads.@threads for i = ProgressBar(1:n)
        # Set random seed
        seed = random_seed[i]
        # Create one history
        mtp_histories[i] = simulator_random(stream, maxstep, seed)
    end

    return mtp_histories
end

