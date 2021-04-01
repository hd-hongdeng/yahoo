#===============================================================================
Title: Function - algorithm : random
Date: 2021-03-29
Description:
Random policy.
===============================================================================#

"""
Multiple Simulator for algorithm: random
"""
function simulator_random_mtp(n::Int, stream::DataFrame, maxstep::Int, random_seed::Vector{Int})

    println("Algorithm: Random, MC: ", n)
    # Initialization-simulator
    mtp_histories = Array{Vector{Int}}(undef, n)

    # Run n simulations
    Threads.@threads for i = ProgressBar(1:n)
        # Update the storage array
        seed = random_seed[i]
        mtp_histories[i] = simulator_random(stream, maxstep, seed)
    end

    return mtp_histories
end


"""
Simulator for algorithm: random
"""
function simulator_random(stream::DataFrame, maxstep::Int, seed::Int)::Vector{Int}

    # Initialization
    #reward, chosen, eventidx = Array{Union{Missing,Float64}}(missing, maxstep),
    #Array{Union{Missing,String}}(missing, maxstep),
    #Array{Union{Missing,Int}}(missing, maxstep)
    selected_events = Array{Union{Missing,Int}}(missing, maxstep)
    Random.seed!(seed)

    # Create the history for algorithm best
    j, i = 0, 0
    while (j < maxstep && i < nrow(stream))
        #println("j = $j, i = $i")

        # Select one candidate event
        i += 1         # One event in stream is used
        candidate_t = stream[i, :]

        # Extract the arm set
        armset_t = collect(skipmissing(candidate_t[r"col"]))

        # Run algorithm
        chosen_t = run_random(armset_t)

        # Factual display
        display_t = candidate_t.display

        # Retain candidate event?
        if chosen_t == display_t

            # One event is added to the history
            j += 1

            # Reveal feedback
            #reward_t = candidate_t.click

            # Update history
            #reward[j] = reward_t
            #chosen[j] = chosen_t
            selected_events[j] = i

        end
    end

    return selected_events

end

"""
Run algorithm: random
"""
function run_random(armset_t::Vector{String})::String

    chosen = rand(armset_t, 1)[1]

    return chosen
end
