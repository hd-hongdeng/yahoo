#===============================================================================
Title: Function - algorithm : optimal policies
Date: 2021-03-01
Description:
===============================================================================#

"""
Simulator for algorithm: homogeneous optimal policy
"""
function simulator_homobest(stream::Union{DataFrame,SubDataFrame}, maxstep::Int, homobest::DataFrame)::Array{Union{Missing, Int64},1}

    # Initialization (history)

    selected_events = Array{Union{Missing,Int}}(missing, maxstep)

    # Create one history for the algorithm
    j, i = 0, 0
    while (j < maxstep && i < nrow(stream))

        # Select one candidate event
        i += 1         # One event in stream is used
        candidate_t = stream[i, :]

        # Extract the arm set
        armset_t = collect(skipmissing(candidate_t[r"col"]))

        # Run algorithm
        chosen_t = run_homobest(armset_t, homobest_ctr)

        # Compare with the factual display
        display_t = candidate_t.display

        # Retain candidate event?
        if chosen_t == display_t

            # One event is added to the history
            j += 1

            # Update history

            selected_events[j] = i

        end
    end

    return selected_events

end

"""
Run algorithm: homogeneous optimal policy
"""
function run_homobest(armset_t::Vector{String}, homobest_ctr::DataFrame)::String

    # Note: Make sure that homobest_ctr is sorted according to CTR

    # What are the CTRs for the available arm?
    ctrs = filter(:display => in(Set(armset_t)), homobest_ctr)

    # Select the article with max. CTR
    chosen = ctrs.display[1]

    return chosen
end
