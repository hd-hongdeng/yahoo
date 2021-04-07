#===============================================================================
Title: Function - algorithm : Heterogeneous optimal policies
Date: 20201-03-01
Description:
===============================================================================#

"""
Simulator for algorithm: Heterogeneous optimal policies with OLS/Logit/LMEM and BPP
"""
function simulator_hetebest(stream::Union{DataFrame,SubDataFrame}, maxstep::Int, strategy::Dict)::Array{Union{Missing, Int64},1}

    # Initialization (history)

    selected_events = Array{Union{Missing,Int}}(missing, maxstep)

    # Create the history for the algorithm
    j, i = 0, 0
    while (j < maxstep && i < nrow(stream))

        # Select one candidate event
        i += 1         # One event in stream is used
        candidate_t = stream[i, :]

        # Extract the arm set
        armset_t = collect(skipmissing(candidate_t[r"col"]))

        # Extract the profile
        profile_t = candidate_t.profile

        # Run algorithm
        chosen_t = run_hetebest(armset_t, profile_t, strategy)

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
Run algorithm: Heterogeneous optimal policies with OLS/Logit/LMEM and BPP
"""
function run_hetebest(armset_t::Vector{String}, profile_t::Tuple, bpp_ctr::Dict)::String

    # Note: Make sure that bpp_ctr is sorted according to CTR

    # What are the CTRs for the available arm, given the current profile?
    ctrs = filter(:display => in(Set(armset_t)), bpp_ctr[profile_t])

    # Select the article with max. CTR
    chosen = ctrs.display[1]

    return chosen
end
