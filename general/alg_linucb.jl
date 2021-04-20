#===============================================================================
Title: Function - algorithm : linucb
Date: 2021-03-29
Description:
Creating simulated histories for linucb.
===============================================================================#

"""
Decision rule at time t
"""
function run_linucb(
    armset_t::Vector{String},
    display_linucb::DataFrame,
    feature_t::Vector{<:Real},
    α::Real,
)::String

    # Extract information for arm set at time t
    display_linucb_t = filter(:display => in(Set(armset_t)), display_linucb)

    # Check whether the articles are new
    if any(iszero, display_linucb_t.nchosen) == false
        # Compute UCB based on features
        ucb_center =
            map((muhat -> feature_t' * muhat), display_linucb_t.muhat)

        ucb_width =
            map((Phat -> sqrt(feature_t' * inv(Phat) * feature_t)), display_linucb_t.Phat)

        display_linucb_t.ucb = ucb_center .+ α .* ucb_width

        # Select the article with max. est. ucb
        ucb_max = maximum(display_linucb_t.ucb)
        chosen_idx = findall(isequal(ucb_max), display_linucb_t.ucb)
    else 
        # Choose one new article randomly
        chosen_idx = findall(iszero, display_linucb_t.nchosen)  
    end
    
    chosen = display_linucb_t.display[rand(chosen_idx)]    

    return chosen
end

"""
Update algorithm's parameter
"""
function update_linucb(display_linucb_idx::DataFrameRow, 
                       reward_t::Real,
                       feature_t::Vector{<:Real})::DataFrameRow
    # Update the # of times for which article has been chosen
    display_linucb_idx.nchosen += 1
    # Update the algorithm's parameters
    display_linucb_idx.c += feature_t * reward_t
    display_linucb_idx.Phat += feature_t * feature_t'
    display_linucb_idx.muhat = inv(display_linucb_idx.Phat) * display_linucb_idx.c

    return display_linucb_idx
end

"""
Create one simulated history
"""
function simulator_linucb(
    stream::DataFrame,
    maxstep::Int,
    seed::Int,
    p::Int;
    α::Real = sqrt(2)
)::Array{Union{Missing, Int64},1}

    # Initialization: one simulated history
    selected_events = Array{Union{Missing,Int}}(missing, maxstep)
    # Set random seed
    Random.seed!(seed)

    # Initialization: algorithm's prior
    display_linucb = DataFrame(
        :display => unique(stream.display),
        :nchosen => 0,
        :c => Ref(zeros(p)),
        :Phat => Ref(Array{Float64}(Matrix(I, p, p))),
        :muhat => Ref(zeros(p))
    )

    # Create one history for the algorithm
    j, i = 0, 0
    while (j < maxstep && i < nrow(stream))

        # Select one candidate event
        i += 1         # One event in stream is used
        candidate_t = stream[i, :]

        # Extract the arm set
        armset_t = collect(skipmissing(candidate_t[r"col"]))

        # Feed feature
        feature_t = candidate_t.profile |> collect

        # Run algorithm
        chosen_t = run_linucb(
            armset_t,
            display_linucb,
            feature_t,
            α)

        # Factual display
        display_t = candidate_t.display

        # Retain candidate event?
        if chosen_t == display_t

            # One event is added to the history
            j += 1

            # Reveal feedback
            reward_t = candidate_t.click

            # Store the index of the retained event
            selected_events[j] = i

            # Update algorithm's parameter
            # Find the chosen article
            idx = findfirst(isequal(chosen_t), display_linucb.display)
            # Update chosen article's parameters
            display_linucb[idx,:] = update_linucb(display_linucb[idx,:], 
                                                reward_t,
                                                feature_t)

        end
    end

    return selected_events

end


"""
Create multiple simulated histories with given steps 
"""
function simulator_linucb_mtp(
    n::Int,
    stream::DataFrame,
    maxstep::Int,
    random_seed::Vector{Int},
    p::Int;
    α::Real = sqrt(2)
    )::Array{Array{Union{Missing, Int64},1},1}

    # Initialization: `n` histories with length `maxstep`
    mtp_histories = Array{Array{Union{Missing,Int}}}(undef, n)

    # Run n simulations
    Threads.@threads for i = ProgressBar(1:n)
        # Set random seed
        seed = random_seed[i]
        # Create one history
        mtp_histories[i] = simulator_linucb(stream, maxstep, seed, p; α = α)
    end

    return mtp_histories
end

