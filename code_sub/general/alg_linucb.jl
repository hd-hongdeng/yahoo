#===============================================================================
Title: Function - algorithm : linucb
Date: 2021-03-29
Description:
Creating simulated histories for linucb.
===============================================================================#


"""
Multiple Simulator for algorithm: linUCB
"""
function simulator_linucb_mtp(
    n::Int,
    stream::DataFrame,
    maxstep::Int,
    random_seed::Vector{Int},
    p::Int;
    α::Real = sqrt(2)
    )::Vector{Array{Union{Missing,Int}}}

    println("Algorithm: Lin-UCB, MC: ", n)

    # Initialization-simulator
    mtp_histories = Array{Array{Union{Missing,Int}}}(undef, n)
    # Run n simulations
    Threads.@threads for i = ProgressBar(1:n)
        # Update the storage array
        seed = random_seed[i]
        mtp_histories[i] = simulator_linucb(stream, maxstep, seed, p; α = α)
    end
    return mtp_histories
end

"""
Simulator for algorithm: linUCB
"""
function simulator_linucb(
    stream::DataFrame,
    maxstep::Int,
    seed::Int,
    p::Int;
    α::Real = sqrt(2)
)::Array{Union{Missing,Int}}

    # Initialization

    selected_events = Array{Union{Missing,Int}}(missing, maxstep)
    Random.seed!(seed)

    # Initialization-linucb
    display_linucb = DataFrame(
        :display => unique(stream.display),
        :nchosen => 0,
        :X => Ref(zeros(1, p)),
        :y => Ref(zeros(1)),
        :c => Ref(zeros(p)),
        :Phat => Ref(Array{Float64}(Matrix(I, p, p))),
        :muhat => Ref(zeros(p))
    )

    # Create the history for algorithm limucb
    j, i = 0, 0
    while (j < maxstep && i < nrow(stream))
        #println("j = $j, i = $i")

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
            #println("j=$j")

            # Reveal feedback
            reward_t = candidate_t.click

            # Update history

            selected_events[j] = i

            # Update algorithm
            # Find the article that was chosen
            idx = findfirst(isequal(chosen_t), display_linucb.display)
            # Log its update
            display_linucb[idx, :nchosen] += 1
            # Update the algorithm

            display_linucb[idx, :c] += feature_t * reward_t
            display_linucb[idx, :Phat] += feature_t * feature_t'
            μ̂ = inv(display_linucb[idx, :Phat]) * display_linucb[idx, :c]

            display_linucb[idx, :muhat] = μ̂

        end
    end


    return selected_events

end

"""
Run algorithm: linucb
"""
function run_linucb(
    armset_t::Vector{String},
    display_linucb::DataFrame,
    feature_t::Vector{<:Real},
    α::Real,
)::String

    # Extract information for arm set at time t
    display_linucb_t = filter(:display => in(Set(armset_t)), display_linucb)

    # Compute UCB based on features
    ucb_center =
        map((x -> feature_t' * x), display_linucb_t.muhat)

    ucb_width =
        map((x -> sqrt(feature_t' * inv(x) * feature_t)), display_linucb_t.Phat)

    display_linucb_t.ucb = ucb_center .+ α .* ucb_width

    # Select the article with max. est. ucb
    ucb_max = maximum(display_linucb_t.ucb)
    chosen_idx = findall(isequal(ucb_max), display_linucb_t.ucb)
    chosen = display_linucb_t.display[rand(chosen_idx,1)[1]]

    return chosen
end
