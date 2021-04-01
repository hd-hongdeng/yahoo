#===============================================================================
Title: Function - algorithm : LIME-UCB
Date: 2021-03-29
Description:
Compute the index given new feature vector;
Create a sythetic history;
Create multiple sythetic histories.
===============================================================================#

#--- (0) Essential functions

"""
Compute the UCB for one article, given a new feature vector
"""
function ucb_limeucb(
    𝒙::Vector{<:Real},  # New feature vector
    μ̃ᵦ::Vector{<:Real}, # FE parameters
    Ω̃ᵦ::Matrix{<:Real},
    ρ::Matrix{<:Real},  # Arm-specific parameters
    b̂::Vector{<:Real},
    Ω̃::Matrix{<:Real},
    α::Real,            # Learning rate
)::NamedTuple
    # Compute UCB for one arm
    l = 𝒙 - ρ' * 𝒙
    ucb_center = l' * μ̃ᵦ + 𝒙' * b̂
    ucb_width = sqrt(l' * Ω̃ᵦ * l + 𝒙' * Ω̃ * 𝒙)
    ucb = ucb_center + α * ucb_width
    return (ucb = ucb, center = ucb_center, width = ucb_width)
end

"""
Select one artcile from the available articles by LIME-UCB algorithm
"""
function run_limeucb(
    armset_t::Vector{String}, # Available articles
    display_limeucb::DataFrame, # Statistics of LIME-UCB (article-specific parameters)
    fe_limeucb::Dict, # Statistics of LIME-UCB (population-level parameters)
    feature_t::Vector, # New feature vector
    α::Real, # Learning rate
)::String
    # Keep only available articles for the following computation
    display_limeucb_t = filter(:display => in(Set(armset_t)), display_limeucb)

    # Compute UCBs for all available articles based on the new feature vector
    ucbs = [
        ucb_limeucb(
            feature_t,
            fe_limeucb["fe_mu"],
            fe_limeucb["fe_var"],
            row.rho,
            row.bhat,
            row.omg,
            α,
        ).ucb for row in eachrow(display_limeucb_t)
    ]

    # Select the article with max. UCB arbitrarily
    max_ucb = maximum(ucbs)
    chosen_idx = rand(findall(isequal(max_ucb), ucbs), 1)[1]
    chosen = display_limeucb_t.display[chosen_idx]

    return chosen
end

#--- (1) Complete version of LIME-UCB
"""
Create one history when implementing LIME-UCB (Complete Version)
"""
function simulator_limeucb(
    stream::DataFrame, # A stream of events to go through
    maxstep::Int, # Desired number of steps for one sythetic history
    #display_limeucb::DataFrame, # Statistics of LIME-UCB (article-specific parameters)
    #fe_limeucb::Dict, # Statistics of LIME-UCB (population-level parameters)
    #featurenames::Array, # Names of features in the data stream
    seed::Int,
    prior_limeucb::Dict,
    p::Int; # Initialization of LIME-UCB;
    α::Real = sqrt(2), # Learning rate
    update_fe::Bool = false
)::NamedTuple

    # Initialization: a collection of event indices
    selected_events = Array{Union{Missing,Int}}(missing, maxstep)
    Random.seed!(seed)
    # A set of articles (in the pool) seen by the algorithm
    articleseen = Set{String}()

    # Initialization of LIME-UCB
    display_limeucb = DataFrame(
        :display => unique(stream.display),
        :nchosen => 0,
        :rho => Ref(zeros(p, p)),
        :bhat => Ref(zeros(p)),
        :omg => Ref(prior_limeucb["re_var0"]),
        :X => Ref(zeros(1, p)),
        :y => Ref(zeros(1)),
        :c_var => Ref(zeros(p, p)),
        :c_mu => Ref(zeros(p)),
    ) # 0.05 seconds
    fe_limeucb =
        Dict("fe_mu" => prior_limeucb["fe_mu0"],
             "fe_var" => prior_limeucb["fe_var0"])

    # Create one history for the algorithm
    j, i = 0, 0
    printstep = Int(ceil(quantile(1:maxstep, 0.25)))
    while (j < maxstep && i < nrow(stream))

        # Select one candidate event
        i += 1         # One event in the stream is used
        candidate_t = stream[i, :]

        # Extract the arm set
        armset_t = collect(skipmissing(candidate_t[r"col"]))

        # Feed feature
        feature_t = candidate_t.profile |> collect

        # Run algorithm
        chosen_t = run_limeucb(armset_t, display_limeucb, fe_limeucb, feature_t, α)

        # Factual display
        display_t = candidate_t.display

        # Retain candidate event?
        if chosen_t == display_t

            # One event is added to the history
            j += 1
            # Print steps
            #(mod(j, printstep) == 0) && println("Progress: $(round(j/maxstep*100))")

            # Reveal feedback
            reward_t = candidate_t.click

            # Store the event index
            selected_events[j] = i

            # Label articles that have been seen in the pool
            articleseen = union(articleseen, Set(armset_t))

            # Update algorithm
            # Find the article that was chosen
            idx = findfirst(isequal(chosen_t), display_limeucb.display)
            # A dataframerow that contains information on the chosen articles
            row = display_limeucb[idx, :]
            # Log its update
            T = row.nchosen + 1
            # Update the design matrix
            design = vcat(row.X, feature_t')
            # Update the reward vector
            reward = vcat(row.y, reward_t)
            # Update RE parameters
            𝐗 = design[Not(1), :] # The first element is for initialization
            𝐲 = reward[Not(1)]   # The first element is for initialization
            σ² = prior_limeucb["noise_var0"]
            Ω = prior_limeucb["re_var0"]
            Ω̃ = inv((1 / σ²) * 𝐗' * 𝐗 + inv(Ω))
            ρ = Ω̃ * ((1 / σ²) * 𝐗' * 𝐗)
            b̂ = Ω̃ * ((1 / σ²) * 𝐗' * 𝐲)

            if update_fe == false
                # Store the updated RE parameters
                display_limeucb[idx, Not([:c_var, :c_mu])] =
                    (chosen_t, T, ρ, b̂, Ω̃, design, reward)
            else
                # Prepare the two consts for updating FE parameter
                c_lhs = 𝐗' * inv(𝐗 * Ω * 𝐗' + σ² * Matrix(I, T, T))
                c_var = c_lhs * 𝐗
                c_mu = c_lhs * 𝐲
                # Update FE parameters
                # Select the articles that have been seen
                df_c = filter(
                    :display => in(articleseen),
                    display_limeucb[!, [:display, :c_var, :c_mu]],
                )
                c_vars = sum(df_c.c_var)
                c_mus = sum(df_c.c_mu)
                Ωᵦ = prior_limeucb["fe_var0"]
                μᵦ = prior_limeucb["fe_mu0"]
                Ω̃ᵦ = inv(c_vars + inv(Ωᵦ))
                μ̃ᵦ = Ω̃ᵦ * (c_mus + inv(Ωᵦ) * μᵦ)

                # Store the updated FE parameters
                fe_limeucb["fe_var"] = Ω̃ᵦ
                fe_limeucb["fe_mu"] = μ̃ᵦ

                # Store the updated RE parameters
                display_limeucb[idx, :] =
                    (chosen_t, T, ρ, b̂, Ω̃, design, reward, c_var, c_mu)
            end


        end
    end
    #println("share of target steps = $(round(j/maxstep*100))")
    #println("share of stream = $(round(i/nrow(stream)*100)) \n------\n")


    # Output in a NamedTuple
    result = (
        selected_events = selected_events,
        display_limeucb = display_limeucb, # RE parameter
        fe_limeucb = fe_limeucb,           # FE parameter
        articleseen = articleseen
    )

    return result

end

"""
Create n histories for evaluating LIME-UCB
"""
function simulator_limeucb_mtp(
    n::Int, # Number of MC simulations
    stream::DataFrame, # A stream of events to go through
    maxstep::Int, # Desired number of steps for one sythetic history
    random_seed::Vector{Int},
    prior_limeucb::Dict,
    p::Int;
    α::Real = sqrt(2), # Learning rate
    update_fe = false
)::Vector{NamedTuple}

    println("Algorithm: LIME-UCB, MC: ", n)
    # Initialization: for multiple histories
    mtp_histories = Array{NamedTuple}(undef, n)
    # Create n histories
    Threads.@threads for i = ProgressBar(1:n)
        # Update the storage array
        seed = random_seed[i]
        mtp_histories[i] = simulator_limeucb(
            stream,
            maxstep,
            seed,
            prior_limeucb,
            p;
            α = α,
            update_fe = update_fe
        )
    end

    return mtp_histories
end



"""
Tentative: Compute the UCBs for available articles, given a new feature vector
"""
function ucbs_limeucb(
    armset_t::Vector{String}, # Available articles
    display_limeucb::DataFrame, # Statistics of LIME-UCB (article-specific parameters)
    fe_limeucb::Dict, # Statistics of LIME-UCB (population-level parameters)
    feature_t::Vector, # New feature vector
    α::Real, # Learning rate
)::NamedTuple
    # Keep only available articles for the following computation
    display_limeucb_t = filter(:display => in(Set(armset_t)), display_limeucb)

    # Compute UCBs for all available articles based on the new feature vector
    ucbs = [
        ucb_limeucb(
            feature_t,
            fe_limeucb["fe_mu"],
            fe_limeucb["fe_var"],
            row.rho,
            row.bhat,
            row.omg,
            α,
        ) for row in eachrow(display_limeucb_t)
    ]

    return (article = display_limeucb_t.display, value = ucbs)
end
