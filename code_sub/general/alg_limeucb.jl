#===============================================================================
Title: Function - algorithm : LIME-UCB
Date: 2021-03-29
Description:
Create a sythetic history;
Create multiple sythetic histories.
===============================================================================#

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
Decision rule at time t
"""
function run_limeucb(
    armset_t::Vector{String}, # Available articles
    display_limeucb::DataFrame, # Statistics of LIME-UCB (article-specific parameters)
    fe_limeucb::Dict, # Statistics of LIME-UCB (population-level parameters)
    feature_t::Vector{<:Real}, # New feature vector
    α::Real, # Learning rate
)::String

    # Keep only available articles for the following computation
    display_limeucb_t = filter(:display => in(Set(armset_t)), display_limeucb)

    # Check whether the articles are new
    if any(iszero, display_limeucb_t.nchosen) == false
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
        chosen_idx = findall(isequal(max_ucb), ucbs)
    else
        # Choose one new article randomly
        chosen_idx = findall(iszero, display_limeucb_t.nchosen)    
    end
        
    chosen = display_limeucb_t.display[rand(chosen_idx)]

    return chosen
end

"""
Update algorithm's parameter
"""
# RE
function update_limeucb_re(display_limeucb_idx::DataFrameRow, 
                           reward_t::Real,
                           feature_t::Vector{<:Real},
                           prior_limeucb::Dict;
                           update_fe::Bool = false
                           )::DataFrameRow
    # Update the # of times for which article has been chosen
    display_limeucb_idx.nchosen += 1
    # Update the algorithm's parameters
    # Update the design matrix
    display_limeucb_idx.X = vcat(display_limeucb_idx.X, feature_t')
    # Update the reward vector
    display_limeucb_idx.y = vcat(display_limeucb_idx.y, reward_t)
    # Update RE parameters
    𝐗 = display_limeucb_idx.X[Not(1), :] # The first element is for initialization
    𝐲 = display_limeucb_idx.y[Not(1)]   # The first element is for initialization
    σ² = prior_limeucb["noise_var0"]
    Ω = prior_limeucb["re_var0"]
    Ω̃ = inv((1 / σ²) * 𝐗' * 𝐗 + inv(Ω))
    display_limeucb_idx.omg = Ω̃
    display_limeucb_idx.rho = Ω̃ * ((1 / σ²) * 𝐗' * 𝐗)
    display_limeucb_idx.bhat = Ω̃ * ((1 / σ²) * 𝐗' * 𝐲)

    if update_fe == true
        T = display_limeucb_idx.nchosen
        c_lhs = 𝐗' * inv(𝐗 * Ω * 𝐗' + σ² * Matrix(I, T, T))
        display_limeucb_idx.c_var = c_lhs * 𝐗
        display_limeucb_idx.c_mu = c_lhs * 𝐲
    end 

    return display_limeucb_idx
end

# FE
function update_limeucb_fe(display_limeucb::DataFrame,
                           prior_limeucb::Dict)::Dict
        
        # Update FE parameters
        # Select the articles that have been chosen
        df_c = @pipe filter(:nchosen => >(0), display_limeucb) |>
                   select(_, :display, :nchosen, :c_var, :c_mu)
        # Compute new FE across all articles
        c_vars = sum(df_c.c_var)
        c_mus = sum(df_c.c_mu)
        Ωᵦ = prior_limeucb["fe_var0"]
        μᵦ = prior_limeucb["fe_mu0"]
        Ω̃ᵦ = inv(c_vars + inv(Ωᵦ))
        μ̃ᵦ = Ω̃ᵦ * (c_mus + inv(Ωᵦ) * μᵦ)

        # Store the updated FE parameters
        fe_limeucb = Dict(
            "fe_var" => Ω̃ᵦ,
            "fe_mu" => μ̃ᵦ
        )
    return fe_limeucb
end

"""
Create one simulated history
"""
function simulator_limeucb(
    stream::DataFrame, # A stream of events to go through
    maxstep::Int, # Desired number of steps for one sythetic history
    seed::Int,
    prior_limeucb::Dict; # Initialization of LIME-UCB;
    α::Real = sqrt(2), # Learning rate
    update_fe::Bool = false
)::Array{Union{Missing, Int64},1}

    # Initialization: one simulated history
    selected_events = Array{Union{Missing,Int}}(missing, maxstep)
    # Set random seed
    Random.seed!(seed)
    # A set of articles (in the pool) seen by the algorithm
    #articleseen = Set{String}()

    # Initialization: algorithm's prior
    p = length(prior_limeucb["fe_mu0"])
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
        chosen_t = run_limeucb(
                    armset_t, 
                    display_limeucb, 
                    fe_limeucb, 
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

            # Store the event index
            selected_events[j] = i

            # Label articles that have been seen in the pool
            #articleseen = union(articleseen, Set(armset_t))

            # Update algorithm's parameter
            # Find the article that was chosen
            idx = findfirst(isequal(chosen_t), display_limeucb.display)
            # Update chosen article's parameters
            display_limeucb[idx,:] = update_limeucb_re(display_limeucb[idx,:], 
                                                    reward_t,
                                                    feature_t,
                                                    prior_limeucb;
                                                    update_fe = update_fe)
            if update_fe == true  
                fe_limeucb = update_limeucb_fe(display_limeucb, 
                                                prior_limeucb)
            end                                  

        end
    end

    return selected_events

end

"""
Create multiple simulated histories with given steps 
"""
function simulator_limeucb_mtp(
    n::Int, # Number of MC simulations
    stream::DataFrame, # A stream of events to go through
    maxstep::Int, # Desired number of steps for one sythetic history
    random_seed::Vector{Int},
    prior_limeucb::Dict;
    α::Real = sqrt(2), # Learning rate
    update_fe::Bool = false
)::Array{Array{Union{Missing, Int64},1},1}

    # Initialization: `n` histories with length `maxstep`
    mtp_histories = Array{Array{Union{Missing,Int}}}(undef, n)
    # Create n histories
    Threads.@threads for i = ProgressBar(1:n)
        # Set random seed
        seed = random_seed[i]
        # Create one history
        mtp_histories[i] = simulator_limeucb(
            stream,
            maxstep,
            seed,
            prior_limeucb;
            α = α,
            update_fe = update_fe
        )
    end

    return mtp_histories
end


"""
Tentative: Compute the UCBs for available articles, given a new feature vector
"""
function get_ucbs(
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
