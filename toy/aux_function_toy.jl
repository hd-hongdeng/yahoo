#--- Functions for the script "update.jl"

struct OneInstance
    reward::Float64
    feature::Vector{Float64}
end

# Necessary statistics
struct StatUCB
    nchosen::Int
    estmu::Float64
    prior::NamedTuple{(:estmu0, :noise_var0),Tuple{Real,Real}}
end


# Necessary statistics
struct StatLIMERe
    nchosen::Int
    feature::Matrix{Float64}
    reward::Vector{Float64} # Do not initilize with Real[], but use Float64[]!
    re_omg::Matrix{Float64}
    re_rho::Matrix{Float64}
    re_bhat::Vector{Float64}
    C_var::Matrix{Float64}
    C_mu::Vector{Float64}
    prior::NamedTuple{(:re_var0, :noise_var0),Tuple{Matrix{Float64},Real}}
end

struct StatLIMEFe
    fe_var::Matrix{Float64}
    fe_mu::Vector{Float64}
    prior::NamedTuple{
        (:fe_var0, :fe_mu0, :noise_var0),
        Tuple{Matrix{Float64},Vector{Float64},Real},
    }
end

# Update formula
"""
    update(data::OneInstance, stat::Beta)

Given an instance of data, update the statistics for Beta prior.
"""
function update(data::OneInstance, stat::Beta)
    return Beta(stat.α + data.reward, stat.β + (1 - data.reward))
end

"""
    update(data::OneInstance, stat::StatUCB)

Given an instance of data, update the statistics for simple UCB algorithm.
"""
function update(data::OneInstance, stat::StatUCB)

    # Update the # of times for which article has been chosen
    T = stat.nchosen + 1
    # Update the estimated mean reward for this arm
    μ̂ = stat.estmu + (data.reward - stat.estmu) / T

    # Return the updated statistics
    return StatUCB(T, μ̂, stat.prior)
end

"""
    update(data::OneInstance, stat_re::StatLIMERe)

Given an instance of data, update the RE statistics for LIME-UCB algorithm.
"""
function update(
    data::OneInstance,
    stat_re::StatLIMERe;
    fe_component::Bool = false,
)

    # Update the # of times for which article has been chosen
    T = stat_re.nchosen + 1

    # Stack the new instance to the design matrix and reward vector
    𝐗 = vcat(stat_re.feature, data.feature)
    𝐲 = vcat(stat_re.reward, data.reward)

    # Extract priors on noise variance and RE variance
    σ² = stat_re.prior.noise_var0
    Ω = stat_re.prior.re_var0

    # Update RE parameters
    Ω̃ = inv((1 / σ²) * 𝐗' * 𝐗 + inv(Ω))
    𝛒 = Ω̃ * ((1 / σ²) * 𝐗' * 𝐗)
    b̂ = Ω̃ * ((1 / σ²) * 𝐗' * 𝐲)

    if fe_component == false
        # Return the updated statistics
        return StatLIMERe(
            T,
            𝐗,
            𝐲,
            Ω̃,
            𝛒,
            b̂,
            stat_re.C_var,
            stat_re.C_mu,
            stat_re.prior,
        )
    else
        # Compute RE's contribution to FE parameters
        LHS = 𝐗' * inv(𝐗 * Ω * 𝐗' + σ² * Matrix(I, T, T))
        # Return the updated statistics
        return StatLIMERe(T, 𝐗, 𝐲, Ω̃, 𝛒, b̂, LHS * 𝐗, LHS * 𝐲, stat_re.prior)
    end
end

"""
    freshfe(stats_re::Vector{StatLIMERe}, prior::NamedTuple)

Compute the FE statistics for LIME-UCB algorithm from all RE statistics.
"""
function freshfe(
    stats_re::Vector{StatLIMERe},
    prior::NamedTuple{
        (:fe_var0, :fe_mu0, :noise_var0),
        Tuple{Matrix{Float64},Vector{Float64},Real},
    },
)

    # Extract priors on FE parameters
    Ωᵦ = prior.fe_var0
    μᵦ = prior.fe_mu0

    # Sum up the contributions from all arms
    Cs_var = getfield.(stats_re, :C_var) |> sum
    Cs_mu = getfield.(stats_re, :C_mu) |> sum

    # Compute FE Posterior
    Ω̃ᵦ = inv(Cs_var + inv(Ωᵦ))
    μ̃ᵦ = Ω̃ᵦ * (Cs_mu + inv(Ωᵦ) * μᵦ)

    # Return the updated statisticss
    return StatLIMEFe(Ω̃ᵦ, μ̃ᵦ, prior)
end

"""
    onearm(datastream::Vector{OneInstance}, stat0::Beta)

Experiment how Beta updates statistics for a sequence of data instances with only one arm.
"""
function onearm(datastream::Vector{OneInstance}, stat0::Beta)

    # Pre-allocate a vector to store updated statistics for every step (incl. initial values)
    step = length(datastream)
    stats = Array{Beta}(undef, step + 1)
    # Add initial values as the first element of the resulted vector
    stats[1] = stat0
    # Loop over each data point
    for i = 1:step
        stats[i+1] = update(datastream[i], stats[i])
    end

    # Return the sequence of updated statistics
    return stats
end

"""
    onearm(datastream::Vector{OneInstance}, stat0::StatUCB)

Experiment how simple UCB algorithm updates statistics for a sequence of data instances with only one arm.
"""
function onearm(datastream::Vector{OneInstance}, stat0::StatUCB)

    # Pre-allocate a vector to store updated statistics for every step (incl. initial values)
    step = length(datastream)
    stats = Array{StatUCB}(undef, step + 1)
    # Add initial values as the first element of the resulted vector
    stats[1] = stat0
    # Loop over each data point
    for i = 1:step
        stats[i+1] = update(datastream[i], stats[i])
    end

    # Return the sequence of updated statistics
    return stats
end

"""
    onearm(datastream::Vector{OneInstance}, stat_re0::StatLIMERe)

Experiment how LIME-UCB algorithm updates (only) RE statistics for a sequence of data instances with only one arm.
"""
function onearm(datastream::Vector{OneInstance}, stat_re0::StatLIMERe)

    # Pre-allocate a vector to store updated statistics for every step (incl. initial values)
    step = length(datastream)
    stats_re = Array{StatLIMERe}(undef, step + 1)
    # Add initial values as the first element of the resulted vector
    stats_re[1] = stat_re0
    # Loop over each data point
    for i = 1:step
        stats_re[i+1] = update(datastream[i], stats_re[i])
    end

    # Return the sequence of updated statistics
    return stats_re
end

"""
    onearm(datastream::Vector{OneInstance}, stat_re0::StatLIMERe, stat_fe0::StatLIMEFe)

Experiment how LIME-UCB algorithm updates RE and FE statistics for a sequence of data instances with only one arm.
"""
function onearm(
    datastream::Vector{OneInstance},
    stat_re0::StatLIMERe,
    stat_fe0::StatLIMEFe,
)

    # Pre-allocate a vector to store updated statistics for every step (incl. initial values)
    step = length(datastream)
    stats_re = Array{StatLIMERe}(undef, step + 1)
    stats_fe = Array{StatLIMEFe}(undef, step + 1)
    # Add initial values as the first element of the resulted vector
    stats_re[1] = stat_re0
    stats_fe[1] = stat_fe0
    feprior = stat_fe0.prior
    # Loop over each data point
    for i = 1:step
        stats_re[i+1] = update(datastream[i], stats_re[i]; fe_component = true)
        stats_fe[i+1] = freshfe([stats_re[i+1]], feprior)
    end

    # Return the sequence of updated statistics
    return (stats_re = stats_re, stats_fe = stats_fe)
end

"""
    getucb(stat::Beta; q::Float64 = 0.95)

Compute q-quantil for the Beta distribution.
"""
function getucb(stat::Beta; q::Float64 = 0.95)
    center = mean(stat)
    ub = quantile(stat, q)
    width = ub - center
    return (ub = ub, center = center, width = width)
end

"""
    getucb(stat::StatUCB; α::Float64 = 1.0)

Compute the upper confidence bound for simple UCB algorithm.
"""
function getucb(stat::StatUCB; α::Float64 = 1.0)

    center = stat.estmu

    variance = stat.prior.noise_var0 / stat.nchosen

    width = α * sqrt(variance)

    ub = center + width

    return (ub = ub, center = center, width = width)
end

"""
    getucb(stat_re::StatLIMERe, stat_fe::StatLIMEFe, feature::Vector{Float64}; α::Float64 = 1.0)

Compute the upper confidence bound for LIME-UCB algorithm.
"""
function getucb(
    stat_re::StatLIMERe,
    stat_fe::StatLIMEFe,
    feature::Vector{Float64};
    α::Float64 = 1.0,
)

    # Extract contextual features and necessary statistics
    𝒙 = feature
    Ω̃ᵦ = stat_fe.fe_var
    μ̃ᵦ = stat_fe.fe_mu
    Ω̃ = stat_re.re_omg
    𝛒 = stat_re.re_rho
    b̂ = stat_re.re_bhat

    # Compute the upper confidence bound
    l = 𝒙 - 𝛒' * 𝒙
    center_fe = l' * μ̃ᵦ
    center_re = 𝒙' * b̂
    center = center_fe + center_re

    var_fe = l' * Ω̃ᵦ * l
    var_re = 𝒙' * Ω̃ * 𝒙
    variance = var_fe + var_re

    width = α * sqrt(variance)

    ub = center + width

    return (ub = ub, center = center, width = width)
end

"""
    gen_reward(n::Int, reward_mean::Float64)

Generate a sequance of binary rewards with specified mean value of the reward.
"""
function gen_reward(n::Int, reward_mean::Float64)
    # Create a vector of zeros with length n
    rewardseq = fill(Float64(0), n)
    # Pick the indices for reopalcement according to the mean value
    setone = sample(1:n, Int(round(n * reward_mean)), replace = false)
    # Fill in ones
    rewardseq[setone] .= 1

    return rewardseq
end

"""
    decompose_lime_stat(stat_fe::StatLIMEFe, stat_re::StatLIMERe)

Compute the contributions made by FE and RE separately in LIME-UCB's bound.
"""
function decompose_lime_stat(stat_fe::StatLIMEFe, stat_re::StatLIMERe)

    # Extract basic statistics
    Ω̃ᵦ = stat_fe.fe_var
    μ̃ᵦ = stat_fe.fe_mu
    Ω̃ = stat_re.re_omg
    𝛒 = stat_re.re_rho
    b̂ = stat_re.re_bhat

    # Decompose into FE and RE contributions
    re_contri_center = b̂ - 𝛒 * μ̃ᵦ

    re_contri_variance = Ω̃ + 𝛒 * Ω̃ᵦ * 𝛒'

    corr_in_variance = (-2) * 𝛒 * Ω̃ᵦ

    re_contri_variance_total = re_contri_variance + corr_in_variance


    return (
        fe_contri_center = μ̃ᵦ,
        re_contri_center = re_contri_center,
        fe_contri_variance = Ω̃ᵦ,
        re_contri_variance = re_contri_variance,
        corr_in_variance = corr_in_variance,
        re_contri_variance_total = re_contri_variance_total,
    )
end

"""

Plot upper bounds, centers and widths for all steps.
"""
function plotindex(
    collection::Array{
        NamedTuple{(:ub, :center, :width),Tuple{Float64,Float64,Float64}},
        1,
    };
    linecolor = :Green,
)

    # Extract upper bounds, centers and widths for every step
    b = getproperty.(collection, :ub)
    c = getproperty.(collection, :center)
    s = getproperty.(collection, :width)
    T = length(b)

    # Plot upper bounds
    plt = plot(
        0:1:T-1,
        b,
        lw = 1.5,
        lc = linecolor,
        ls = :dash,
        #label = "Bound",
        leg = false,
        xlabel = "Step",
        title = "UCB (Final Bound = $(round(b[end], digits = 2)), Center = $(round(c[end], digits = 2)), Width = $(round(s[end], digits = 2))) "
    )

    # Plot centers and widths
    plot!(
        0:1:T-1,
        c,
        yerror = (fill(0, T), s),
        lw = 1.5,
        lc = linecolor,
        ls = :solid,
        #label = "Center",
    )

    return plt
end

function slider_lime(reward_mean, n, p, fe_mu0, fe_var0, re_var0, noise_var0, α; seed = 11223344)

    # Data
    # Set seed!
    Random.seed!(seed)
    rewardseq = gen_reward(n, reward_mean)
    datastream = [OneInstance(r, [1]) for r in rewardseq]

    # Piror
    stat_fe0 = StatLIMEFe(
        fe_var0, # fe_var
        fill(0, p), # fe_mu
        (fe_var0 = fe_var0, fe_mu0 = fe_mu0, noise_var0 = noise_var0),
    )
    stat_re0 = StatLIMERe(
        0, # nchosen
        zeros(0, p), # feature/design matrix
        Bool[], # reward
        re_var0, # re_omg
        zeros(p, p), # re_rho
        zeros(p), # re_bhat
        zeros(p, p), # C_var
        zeros(p), # C_mu
        (re_var0 = re_var0, noise_var0 = noise_var0),
    )

    # Update LIME-UCB's parameters
    stats_lime_re, stats_lime_fe = onearm(datastream, stat_re0, stat_fe0)

    # Compute uppper bounds, centers, and widths for each step
    datastream_add0 = vcat(OneInstance(0, fill(0, p)), datastream)
    inds_lime = [
        getucb(
            stats_lime_re[i],
            stats_lime_fe[i],
            datastream_add0[i].feature,
            α = α,
        ) for i = 1:(n+1)
    ]

    # Decompose LIME-UCB's parameters for each step
    decomp_lime = [
        decompose_lime_stat(stats_lime_fe[i], stats_lime_re[i]) for
        i = 1:(n+1)
    ]

    # Specify steps from 0
    xs = range(0, n, step = 1)

    # Plot the dynamics
    plt1 = plot(
        xs,
        getindex.(getproperty.(decomp_lime, :fe_contri_center), 1),
        xlabel = "Step",
        ylabel = L"\tilde{\mu}_{\beta}^\star",
        lc = :Blue,
        title = "FE's impact on the center",
        leg = false,
    )
    plt2 = plot(
        xs,
        getindex.(getproperty.(decomp_lime, :re_contri_center), 1),
        xlabel = "Step",
        ylabel = L"\hat{b}_{i}-{\rho}_{i}\tilde{\mu}_{\beta}^\star",
        lc = :Red,
        title = "RE's impact on the center",
        leg = false,
    )
    plt3 = plot(
        xs,
        getindex.(getproperty.(decomp_lime, :fe_contri_variance), 1),
        xlabel = "Step",
        ylabel = L"\tilde{\Omega}_{\beta}^\star",
        lc = :Blue,
        title = "FE's impact on the variance",
        leg = false,
    )
    plt4 = plot(
        xs,
        getindex.(getproperty.(decomp_lime, :re_contri_variance_total), 1),
        xlabel = "Step",
        ylabel = L"\tilde{\Omega}_{b_i} + \rho_i \tilde{\Omega}_{\beta}^\star \rho_i^\prime - 2 \rho_i \tilde{\Omega}_{\beta}^\star ",
        lc = :Red,
        title = "RE's impact on the variance",
        leg = false,
    )
    plt5 = plotindex(inds_lime)
    plt6 = plot(
        xs,
        getproperty.(inds_lime, :center),
        xlabel = "Step",
        lc = :Green,
        title = "Center",
        leg = false,
    )
    plt7 = plot(
        xs,
        getproperty.(inds_lime, :width),
        xlabel = "Step",
        lc = :Black,
        title = "Width",
        leg = false,
    )

    plot(
        plt5,
        plt1,
        plt2,
        plt6,
        plt3,
        plt4,
        plt7,
        thickness_scaling = 1.2,
        guidefontsize = 8,
        legendfontsize = 8,
        tickfontsize = 8,
        titlefontsize = 9,
        size = (1200, 800),
        layout = @layout [ a{0.3h}
                           grid(2,3){0.7h} ]
    )
end


function test_beta(datastream; q = 0.95, stat_beta0 = Beta(1, 1))

    # Update for one-armed bandit problem
    stats_beta = onearm(datastream, stat_beta0)
    # Compute uppper bounds, centers, and widths for each step
    inds_beta = [getucb(stats_beta[i], q = q) for i = 1:length(stats_beta)]

    return (stats = stats_beta, inds = inds_beta)
end

function test_ucb(
    datastream;
    α = 1.0,
    stat_ucb0 = StatUCB(
        0, # nchosen
        0, # estmu
        (estmu0 = 0, noise_var0 = 1),
    ),
)
    # Update for one-armed bandit problem
    stats_ucb = onearm(datastream, stat_ucb0)
    # Compute uppper bounds, centers, and widths for each step
    inds_ucb = [getucb(stats_ucb[i], α = α) for i = 1:length(stats_ucb)]

    return (stats = stats_ucb, inds = inds_ucb)
end

function test_lime_fixed(
    datastream,
    p;
    α = 1.0,
    stat_fe0 = StatLIMEFe(
        Matrix(I, p, p), # fe_var
        zeros(p), # fe_mu
        (fe_var0 = Matrix(I, p, p), fe_mu0 = zeros(p), noise_var0 = 1),
    ),
    stat_re0 = StatLIMERe(
        0, # nchosen
        zeros(0, p), # feature/design matrix
        Bool[], # reward
        Matrix(I, p, p), # re_omg
        zeros(p, p), # re_rho
        zeros(p), # re_bhat
        zeros(p, p), # C_var
        zeros(p), # C_mu
        (re_var0 = Matrix(I, p, p), noise_var0 = 1),
    ),
)
    # Update for one-armed bandit problem
    stats_lime_re = onearm(datastream, stat_re0)
    # Compute uppper bounds, centers, and widths for each step
    datastream_add0 = vcat(OneInstance(0, fill(0, p)), datastream)
    inds_lime = [
        getucb(stats_lime_re[i], stat_fe0, datastream_add0[i].feature, α = α) for i = 1:length(stats_lime_re)
    ]

    return (stats = stats_lime_re, inds = inds_lime)
end

function test_lime(
    datastream,
    p;
    α = 1.0,
    stat_fe0 = StatLIMEFe(
        1 * Matrix(I, p, p), # fe_var
        fill(0, p), # fe_mu
        (fe_var0 = 1 * Matrix(I, p, p), fe_mu0 = fill(0, p), noise_var0 = 1),
    ),
    stat_re0 = StatLIMERe(
        0, # nchosen
        zeros(0, p), # feature/design matrix
        Bool[], # reward
        1 * Matrix(I, p, p), # re_omg
        zeros(p, p), # re_rho
        zeros(p), # re_bhat
        zeros(p, p), # C_var
        zeros(p), # C_mu
        (re_var0 = 1 * Matrix(I, p, p), noise_var0 = 1),
    ),
)
    # Update for one-armed bandit problem
    stats_lime_re, stats_lime_fe = onearm(datastream, stat_re0, stat_fe0)
    # Compute uppper bounds, centers, and widths for each step
    datastream_add0 = vcat(OneInstance(0, fill(0, p)), datastream)
    inds_lime = [
        getucb(
            stats_lime_re[i],
            stats_lime_fe[i],
            datastream_add0[i].feature,
            α = α,
        ) for i = 1:length(stats_lime_re)
    ]

    return (stats = (re = stats_lime_re, fe = stats_lime_fe), inds = inds_lime)
end
