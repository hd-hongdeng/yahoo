#--- Functions for the script "toy.jl"

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
    𝐗 = vcat(stat_re.feature, data.feature')
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
    },
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
        lw = 2,
        lc = :orange,
        ls = :solid,
        label = "Bound",
        leg = :outertopright,
        xlabel = "Step",
        title = "UCB (Final Bound = $(round(b[end], digits = 2)), Center = $(round(c[end], digits = 2)), Width = $(round(s[end], digits = 2))) ",
    )

    # Plot centers and widths
    plot!(
        0:1:T-1,
        c,
        yerror = (fill(0, T), s),
        lw = 1.5,
        lc = :green,
        ls = :solid,
        label = "Center",
    )

    return plt
end

function slider_lime(
    reward_mean,
    n,
    p,
    fe_mu0,
    fe_var0,
    re_var0,
    noise_var0,
    α;
    seed = 11223344,
)

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
        Float64[], # reward
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
        decompose_lime_stat(stats_lime_fe[i], stats_lime_re[i]) for i = 1:(n+1)
    ]

    # What would happen if we use offline calibration
    stats_lime_re_base, stats_lime_fe_base = onearm(
        datastream,
        StatLIMERe(
            0, # nchosen
            zeros(0, p), # feature/design matrix
            Float64[], # reward
            4e-4 * Matrix(I, p, p), # re_omg
            zeros(p, p), # re_rho
            zeros(p), # re_bhat
            zeros(p, p), # C_var
            zeros(p), # C_mu
            (re_var0 = 4e-4 * Matrix(I, p, p), noise_var0 = 0.04),
        ),
        StatLIMEFe(
            1e-6 * Matrix(I, p, p), # fe_var
            fill(0.04, p), # fe_mu
            (
                fe_var0 = 1e-6 * Matrix(I, p, p),
                fe_mu0 = fill(0.04, p),
                noise_var0 = 0.04,
            ),
        ),
    )
    inds_lime_base = [
        getucb(
            stats_lime_re_base[i],
            stats_lime_fe_base[i],
            datastream_add0[i].feature,
            α = α,
        ) for i = 1:(n+1)
    ]
    decomp_lime_base = [
        decompose_lime_stat(stats_lime_fe_base[i], stats_lime_re_base[i])
        for i = 1:(n+1)
    ]

    # Specify steps from 0
    xs = range(0, n, step = 1)

    # Plot the dynamics
    plt1 = plot(
        xs,
        getindex.(getproperty.(decomp_lime_base, :fe_contri_center), 1),
        lc = :Gray,
    )
    plot!(
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
        getindex.(getproperty.(decomp_lime_base, :re_contri_center), 1),
        lc = :Gray,
    )
    plot!(
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
        getindex.(getproperty.(decomp_lime_base, :fe_contri_variance), 1),
        lc = :Gray,
    )
    plot!(
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
        getindex.(getproperty.(decomp_lime_base, :re_contri_variance_total), 1),
        lc = :Gray,
    )
    plot!(
        xs,
        getindex.(getproperty.(decomp_lime, :re_contri_variance_total), 1),
        xlabel = "Step",
        ylabel = L"\tilde{\Omega}_{b_i} + \rho_i \tilde{\Omega}_{\beta}^\star \rho_i^\prime - 2 \rho_i \tilde{\Omega}_{\beta}^\star ",
        lc = :Red,
        title = "RE's impact on the variance",
        leg = false,
    )
    plt5 = plotindex(inds_lime)
    hline!(
        [reward_mean],
        lc = :green,
        ls = :dash,
        label = "r̄ ($(reward_mean))",
    )

    plt6 = plot(xs, getproperty.(inds_lime_base, :center), lc = :Gray)
    plot!(
        xs,
        getproperty.(inds_lime, :center),
        xlabel = "Step",
        lc = :Green,
        title = "Center",
        leg = false,
    )
    hline!([reward_mean], lc = :green, ls = :dash)
    plt7 = plot(xs, getproperty.(inds_lime_base, :width), lc = :Gray)
    plot!(
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

function slider_profile(
    β,
    n,
    p,
    pf_learn,
    pf_infer,
    fe_mu0,
    fe_var0,
    re_var0,
    noise_var0,
    α;
    seed = 112233,
)
    # Random seed
    rng = MersenneTwister(seed)
    # Error
    d = Normal(0, sqrt(0.01))
    noise = rand(rng, d, n)
    # Mean reward
    r̄1 = β' * pf_learn
    r̄2 = β' * pf_infer
    # Reward sequance
    rewardseq = r̄1 .+ noise
    # Data stream
    datastream1 = [OneInstance(r, pf_learn) for r in rewardseq]
    datastream2 = [OneInstance(0, pf_infer) for _ = 1:n]
    datastream1_add0 = vcat(OneInstance(0, fill(0, p)), datastream1)
    datastream2_add0 = vcat(OneInstance(0, fill(0, p)), datastream2)


    # Learn from profile_learn
    stats_lime_re_learn, stats_lime_fe_learn = onearm(
        datastream1,
        StatLIMERe(
            0, # nchosen
            zeros(0, p), # feature/design matrix
            Float64[], # reward
            re_var0, # re_omg
            zeros(p, p), # re_rho
            zeros(p), # re_bhat
            zeros(p, p), # C_var
            zeros(p), # C_mu
            (re_var0 = re_var0, noise_var0 = noise_var0),
        ),
        StatLIMEFe(
            fe_var0, # fe_var
            fe_mu0, # fe_mu
            (fe_var0 = fe_var0, fe_mu0 = fe_mu0, noise_var0 = noise_var0),
        ),
    )
    # Compute UCBs for profile_learn
    inds_lime_learn = [
        getucb(
            stats_lime_re_learn[i],
            stats_lime_fe_learn[i],
            datastream1_add0[i].feature,
            α = α,
        ) for i = 1:(n+1)
    ]
    # Compute UCBs for profile_infer
    inds_lime_infer = [
        getucb(
            stats_lime_re_learn[i],
            stats_lime_fe_learn[i],
            datastream2_add0[i].feature,
            α = α,
        ) for i = 1:(n+1)
    ]

    # Plot UCB
    inds =
        [inds_lime_learn, inds_lime_infer]

    ylim_up = maximum([
        maximum([maximum(getproperty.(coll, :ub)) for coll in inds]),
        r̄1,
        r̄2,
    ])
    ylim_dw = minimum([
        minimum([minimum(getproperty.(coll, :center)) for coll in inds]),
        r̄1,
        r̄2,
    ])

    plts_ucb = [plotindex(coll) for coll in inds]
    plt_ucb = plot(
        plts_ucb...,
        #leg = false,
        ylabel = "UCB",
        title = ["Learning x = $(pf_learn)" "Infering x⁺ = $(pf_infer)"],
        ylims = (ylim_dw, ylim_up),
        #xticks = 0:1:n,
    )
    hline!([r̄1 r̄2], lc = :green, ls = :dash, label = "True r̄")

    # Is width decreasing?
    width_learn = getproperty.(inds_lime_learn, :width)
    width_infer = getproperty.(inds_lime_infer, :width)
    plt_width = plot(
        plot(0:1:n, width_learn),
        plot(0:1:n, width_infer),
        legend = :bottomright,
        label = ["width for learning profile" "width for inferring profile"],
    )

    # Is variance decreasing?
    omg = getfield.(stats_lime_re_learn, :re_omg)
    s1 = [m[1, 1] for m in omg]
    s2 = [m[2, 2] for m in omg]
    s12 = [m[2, 1] for m in omg]
    plt_omg = plot(
        plot(0:1:n, s1),
        plot(0:1:n, s2),
        plot(0:1:n, s12),
        layout = (1, 3),
        label = ["s₀²" "s₁²" "cov"],
        lc = [:blue :red :purple]
    )

    # Is center correct?
    bhat = getfield.(stats_lime_re_learn, :re_bhat)
    b̂₀ = [v[1] for v in bhat]
    b̂₁ = [v[2] for v in bhat]
    plt_bhat = plot(plot(0:1:n, b̂₀), plot(0:1:n, b̂₁), label = ["b̂₀" "b̂₁"], lc = [:blue :red])

    plot(plt_ucb, plt_width, plt_omg, plt_bhat, layout = (4, 1),
    guidefontsize = 8,
    legendfontsize = 8,
    tickfontsize = 8,
    titlefontsize = 8,
    size = (1200, 800),)

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
        Float64[], # reward
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
        Float64[], # reward
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
