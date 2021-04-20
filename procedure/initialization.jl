#= ==============================================================================
Title: Module - Initialization of algorithms' parameters
Updated: 2021-04-13
Description:
- Choose initial values for FE and RE parameters of LIME-UCB, and noise variance;
- Choose initial values for the mean parameters of UCB-1, and noise variance.
============================================================================== =#

# Number of features
user_features = [:u1]
p = length(user_features)

# Load data stream
stream_tun = loadjdf("limeucb_evaluation_2021/data/modify3.jdf") |> DataFrame # In VS code, 36 seconds
propertynames(stream_tun) |> println

# Extra data for initialization
issorted(stream_tun, :date_neg6)
filter!(:date_neg6 => (x -> Date(x) < Date(2011, 10, 12)), stream_tun)
select!(stream_tun, :time, :date_neg6, :display, :click, user_features)
transform!(stream_tun, user_features => ByRow(tuple) => :profile)

# stream_evl = filter(:date_neg6 => (x -> x >= DateTime(2011, 10, 13, 8)), data)
# transform!(stream_evl, [:u1, :u19] => ByRow(tuple) => :profile)
# savejdf("data/stream_evl_1feature.jdf", stream_evl)

# --- (1) Initialization for LIME-UCB

# Run mixed effects model with the estimation data
# fm = @formula(click ~ 1 + u19 + u17 + u16 + zerocorr(1 + u19 + u17 + u16 | display))
# fm = @formula(click ~ 1 + u19 + u17 + u16 + (1 + u19 + u17 + u16 | display))
#fm = @formula(click ~ 1 + u19 + (1 + u19 | display))
fm = @formula(click ~ 1 + (1 | display))
reg_lmem = fit(MixedModel, fm, stream_tun) # 30 seconds

# Initialization based on mixed effects coefficients
# FE parameters
fe_mu0 = reg_lmem.β
fe_var0 = vcov(reg_lmem)
# RE parameters
se = collect(VarCorr(reg_lmem).σρ.display.σ)
corr = collect(VarCorr(reg_lmem).σρ.display.ρ)
# c0 = (VarCorr(fm1).σρ.display.σ) |> collect
# re_var0 = diagm(c0 .^ 2)
re_var0 = diagm(se.^2)
idx = [(i, j) for i in (1 + 1):p for j in 1:i - 1]
for s in 1:(Int(p * (p - 1) / 2))
    position = idx[s]
    rowidx = position[1]
    colidx = position[2]
    cov = corr[s] * se[rowidx] * se[colidx]
    re_var0[rowidx,colidx] = cov
    re_var0[colidx,rowidx] = cov
end
# display(VarCorr(reg_lmem))
# Correlation matrix of RE coefficients
re_rho0 = diagm(fill(1.0, p))
for s in 1:(Int(p * (p - 1) / 2))
    position = idx[s]
    rowidx = position[1]
    colidx = position[2]
    rho = VarCorr(reg_lmem).σρ.display.ρ[s]
    re_rho0[rowidx,colidx] = rho
    re_rho0[colidx,rowidx] = rho
end
# Noise variance
noise_var0 = varest(reg_lmem)

# --- (2) Initialization for UCB-1

# Compute the empirical mean of reward for the tunning data
ucb1estmu0 = mean(stream_tun.click)
# Estimate the noise variance
ucb1estvar0 = var(stream_tun.click)

#--- (3) Store initialization results
result_initial = (
    fe_mu0 = fe_mu0,
    fe_var0 = fe_var0,
    re_var0 = re_var0,
    noise_var0 = noise_var0,
    ucb1estmu0 = ucb1estmu0,
    ucb1estvar0 = ucb1estvar0,
)

open("case_p1/result/result_initial.bin", "w") do io
    serialize(io, result_initial)
end

# To Latex
# println("Initialized FE: ", round.(fe_mu0, digits = 4))
outlatex(fe_mu0, "case_p1/table/table_fe_mu0.tex"; format="%.4f", environment=:array)
# round.(fe_var0,digits = 6) |> display
outlatex(fe_var0, "case_p1/table/table_fe_var0.tex"; format="%.6f", environment=:array)
# round.(re_var0,digits = 6) |> display
outlatex(re_var0, "case_p1/table/table_re_var0.tex"; format="%.6f", environment=:array)
outlatex(re_rho0, "case_p1/table/table_re_rho0.tex"; format="%.2f", environment=:array)
println("Initialized noise variance for LIME-UCB: ", round.(noise_var0, digits=4))
outlatex(noise_var0, "case_p1/table/table_noise_var0.tex"; format="%.2f", environment=:array)
#println("Initialized mean for UCB-1: ", round.(ucb1estmu0, digits=4))
#println("Initialized variance for UCB-1: ", round.(ucb1estvar0, digits=4))
outlatex(ucb1estmu0, "case_p1/table/table_ucb1estmu0.tex"; format="%.2f", environment=:array)
outlatex(ucb1estvar0, "case_p1/table/table_ucb1estvar0.tex"; format="%.2f", environment=:array)


println("\n ========== Initialization ends ========== \n")