#===============================================================================
Title: Module - Initialization of algorithms' parameters
Updated: 2021-03-29
Description:
Choose initial values for FE and RE parameters of LIME-UCB, and noise variance;
Choose initial values for the mean parameters of UCB-1, and noise variance.
===============================================================================#

#--- (1) Initialization for LIME-UCB

# Run mixed effects model with the estimation data
#fm = @formula(click ~ 1 + u19 + u17 + u16 + zerocorr(1 + u19 + u17 + u16 | display))
fm = @formula(click ~ 1 + u19 + u17 + u16 + (1 + u19 + u17 + u16 | display))
reg_lmem = fit(MixedModel, fm, stream_tun) # 30 seconds

# Initialization based on mixed effects coefficients
# FE parameters
fe_mu0 = reg_lmem.β
fe_var0 = vcov(reg_lmem)
# RE parameters
se = collect(VarCorr(reg_lmem).σρ.display.σ)
corr = collect(VarCorr(reg_lmem).σρ.display.ρ)
#c0 = (VarCorr(fm1).σρ.display.σ) |> collect
#re_var0 = diagm(c0 .^ 2)
re_var0 = diagm(se.^2)
idx = [(i,j) for i in (1+1):p for j in 1:i-1]
for s in 1:(Int(p * (p-1) / 2))
    position = idx[s]
    rowidx = position[1]
    colidx = position[2]
    cov = corr[s] * se[rowidx] * se[colidx]
    re_var0[rowidx,colidx] = cov
    re_var0[colidx,rowidx] = cov
end
# Noise variance
noise_var0 = varest(reg_lmem)
#σ² = 1 # To make sure all algorithms sharing the same noise variance
println("Initialized noise variance for LIME-UCB: ", round.(noise_var0, digits = 4))

# To Latex: FE and RE parameters

#println("Initialized FE: ", round.(fe_mu0, digits = 4))
outlatex(fe_mu0, "table/table_fe_mu0.tex"; format = "%.4f", environment = :array)

#round.(fe_var0,digits = 6) |> display
outlatex(fe_var0, "table/table_fe_var0.tex"; format = "%.6f", environment = :array)

#round.(re_var0,digits = 6) |> display
outlatex(re_var0, "table/table_re_var0.tex"; format = "%.6f", environment = :array)

#display(VarCorr(reg_lmem))
# Correlation matrix of RE coefficients
re_rho0 = diagm(fill(1.0,p))
for s in 1:(Int(p * (p-1) / 2))
    position = idx[s]
    rowidx = position[1]
    colidx = position[2]
    rho = VarCorr(reg_lmem).σρ.display.ρ[s]
    re_rho0[rowidx,colidx] = rho
    re_rho0[colidx,rowidx] = rho
end
outlatex(re_rho0, "table/table_re_rho0.tex"; format = "%.2f", environment = :array)

#--- (2) Initialization for UCB-1

# Compute the empirical mean of reward for the tunning data
ucb1estmu0 = mean(stream_tun.click)
# Estimate the noise variance
ucb1estvar0 = var(stream_tun.click)
println("Initialized mean for UCB-1: ", round.(ucb1estmu0, digits = 4))
println("Initialized variance for UCB-1: ", round.(ucb1estvar0, digits = 4))


# Store initialization results
result_initial = (
    fe_mu0 = fe_mu0,
    fe_var0 = fe_var0,
    re_var0 = re_var0,
    noise_var0 = noise_var0,
    ucb1estmu0 = ucb1estmu0,
    ucb1estvar0 = ucb1estvar0,
)

open("data/result_initial.bin", "w") do io
    serialize(io, result_initial)
end
