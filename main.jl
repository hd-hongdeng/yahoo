#===============================================================================
Title: Off-line Evaluation
Updated: 2021-04-13
Description: Test the performance of LIME-UCB against other algorithms.
===============================================================================#

#--- Set up
include("procedure/setup.jl")

#--- Initialization
# Note: initialization results have been stored in the result folder.
include("procedure/initialization.jl")

#--- Tunning
# Note: tunning results have been stored in the result folder.
include("procedure/tunning.jl")

#--- Simualtion
# Note: simulation results have been stored in the result folder.
include("procedure/simulation.jl")

#--- Analysis
# Note: load simulation results first (independent of simulation)
include("procedure/analysis.jl")

#--- Test
# Investigate how UCB, LIME update when a new data point comes in.
include("procedure/update.jl")