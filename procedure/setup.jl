#===============================================================================
Title: Module - Tunning of algorithms' learning rates
Updated: 2021-04-13
Description: 
- Set up multi-threading;
- Specify simulation set up.
===============================================================================#

#--- Working directory
pwd()

#--- Packages
using DataFrames
using DataStructures
using Dates
using Statistics
using LinearAlgebra
using Random
using JDF
using Serialization
using Latexify
using Pipe
using Plots
using StatsPlots
using Plots.PlotMeasures
using GLM
using RegressionTables
using MixedModels
using BenchmarkTools
using ProgressBars
using Distributions


#--- Import algorithms and auxiliary functions
path = readdir("general/"; join = true)
path = string.(pwd(), "/", path)
foreach(include, path[occursin.(r".jl", path)])

#--- Check number of threads at disposal
println("Number of threads: ", Threads.nthreads())

println("\n ========== Set up ends ========== \n")