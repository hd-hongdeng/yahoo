#===============================================================================
Title: Off-line Evaluation
Updated: 2021-03-29
Description: Test the performance of LIME-UCB against other algorithms.
===============================================================================#

#--- 0. Set up

# Working directory
#cd("/Users/hongdeng/OneDrive - Erasmus University Rotterdam/work_file_phd/research/project_lime/limeucb_evaluation_2021/yahoo")
pwd()

# Packages
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
#using .Threads


# Change default setting
ENV["LINE"] = 20
ENV["COLUMNS"] = 20

# Import algorithms and auxiliary functions
subcode = readdir("code_sub/general"; join = true)
foreach(include, subcode[occursin.(r".jl", subcode)])

# Set random seeds
#Random.seed!(112266)

# Check threads
Threads.nthreads()

println("\n ######### Set up ends \n")

#--- Main Procudure
# Note: simulation results have been stored in the data folder.
include("code_sub/procedure.jl")

#--- Analysis
# Note: load simulation results first (independent of the main Ppocudure)
include("code_sub/analysis.jl")
