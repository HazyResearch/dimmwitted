### Adding Folder containing Modules to the path ###
push!(LOAD_PATH, "/home/shubham/Documents/research/columbus/modules/")

### Add processors to parallelize the task ###
addprocs(4)

### Importing Required Modules and making them available to all the cpu cores###
@everywhere using stepdrop_multicore
using julia_ls
println("Modules Loaded")

### Initializing Data ###
NCols=151
NRows=100000
d = rand(NRows, NCols)

#Last Column is the Expected Output
println("Data Initialised")

### Initializing Feature Set for dropping a feature ###
fs = [1:20]

### Dropping Feature using Least Square loss function###
d_new = drop_feature_multicore(least_square, d, fs)

### Output ###
println("features left: ", size(d_new,2))
