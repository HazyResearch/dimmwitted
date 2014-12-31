### Adding Folder containing Modules to the path ###
push!(LOAD_PATH, "/home/shubham/Documents/research/columbus/modules/")

### Add processors to parallelize the task ###
addprocs(4)

### Importing Required Modules and making them available to all the cpu cores###
@everywhere using stepadd_multicore
using julia_ls
println("Modules Loaded")

### Initializing Data ###
NCols=151
NRows=100000
d = rand(NRows, NCols)

#Last Column is the Expected Output
println("Data Initialised")

### Initializing Feature Set for adding a feature: add to fs1, add from fs2 ###
fs1 = [1:20]
fs2 = [30:40]

### Dropping Feature using Least Square loss function###
d_new = add_feature(least_square, d, fs1, fs2)

### Output ###
println("total features: ", size(d_new,2))
