### Adding Folder containing Modules to the path ###
push!(LOAD_PATH, "/home/shubham/Documents/research/columbus/modules/")

### Importing Required Modules###
using sampling
println("Modules Loaded")

### Initializing Data ###
NCols=151
NRows=100000
d = rand(NRows, NCols)

#Last Column is the Expected Output
println("Data Initialised")

### Coreset Sampling ###
data_sampled,sampled = coreset(d)

println("Rows Sampled: ", size(data_sampled,1))